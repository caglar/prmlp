import numpy
import theano

from theano import tensor as T

DEBUGGING = False

class Layer(object):
    def __init__(self, rng, input, n_in, n_out):
        pass

class HiddenLayer(Layer):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The defauilt nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int

        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer.
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')

        if b is None:
            b_values = numpy.zeros((n_out), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        # parameters of the model
        self.params = [self.W, self.b]
        self.activation = activation
        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if self.activation is None
                else self.activation(lin_output))
        
    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output

class LogisticRegressionLayer(Layer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, is_binary=False, rng=None):
        """ Initialize the parameters of the logistic regression
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture 
        (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which 
        the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """

        self.input = input

        if rng is not None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W_values *= 4
            self.W = theano.shared(value=W_values, name='W')
        else:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                dtype=theano.config.floatX),
                name='W')

        self.is_binary = is_binary

        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
            dtype=theano.config.floatX),
            name='b')

        # The number of classes seen
        self.n_classes_seen = numpy.zeros(n_out)
        # The number of wrong classification made for class i
        self.n_wrong_clasif_made = numpy.zeros(n_out)

        # 
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = self.get_class_memberships(self.input)

        if not is_binary:
            # compute prediction as class whose probability is maximal in
            # symbolic form
            self.y_decision = T.argmax(self.p_y_given_x, axis=1)
        else:
            #If the probability is greater than 0.5 assign to the class 1
            # otherwise it is 0
            self.y_decision = T.gt(self.p_y_given_x, 0.5)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                    \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        if self.is_binary:
            -T.mean(T.log(self.p_y_given_x))
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def crossentropy_categorical(self, y):
        """
        Find the categorical crossentropy.
        """
        #TODO this has a bug, fix it, categorical_crossentropy returns nan values.
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def crossentropy(self, y):
        """
        use the theano nnet cross entropy function. Return the mean.
        """
        return T.mean(T.nnet.binary_crossentropy(T.flatten(self.p_y_given_x), y))

    def get_class_memberships(self, x):
        self.input = x
        lin_activation = T.dot(x, self.W) + self.b
        if self.is_binary:
            """If it is binary return the sigmoid."""
            return T.nnet.sigmoid(lin_activation)

        """
            Else return the softmax class memberships.
        """
        return T.nnet.softmax(lin_activation)

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_decision
        if y.ndim != self.y_decision.ndim:
            raise TypeError('y should have the same shape as self.y_decision',
                    ('y', y.type, 'y_decision', self.y_decision.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_decision, y))
        else:
            raise NotImplementedError()


class Costs:
    Crossentropy = "crossentropy"
    NegativeLikelihood = "negativelikelihood"

class MLP:
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self,
            input,
            n_in=64,
            n_hidden=64,
            n_out=11,
            patch_size=(8, 8),
            learning_rate=0.1,
            normalize_inputs=False,
            L1_reg=0.00,
            L2_reg=0.0001,
            batch_size = 100,
            n_epochs = 80,
            rng=None):

        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which
        the labels lie.
        """
        self.input = input
        if rng == None:
            rng = numpy.random.RandomState(1234)
        self.normalize_inputs = normalize_inputs

        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity
        self.rng = rng
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.out_dir = "out/"

        self.learning_rate = learning_rate
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.test_scores = []

        self.n_in = n_in
        self.pretrain_train_probs = []
        self.pretrain_valid_probs = []
        self.pretrain_test_probs = []

        self.setup_hidden_layers(n_in=n_in, n_out=n_out, n_hidden=n_hidden)

    def normalize_in_data(self, data):
        return (data.T/data.argmax(axis=1)).T

    def setup_hidden_layers(self, n_in=0, n_out=0, n_hidden=0):

        if n_in == 0:
            n_in = self.n_in
        if n_out == 0:
            n_out = self.n_out
        if n_hidden == 0:
            n_hidden = self.n_hidden

        self.hiddenLayer = HiddenLayer(rng=self.rng, input=self.input,
                n_in=n_in, n_out=n_hidden,
                activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionLayer(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out,
                rng=None)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # negative log likelihood of the MLP is given by the 
        # crossentropy of the output of the model, computed in the
        # logistic regression layer
        self.crossentropy = self.logRegressionLayer.crossentropy
        self.crossentropy_categorical = self.logRegressionLayer.crossentropy_categorical

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        # Class memberships
        self.class_memberships = self.logRegressionLayer.get_class_memberships(self.hiddenLayer.get_outputs(self.input))

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

    def _shared_dataset(self, data_x, name="x"):
        shared_x = theano.shared(numpy.asarray(list(data_x),
            dtype=theano.config.floatX))
        shared_x.name = name
        return shared_x


    def train(self,
            data=None,
            labels=None,
            save_costs_to_file=False,
            cost_type=Costs.Crossentropy,
            presences=None):

        train_set_patches = self._shared_dataset(data, name="training_set")
        train_set_pre = T.cast(self._shared_dataset(labels, name="train_labels"), 'int32')


        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_patches.get_value(borrow=True).shape[0] / self.batch_size

        pre_minitrain_probs = numpy.zeros((train_set_patches.get_value(borrow=True).shape[0], self.n_out))

        ######################
        # train the MODEL #
        ######################
        print '... training the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of presences

        # construct the MLP class
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically.

        cost = None
        if cost_type == Costs.Crossentropy:
            cost = self.crossentropy_categorical(y) \
                    + self.L1_reg * self.L1 \
                    + self.L2_reg * self.L2_sqr
        elif cost_type == Costs.NegativeLikelihood:
            cost = self.negative_log_likelihood(y) \
                    + self.L1_reg * self.L1 \
                    + self.L2_reg * self.L2_sqr

        p_y_given_x = self.class_memberships

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}

        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - self.learning_rate * gparam

        # compiling a Theano function `train_model` that returns the cost, butx
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index], outputs=[cost, p_y_given_x],
                updates=updates,
                givens={
                    self.input: train_set_patches[index * self.batch_size:(index + 1) * self.batch_size],
                    y: train_set_pre[index * self.batch_size:(index + 1) * self.batch_size]})

        epoch = 0
        costs = []
        Ws = []
        while (epoch < self.n_epochs):
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost, membership_probs = train_model(minibatch_index)
                costs.append(minibatch_avg_cost)
                Ws.append(self.logRegressionLayer.W.get_value())
                pre_minitrain_probs[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size] = membership_probs
            epoch += 1

        import pudb; pudb.set_trace()

        self.train_train_probs = pre_minitrain_probs
        return costs

    def pretest(self,
            dataset=None,
            save_costs_to_file=False,
            presences=None):

        if dataset is None or presences is None:
            raise Exception("Dataset or presences for training can't be None.")

        test_set_patches, test_set_pre = dataset, presences
        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_patches.get_value(borrow=True).shape[0] / self.batch_size

        pre_minitest_probs = numpy.zeros((test_set_patches.get_value(borrow=True).shape[0], self.n_out))

        ######################
        # Testing  the MODEL #
        ######################
        print '... pre-testing the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of presences

        p_y_given_x = self.class_memberships

        test_model = theano.function(
            inputs=[index],
            outputs=[self.errors(y), p_y_given_x],
            givens={
                self.input: test_set_patches[index * self.batch_size : (index + 1) * self.batch_size],
                y: test_set_pre[index * self.batch_size : (index + 1) * self.batch_size]
            }
        )

        #TODO this is wrong, inputs should be the output of hidden layer, fix it.
        """
        class_memberships = theano.function(inputs=[index], outputs=p_y_given_x,
                givens={
                    self.input: test_set_patches[index * self.batch_size : (index + 1) * self.batch_size]})
        """

        test_losses = []
        test_score = 0

        for minibatch_index in xrange(n_test_batches):
            test_losses, membership_probs = test_model(minibatch_index)
            pre_minitest_probs[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size] = membership_probs
            test_score = numpy.mean(test_losses)
            print("Minibatch %i, mean test error %f" % (minibatch_index, test_score))

        import pudb; pudb.set_trace()

        self.test_scores.append(test_score)

#        class_memberships = theano.function(inputs=[index], outputs=p_y_given_x,
#                givens={
#                    self.input: self.hiddenLayer.output[index * self.batch_size : (index + 1) * self.batch_size]},
#                mode="DEBUG_MODE")
#        data = T.matrix('data')
#        p_y_given_x = self.class_memberships(self.input)
#        class_memberships = theano.function(inputs=[data], outputs=p_y_given_x)
#        pre_minitest_probs = class_memberships(self.hiddenLayer.output)
#        for minibatch_index in xrange(n_test_batches):
#            membership_probs = numpy.array(class_memberships(minibatch_index))
#            pre_minitest_probs[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size] = membership_probs

        self.pretrain_test_probs = pre_minitest_probs
#        if save_costs_to_file:
#            numpy.save(cost_file, test_losses)
        return self.pretrain_test_probs

if __name__ == "__main__":

    x = T.matrix('x')
    ncols = 704
    nrows = 1800
    data = numpy.random.randint(2, size=(nrows, ncols))
    labels = numpy.random.randint(11, size=(nrows))
    mlp = MLP(x, n_in=704, n_out=11, n_hidden=400, batch_size=100)
    mlp.train(data=data, labels=labels)

