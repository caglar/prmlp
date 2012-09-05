import numpy
import theano

from theano import tensor as T

from hidden import HiddenLayer

from log_reg import *
import pickle as pkl

DEBUGGING = False

class Costs:
    Crossentropy = "crossentropy"
    NegativeLikelihood = "negativelikelihood"

class PosttrainMLP:
    """Post training:- Second phase MLP.
    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self,
            input,
            n_in,
            n_hidden,
            n_out,
            rng=None,
            is_binary=False,
            params_first_phase=None):
        """
        Initialize the parameters for the multilayer perceptron

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

        if DEBUGGING:
            theano.config.compute_test_value = 'raise'
            self.input.tag.test_value = numpy.random.rand(1800, n_in)

        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity
        self.hiddenLayer = HiddenLayer(rng=rng, input=self.input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)
        self.state = "train"
        self.is_binary = is_binary

        self.out_dir = "out/"

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out,
            is_binary=is_binary,
            rng=rng)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the 
        # crossentropy of the output of the model, computed in the
        # logistic regression layer
        if is_binary:
            self.crossentropy = self.logRegressionLayer.crossentropy
        else:
            self.crossentropy = self.logRegressionLayer.crossentropy_categorical

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # Class memberships
        self.class_memberships = self.logRegressionLayer.get_class_memberships(self.hiddenLayer.get_outputs(self.input))

        self.train_pkl_file = "post_train_n_hidden_" + str(n_hidden) + "n_out_" + str(n_out) 

        self.test_pkl_file = "post_test_n_hidden_" + str(n_hidden) + "n_out_" + str(n_out)

        if params_first_phase is not None:
            self.params = params_first_phase

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.data_dict = { 'Ws':[],
                'costs':[],
                'test_probs':[],
                'train_probs':[],
                'test_scores':[]}

    def _shared_dataset(self, data_x, name="x"):
        shared_x = theano.shared(numpy.asarray(list(data_x), dtype=theano.config.floatX))
        shared_x.name = name
        return shared_x

    def save_data(self):
        output = None

        if self.state == "train":
            output = open(self.out_dir + self.train_pkl_file + ".pkl", 'wb')
        else:
            output = open(self.out_dir + self.test_pkl_file + ".pkl", 'wb')

        pkl.dump(self.data_dict, output)

        self.data_dict['Ws'] = []
        self.data_dict['costs'] = []
        self.data_dict['test_scores'] = []


    def posttrain(self,
             learning_rate=0.1,
             L1_reg=0.00,
             L2_reg=0.0001,
             n_epochs=80,
             data=None,
             labels=None,
             cost_type=Costs.Crossentropy,
             save_exp_data=False,
             batch_size=20):

        if data is None:
            raise Exception("Post-training can't start without pretraining class membership probabilities.")

        if labels is None:
            raise Exception("Post-training can not start without posttraining class labels.")

        self.state = "train"

        self.learning_rate = learning_rate
        train_set_x = self._shared_dataset(data, name="training_set")
        train_set_y = labels

        # compute number of minibatches for training
        n_examples = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches = n_examples / batch_size

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '...postraining the model'
        # allocate symbolic variables for the data
        index = T.lscalar('index')    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of int32

        mode = "FAST_COMPILE" #DEBUG_MODE"

        if DEBUGGING:
            self.input.tag.test_value = numpy.random.rand(self)
            index.tag.test_value = 0
            y.tag.test_value = numpy.ones(n_examples)
            mode = "DEBUG_MODE"

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically.

        cost = None

        if cost_type == Costs.NegativeLikelihood:
            cost = self.negative_log_likelihood(y) \
                    + L1_reg * self.L1 \
                    + L2_reg * self.L2_sqr

        elif cost_type == Costs.Crossentropy:
            cost = self.crossentropy(y) \
                    + L1_reg * self.L1 \
                    + L2_reg * self.L2_sqr

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as selfa dictionary
        updates = {}

        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam

        # compiling a Theano function `train_model` that returns the cost, butx
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        # p_y_given_x = self.class_memberships
        train_model = theano.function(inputs=[index],
            outputs=cost,
            updates = updates,
            givens = {
                self.input: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            mode=mode)

        if DEBUGGING:
            theano.printing.debugprint(train_model)

        epoch = 0
        costs = []
        Ws = []

        while (epoch < n_epochs):
            for minibatch_index in xrange(n_train_batches):
                print "Postraining in Minibatch %i " % (minibatch_index)
                minibatch_avg_cost = train_model(minibatch_index)
                costs.append(float(minibatch_avg_cost))
                Ws.append(self.params[2])
            epoch +=1

        if save_exp_data:
            self.data_dict['Ws'].append(Ws)
            self.data_dict['costs'].append([costs])
            self.save_data()
        return costs

    def posttest(self,
             data=None,
             labels=None,
             save_exp_data = False,
             batch_size=20):

        if data is None:
            raise Exception("Post-training can't start without pretraining class membership probabilities.")

        if labels is None:
            raise Exception("Post-training can not start without posttraining class-membership probabilities.")

        test_set_x = shared_dataset(data)
        test_set_y = labels

        self.state = "test"

        # compute number of minibatches for training, validation and testing
        n_examples = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches = n_examples / batch_size

        print '...post-testing the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch 

        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        mode = "FAST_RUN"

        if DEBUGGING:
            theano.config.compute_test_value = 'raise'
            index.tag.test_value = 0
            y.tag.test_value = numpy.ones(n_examples)
            mode = "DEBUG_MODE"

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically

        # compiling a Theano function `test_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`

        test_model = theano.function(inputs=[index],
            outputs=self.errors(y),
            givens={
                self.input: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]},
            mode=mode)

        ###############
        # TEST MODEL  #
        ###############

        test_losses = []

        for minibatch_index in xrange(n_test_batches):
            test_losses.append(float(test_model(minibatch_index)))
            test_score = numpy.mean(test_losses)
            print("Minibatch %i, mean test error %f" % (minibatch_index, test_losses[minibatch_index] * 100))

        print "In the end test score is %f" %(test_score * 100)
        if save_exp_data:
            self.data_dict['test_scores'].append(test_losses)
            self.save_data()

        return test_score, test_losses

