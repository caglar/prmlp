import math

import numpy
import theano

from theano import tensor as T

from utils import shared_dataset

from layer import HiddenLayer, LogisticRegressionLayer
import pickle as pkl

from mlp import MLP, Costs, NeuralActivations

DEBUGGING = False

class PostMLP(MLP):
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
            n_in=64*11,
            n_hiddens=[500, 400],
            n_out=1,
            normalize_inputs=False,
            use_adagrad=True,
            activation=NeuralActivations.Rectifier,
            exp_id=1,
            rng=None,
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
        if DEBUGGING:
            theano.config.compute_test_value = 'raise'
            self.input.tag.test_value = numpy.random.rand(1800, n_in)

        super(PostMLP, self).__init__(input,
            n_in,
            n_hiddens,
            n_out,
            normalize_inputs,
            use_adagrad,
            activation,
            exp_id,
            rng)

        self.params_first_phase = params_first_phase

    def train(self,
             data=None,
             labels=None,
             **kwargs):

        learning_rate = kwargs["learning_rate"]
        L1_reg = kwargs["L1_reg"]
        L2_reg = kwargs["L2_reg"]
        n_epochs = kwargs["nepochs"]
        cost_type = kwargs["cost_type"]
        save_exp_data = kwargs["save_exp_data"]
        batch_size = kwargs["batch_size"]
        normalize_weights = kwargs["normalize_weights"]
        enable_dropout = kwargs["enable_dropout"]

        if data is None:
            raise Exception("Post-training can't start without pretraining class membership probabilities.")

        if labels is None:
            raise Exception("Post-training can not start without posttraining class labels.")

        self.state = "train"

        self.learning_rate = learning_rate

        train_set_x = shared_dataset(data, name="training_set_x")
        train_set_y = shared_dataset(labels, name="labels")
        train_set_y = T.cast(train_set_y, "int32")

        # compute number of minibatches for training
        n_examples = data.shape[0]
        n_train_batches = int(math.ceil(n_examples / batch_size))

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '...postraining the model'
        # allocate symbolic variables for the data
        index = T.lscalar('index')    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of int32

        mode = "FAST_RUN"
        #import pudb; pudb.set_trace()
        if DEBUGGING:
            index.tag.test_value = 0
            y.tag.test_value = numpy.ones(n_examples)
            mode = "DEBUG_MODE"

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically.
        cost = self.get_cost_function(cost_type, y, L1_reg, L2_reg)
        updates = self.sgd_updates(cost, learning_rate)

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
            print "In da epoch %d" % (epoch)
            for minibatch_index in xrange(n_train_batches):
                print "Postraining in Minibatch %i " % (minibatch_index)
                minibatch_avg_cost = train_model(minibatch_index)
                if enable_dropout:
                    self.dropout()

                if normalize_weights:
                    self.normalize_weights()

                costs.append(float(minibatch_avg_cost))
                Ws.append(self.params[2])
            epoch +=1

        if save_exp_data:
            self.data_dict['Ws'].append(Ws)
            self.data_dict['costs'].append([costs])
            self.save_data()
        return costs

    def test(self,
             data=None,
             labels=None,
             **kwargs):

        save_exp_data = kwargs["save_exp_data"]
        batch_size = kwargs["batch_size"]

        if data is None:
            raise Exception("Post-training can't start without pretraining class membership probabilities.")

        if labels is None:
            raise Exception("Post-training can not start without posttraining class-membership probabilities.")

        test_set_x = shared_dataset(data)
        test_set_y = shared_dataset(labels)
        test_set_y = T.cast(test_set_y, "int32")

        self.state = "test"

        # compute number of minibatches for training, validation and testing
        n_examples = data.shape[0]
        n_test_batches = int(math.ceil(n_examples / batch_size))

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
            print("Minibatch %i, mean test error %f" % (minibatch_index, test_score * 100))

        if save_exp_data:
            self.data_dict['test_scores'].append(test_losses)
            self.save_data()

        return test_score, test_losses
