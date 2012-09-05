from __future__ import division

import numpy
import theano

from layer import LogisticRegressionLayer, HiddenLayer
from dataset import *

from theano import tensor as T

from utils import as_floatX, safe_update
import pickle as pkl


class Costs:
    Crossentropy = "crossentropy"
    NegativeLikelihood = "negativelikelihood"


class NeuralActivations:
    Tanh = "tanh"
    Logistic = "sigmoid"
    Rectifier = "rectifier"


class MLP(object):

    """
    Multi-Layer Perceptron Class with multiple hidden layers. This class is
    used for pretraining the second phase neural network. Intermediate layers
    have tanh activation function or the sigmoid function (defined here by a
    ``SigmoidalLayer`` class)  while the top layer is a softmax layer (defined
    here by a ``LogisticRegression`` class).
    """
    def __init__(self,
            input,
            n_in=64,
            n_hiddens=[400, 500],
            n_out=11,
            normalize_inputs=False,
            use_adagrad=True,
            activation=NeuralActivations.Rectifier,
            exp_id=1,
            quiet=False,
            rng=None):
        """
        Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType

        :param input: symbolic variable that describes the input of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie.

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

        self.learning_rate = 0.001

        self.exp_id = exp_id

        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity

        self.rng = rng
        self.ds = Dataset()
        self.n_out = n_out
        self.n_in = n_in

        self.n_hiddens = n_hiddens
        self.n_hidden_layers = len(n_hiddens)

        self.hiddenLayers = []
        self.state = "train"

        self.out_dir = "out/"
        self.grads = []
        self.test_scores = []

        #Whether to turn on or off the messages.
        self.quiet = quiet
        self.setup_pkl_paths()
        self.setup_hidden_layers(activation, n_in=n_in, n_out=n_out, n_hiddens=n_hiddens)

        self.use_adagrad = use_adagrad

        #Error for patches with object in it:
        self.obj_patch_error_percent = 0

        self.reset_classified_patches()
        self.setup_patch_save_file()

        self.data_dict = {
            'Ws':[],
            'costs':[],
            'test_probs':[],
            'train_probs':[],
            'test_scores':[]
        }

    def setup_patch_save_file(self,
        path="/RQusagers/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_clean/out/"):

        self.npy_well_classified = path + "_well_classified_patches.npy"
        self.npy_misclassified = path + "_misclassified_patches.npy"

    def setup_pkl_paths(self, name_prefix="premlp_train_"):
        self.train_pkl_file = name_prefix + "n_hidden_" + str(self.n_hiddens) + \
        "_n_out_" + str(self.n_out) + "_learning_rate_" + str(self.learning_rate) + \
        "_exp_id_" + str(self.exp_id)

        self.test_pkl_file = name_prefix + "n_hidden_" + str(self.n_hiddens) + \
        "_n_out_" + str(self.n_out) + "_learning_rate_" + str(self.learning_rate) + \
        "_exp_id_" + str(self.exp_id)

    def setup_hidden_layers(self, activation, n_in=0, n_out=0, n_hiddens=0):
        """
        Setup the hidden layers with the specified number of hidden units.
        """
        act_fn = T.tanh
        if activation == NeuralActivations.Rectifier:
            act_fn = self.rectifier_act

        if n_in == 0:
            n_in = self.n_in
        if n_out == 0:
            n_out = self.n_out
        if n_hiddens == 0:
            n_hiddens = self.n_hiddens

        self.rng.seed(1985)
        #Create the hidden layers.
        self.hiddenLayers.append(HiddenLayer(rng=self.rng,
                input=self.input,
                n_in=n_in,
                n_out=n_hiddens[0],
                activation=act_fn))

        for i in xrange(1, self.n_hidden_layers):
            self.rng.seed(2012)
            self.hiddenLayers.append(HiddenLayer(rng=self.rng,
                                       input=self.hiddenLayers[i-1].output,
                                       n_in=n_hiddens[i-1],
                                       n_out=n_hiddens[i],
                                       activation=act_fn))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegressionLayer(
                input=self.hiddenLayers[-1].output,
                n_in=n_hiddens[-1],
                n_out=n_out,
                rng=self.rng)

        self.initialize_regularization()

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

        self.raw_prediction_errors =\
        self.logRegressionLayer.raw_prediction_errors

        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        # Class memberships
        hidden_outputs = self.hiddenLayers[0].get_outputs(self.input)
        for i in xrange(1, self.n_hidden_layers):
            hidden_outputs = self.hiddenLayers[i].get_outputs(hidden_outputs)

        self.class_memberships = self.logRegressionLayer.get_class_memberships(hidden_outputs)
        self.initialize_params()

    def initialize_params(self):
        # the parameters of the model are the parameters of the all hidden
        # and logistic regression layers it is made out of
        self.params = self.hiddenLayers[0].params

        for i in xrange(1, self.n_hidden_layers):
            self.params += self.hiddenLayers[i].params
        self.params += self.logRegressionLayer.params

    def initialize_regularization(self):
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0

        # square of L2 norm;
        # one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0

        for i in xrange(self.n_hidden_layers):
            self.L1 += abs(self.hiddenLayers[i].W).sum()
            self.L2_sqr += (self.hiddenLayers[i].W ** 2).sum()

        self.L1 += abs(self.logRegressionLayer.W).sum()
        self.L2_sqr += (self.logRegressionLayer.W ** 2).sum()

    def normalize_in_data(self, data):
        return (data.T / data.max(axis=1)).T

    def rectifier_act(self, x, mask=False):
        """
            Activation function for rectifier hidden units.
        """
        if mask:
            activation = numpy.asarray(numpy.sign(numpy.random.uniform(low=-1,
                high=1, size=(10,))), dtype=theano.config.floatX) * T.maximum(x, 0)
        else:
            activation = T.maximum(x, 0)
        return activation

    def do_brain_transfer(self,
                         t_params):
        """
        Copy the weights of one neural network to another one.
        """
        cnt_nonparams = 0
        for i in xrange(len(self.params)):
            if self.params[i].name == "W":
                self.params[i].set_value(t_params[i - cnt_nonparams].get_value())
            else:
                cnt_nonparams += 1

    def sgd_updates(self,
                    cost,
                    learning_rate):
        """
                Using this function, specify how to update the parameters of the model as a dictionary
        """
        updates = {}
        if self.use_adagrad:
                updates = self.sgd_updates_adagrad(cost, learning_rate)
        else:
                for param in self.params:
                    grad = T.grad(cost, param)
                    self.grads.append(grad)
                # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
                # same length, zip generates a list C of same size, where each element
                # is a pair formed from the two lists :
                #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
                for param, grad in zip(self.params, self.grads):
                    updates[param] = param - learning_rate * grad
        return updates

    def sgd_updates_adagrad(self,
                    cost,
                    learning_rate):
        """
        Return the dictionary of parameter specific learning rate updates using adagrad algorithm.
        """
        #Initialize the variables
        accumulators = {}
        e0s = {}
        learn_rates = []
        ups = {}

        #initialize the accumulator and the epsilon_0
        for param in self.params:
                accumulators[param] = theano.shared(value=as_floatX(0.), name="acc_%s" % param.name)
                e0s[param] = as_floatX(learning_rate)

        self.grads = [T.grad(cost, p) for p in self.params]

        #Compute the learning rates
        for param, gp in zip(self.params, self.grads):
                acc = accumulators[param]
                ups[acc] = T.sqrt((gp ** 2).sum())
                learn_rates.append(e0s[param] / ups[acc])

        #Find the updates based on the parameters
        updates = [(p, p - step * gp) for (step, p, gp) in zip(learn_rates,
        self.params, self.grads)]
        p_up = dict(updates)
        safe_update(ups, p_up)
        return ups

    def standardize_weights(self):
        for param in self.params:
            if param.name == "W":
                val = param.get_value(borrow=True)
                if val.mean(0).nonzero()[0].shape[0] > 0:
                    val = (val - val.mean(0))
                if val.std(0).nonzero()[0].shape[0] > 0:
                    val /= val.std(0)
                param.set_value(val)

    def normalize_weights(self):
        """
        TODO: Add the weight constraint for the weight normalization that G.
        Hinton mentions about.
        """
        for param in self.params:
            if param.name == "W":
                val = param.get_value(borrow=True)
                val = self.normalize_in_data(val)
                param.set_value(val)

    def dropout(self, prob=0.5):
        """
        Randomly dropout the hidden units wrt given probability as a binomial
        prob.
        """
        for param in self.params:
            if param.name == "W":
                val = param.get_value(borrow=True)
                #throw a coin for weight:
                dropouts = numpy.random.binomial(1, prob, (val.shape[0], val.shape[1]))
                new_param = param * dropouts
                param.set_value(val)

    def dropout_raw(self, prob=0.5):
        """
        Randomly dropout the hidden units wrt given probability as a binomial
        prob.Avoid using that function, unoptimized and very naive. Just for
        testing.
        """
        for param in self.params:
            if param.name == "W":
                val = param.get_value(borrow=True)
                #throw a coin for weight:
                dropouts = numpy.random.binomial(1, prob, (val.shape[0], val.shape[1]))
                for i in xrange(val.shape[0]):
                    for j in xrange(val.shape[1]):
                        if dropouts[i][j] == 0:
                            val[i][j] = 0
                param.set_value(val)

    def tangent_propagation(self, learning_rate):
        """
        TODO:
        Implement the tangent propagation for invariant recognition.
        """
        return NotImplementedError()

    def get_cost_function(self,
        cost_type,
        y,
        L1_reg,
        L2_reg):
        if cost_type == Costs.Crossentropy:
            if self.n_out == 1:
                cost = self.crossentropy(y) \
                    + L1_reg * self.L1 \
                    + L2_reg * self.L2_sqr
            else:
                cost = self.crossentropy_categorical(y) \
                        + L1_reg * self.L1 \
                        + L2_reg * self.L2_sqr
        elif cost_type == Costs.NegativeLikelihood:
            cost = self.negative_log_likelihood(y) \
                    + L1_reg * self.L1 \
                    + L2_reg * self.L2_sqr
        return cost

    def flip_sign(self,
        val,
        prob=0.1):
        p = 1 - prob
        return -val if numpy.random.rand() > p else val

    def get_inhibitory_mat(self,
        size=None,
        prob=0.1):
        if size is None:
            raise Exception("Please enter a shape value")
        id_mat = numpy.identity(size)
        for i in xrange(size):
            id_mat[i] = self.flip_sign(id_mat[i], prob=prob)
        return id_mat

    def flip_weight_matrices_sign(self, prob=0.1):
        for i in xrange(len(self.params)):
            if self.params[i].name == "W":
                weights = self.params[i].get_value()
                inhibit_mat = self.get_inhibitory_mat(size=weights.shape[1],
                prob=prob)
                weights = numpy.asarray(numpy.dot(weights, inhibit_mat), dtype=theano.config.floatX)
                self.params[i].set_value(weights)

    def record_classified_examples(self,
        patches,
        errors):

#        import ipdb; ipdb.set_trace()

        for i in xrange(errors.shape[0]):
            if errors[i] == 1:
                self.misclassifieds.append(patches[i])
            else:
                self.well_classifieds.append(patches[i])

    def save_classified_patches(self):
        misclassifieds = numpy.asarray(self.misclassifieds, dtype="uint8")
        well_classifieds = numpy.asarray(self.well_classifieds, dtype="uint8")
        numpy.save(self.npy_well_classified, well_classifieds)
        numpy.save(self.npy_misclassified, misclassifieds)

    def save_data(self):
        output = None
        if self.state == "train":
            output = open(self.out_dir + self.train_pkl_file + ".pkl", 'wb')
        else:
            output = open(self.out_dir + self.test_pkl_file + ".pkl", 'wb')
        pkl.dump(self.data_dict, output)
        self.data_dict['Ws'] = []
        self.data_dict['costs'] = []
        self.data_dict['test_probs'] = []
        self.data_dict['train_probs'] = []
        self.data_dict['test_scores'] = []

    def report_object_patch_statistics(self):
        """
        Report the generalization error for the patches containing objects.
        """
        conf_mat = self.logRegressionLayer.conf_mat
        #import ipdb; ipdb.set_trace()
        correct_class = 0
        incorrect_class = 0

        for i in xrange(1, self.n_out):
            for j in xrange(self.n_out):
                if i == j:
                    correct_class += conf_mat[i][j]
                else:
                    incorrect_class += conf_mat[i][j]
        self.obj_patch_error_percent = (correct_class/ (incorrect_class + correct_class)) * 100
        print "Number of incorrectly classified patch objects: " + str(incorrect_class)
        print "Percentage of correctly classified patch objects: " + str(self.obj_patch_error_percent)

    def reset_confusion_mat(self):
        """
        Reset the confusion matrix.
        """
        self.logRegressionLayer.reset_conf_mat()

    def reset_classified_patches(self):
        self.well_classifieds = []
        self.misclassifieds = []
