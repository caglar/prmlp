from __future__ import division
import math

import numpy
import theano

from dataset import *

from theano import tensor as T

from utils import shared_dataset
from mlp import MLP, Costs, NeuralActivations

import pickle as pkl

class PatchBasedMLP(MLP):

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
            patch_size=(8, 8),
            normalize_inputs=False,
            no_of_patches=64,
            use_adagrad=True,
            activation=NeuralActivations.Rectifier,
            exp_id=1,
            output=0,
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
        :param n_out: number of output units, the dimension of the
        space in which
        the labels lie.
        """

        self.no_of_patches = no_of_patches
        self.patch_size = patch_size
        # If output is zero concatenate the outputs
        # If output is one, sum the outputs of all 
        #patches in the image(mixture of experts).
        # Otherwise take the geometric mean of all the outputs(product of
        # experts).
        self.output = output

        super(PatchBasedMLP, self).__init__(input,
            n_in,
            n_hiddens,
            n_out,
            normalize_inputs,
            use_adagrad,
            activation,
            exp_id,
            quiet=quiet,
            rng=rng)

    def train(self,
            data=None,
            presences=None,
            **kwargs):
        """
        Pretrain the MLP on the patches of images.
        """

        learning_rate = kwargs["learning_rate"]
        L1_reg = kwargs["L1_reg"]
        L2_reg = kwargs["L2_reg"]
        n_epochs = kwargs["nepochs"]
        cost_type = kwargs["cost_type"]
        save_exp_data = kwargs["save_exp_data"]
        batch_size = kwargs["batch_size"]
        normalize_weights = kwargs["normalize_weights"]

        presences = numpy.asarray(presences.tolist(), dtype="uint8")
        self.learning_rate = learning_rate

        # Assign the state of MLP:
        self.state = "train"

        if data is None or presences is None:
            raise Exception("Dataset or presences for pretraining can't be None.")

        if data.shape[0] != presences.shape[0]:
            raise Exception("Dataset and presences shape mismatch.")

        train_set_patches = shared_dataset(data, name="train_set_x")
        train_set_pre = shared_dataset(presences, name="train_set_pre")
        train_set_pre = T.cast(train_set_pre, "int32")

        # compute number of minibatches for training, validation and testing
        n_train_batches = int(math.ceil(data.shape[0] / batch_size))
        if self.output == 1 or self.output == 2:
            pre_train_probs =\
            numpy.zeros((data.shape[0], self.n_out))
        else:
            pre_train_probs =\
            numpy.zeros((data.shape[0], self.n_out * self.no_of_patches))

        ######################
        # Pretrain the MODEL #
        ######################
        print '... pretraining the model'

        # allocate symbolic variables for the data
        index = T.lscalar('index')    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of presences
        pindex = T.lscalar('pindex')

        #construct the MLP class
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically.
        cost = self.get_cost_function(cost_type, y, L1_reg, L2_reg)
        p_y_given_x = self.class_memberships

        updates = self.sgd_updates(cost, learning_rate)

        # compiling a Theano function `train_model` that returns the cost, butx
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index, pindex], outputs=[cost, p_y_given_x],
                updates=updates,
                givens={
                    self.input: train_set_patches[index * batch_size:(index + 1) * batch_size, pindex],
                    y: train_set_pre[index * batch_size:(index + 1) * batch_size, pindex]
                    }
                )

        epoch = 0
        costs = []
        Ws = []

        while (epoch < n_epochs):
            epoch_costs = []
            if normalize_weights:
                if epoch != 0:
                    self.normalize_weights()
            if not self.quiet:
                print "Training epoch %d has started." % (epoch)

            for minibatch_index in xrange(n_train_batches):
                minibatch_costs = []
                for pidx in xrange(self.no_of_patches):
                    minibatch_avg_cost, membership_probs = train_model(minibatch_index, pidx)
                    minibatch_costs.append(float(minibatch_avg_cost.tolist()))
                    if self.output == 1:
                        pre_train_probs[minibatch_index * batch_size:\
                        (minibatch_index + 1) * batch_size] += membership_probs
                    if self.output == 2:
                        pre_train_probs[minibatch_index * batch_size:\
                        (minibatch_index + 1) * batch_size] *= 10 * membership_probs
                    else:
                        pre_train_probs[minibatch_index * batch_size: (minibatch_index + 1) * batch_size, pidx * self.n_out: (pidx + 1) * self.n_out] = membership_probs
                if self.output == 2:
                    pre_train_probs = numpy.sqrt(pre_train_probs)

                Ws.append(self.params[2])
                epoch_costs.append(minibatch_costs)

            costs.append(epoch_costs)
            if not self.quiet:
                print "Normalizing the weights"
            epoch += 1

        self.data_dict['costs'].append([costs])
        self.data_dict['train_probs'].append(pre_train_probs)
        return costs, pre_train_probs

    def test(self,
            dataset=None,
            presences=None,
            **kwargs):
        """
        Test the mlp on the given dataset with the presences.
        """
        save_costs_to_file = kwargs["save_exp_data"]
        batch_size = kwargs["batch_size"]
        save_patch_examples = False

        if kwargs.has_key("save_classified_patches"):
            save_patch_examples = kwargs["save_classified_patches"]

        if dataset is None or presences is None:
            raise Exception("Dataset or presences for pretraining can't be None.")

        self.state = "test"
        test_set_patches = shared_dataset(dataset, name="test_set_x")
        presences = numpy.asarray(presences.tolist(), dtype="int32")
        test_set_pre = shared_dataset(presences, name="test_set_pre")
        test_set_pre = T.cast(test_set_pre, 'int32')

        # compute number of minibatches for training, validation and testing
        n_test_batches = int(math.ceil(dataset.shape[0] / batch_size))

        if self.output == 1 or self.output == 2:
            pre_minitest_probs = numpy.zeros((dataset.shape[0], self.n_out))
        else:
            pre_minitest_probs = numpy.zeros((dataset.shape[0], self.n_out * self.no_of_patches))

        ######################
        # Testing the MODEL. #
        ######################
        print '... pre-testing the model'

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        y = T.ivector('y')  # the labels are presented as 1D vector of presences
        pindex = T.lscalar('pindex')

        p_y_given_x = self.class_memberships

        if save_patch_examples:
            test_model = theano.function(
                inputs=[index, pindex],
                outputs=[self.errors(y), p_y_given_x, self.raw_prediction_errors(y)],
                givens={
                    self.input: test_set_patches[index * batch_size: (index + 1) * batch_size, pindex],
                    y: test_set_pre[index * batch_size: (index + 1) * batch_size, pindex]
                }
            )
        else:
            test_model = theano.function(
                inputs=[index, pindex],
                outputs=[self.errors(y), p_y_given_x],
                givens={
                    self.input: test_set_patches[index * batch_size: (index + 1) * batch_size, pindex],
                    y: test_set_pre[index * batch_size: (index + 1) * batch_size, pindex]
                }
            )

        test_losses = []
        test_score = 0

        for minibatch_index in xrange(n_test_batches):
            for pidx in xrange(self.no_of_patches):
                if save_patch_examples:
                    test_loss, membership_probs, raw_errors = test_model(minibatch_index, pidx)
                    patches = dataset[minibatch_index * batch_size: (minibatch_index + 1) * batch_size, pidx]
                    self.record_classified_examples(patches, raw_errors)
                else:
                    test_loss, membership_probs = test_model(minibatch_index, pidx)

                test_losses.append(test_loss)
                test_score = numpy.mean(test_loss)
                pre_batch_vals = presences[minibatch_index * batch_size:\
                    (minibatch_index + 1) * batch_size, pidx]
                if self.output == 1:
                    pre_minitest_probs[minibatch_index * batch_size:\
                    (minibatch_index + 1) * batch_size] +=\
                    membership_probs
                if self.output == 2:
                    pre_minitest_probs[minibatch_index * batch_size:\
                    (minibatch_index + 1) * batch_size] *=\
                    10 * membership_probs
                else:
                    pre_minitest_probs[minibatch_index * batch_size:\
                    (minibatch_index + 1) * batch_size, pidx * self.n_out:\
                    (pidx + 1) * self.n_out] = membership_probs

                self.logRegressionLayer.update_conf_mat(pre_batch_vals, membership_probs)

                if not self.quiet:
                    print("Minibatch %i and its test error %f percent on patch %i" % (minibatch_index, test_score * 100, pidx))
            if self.output == 2:
                pre_minitest_probs = numpy.sqrt(pre_minitest_probs)

        self.save_classified_patches()

        print "Confusion matrix:"
        print self.logRegressionLayer.conf_mat

        self.report_object_patch_statistics()

        fin_test_score = numpy.mean(test_losses)

        print("In the end final test score on whole image is %f\n" % (fin_test_score * 100))

        self.data_dict['test_scores'].append(test_losses)
        self.data_dict['test_probs'].append(pre_minitest_probs)

        return fin_test_score, pre_minitest_probs

if __name__=="__main__":

    print "Loading the dataset"
    ds = Dataset()
    patch_size = (16, 16)

    ds.setup_pretraining_dataset(data_path="/RQusagers/gulcehre/dataset/pentomino/rnd_pieces/pento64x64_20k_seed_39112222_16patches_rnd.npy",
    patch_size=patch_size, normalize_inputs=False)
    print "Dataset successfully loaded to memory"

    cs_args = {
        "train_args": {
         "L1_reg": 1e-6,
         "learning_rate": 0.05,
         "L2_reg": 1e-5,
         "nepochs": 2,
         "cost_type": "negativelikelihood",
         "save_exp_data": False,
         "batch_size": 100,
         "normalize_weights": False
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 100
        }
    }

    print "Starting the cross-validation"

    x = T.matrix('x')

    n_hiddens = [2048, 2048]

    train_set_patches, train_set_pre = ds.Xtrain_patches, ds.Xtrain_presences
    test_set_patches, test_set_pre = ds.Xtest_patches, ds.Xtest_presences

    prmlp = PatchBasedMLP(x, n_in=16*16, n_hiddens=n_hiddens, n_out=11,
    no_of_patches=16, activation=NeuralActivations.Rectifier, use_adagrad=False)

    prmlp.train(train_set_patches, train_set_pre, **cs_args["train_args"])
    prmlp.save_data()

    print "Testing on the training dataset."
    prmlp.test(train_set_patches, train_set_pre, **cs_args["test_args"])

    print "Testing on the test dataset."
    prmlp.test(test_set_patches, test_set_pre, **cs_args["test_args"])

#   valid_errs, valid_patch_errs, test_errs, test_patch_errs = \
#   kfoldCrossvalidation.crossvalidate_prmlp(train_set_patches, train_set_pre, \
#   test_set_patches, test_set_pre, prmlp, **cs_args)
