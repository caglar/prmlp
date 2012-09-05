from __future__ import division

import numpy
import cPickle as pkl

from pretrained_mlp.prmlp_clean.postmlp import PostMLP
from pretrained_mlp.prmlp_clean.patch_mlp import PatchBasedMLP
from pretrained_mlp.prmlp_clean.mlp import NeuralActivations

from pretrained_mlp.prmlp_clean.dataset import Dataset

import theano.tensor as T
import theano

"""
    This is the code for testing the two-phase architecture with the random
    initialization.
"""
def normalize(mat):
    new_mat = []
    for i in xrange(mat.shape[0]):
        new_mat.append(mat[i]/mat[i].sum())
    return numpy.array(new_mat)

def normalize_data(data):
    """
    Normalize the data with respect to finding the mean and standard deviation of it
    and dividing by mean and standard deviation.
    """
    mu = numpy.mean(data, axis=0)
    sigma = numpy.std(data, axis=0)

    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data

def get_summed_probs(probs, no_of_patches=64, no_of_classes=11):
    return numpy.sum(probs.reshape((probs.shape[0], no_of_patches, no_of_classes)), axis=1)

def shared_train_test_labels(data_train, data_test):
    shared_train = theano.shared(numpy.asarray(list(data_train), dtype=theano.config.floatX).flatten())
    shared_train.name = "train_labels"

    shared_test = theano.shared(numpy.asarray(list(data_test), dtype=theano.config.floatX).flatten())
    shared_test.name = "test_labels"

    return T.cast(shared_train, 'int32'), T.cast(shared_test, 'int32')

def get_binary_labels(labels, no_of_classes=11):
    bin_lbl = []
    for label in labels:
        if label == no_of_classes:
            bin_lbl.append(0)
        else:
            bin_lbl.append(1)
    return numpy.array(bin_lbl)

def load_file(file):
    if file.endswith("pkl"):
        data = pkl.load(open(file))
    elif file.endswith("npy"):
        data=numpy.load(file)
    return data

def pre_training(post_mlp,
    train_probs,
    test_train_probs,
    test_test_probs,
    train_lbls,
    test_lbls):
#    import ipdb; ipdb.set_trace()
    print "Loading the dataset"

    post_cs_args = {
        "train_args":{
         "L1_reg": 1e-6,
         "learning_rate": 0.1,
         "L2_reg": 1e-6,
         "nepochs": 8,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 200,
         "normalize_weights": False,
         "enable_dropout": False
        },
        "test_args":{
         "save_exp_data": False,
         "batch_size": 200
        }
    }

    test_lbls = get_binary_labels(test_lbls)
    train_lbls = get_binary_labels(train_lbls)

    print "Normalizing patch-mlp's outputs"
    train_probs = get_summed_probs(normalize_data(train_probs))
    test_train_probs = get_summed_probs(normalize_data(test_train_probs))
    test_test_probs = get_summed_probs(normalize_data(test_test_probs))

    print "Training post-mlp"
    post_mlp.train(data=train_probs, labels=train_lbls, **post_cs_args["train_args"])

    print "starting post-testing on training dataset"
    post_mlp.test(data=test_train_probs, labels=train_lbls, **post_cs_args["test_args"])

    print "starting post-testing on the  dataset"
    post_mlp.test(data=test_test_probs, labels=test_lbls, **post_cs_args["test_args"])


if __name__=="__main__":
    print "Loading the dataset"

    ds = Dataset()

    x = T.matrix('x')
    no_of_patches = 64
    no_of_classes = 11

    dir = "/RQexec/gulcehre/datasets/pentomino/second_level_ins/"

    train_file = dir + "train_probs_60k_wlbls.pkl"
    train_test_file = dir + "test_ontrain_probs_60k_wlbls.pkl"
    test_file = dir + "test_ontest_probs_60k_wlbls.pkl"

    train_data, train_lbls = load_file(train_file)
    test_train_data = load_file(train_test_file)[0]
    test_test_data, test_lbls = load_file(test_file)

    post_mlp = PostMLP(x, n_in=no_of_classes, n_hiddens=[128],
    activation=NeuralActivations.Rectifier, n_out=1, use_adagrad=False)

    pre_training(post_mlp,
    train_data,
    test_train_data,
    test_test_data,
    train_lbls,
    test_lbls)
