from __future__ import division

import numpy

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

def pre_training_on_multi_datasets(patch_mlp=None, post_mlp=None, train_ds=None, test_ds=None):

    ds = Dataset()
    print "Loading the dataset"

    train_set_patches, train_set_pre, train_set_labels =\
    ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain

    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytrain

    cs_args = {
        "train_args":{
         "L1_reg": 1e-6,
         "learning_rate": 0.75,
         "L2_reg": 1e-5,
         "nepochs": 2,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 250,
         "normalize_weights": False
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 250
        }
    }

    post_cs_args = {
        "train_args":{
         "L1_reg": 1e-6,
         "learning_rate": 0.08,
         "L2_reg": 1e-5,
         "nepochs": 10,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 250,
         "normalize_weights": False
        },
        "test_args":{
         "save_exp_data": False,
         "batch_size": 250
        }
    }

    test_dataset = test_ds[0]

    ds.setup_pretraining_dataset(
        data_path=test_dataset,
        train_split_scale=0.4,
        patch_size=(8,8))
    test_patches, test_pre, test_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytest

    for i in xrange(len(train_ds)):

        print "Current training dataset is %s \n" % (train_ds[i])
        ds.setup_pretraining_dataset(
        data_path=train_ds[i],
        train_split_scale=1.0,
        patch_size=(8,8))

        train_patches, train_pre, train_labels = ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain

        test_labels = get_binary_labels(test_labels)
        train_labels = get_binary_labels(train_labels)

        print "Starting the pretraining phase."
        (costs, pre_train_probs) = prmlp.train(train_patches, train_pre, **cs_args["train_args"])
        prmlp.save_data()

        print "Testing on the training dataset."
        (costs, pre_test_train_probs) = prmlp.test(train_patches, train_pre, **cs_args["test_args"])

        print "Testing on the test dataset."
        (costs, pre_test_test_probs) = prmlp.test(test_patches, test_pre, **cs_args["test_args"])

        print "Normalizing patch-mlp's outputs"
        pre_train_probs = normalize_data(pre_train_probs)
        pre_test_train_probs = normalize_data(pre_test_train_probs)
        pre_test_test_probs = normalize_data(pre_test_test_probs)

        print "Training post-mlp"
        post_mlp.train(data=pre_train_probs, labels=train_labels, **post_cs_args["train_args"])

        print "starting post-testing on training dataset"
        post_mlp.test(data=pre_test_train_probs, labels=train_labels, **post_cs_args["test_args"])

        print "starting post-testing on the  dataset"
        post_mlp.test(data=pre_test_test_probs, labels=test_labels, **post_cs_args["test_args"])


if __name__=="__main__":
    print "Loading the dataset"

    data_train =[
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_seed_7463810918_64patches.npy",
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_64patches_seed_975168712_64patches.npy",
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_seed_23111298122_64patches.npy",
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_seed_392122_64patches.npy",
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_seed_98313722_64patches.npy",
    ]

    data_test = [
        "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_20k_64patches_seed_112168712_64patches.npy"
    ]

    patch_size=(8, 8)

    x = T.matrix('x')
    n_hiddens = [2048]
    no_of_patches = 64
    no_of_classes = 11

    prmlp = PatchBasedMLP(x, n_in=patch_size[0] * patch_size[1],
    n_hiddens=n_hiddens, n_out=11, no_of_patches=no_of_patches,
    activation=NeuralActivations.Rectifier, use_adagrad=False)

    post_mlp = PostMLP(x, n_in=no_of_patches*no_of_classes, n_hiddens=[1024], activation=NeuralActivations.Tanh, n_out=1, use_adagrad=False)
    pre_training_on_multi_datasets(patch_mlp=prmlp, post_mlp=post_mlp, train_ds=data_train, test_ds=data_test)
