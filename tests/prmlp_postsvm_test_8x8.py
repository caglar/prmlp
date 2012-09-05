from __future__ import division

import numpy

from pretrained_mlp.prmlp_clean.patch_mlp import PatchBasedMLP
from pretrained_mlp.prmlp_clean.mlp import NeuralActivations
from pretrained_mlp.prmlp_clean.svm import CSVM

from pretrained_mlp.prmlp_clean.dataset import Dataset

import theano.tensor as T
import theano

def normalize(mat):
    """
    This is the code for testing the two-phase architecture with the random 
    initialization.
    """
    new_mat = []
    for i in xrange(mat.shape[0]):
        new_mat.append(mat[i]/mat[i].sum())
    return numpy.array(new_mat)

def shared_train_test_labels(data_traintest):
    data_train, data_test = data_traintest
    shared_train = theano.shared(numpy.asarray(list(data_train), dtype=theano.config.floatX).flatten())
    shared_train.name = "train_labels"

    shared_test = theano.shared(numpy.asarray(list(data_test), dtype=theano.config.floatX).flatten())
    shared_test.name = "test_labels"
    return T.cast(shared_train, 'int32'), T.cast(shared_test, 'int32')

def normalize_data(data):
    """
    Normalize the data with respect to finding the mean and standard deviation of it
    and dividing by mean and standard deviation.
    """
    mu = numpy.mean(data, axis=0)
    sigma = numpy.std(data, axis=0)
    norm_data = (data - mu) / sigma
    return norm_data

def get_binary_labels(labels, no_of_classes=11):
    bin_lbl = []
    for label in labels:
        if label == no_of_classes:
            bin_lbl.append(0)
        else:
            bin_lbl.append(1)
    return numpy.array(bin_lbl)

def pre_training(patch_mlp=None, csvm=None, ds=None):
    print "Loading the dataset"
    patch_size=(16,16)

    train_set_patches, train_set_pre, train_set_labels =\
    ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain

    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytrain

    cs_args = {
        "train_args":{
         "L1_reg": 1e-6,
         "learning_rate": 0.75,
         "L2_reg": 1e-7,
         "nepochs": 2,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 100,
         "normalize_weights": False
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 100
        }
    }

    post_cs_args = {
        "train_args":{
         "kern":"rbf",
         "gamma": 0.01,
         "C": 10
        },
        "test_args":{
         "binary_data":True
        }
    }

    test_set_labels = get_binary_labels(test_set_labels)
    train_set_labels = get_binary_labels(train_set_labels)

    print "Starting the pretraining phase."
    (costs, pre_train_probs) = prmlp.train(train_set_patches, train_set_pre, **cs_args["train_args"])
    prmlp.save_data()

    print "Testing on the training dataset."
    (costs, pre_test_train_probs) = prmlp.test(train_set_patches, train_set_pre, **cs_args["test_args"])

    print "Testing on the test dataset."
    (costs, pre_test_test_probs) = prmlp.test(test_set_patches, test_set_pre, **cs_args["test_args"])

    print "Normalizing patch-mlp's outputs"
    pre_train_probs = normalize_data(pre_train_probs)
    pre_test_train_probs = normalize_data(pre_test_train_probs)
    pre_test_test_probs = normalize_data(pre_test_test_probs)

#    import ipdb; ipdb.set_trace()

    csvm.train(pre_train_probs, train_set_labels, **post_cs_args["train_args"])
    print "starting post-testing on training dataset"
    train_error = csvm.test(pre_test_train_probs, train_set_labels, **post_cs_args["test_args"])
    print "For training %s" %(train_error)

    print "starting post-testing on the  dataset"
    test_error = csvm.test(pre_test_test_probs, test_set_labels, **post_cs_args["test_args"])
    print "For testing %s" %(test_error)

#    import ipdb; ipdb.set_trace()

if __name__=="__main__":
    print "Task has just started."
    print "Loading the dataset"
    ds = Dataset()
    patch_size=(8,8)

    ds_path = \
    "/RQusagers/gulcehre/dataset/pentomino/experiment_data/pento64x64_80k_seed_39112222.npy"
    data_new =\
    "/RQusagers/gulcehre/dataset/pentomino/rnd_pieces/pento64x64_5k_seed_43112222_64patches_rnd.npy"

    data_new_40k =\
    "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_64patches_seed_975168712_64patches.npy"

    ds.setup_pretraining_dataset(data_path=data_new_40k, patch_size=patch_size, normalize_inputs=False)
    pre_input = T.matrix('pre_input')
    n_hiddens = [2048]

    prmlp = PatchBasedMLP(pre_input, n_in=8*8, n_hiddens=n_hiddens, n_out=11,
    no_of_patches=64, activation=NeuralActivations.Rectifier, use_adagrad=False)

    csvm = CSVM()
    pre_training(prmlp, csvm, ds)
