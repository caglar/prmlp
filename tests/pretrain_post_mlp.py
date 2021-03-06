from __future__ import division

import numpy

from postmlp import PostMLP
from patch_mlp import PatchBasedMLP
from mlp import NeuralActivations

from dataset import Dataset

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

def shared_train_test_labels(data_traintest):
    data_train, data_test = data_traintest
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

def pre_training(patch_mlp=None, post_mlp=None, ds=None):
    print "Loading the dataset"
    patch_size=(16,16)

    train_set_patches, train_set_pre, train_set_labels = ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain
    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytrain

    cs_args = {
        "train_args":{
         "L1_reg": 1e-6,
         "learning_rate": 0.025,
         "L2_reg": 1e-5,
         "nepochs": 8,
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
         "L1_reg": 1e-6,
         "learning_rate": 0.1,
         "L2_reg": 1e-5,
         "nepochs": 8,
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
    pre_train_probs = normalize(pre_train_probs)
    pre_test_train_probs = normalize(pre_test_train_probs)
    pre_test_test_probs = normalize(pre_test_test_probs)

    post_mlp.train(data=pre_train_probs, labels=train_set_labels, **cs_args["train_args"])

    print "starting post-testing on training dataset"
    post_mlp.test(data=pre_test_train_probs, labels=train_set_labels, **cs_args["test_args"])

    print "starting post-testing on the  dataset"
    post_mlp.test(data=pre_test_test_probs, labels=test_set_labels, **cs_args["test_args"])


if __name__=="__main__":
    print "Loading the dataset"
    ds = Dataset()
    patch_size=(16,16)
    ds.setup_pretraining_dataset(data_path="/RQusagers/gulcehre/dataset/pentomino/rnd_pieces/pento64x64_40k_seed_39112222_16patches_rnd.npy", patch_size=patch_size, normalize_inputs=False)
    x = T.matrix('x')
    n_hiddens = [1024, 768]

    prmlp = PatchBasedMLP(x, n_in=16*16, n_hiddens=n_hiddens, n_out=11,
    no_of_patches=16, activation=NeuralActivations.Rectifier, use_adagrad=False)

    post_mlp = PostMLP(x, n_in=16*11, n_hiddens=[512, 256], n_out=1, use_adagrad=False)
    pre_training(patch_mlp=prmlp, post_mlp=post_mlp, ds=ds)
