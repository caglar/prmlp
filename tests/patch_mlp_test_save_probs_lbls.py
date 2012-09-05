from __future__ import division

import theano.tensor as T

from  pretrained_mlp.prmlp_clean.dataset import *

from pretrained_mlp.prmlp_clean.patch_mlp import PatchBasedMLP
from pretrained_mlp.prmlp_clean.mlp import NeuralActivations
import cPickle as pkl

def save_probs(data, pkl_file):
    filehandler = open(pkl_file, "wb")
    pkl.dump(data, filehandler)

if __name__ == "__main__":
    print "Loading the dataset"
    ds = Dataset()
    patch_size = (8, 8)
    no_of_patches = 64

    filename =\
    "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/pento64x64_40k_64patches_seed_975168712_64patches.npy"
    ds.setup_pretraining_dataset(data_path=filename, patch_size=patch_size, normalize_inputs=False)

    dir = "/RQexec/gulcehre/datasets/pentomino/second_level_ins/"

    train_file = dir + "train_probs_40k_wlbls.pkl"
    train_test_file = dir + "test_ontrain_probs_40k_wlbls.pkl"
    test_file = dir + "test_ontest_probs_40k_wlbls.pkl"

    print "Dataset successfully loaded to memory"

    cs_args = {
        "train_args": {
         "L1_reg": 1e-6,
         "learning_rate": 0.75,
         "L2_reg": 1e-5,
         "nepochs": 2,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 200,
         "normalize_weights": False
        },
        "test_args": {
         "save_exp_data": False,
         "batch_size": 200
        }
    }

    print "Starting the cross-validation"

    x = T.matrix('x')

    n_hiddens = [2048]

    train_set_patches, train_set_pre, train_set_labels = ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain
    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytest

    prmlp = PatchBasedMLP(x, n_in=patch_size[0] * patch_size[1],
    n_hiddens=n_hiddens, n_out=11, no_of_patches=no_of_patches,
    activation=NeuralActivations.Rectifier, use_adagrad=False, quiet=True)

    costs, pretrain_probs = prmlp.train(train_set_patches, train_set_pre, **cs_args["train_args"])

    save_probs((pretrain_probs, train_set_labels), train_file)

    print "Testing on the training dataset."
    fin_test_score, post_test_train_probs = prmlp.test(train_set_patches, train_set_pre, **cs_args["test_args"])
    save_probs((post_test_train_probs, train_set_labels), train_test_file)

    print "Testing on the test dataset."
    fin_test_score, post_test_probs = prmlp.test(test_set_patches, test_set_pre, **cs_args["test_args"])
    save_probs((post_test_probs, test_set_labels), test_file)

