from __future__ import division
import theano.tensor as T

from  pretrained_mlp.prmlp_clean.dataset import *

from pretrained_mlp.prmlp_clean.patch_mlp import PatchBasedMLP
from pretrained_mlp.prmlp_clean.mlp import NeuralActivations

if __name__ == "__main__":
    print "Loading the dataset"
    ds = Dataset()
    patch_size = (8, 8)
    no_of_patches = 64

    ds.setup_pretraining_dataset(data_path="/RQusagers/gulcehre/dataset/pentomino/experiment_data/pento64x64_5k_seed_23112222.npy",
    patch_size=patch_size,
    normalize_inputs=False)

    print "Dataset successfully loaded to memory"

    cs_args = {
        "train_args": {
         "L1_reg": 1e-6,
         "learning_rate": 0.2,
         "L2_reg": 1e-5,
         "nepochs": 1,
         "cost_type": "negativelikelihood",
         "save_exp_data": False,
         "batch_size": 100,
         "normalize_weights": False
        },
        "test_args": {
         "save_exp_data": False,
         "batch_size": 100,
         "save_classified_patches": True
        }
    }

    print "Starting the cross-validation"

    x = T.matrix('x')

    n_hiddens = [2048]

    train_set_patches, train_set_pre = ds.Xtrain_patches, ds.Xtrain_presences
    test_set_patches, test_set_pre = ds.Xtest_patches, ds.Xtest_presences

    prmlp = PatchBasedMLP(x,
    n_in=patch_size[0] * patch_size[1],
    n_hiddens=n_hiddens,
    n_out=11,
    no_of_patches=no_of_patches,
    activation=NeuralActivations.Rectifier,
    use_adagrad=False,
    quiet=True)

    prmlp.train(train_set_patches, train_set_pre, **cs_args["train_args"])
    prmlp.save_data()

    print "Testing on the training dataset."
    prmlp.test(train_set_patches, train_set_pre, **cs_args["test_args"])

    print "Testing on the test dataset."
    prmlp.test(test_set_patches, test_set_pre, **cs_args["test_args"])
