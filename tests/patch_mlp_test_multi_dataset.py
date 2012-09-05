from __future__ import division

import numpy

from pretrained_mlp.prmlp_clean.patch_mlp import PatchBasedMLP

from pretrained_mlp.prmlp_clean.dataset import Dataset

import theano.tensor as T
import theano

data_dir =\
    "/RQusagers/gulcehre/dataset/pentomino/pieces/"

file_suffix = ".npy"

def train_prmlp(prmlp, dataset, presences, **kwargs):
    pre_train_probs = None
    costs, pre_train_probs = prmlp.train(data=dataset, presences=presences, **kwargs)
    return costs

def test_prmlp(prmlp, dataset, presences, no_of_patches=64, **kwargs):
    test_scores = prmlp.test(dataset=dataset, presences=presences, **kwargs)
    prmlp.reset_confusion_mat()
    print "Testing finished!"
    return test_scores

def incremental_data_experiment(prmlp,
    train_datasets,
    test_datasets,
    no_of_patches=64,
    patch_size=(8, 8),
    **kwargs):

    ds_train = Dataset()
    ds_test = Dataset()

    costs = []
    test_scores = []

    test_ds_name = data_dir + test_datasets[0] + file_suffix
    print "Loading the test dataset"
    ds_test.setup_pretraining_dataset(data_path=test_ds_name,
    train_split_scale=0.5, patch_size=patch_size, normalize_inputs=False)

    """
    Perform the test on test dataset for each learnt training dataset.
    """
    for x_t_idx in xrange(len(train_datasets)):
        train_ds_name = data_dir + train_datasets[x_t_idx] + file_suffix
        print "Loading the dataset %s " % (train_ds_name)

        ds_train.setup_pretraining_dataset(data_path=train_ds_name,
        train_split_scale=1, patch_size=patch_size, normalize_inputs=False)
        print "Training on the dataset %d " % (x_t_idx)
        cost = train_prmlp(prmlp, ds_train.Xtrain_patches, ds_train.Xtrain_presences, **kwargs["train_args"])
        costs.append(cost)

        print "Testing on the test dataset."
        test_scores_per_ds = test_prmlp(prmlp, ds_test.Xtest_patches, ds_test.Xtest_presences, no_of_patches, **kwargs["test_args"])
        test_score_patch_based = prmlp.obj_patch_error_percent
        test_scores.append(test_score_patch_based)

    all_data_dict = {
        "test_scores": test_scores,
        "costs": costs
    }

    numpy.save("/RQusagers/gulcehre/codes/python/experiments/arcade_ds_exps/pretrained_mlp/prmlp_multi_datasets/out/multi_hidden_mlp_240k_lrate_0.025_3hidden_8x8_1epoch.npy", all_data_dict)

if __name__=="__main__":
    pre_input = T.matrix('pre_input')
    train_datasets = ["pento64x64_40k_seed_12981222",
                      "pento64x64_40k_seed_23112222",
                      "pento64x64_40k_seed_39112222",
                      "pento64x64_40k_seed_39123122",
                      "pento64x64_40k_seed_51161222",
                      "pento64x64_40k_seed_83711222",
                      "pento64x64_40k_seed_91712222"]

    test_datasets = ["pento64x64_40k_seed_43112222"]
    patch_size = (8, 8)

    cs_args = {
        "train_args":{
         "L1_reg": 1e-06,
         "learning_rate": 0.05,
         "L2_reg": 1e-05,
         "nepochs": 1,
         "cost_type": "crossentropy",
         "save_exp_data": False,
         "batch_size": 200,
         "normalize_weights": False
        },
        "test_args":{
         "save_exp_data":False,
         "batch_size": 200
        }
    }

    no_of_patches = 64
    print "starting pretrain"

    prmlp = PatchBasedMLP(pre_input,
    n_in=patch_size[0] * patch_size[1],
    n_hiddens=[2048, 2048],
    n_out=11,
    patch_size=patch_size)

    incremental_data_experiment(prmlp, train_datasets, test_datasets, no_of_patches=no_of_patches,
    patch_size=patch_size, **cs_args)
