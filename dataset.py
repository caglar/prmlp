from __future__ import division

import pickle as pkl
import math
import numpy as np

from utils import get_dataset_patches, get_dataset_obj_patches

class Dataset(object):

    def __init__(self, is_binary=False):

        self.is_binary = is_binary
        #Examples
        self.Xtrain = None
        self.Xtrain_presences = None
        self.Xtrain_patches = None

        self.Xtest = None
        self.Xtest_presences = None
        self.Xtest_patches = None

        #Labels
        self.Ytrain = None
        self.Ytest = None

        self.sparsity = 0.0
        self.n_examples = 0

    def _get_data(self, data_path):
        if data_path.endswith("pkl") or data_path.endswith("pickle"):
            data = pkl.load(open(data_path, "rb"))
        else:
            data = np.load(data_path)
        return data

    def _normalize_data(self, data):
        """
        Normalize the data with respect to finding the mean and standard deviation of it
        and dividing by mean and standard deviation.
        """

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        norm_data = (data - mu) / sigma
        return norm_data

    def binarize_labels(self, labels=None):
        """
        Convert the labels into the binary format for the second phase task.
        """

        #Largest label is for the images without different object.
        last_lbl = np.max(labels)
        binarized_lbls = []
        if self.is_binary:
            for label in labels:
                if label == last_lbl:
                    binarized_lbls.append(0)
                else:
                    binarized_lbls.append(1)
        return binarized_lbls

    def setup_dataset(self,
            data_path=None,
            train_split_scale = 0.8):
        """
        Setup the dataset given the data path with respect to the split scale.
        """

        data = self._get_data(data_path)

        self.n_examples = data[0].shape[0]
        ntrain = math.floor(self.n_examples * train_split_scale)

        self.Xtrain = data[0][:ntrain]
        self.Xtest = data[0][ntrain:]

        self.Ytrain = np.array(self.binarize_labels(data[1][:ntrain].flatten()) \
        if self.is_binary else data[1][:ntrain].flatten())

        self.Ytest = np.array(self.binarize_labels(data[1][ntrain:].flatten()) \
        if self.is_binary else data[1][ntrain:].flatten())

    def setup_pretraining_dataset(self,
        data_path=None,
        train_split_scale=0.8,
        patch_size=(8, 8),
        normalize_inputs=False):
        """
        Load the pretraining datasets for the pretraining neural networks.
        """

        data = self._get_data(data_path)
        self.n_examples = data[0].shape[0]
        ntrain = math.floor(self.n_examples * train_split_scale)

        self.Xtrain = data[0][:ntrain]
        self.Xtest = data[0][ntrain:]

        if normalize_inputs:
            self.Xtrain = self._normalize_data(self.Xtrain)
            self.Xtest = self._normalize_data(self.Xtest)

        self.Ytrain = np.array(self.binarize_labels(data[1][:ntrain].flatten()) \
        if self.is_binary else data[1][:ntrain].flatten())
        self.Ytrain += 1

        self.Ytest = np.array(self.binarize_labels(data[1][ntrain:].flatten()) \
        if self.is_binary else data[1][ntrain:].flatten())
        self.Ytest += 1

        self.Xtrain_presences = data[2][:ntrain]
        self.Xtrain_presences += 1

        self.Xtrain_patches = get_dataset_patches(self.Xtrain, patch_size=patch_size)

        self.Xtest_presences = data[2][ntrain:]
        self.Xtest_presences += 1
        self.Xtest_patches = get_dataset_patches(self.Xtest, patch_size=patch_size)

    def setup_pretraining_obj_patch_dataset(self,
            data_path=None,
            train_split_scale=0.8,
            patch_size=(8, 8),
            normalize_inputs=False):
        """
        Load the pretraining datasets for the pretraining neural networks.
        """

        data = self._get_data(data_path)
        self.n_examples = data[0].shape[0]
        ntrain = math.floor(self.n_examples * train_split_scale)

        self.Xtrain = data[0][:ntrain]
        self.Xtest = data[0][ntrain:]

        if normalize_inputs:
            self.Xtrain = self._normalize_data(self.Xtrain)
            self.Xtest = self._normalize_data(self.Xtest)

        self.Ytrain = np.array(self.binarize_labels(data[1][:ntrain].flatten()) \
        if self.is_binary else data[1][:ntrain].flatten())
        self.Ytrain += 1

        self.Ytest = np.array(self.binarize_labels(data[1][ntrain:].flatten()) \
        if self.is_binary else data[1][ntrain:].flatten())
        self.Ytest += 1

        self.Xtrain_presences = data[2][:ntrain]
        self.Xtrain_presences += 1

        self.Xtrain_patches, self.Xtrain_presences = get_dataset_obj_patches(self.Xtrain, self.Xtrain_presences, patch_size=patch_size)

        self.Xtest_presences = data[2][ntrain:]
        self.Xtest_presences += 1
        self.Xtest_patches, self.Xtest_presences = get_dataset_obj_patches(self.Xtest, self.Xtest_presences, patch_size=patch_size)

    def comp_sparsity(self):
        """
        Compute the sparsity level for the dataset.
        """

        num_sparse_els = 0
        for el in self.Xtrain.flatten():
            if el == 0:
                num_sparse_els+=1
        for el in self.Xtest.flatten():
            if el == 0:
                num_sparse_els+=1
        self.sparsity = (num_sparse_els/self.n_examples)
        return self.sparsity

