from __future__ import division

import logging
import numpy as np
from sklearn import svm

from learning_algo import LearningAlgorithm
from error import ErrorMeasure

import pickle as pkl
import math

class Dataset(object):

    def __init__(self, is_binary=False):
        self.is_binary = is_binary

        #Examples
        self.Xtrain = None
        self.Xtest = None

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

    def binarize_labels(self, labels=None):
        last_lbl = np.max(labels)
        binarized_lbls = []
        if self.is_binary:
            for label in labels:
                if label == last_lbl:
                    binarized_lbls.append(0)
                else:
                    binarized_lbls.append(1)
        return binarized_lbls

    def setup_dataset(self, data_path=None, train_split_scale = 0.8):
        data = self._get_data(data_path)
        self.n_examples = data[0].shape[0]
        ntrain = math.floor(self.n_examples * train_split_scale)

        self.Xtrain = data[0][:ntrain]
        self.Xtest = data[0][ntrain:]

        self.Ytrain = np.array(self.binarize_labels(data[1][:ntrain].flatten()) \
        if self.is_binary else data[1][:ntrain].flatten())

        self.Ytest = np.array(self.binarize_labels(data[1][ntrain:].flatten()) \
        if self.is_binary else data[1][ntrain:].flatten())

    def comp_sparsity(self):
        num_sparse_els = 0
        for el in self.Xtrain.flatten():
            if el == 0:
                num_sparse_els+=1
        for el in self.Xtest.flatten():
            if el == 0:
                num_sparse_els+=1
        self.sparsity = (num_sparse_els/self.n_examples)
        return self.sparsity

class CSVM(LearningAlgorithm):

    def __init__(self):
        self.clf = None

    def train(self, Xtrain, Ytrain, **kwargs):
        print "Training on data has started"
        kern = kwargs["kern"]
        gamma = kwargs["gamma"]
        C = kwargs["C"]

        kern = kern.encode("ascii", "ignore")

        self.clf = svm.SVC(kernel=kern, gamma=gamma, C=C)
        self.clf.fit(list(Xtrain), list(Ytrain))

    def test(self, Xtest, Ytest, **kwargs):
        print "Testing on data has started"

        is_binary_data = kwargs["binary_data"]

        error_comp = ErrorMeasure(binary_measure=is_binary_data)
        predictions = self.clf.predict(list(Xtest))

        for i, ex in enumerate(Xtest):

            error_comp.add_new_measure(Ytest[i], predictions[i])

        return error_comp

    def get_logger(self):
        logger = logging.getLogger('Crossvalidation')
        logger.setLevel(logging.INFO)

        # create file handler which logs even debug messages
        fh = logging.FileHandler('svm_tetromino_crossvalidation.log')
        fh.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

if __name__=="__main__":
    from kcv import KfoldCrossvalidation

    DS = Dataset(is_binary=True)
    DS.setup_dataset(data_path="/home/gulcehre/dataset/pentomino/pieces/pento64x64_4k_task4_seed_98981222.npy")
    kfoldCrossValidation = KfoldCrossvalidation(no_of_folds=2)

    cs_args = {
        "train_args":{
         "kern":"rbf",
         "gamma": 0.01,
         "C": 10
        },
        "test_args":{
         "binary_data":True
        }
    }

    csvm = CSVM()
    valid_errs, test_errs = kfoldCrossValidation.crossvalidate(DS.Xtrain,
    DS.Ytrain, DS.Xtest, DS.Ytest, csvm, **cs_args)

