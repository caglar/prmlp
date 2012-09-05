import numpy

from prmlp import PreMLP
from post_mlp import PosttrainMLP

from dataset import Dataset

import theano.tensor as T
import theano

def train_prmlp(prmlp, data, presences, no_of_patches=64):
    pre_train_probs = None
    for i in xrange(no_of_patches):
        shared_data = theano.shared(data.get_value()[:, i])
        shared_presences = T.cast(theano.shared(presences.get_value()[:, i]), 'int32')
        prmlp.pretrain(dataset=shared_data, presences=shared_presences)
        if pre_train_probs == None:
            pre_train_probs = prmlp.pretrain_train_probs
        else:
            pre_train_probs = numpy.column_stack((pre_train_probs, prmlp.pretrain_train_probs))
#    import pudb; pudb.set_trace()
    return pre_train_probs

def test_prmlp(prmlp, data, presences, no_of_patches=64):
    pre_test_probs = None
    test_scores = []
    for i in xrange(no_of_patches):
        shared_data = theano.shared(data.get_value()[:, i])
        shared_presences = T.cast(theano.shared(presences.get_value()[:, i]), 'int32')
        prmlp.pretest(dataset=shared_data, presences=shared_presences)
        test_scores.append(numpy.mean(prmlp.test_scores))
        if pre_test_probs == None:
            pre_test_probs = prmlp.pretrain_test_probs
        else:
            pre_test_probs = numpy.column_stack((pre_test_probs, prmlp.pretrain_test_probs))
    print "In the end the test score is %f " % (numpy.mean(test_scores))
    return pre_test_probs

if __name__=="__main__":

    x = T.matrix('x')
    dataset = "/u/gulcehrc/pento64x64_simple_2k_seed_312555.npy"

    ds = Dataset()

    print "starting pretrain"

    ds.setup_pretraining_dataset(data_path=dataset)

    train_set_patches, train_set_pre, train_set_labels = ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain
    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytest
    

    prmlp = PreMLP(x, n_epochs=2)
    post_mlp = PosttrainMLP(x, n_in=64*11, n_hidden=400, n_out=10)

    print "starting pre-training"
    pre_train_probs = train_prmlp(prmlp, train_set_patches, train_set_pre)

    print "starting the pre-testing"
    pre_test_probs = test_prmlp(prmlp, test_set_patches, test_set_pre)
    
    print "starting post-training"
    post_mlp.posttrain(learning_rate=0.001, data=pre_train_probs, n_epochs=4, labels=train_set_labels, batch_size=60, save_costs_file=True, cost_type="negativelikelihood")

    print "starting post-testing"
    post_mlp.posttest(data=pre_test_probs, labels=test_set_labels, save_costs_file=True)
