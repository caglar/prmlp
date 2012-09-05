import numpy

from prmlp import PreMLP
from post_mlp import PosttrainMLP

from dataset import Dataset

import theano.tensor as T
import theano

def train_prmlp(prmlps, data, presences):
    pre_train_probs = None
    for i in xrange(len(prmlps)):
        shared_data = theano.shared(data.get_value()[:, i])
        shared_presences = T.cast(theano.shared(presences.get_value()[:, i]), 'int32')
        prmlps[i].pretrain(dataset=shared_data, presences=shared_presences)
        
        if pre_train_probs == None:
            pre_train_probs = prmlps[i].pretrain_train_probs[0]
        else:
            pre_train_probs = numpy.column_stack((pre_train_probs, prmlps[i].pretrain_train_probs[0]))
#    import pudb; pudb.set_trace()
    return pre_train_probs 

def test_prmlp(prmlps, data, presences):
    pre_test_probs = None
    test_scores = []
    for i in xrange(len(prmlps)):
        shared_data = theano.shared(data.get_value()[:, i])
        shared_presences = T.cast(theano.shared(presences.get_value()[:, i]), 'int32')
        prmlps[i].pretest(dataset=shared_data, presences=shared_presences)
        test_scores.append(numpy.mean(prmlps[i].test_scores))
        if pre_test_probs == None:
            pre_test_probs = prmlps[i].pretrain_test_probs
        else:
            pre_test_probs = numpy.column_stack((pre_test_probs, prmlps[i].pretrain_test_probs))
    print "In the end the test score is %f " % (numpy.mean(test_scores))
    return pre_test_probs

if __name__=="__main__":
    x = T.matrix('x')
    dataset = "/data/lisa/data/pentomino/pentomino64x64_4k_pre.npy"
    ds = Dataset()

    print "starting pretrain"

    ds.setup_pretraining_dataset(data_path=dataset)
    
    train_set_patches, train_set_pre, train_set_labels = ds.Xtrain_patches, ds.Xtrain_presences, ds.Ytrain
    test_set_patches, test_set_pre, test_set_labels = ds.Xtest_patches, ds.Xtest_presences, ds.Ytest

    prmlps = [PreMLP(x) for each in xrange(64)]
    
    print "starting pretest"
    pre_train_probs = train_prmlp(prmlps, train_set_patches, train_set_pre)
    pre_test_probs = test_prmlp(prmlps, test_set_patches, test_set_pre)

    post_mlp = PosttrainMLP(x, n_in=64*11, n_hidden=200, n_out=10)

    post_mlp.posttrain(data=pre_train_probs, labels=train_set_labels, batch_size=80)
    post_mlp.posttest(data=pre_test_probs, labels=test_set_labels)

