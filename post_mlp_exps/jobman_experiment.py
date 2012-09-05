from kcv import KfoldCrossvalidation
from pretrained_mlp.prmlp_clean.postmlp import PostrainMLP
from pretrained_mlp.prmlp_clean.dataset import Dataset
import theano.tensor as T


"""
    Experiment function to run on jobman.
"""
def experiment(state, channel):

    DS = Dataset(is_binary=True)
    DS.setup_dataset(data_path=state.dataset)

    kfoldCrossValidation = KfoldCrossvalidation(no_of_folds=state.no_of_folds)

    cs_args = {
        "train_args":{
         "L1_reg": state.l1_reg,
         "learning_rate": state.learning_rate,
         "L2_reg": state.l2_reg,
         "nepochs":state.n_epochs,
         "cost_type": state.cost_type,
         "save_exp_data": state.save_exp_data,
         "batch_size": state.batch_size
        },
        "test_args":{
         "save_exp_data": state.save_exp_data,
         "batch_size": state.batch_size
        }
    }

    post_input = T.matrix('post_input')
    mlp = PostMLP(post_input, n_in=state.n_in, n_hiddens=state.n_hiddens,
    n_out=state.n_out, n_hidden_layers=state.n_hidden_layers,
    is_binary=True, exp_id=state.exid)

    valid_errs, test_errs = kfoldCrossValidation.crossvalidate(DS.Xtrain, \
    DS.Ytrain, DS.Xtest, DS.Ytest, mlp, **cs_args)

    errors = \
    kfoldCrossValidation.get_best_valid_scores(valid_errs, test_errs)

    state.best_valid_error = errors["valid_scores"]["error"]

    state.best_test_error = errors["test_scores"]["error"]

    return channel.COMPLETE
