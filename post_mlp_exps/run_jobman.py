from jobman import api0, sql
from jobman.tools import DD, flatten

from jobman_experiment import experiment
import numpy

NO_OF_TRIALS = 100

def create_jobman_jobs():

    #Database operations
    TABLE_NAME = "arcade_post_mlp_cv_binary_8x8_40k"

    db = api0.open_db('postgresql://gulcehrc@gershwin.iro.umontreal.ca/gulcehrc_db?table=' + TABLE_NAME)

    ri = numpy.random.random_integers

    # Default values
    state = DD()
    state.dataset = \
    "/home/gulcehre/dataset/pentomino/experiment_data/pento64x64_40k_seed_23112222.npy"

    state.no_of_folds = 5
    state.exid = 0

    state.n_hiddens = [100, 200, 300]
    state.n_hidden_layers = 3

    state.learning_rate = 0.001
    state.l1_reg = 1e-5
    state.l2_reg = 1e-3
    state.n_epochs = 2
    state.batch_size = 120
    state.save_exp_data = True
    self.no_of_patches = 64
    state.cost_type = "crossentropy"
    state.n_in = 8*8
    state.n_out = 1

    state.best_valid_error = 0.0

    state.best_test_error = 0.0

    state.valid_obj_path_error = 0.0
    state.test_obj_path_error = 0.0

    l1_reg_values = [0., 1e-6, 1e-5, 1e-4]
    l2_reg_values = [0., 1e-5, 1e-4]

    learning_rates = numpy.logspace(numpy.log10(0.0001), numpy.log10(1), 36)
    num_hiddens = numpy.logspace(numpy.log10(256), numpy.log10(2048), 24)

    for i in xrange(NO_OF_TRIALS):
        state.exid = i
        state.n_hidden_layers = ri(4)
        n_hiddens = []

        for i in xrange(state.n_hidden_layers):
            n_hiddens.append(int(num_hiddens[ri(num_hiddens.shape[0]) - 1]))

        state.n_hiddens = n_hiddens

        state.learning_rate = learning_rates[ri(learning_rates.shape[0]) - 1]
        state.l1_reg = l1_reg_values[ri(len(l1_reg_values)) - 1]
        state.l2_reg = l2_reg_values[ri(len(l2_reg_values)) - 1]
        sql.insert_job(experiment, flatten(state), db)

    db.createView(TABLE_NAME + "_view")

create_jobman_jobs()
