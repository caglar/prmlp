import theano
import utils

#Adagrad annealing:
base_learning_rate = 0

base_lr = 0.1

params = [] #Parameters of the model.

accumulators = {}
e0s = {}

cost = CrossEntropy

for param in self.params:
    self.accumulators[param] = theano.shared(value=as_floatX(0.), name="acc_%s" % param.name)
    self.e0s[param] = as_floatX(base_lr)

learn_rates = []
ups = {}

grads = [tensor.grad(cost, p) for p in self.params]

for param, gp in zip(self.params, grads):
        acc = accumulators[param]
        ups[acc] = acc + (gp ** 2).sum()
        learn_rates.append(e0s[param] / (ups[acc] ** .5))


updates = [(p, p - step * gp) for (step, p, gp) in zip(learn_rates, params, grads)]

p_up = dict(updates)

safe_update(ups, p_up)
