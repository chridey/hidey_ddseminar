import collections

import theano
import theano.tensor as T

class Adagrad:

    def __init__(self, params):
        #params must be a list of Theano shared variables

        self.params = params
        self.eps = 1e-3

        # initial learning rate
        self.learning_rate = 0.05

        # stores sum of squared gradients
        self.h = [theano.shared(value=p.get_value()*0.) for p in self.params]

        gradients = [p.type() for p in self.params]
        gradient_updates = [gradients[i] * self.learning_rate / (T.sqrt(self.h[i] + gradients[i]**2) + self.eps) for i in range(len(gradients))]

        self.updates=collections.OrderedDict((p, p-g) for p,g in zip(self.params,gradient_updates))
        self.updates.update((h, h + g**2) for h,g in zip(self.h,gradients))
        self.gradient_descent = theano.function(inputs=gradients,
                                                updates=self.updates)
                        
    def reset_weights(self):
        for i in self.h:
            i.set_value(i.get_value()*0.)
