import theano
import theano.tensor as T
import theano.sparse as sp

import numpy as np

from dependencyRNN.util.activation import normalized_tanh

from dependencyRNN.util.adagrad import Adagrad

from sklearn.metrics import precision_recall_fscore_support

class EventContextRNN:
    def __init__(self, d, V, r,
                 padding=5,
                 embeddings=None):
        
        #d = dimensionality of embeddings
        #V = size of vocabulary
        #r = number of dependency relations

        self.d = d
        
        #|V| x d embedding matrix for event and context
        if embeddings is None:
            self.Wx = theano.shared(name='Wx',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
        else:
            self.Wx = theano.shared(name='Wx',
                                    value=embeddings,
                                    borrow=True,
                                    ).astype(theano.config.floatX)

        #d x d map from embedding to hidden vector
        #one for each of event and context
        self.Wve = theano.shared(name='Wve',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                                 ).astype(theano.config.floatX)
        self.Wvc = theano.shared(name='Wvc',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                                 ).astype(theano.config.floatX)

        #r x d x d tensor (matrix for each dependency relation)
        #one for each of event and context
        self.Wre = theano.shared(name='Wre',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (r, d, d))
                                 ).astype(theano.config.floatX)
        self.Wrc = theano.shared(name='Wrc',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (r, d, d))
                                 ).astype(theano.config.floatX)

        self.params = [self.Wx, self.Wve, self.Wvc, self.Wre, self.Wrc]

        x_idxs = T.ivector('x')
        c_idxs = T.ivector('c')
        n_idxs = T.ivector('n')

        x_child_idxs = T.ivector('x_child')
        c_child_idxs = T.ivector('c_child')
        n_child_idxs = T.ivector('n_child')        

        x_rel_idxs = T.ivector('x_rel')
        c_rel_idxs = T.ivector('c_rel')
        n_rel_idxs = T.ivector('n_rel')        

        x_mask = T.ivector('x_mask')
        c_mask = T.ivector('c_mask')
        n_mask = T.ivector('n_mask')

        x_child_mask = T.ivector('x_child_mask')
        c_child_mask = T.ivector('c_child_mask')
        n_child_mask = T.ivector('n_child_mask')
        
        X = self.Wx[x_idxs]
        C = self.Wx[c_idxs]
        N = self.Wx[n_idxs]

        X_child = self.Wx[x_child_idxs].T*x_child_mask
        C_child = self.Wx[c_child_idxs].T*c_child_mask
        N_child = self.Wx[n_child_idxs].T*n_child_mask

        X_rels = self.Wre[x_rel_idxs]
        C_rels = self.Wrc[c_rel_idxs]
        N_rels = self.Wrc[n_rel_idxs]
            
        def normalized_tanh(x):
            s = T.tanh(x)
            return s/((s**2).sum())

        #self.f = normalized_tanh
        self.f = T.tanh

        H_x_child = self.f(T.dot(self.Wve, X_child))
        H_c_child = self.f(T.dot(self.Wvc, C_child))
        H_n_child = self.f(T.dot(self.Wvc, N_child))

        H_x_sum = T.batched_dot(X_rels, H_x_child.T).reshape((x_idxs.shape[0],
                                                            x_child_idxs.shape[0] // x_idxs.shape[0],
                                                            d)).sum(axis=1)
        H_c_sum = T.batched_dot(C_rels, H_c_child.T).reshape((c_idxs.shape[0],
                                                            c_child_idxs.shape[0] // c_idxs.shape[0],
                                                            d)).sum(axis=1)
        H_n_sum = T.batched_dot(N_rels, H_n_child.T).reshape((n_idxs.shape[0],
                                                            n_child_idxs.shape[0] // n_idxs.shape[0],
                                                            d)).sum(axis=1)
        
        H_x = self.f(T.dot(self.Wve, X.T) + H_x_sum.T)
        H_c = self.f(T.dot(self.Wvc, C.T) + H_c_sum.T)
        H_n = self.f(T.dot(self.Wvc, N.T) + H_n_sum.T)
        
        cost = -T.mean(T.log(T.nnet.sigmoid(T.batched_dot(H_x.T, H_c.T))) + \
                       T.log(T.nnet.sigmoid(-T.batched_dot(H_x.T, H_n.T))))
        
        grad = T.grad(cost, self.params)

        learning_rate = T.scalar('learning_rate')
        updates = [(self.params[i], self.params[i] - learning_rate * grad[i]) for i in range(len(self.params))]

        #adagrad = Adagrad(self.params)

        #updates = adagrad.updates

        self.train = theano.function(inputs = [x_idxs, x_child_idxs, x_rel_idxs, x_child_mask,
                                               c_idxs, c_child_idxs, c_rel_idxs, c_child_mask,
                                               n_idxs, n_child_idxs, n_rel_idxs, n_child_mask,
                                               learning_rate],
                                     outputs = cost,
                                     updates = updates,
                                     allow_input_downcast=True)

        self.cost = theano.function(inputs = [x_idxs, x_child_idxs, x_rel_idxs, x_child_mask,
                                              c_idxs, c_child_idxs, c_rel_idxs, c_child_mask,
                                              n_idxs, n_child_idxs, n_rel_idxs, n_child_mask],
                                    outputs = cost,
                                    allow_input_downcast=True)

        self.event_states = theano.function(inputs = [x_idxs, x_child_idxs, x_rel_idxs, x_child_mask],
                                            outputs = H_x,
                                            allow_input_downcast=True)

        self.context_states = theano.function(inputs = [c_idxs, c_child_idxs, c_rel_idxs, c_child_mask],
                                              outputs = H_c,
                                              allow_input_downcast=True)
                                                      

    def save(self, prefix):
        #save all the weights and hyperparameters to a file
        kwds = {}
        for param in self.params:
            np.save(prefix + param.name + '.npy', param.get_value())

    @classmethod
    def load(cls, prefix):
        params = {}
        for paramName in ['Wx', 'Wve', 'Wvc', 'Wre', 'Wrc']:
            params[paramName] = np.load(prefix + paramName + '.npy')

        d = params['Wx'].shape[1]
        V = params['Wx'].shape[0]
        r = params['Wrc'].shape[0]

        d = cls(d, V, r, embeddings=params['Wx'])

        for param in d.params:
            param.set_value(params[param.name])

        return d

    
        
