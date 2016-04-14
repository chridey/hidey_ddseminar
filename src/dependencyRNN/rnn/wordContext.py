import theano
import theano.tensor as T
import theano.sparse as sp

import numpy as np

from dependencyRNN.util.activation import normalized_tanh

from sklearn.metrics import precision_recall_fscore_support

class WordContext:
    def __init__(self, d, V,
                 embeddings=None):
        
        #d = dimensionality of embeddings
        #V = size of vocabulary
        
        #|V| x d embedding matrix for event and context
        if embeddings is None:
            self.Wx = theano.shared(name='word_embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
            self.Wc = theano.shared(name='context_embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
        else:
            self.Wx = theano.shared(name='embeddings',
                                    value=embeddings
                                    ).astype(theano.config.floatX)
            self.Wc = theano.shared(name='embeddings',
                                    value=embeddings
                                    ).astype(theano.config.floatX)
        
        self.params = [self.Wx, self.Wc]

        x_idxs = T.ivector('x')
        c_idxs = T.ivector('c')
        n_idxs = T.ivector('n')
        
        X = self.Wx[x_idxs]
        C = self.Wc[c_idxs]
        N = self.Wc[n_idxs]

        #cost = -T.mean(T.log(T.nnet.sigmoid(T.dot(X, C.T))) + T.log(T.nnet.sigmoid(-T.dot(X,N.T))))
        cost = -T.mean(T.log(T.nnet.sigmoid(T.batched_dot(X, C))) + \
                       T.log(T.nnet.sigmoid(-T.batched_dot(X, N))))
        
        grad = T.grad(cost, self.params)

        learning_rate = T.scalar('learning_rate')
        updates = [(self.params[i], self.params[i] - learning_rate * grad[i]) for i in range(len(self.params))]

        self.train = theano.function(inputs = [x_idxs, c_idxs, n_idxs, learning_rate],
                                     outputs = cost,
                                     updates = updates,
                                     allow_input_downcast=True)

        self.cost = theano.function(inputs = [x_idxs, c_idxs, n_idxs, learning_rate],
                                    outputs = cost,
                                    allow_input_downcast=True)
