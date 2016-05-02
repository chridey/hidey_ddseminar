import theano
import theano.tensor as T
import theano.sparse as sp

import numpy as np

from dependencyRNN.util.adagrad import Adagrad
from dependencyRNN.util.activation import normalized_tanh

from sklearn.metrics import precision_recall_fscore_support

class AltlexDiscourseRNN:
    def __init__(self, d, V, r, #nf,
                 embeddings=None,
                 nc=2,
                 nf=0,
                 pairwise_constraint=False,
                 lambda_w=0.01,
                 lambda_e=0.01,
                 lambda_f=0.01,
                 learning_rate='optimal',
                 rnn=True,
                 l1_ratio=0.15,
                 beta=None,
                 fixed_beta=True):

        assert(0 <= l1_ratio <= 1)

        if not rnn:
            print('skipping rnn...')

        #d = dimensionality of embeddings
        #V = size of vocabulary
        #r = number of dependency relations
        #nc = number of classes for classification

        self.learning_rate = learning_rate
        
        #|V| x d embedding matrix
        if embeddings is None:
            self.We = theano.shared(name='embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
        else:
            self.We = theano.shared(name='embeddings',
                                    value=embeddings,
                                    borrow=True
                                    ).astype(theano.config.floatX)
            
        #r x d x d tensor (matrix for each dependency relation)
        self.Wr = theano.shared(name='dependencies',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (r, d, d))
                                ).astype(theano.config.floatX)
        
        #d x d map from embedding to hidden vector
        self.Wv = theano.shared(name='Wv',
                                value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                                ).astype(theano.config.floatX)

        #d long bias vector
        self.b = theano.shared(name='b',
                               value=np.zeros(d, dtype=theano.config.floatX))

        if nc > 2:
            self.gamma = theano.shared(name='gamma', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0, (d, nc))
                                       ).astype(theano.config.floatX)
            if nf > 0:
                #weights for fine grained features plus bias
                self.beta = theano.shared(name='beta', 
                                          value=0.2 * np.random.uniform(-1.0, 1.0, (nf, nc))
                                          ).astype(theano.config.floatX)                
        else:
            self.gamma = theano.shared(name='gamma', 
                                       value=0.2 * np.random.uniform(-1.0, 1.0, (d))
                                       ).astype(theano.config.floatX)
            if nf > 0:
                #weights for fine grained features plus bias
                self.beta = theano.shared(name='beta', 
                                          value=0.2 * np.random.uniform(-1.0, 1.0, (nf))
                                          ).astype(theano.config.floatX)                        

        if nf > 0 and beta is not None:
            self.beta = theano.shared(name='beta', 
                                      value=beta
                                      ).astype(theano.config.floatX)
            
        self.params = []
        if rnn:
            self.params += [self.We, self.Wr, self.Wv, self.b, self.gamma]
        if nf > 0 and (beta is None or not fixed_beta):
            self.params += [self.beta]

        if learning_rate == 'adagrad':
            self.descender = Adagrad(self.params)
            
        #self.f = T.tanh
        self.f = normalized_tanh

        def recurrence(k, hidden_states, hidden_sums, x, r, p, mask):
            #at each node n in the tree, calculate Wr(p,n) \dot f(W_v \dot We_word(n) + b + sum_n) and add to sum_p
            h_k = self.f((T.dot(self.Wv, x[k].T) + hidden_sums[k].T).T + self.b).T*mask[k] #D x N
            sum_k = T.batched_dot(r[k], h_k.T) #N x D
            
            return T.set_subtensor(hidden_states[k],
                                   h_k.T), T.inc_subtensor(hidden_sums[p[k],
                                                                       T.arange(sum_k.shape[0])],
                                                           sum_k)

        y = T.ivector('y')

        #all N x K matrices, where N is batch size and K is max sentence length (padded)
        x_idxs = T.imatrix('x')
        x_parents = T.imatrix('x_parents')
        x_rel_idxs = T.imatrix('x_rel')
        x_mask = T.imatrix('x_mask')

        #now these are K x N x D tensors
        X = self.We[x_idxs.T]

        #these are K x N x D x D tensors
        X_rel = self.Wr[x_rel_idxs.T]
        
        X_hidden_states = T.zeros((x_idxs.shape[1],
                                   x_idxs.shape[0],
                                   d),
                                  dtype=theano.config.floatX)

        X_hidden_sums = T.zeros((x_idxs.shape[1]+1,
                                 x_idxs.shape[0],
                                 d),
                                dtype=theano.config.floatX)

        #these are K(+1) x K x N x D
        [X_h, X_s], updates = theano.scan(fn=recurrence,
                                          sequences=T.arange(x_idxs.shape[1]),
                                          outputs_info=[X_hidden_states, X_hidden_sums],
                                          non_sequences=[X, X_rel, x_parents.T, x_mask.T])

        phi = sp.csc_fmatrix('phi')

        #X_h[-1, -1] is N x D
        base = 0
        if rnn:
            base = base + T.dot(X_h[-1, -1], self.gamma)
        if nf > 0:
            if nc > 2:
                base = base + sp.structured_dot(phi, self.beta)
            else:
                base = base + sp.structured_dot(phi, self.beta.dimshuffle(0, 'x')).flatten()

        if nc > 2:
            p_y_given_x = T.nnet.softmax(base)
            y_pred = T.argmax(p_y_given_x, axis=1)
            costs = -T.log(p_y_given_x)[T.arange(y.shape[0]), y]
        else:
            p_y_given_x = T.nnet.sigmoid(base)
            y_pred = p_y_given_x > 0.5
            costs = -y * T.log(p_y_given_x) - (1-y) * T.log(1-p_y_given_x)

        cost = costs.mean()
        if rnn:
            cost = cost + lambda_w * (self.We ** 2).sum() + lambda_w * (self.Wr ** 2).sum() + lambda_w * (self.Wv ** 2).sum() + lambda_w * (self.b ** 2).sum() + lambda_w * (self.gamma ** 2).sum()
            if pairwise_constraint:
                cost = cost - lambda_e * T.batched_dot(X_h[-1, -1][::2], X_h[-1, -1][1::2]).mean()

        if nf > 0 and (beta is None or not fixed_beta):
            cost = cost + lambda_f*(l1_ratio*T.abs_(self.beta).sum() + (1-l1_ratio) * (self.beta ** 2).sum())
    
        grad = T.grad(cost, self.params)

        if learning_rate == 'optimal':
            def dloss(p, y):
                z = p * y
                if z > 18:
                    return np.exp(-z) * -y
                if z < -18:
                    return -y
                return -y / (np.exp(z) + 1)
            
            typw = np.sqrt(1.0 / np.sqrt(lambda_w))            
            initial_eta0 = typw / max(1.0, dloss(-typw, 1.0))
            optimal_init = 1.0 / (initial_eta0 * lambda_w)
            print(typw, initial_eta0, optimal_init)
            self.t = theano.shared(name='t',
                                   value=0.).astype(theano.config.floatX)
            eta = 1.0 / (lambda_w * (optimal_init + self.t))
            updates = [(p, p - eta*g) for p,g in zip(self.params, grad)]
        else:
            updates = []

        inputs = []
        if rnn:
            inputs += [x_idxs, x_parents, x_rel_idxs, x_mask]
        if nf > 0:
            inputs += [phi]
        inputs += [y]

        self.cost_and_grad = theano.function(inputs=inputs,
                                             outputs=[cost] + grad,
                                             updates=updates,
                                             allow_input_downcast=True)
                                             
        self.sums = theano.function(inputs=[x_idxs, x_parents, x_rel_idxs, x_mask],
                                    outputs=X_s,
                                    allow_input_downcast=True)
        self.states = theano.function(inputs=[x_idxs, x_parents, x_rel_idxs, x_mask],
                                      outputs=X_h,
                                     allow_input_downcast=True)

        self.classify = theano.function(inputs=inputs[:-1],
                                        outputs=y_pred,
                                        allow_input_downcast=True)
        

    def gradient_descent(self, new_gradients):
        self.descender.gradient_descent(*new_gradients)

    #batch consists of tuples of word indices, relation indices, parent indices, and an answer index
    def train(self, batch, lambda_w=1, lambda_a=1, lambda_beta=1, lambda_e=1):
        cost_and_grad = self.cost_and_grad(*batch)                

        if self.learning_rate == 'adagrad':
            #update gradients from cost_and_grad[1:]
            self.gradient_descent([j/len(batch[-1]) for i,j in enumerate(cost_and_grad[1:])])
        if self.learning_rate == 'optimal':
            print(self.t.get_value())
            self.t.set_value(self.t.get_value() + len(batch[-1]))

        return cost_and_grad[0]/len(batch[-1])

    def metrics(self, test):
        y_true = [datum[-1] for datum in test]
        y_pred = self.classify(*test[:-1])

        return precision_recall_fscore_support(y_true, y_pred)

    def save(self, filename):
        #save all the weights and hyperparameters to a file
        kwds = {}
        for param in self.params:
            kwds[param.name] = param.get_value()

        with open(filename, 'wb') as f:
            np.savez(f, **kwds)

    @classmethod
    def load(cls, filename, pairwise_constraint=False):
        with open(filename) as f:
            npzfile = np.load(f)

            d = npzfile['embeddings'].shape[1]
            V = npzfile['embeddings'].shape[0]
            r = npzfile['dependencies'].shape[0]
            #nf = npzfile['beta'].shape[1]
        
            d = cls(d, V, r) #, nf, pairwise_constraint=pairwise_constraint)        
        
            for param in d.params:
                param.set_value(npzfile[param.name])

        return d
