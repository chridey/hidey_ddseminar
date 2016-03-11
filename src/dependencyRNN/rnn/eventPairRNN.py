import theano
import theano.tensor as T
import theano.sparse as sp

import numpy as np

from dependencyRNN.util.adagrad import Adagrad
from dependencyRNN.util.activation import normalized_tanh

from sklearn.metrics import precision_recall_fscore_support

class EventPairRNN:
    def __init__(self, d, V, r, nc, nf,
                 pairwise_constraint=False,
                 embeddings=None,
                 fix_embeddings=False):
        #d = dimensionality of embeddings
        #V = size of vocabulary
        #r = number of dependency relations
        #nc = number of classes for classification
        
        #|V| x d embedding matrix
        if embeddings is None:
            self.We = theano.shared(name='embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
        else:
            self.We = theano.shared(name='embeddings',
                                    value=embeddings
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

        #weights for fine grained features plus bias
        self.beta = theano.shared(name='beta', 
                                  value=0.2 * np.random.uniform(-1.0, 1.0, (nc, nf))
                                  ).astype(theano.config.floatX)
                                  
        #low dimension approximation to classification parameters
        self.a = []
        for i in range(nc):
            a = []
            for j in range(3):
                a.append(theano.shared(name='a_{}_{}'.format(i, j),
                                       value=0.2 * np.random.uniform(-1.0, 1.0, d)
                                       ).astype(theano.config.floatX))
                                       #value=np.zeros(d, dtype=theano.config.floatX)))
            self.a.append(a)

        self.pairwise_constraint = pairwise_constraint

        if fix_embeddings:
            self.params = [self.Wr, self.Wv, self.b] + [j for i in self.a for j in i] + [self.beta]
        else:
            self.params = [self.We, self.Wr, self.Wv, self.b] + [j for i in self.a for j in i] + [self.beta]     

        self.descender = Adagrad(self.params)

        #self.f = T.tanh
        self.f = normalized_tanh

        def recurrence(n, hidden_states, hidden_sums, x, r, p):
            #at each node n in the tree, calculate Wr(p,n) \dot f(W_v \dot We_word(n) + b + sum_n) and add to sum_p
            h_n = self.f(T.dot(self.Wv, x[n]) + self.b + hidden_sums[n])
            sum_n = T.dot(r[n], h_n)
            
            return T.set_subtensor(hidden_states[n], h_n), T.inc_subtensor(hidden_sums[p[n]], sum_n)

        idxs = []
        x = []
        rel_idxs = []
        r = []
        p = []
        hidden_sums = []
        hidden_states = []
        h = []
        s = []
        if pairwise_constraint:
            num_events = 4
        else:
            num_events = 2

        for i in range(num_events):
            idxs.append(T.ivector('idxs'))
            x.append(self.We[idxs[i]])

            rel_idxs.append(T.ivector('rel_idxs'))
            r.append(self.Wr[rel_idxs[i]])

            p.append(T.ivector('parents'))

            hidden_states.append(T.zeros((idxs[i].shape[0], d), dtype=theano.config.floatX))
            #needs to be sent_length + 1 to store final sum
            hidden_sums.append(T.zeros((idxs[i].shape[0]+1, d), dtype=theano.config.floatX))

            h.append(None)
            s.append(None)
            [h[i], s[i]], updates = theano.scan(fn=recurrence,
                                                sequences=T.arange(x[i].shape[0]),
                                                outputs_info=[hidden_states[i], hidden_sums[i]],
                                                non_sequences=[x[i], r[i], p[i]])

        #A = T.dot(self.a_1, self.a_2.reshape((1, d))) + T.nlinalg.diag(self.a_3)
        #cost = T.dot(T.dot(h[0][-1, -1], A), h[1][-1, -1])
        #cost = T.dot(h[0][-1, -1], h[1][-1, -1])
        #grad = T.grad(cost, self.params)
        #self.cost_and_grad = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1]],
        #                                     outputs=[cost] + grad)

        A_stack = []
        for i in range(len(self.a)):
            A_stack.append(T.dot(self.a[i][0].reshape((d, 1)), self.a[i][1].reshape((1, d))) + T.nlinalg.diag(self.a[i][2]))
        A = T.vertical_stack(*A_stack).reshape((d, d, nc))

        self.states = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1]],
                                      outputs=[h[0], h[1]])
        
        #add fine-grained features
        phi = T.vector('phi')

        p_y_given_x = T.nnet.softmax(T.dot(h[0][-1, -1], A).T.dot(h[1][-1, -1]) + T.dot(self.beta, phi))
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        self.classify = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1], phi],
                                        outputs=y_pred)

        y = T.iscalar('y')

        if not pairwise_constraint:        
            sentence_nll = -(T.log(p_y_given_x)[0,y])

            grad = T.grad(sentence_nll, self.params)

        
            self.cost_and_grad = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1], phi, y],
                                                 outputs=[sentence_nll] + grad)
        else:
            lambda_e = T.scalar('lambda_e')

            phi2 = T.vector('phi2')
            p_y_given_x1 = T.nnet.softmax(T.dot(h[0][-1, -1], A).T.dot(h[1][-1, -1]) + T.dot(self.beta, phi))
            p_y_given_x2 = T.nnet.softmax(T.dot(h[2][-1, -1], A).T.dot(h[3][-1, -1]) + T.dot(self.beta, phi2))

            sentence_nll = -(T.log(p_y_given_x1)[0,y]) - (T.log(p_y_given_x2)[0,y])

            #add constraint that events should be maximally similar
            cost = sentence_nll - lambda_e*T.dot(h[0][-1, -1], h[2][-1, -1]) - lambda_e*T.dot(h[1][-1, -1], h[3][-1, -1])

            #grad = T.grad(sentence_nll, self.params[:4] + [A])
            grad = T.grad(cost, self.params)

            self.cost_and_grad = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1], phi, 
                                                         idxs[2], rel_idxs[2], p[2], idxs[3], rel_idxs[3], p[3], phi2,
                                                         y, theano.In(lambda_e, value=1)],
                                                 outputs=[cost] + grad)

    def gradient_descent(self, new_gradients):
        self.descender.gradient_descent(*new_gradients)

    #batch consists of tuples of word indices, relation indices, parent indices, and an answer index
    def train(self, batch, lambda_w=1, lambda_a=1, lambda_beta=1, lambda_e=1):
        total_cost_and_grad = None

        #split data into batches, then into minibatches for multiprocessing
        #TODO: multiprocessing
        
        #lambda_W, lambda_A, lambda_beta
        reg_cost = 0.0
        regularization = []
        for i,j in enumerate(self.params):
            if i < 4:
                lamda = lambda_w
            elif i == len(self.params)-1:
                lamda = lambda_beta
            else:
                lamda = lambda_a

            regularization.append(j.get_value()*lamda)
            reg_cost += 0.5*lamda*np.linalg.norm(j.get_value())
            
        for index,datum in enumerate(batch):
            if self.pairwise_constraint:
                cost_and_grad = self.cost_and_grad(*datum, lambda_e=lambda_e)
            else:
                cost_and_grad = self.cost_and_grad(*datum)                
            if total_cost_and_grad is None:
                total_cost_and_grad = [0.] + [np.zeros(i.shape) for i in cost_and_grad[1:]]
            for i in range(len(cost_and_grad)):
                total_cost_and_grad[i] += cost_and_grad[i]

        if total_cost_and_grad is None:
            return 0
        #update gradients from total_cost_and_grad[1:]
        self.gradient_descent([j/len(batch) + regularization[i] for i,j in enumerate(total_cost_and_grad[1:])])

        return total_cost_and_grad[0]/len(batch) + reg_cost

    def metrics(self, test):
        y_true = []
        y_pred = []
        for i,datum in enumerate(test):
            
            #try:
            y_pred.append(self.classify(*datum[:-1]))
            #except Exception:
            #    print(i)
            #    continue
            y_true.append(datum[-1])
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
            nc = npzfile['beta'].shape[0]
            nf = npzfile['beta'].shape[1]
        
            d = cls(d, V, r, nc, nf, pairwise_constraint=pairwise_constraint)        
        
            for param in d.params:
                param.set_value(npzfile[param.name])

        return d
