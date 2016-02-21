import theano
import theano.tensor as T

import numpy as np

from util.adagrad import Adagrad
from util.activation import normalized_tanh

class DependencyRNN:
    def __init__(self, d, V, r, nc, embeddings=None):
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
            
        self.params = [self.We, self.Wr, self.Wv, self.b] + [j for i in self.a for j in i]

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
        for i in range(2):
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
        
        #A = theano.shared(name='A',
        #                  value=np.zeros((d, d, nc), dtype=theano.config.floatX))
        p_y_given_x = T.nnet.softmax(T.dot(h[0][-1, -1], A).T.dot(h[1][-1, -1]))

        y_pred = T.argmax(p_y_given_x, axis=1)
        y = T.iscalar('y')
        sentence_nll = -(T.log(p_y_given_x)[0,y])

        #grad = T.grad(sentence_nll, self.params[:4] + [A])
        grad = T.grad(sentence_nll, self.params)

        self.classify = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1]],
                                        outputs=y_pred)

        self.cost_and_grad = theano.function(inputs=[idxs[0], rel_idxs[0], p[0], idxs[1], rel_idxs[1], p[1], y],
                                             outputs=[sentence_nll] + grad)

    def gradient_descent(self, new_gradients):
        self.descender.gradient_descent(*new_gradients)

    #batch consists of tuples of word indices, relation indices, parent indices, and an answer index                                                                                                        
    def train(self, batch):
        total_cost_and_grad = None

        #split data into batches, then into minibatches for multiprocessing                                                                                                                                 

        #TODO: multiprocessing                                                                                                                                                                              
        for datum in batch:
            cost_and_grad = self.cost_and_grad(*datum)
            if total_cost_and_grad is None:
                total_cost_and_grad = [0.] + [np.zeros(i.shape) for i in cost_and_grad[1:]]
            for i in range(len(cost_and_grad)):
                total_cost_and_grad[i] += cost_and_grad[i]

        #update gradients from total_cost_and_grad[1:]                                                                                                                                                      
        self.gradient_descent([i/len(batch) for i in total_cost_and_grad[1:]])

        return total_cost_and_grad[0]/len(batch)

    def metrics(self, test):
        y_true = []
        y_pred = []
        for i,datum in enumerate(test):                                                                                                                                                                     
            try:
                y_pred.append(self.classify(*datum[:-1]))                                                                                                                                                   
            except Exception:
                print(i)
                continue
            y_true.append(datum[-1])
        return precision_recall_fscore_support(y_true, y_pred)

