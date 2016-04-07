import theano
import theano.tensor as T
import theano.sparse as sp

import numpy as np

from dependencyRNN.util.adagrad import Adagrad
from dependencyRNN.util.activation import normalized_tanh

from sklearn.metrics import precision_recall_fscore_support

class EventContextRNN:
    def __init__(self, d, V, r,
                 embeddings=None,
                 fix_embeddings=False):
        #d = dimensionality of embeddings
        #V = size of vocabulary
        #r = number of dependency relations
        
        #|V| x d embedding matrix for event and context
        if embeddings is None:
            self.We = theano.shared(name='embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0, (V, d))
                                    ).astype(theano.config.floatX)
        else:
            self.We = theano.shared(name='embeddings',
                                    value=embeddings
                                    ).astype(theano.config.floatX)
            
        #r x d x d tensor (matrix for each dependency relation)
        #one for each of event and context
        self.Wre = theano.shared(name='Wre',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (r, d, d))
                                 ).astype(theano.config.floatX)
        self.Wrc = theano.shared(name='Wrc',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (r, d, d))
                                 ).astype(theano.config.floatX)
        
        #d x d map from embedding to hidden vector
        #one for each of event and context
        self.Wve = theano.shared(name='Wve',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                                 ).astype(theano.config.floatX)
        self.Wvc = theano.shared(name='Wvc',
                                 value=0.2 * np.random.uniform(-1.0, 1.0, (d, d))
                                 ).astype(theano.config.floatX)

        #d long bias vector
        #one for each of event and context
        self.be = theano.shared(name='be',
                               value=np.zeros(d, dtype=theano.config.floatX))
        self.bc = theano.shared(name='bc',
                               value=np.zeros(d, dtype=theano.config.floatX))

        if fix_embeddings:
            self.params = [self.Wre, self.Wrc, self.Wve, self.Wvc, self.be, self.bc]
        else:
            self.params = [self.We, self.Wre, self.Wrc, self.Wve, self.Wvc, self.be, self.bc]

        self.descender = Adagrad(self.params)

        #self.f = T.tanh
        self.f = normalized_tanh

        def event_recurrence(n, hidden_states, hidden_sums, x, r, p):
            #at each node n in the tree, calculate Wr(p,n) \dot f(W_v \dot We_word(n) + b + sum_n) and add to sum_p
            h_n = self.f(T.dot(self.Wve, x[n]) + self.be + hidden_sums[n])
            sum_n = T.dot(r[n], h_n)
            
            return T.set_subtensor(hidden_states[n], h_n), T.inc_subtensor(hidden_sums[p[n]], sum_n)

        def context_recurrence(n, hidden_states, hidden_sums, x, r, p):
            #at each node n in the tree, calculate Wr(p,n) \dot f(W_v \dot We_word(n) + b + sum_n) and add to sum_p
            h_n = self.f(T.dot(self.Wvc, x[n]) + self.bc + hidden_sums[n])
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

        #we have an event, a positive context, and a negative context
        num_contexts = 3
        
        for i in range(num_contexts):
            if i == 0:
                Wr = self.Wre
                recurrence = event_recurrence
            else:
                Wr = self.Wrc
                recurrence = context_recurrence                

            idxs.append(T.ivector('idxs'))
            x.append(self.We[idxs[i]])

            rel_idxs.append(T.ivector('rel_idxs'))
            r.append(Wr[rel_idxs[i]])

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

        cost = T.log(T.nnet.sigmoid(T.dot(h[0][-1, -1], h[1][-1, -1]))) + T.log(T.nnet.sigmoid(T.dot(h[0][-1, -1], -h[2][-1, -1])))
        grad = T.grad(cost, self.params)
        self.cost_and_grad = theano.function(inputs=[idxs[0], rel_idxs[0], p[0],
                                                     idxs[1], rel_idxs[1], p[1],
                                                     idxs[2], rel_idxs[2], p[2]],
                                             outputs=[cost] + grad)
        final_event_state = h[0][-1]
        self.event_state = theano.function(inputs=[idxs[0], rel_idxs[0], p[0]], outputs=final_event_state)
        final_context_state = h[1][-1]
        self.context_state = theano.function(inputs=[idxs[1], rel_idxs[1], p[1]], outputs=final_context_state)

        #these functions should take in the idxs, rel_idxs, and p for the event and 1+negative_samples matrix of these for contexts
        
        #TODO: add supervised examples

    def gradient_descent(self, new_gradients):
        self.descender.gradient_descent(*new_gradients)

    #batch consists of tuples of word indices, relation indices, parent indices, and an answer index
    def train(self, batch, eventSampler):
        total_cost_and_grad = None

        #split data into batches, then into minibatches for multiprocessing
        #TODO: multiprocessing
        
        for index,datum in enumerate(batch):
            negative_sample = eventSampler.sampleEvent()
            datum_plus_sample = datum + negative_sample
            cost_and_grad = self.cost_and_grad(*datum_plus_sample)                
            if total_cost_and_grad is None:
                total_cost_and_grad = [0.] + [np.zeros(i.shape) for i in cost_and_grad[1:]]
            for i in range(len(cost_and_grad)):
                total_cost_and_grad[i] += cost_and_grad[i]

        if total_cost_and_grad is None:
            return 0
        #update gradients from total_cost_and_grad[1:]
        self.gradient_descent([j/len(batch) for i,j in enumerate(total_cost_and_grad[1:])])

        return total_cost_and_grad[0]/len(batch)

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
            r = npzfile['Wre'].shape[0]
        
            d = cls(d, V, r, embeddings=npzfile['embeddings'])
        
            for param in d.params:
                param.set_value(npzfile[param.name])

        return d
