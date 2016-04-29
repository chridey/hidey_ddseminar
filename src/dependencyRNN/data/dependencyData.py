import numpy as np

import toposort
      
class DependencyData:
   '''
    class for handling dependency trees for questions and answers
    take in a list of dependency trees, topologically sort them,
    and for each tree return a list of word indices, relation indices, parent node indices, and an answer index
   '''

   def __init__(self):
	self.vocab = {}
	self.relations = {}
	    
   def scan_vocab(self, corpus, train_index=0):
       ret = []
       for datasetIndex,data in enumerate(corpus.datasets):
            ret.append(self.transform(data, True))
       return ret

   def sort_datum(self, datum):

      return toposort.toposort_flatten({i:{j[2]} for i,j in enumerate(datum) if j[2] is not None})

   def match_embeddings(self, model):
      '''
      takes in a word2vec model and return a matrix of embeddings
      
      model - gensim Word2Vec object
      '''

      embeddings = [None for i in range(len(self.vocab))]

      for word in self.vocab:
         if word in model:
            embedding = model[word]
         else:
            try:
               print('Could not find word {} in model'.format(word))
            except Exception:
               pass
            embedding = np.random.uniform(-.2, .2, model.layer1_size)
         embeddings[self.vocab[word]] = embedding

      return np.array(embeddings, dtype=np.float64)

   def stop_indices(self, stop_words):
      '''given a list of stop words, return their indices'''
      
      stop_indices = set()
      for word in stop_words:
         if word in self.vocab:
            stop_indices.add(self.vocab[word])
      return stop_indices
   
   def transform(self, data, initialize=False, minLength=0):
      '''return the indices for the words, relations, parents, and answer'''
      output = []
      for datum,label in data:
          #do topological sort here, also remove nodes that don't have parents (including ROOT)
          #order the nodes left to right with the root as the rightmost node
          indices = self.sort_datum(datum)
          idxs, rel_idxs, p = [], [], []
          parentLookup = {}

          if minLength > len(indices):
             #pad out to the minimum length
             idxs = [0] * minLength-len(indices)
             rel_idxs = [0] * minLength-len(indices)
             p = [0] * minLength-len(indices)             
             length = minLength
             padding = minLength-len(indices)
          else:
             length = len(indices)
             padding = 0
             
          for index in indices:
              if datum[index][2] is None:
                  parentLookup[index] = length-1
                  continue
              if index not in parentLookup:
                  parentLookup[index] = length-len(parentLookup)-1

              word, relation, parent = datum[index]

              if initialize:
                 if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                 if relation not in self.relations:
                    self.relations[relation] = len(self.relations)

              idxs.insert(padding, self.vocab[word])
              rel_idxs.insert(padding, self.relations[relation])
              p.insert(padding, parentLookup[parent])

          output.append((idxs, rel_idxs, p, [1 for i in range(len(idxs))], label))

      return output
