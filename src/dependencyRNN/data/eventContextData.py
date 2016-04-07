import collections
import os
import gzip
import json
import time

import numpy as np

from scipy.stats import rv_discrete

def countDictToProbList(counts):
    values = probs.values()
    total = sum(values)
    keys = probs.keys()
    for i in range(len(values)):
        values[i] /= total

    return [keys, values]

def makeEvent(predicate, subjects, objects, indirectObjects):
    predicate_index = len(subjects) + len(objects) + len(indirectObjects)
    idxs = []
    rel_idxs = []
    p = []

    for childGroup,relation in ((subjects, 1),
                               (objects, 2),
                               (indirectObjects, 3)):
        for child in childGroup:
            idxs.append(child)
            rel_idxs.append(relation)
            p.append(predicate_index)

    idxs.append(predicate)
    rel_idxs.append(0)
    p.append(predicate_index+1)

    return idxs, rel_idxs, p

class EventContextData:
    def __init__(self, indir):
        self.indir = indir
        #self.processEvents()
        self.predicateCounts = collections.defaultdict(float)
        self.subjectCounts = collections.defaultdict(dict)
        self.objectCounts = collections.defaultdict(dict)
        self.indirectObjectCounts = collections.defaultdict(dict)
        self.argumentCounts = [self.predicateCounts,
                               self.subjectCounts,
                               self.objectCounts,
                               self.indirectObjectCounts]
        
    def iterArticles(self, shuffle=False, verbose=False):
        filenames = sorted(os.listdir(self.indir))
        if shuffle:
            np.shuffle(filenames)

        for filename in filenames:
            if verbose:
                print(filename)
            if filename.startswith('event') and filename.endswith('.json.gz'):
                with gzip.open(os.path.join(self.indir, filename)) as f:
                    batch = json.load(f)

                z = [j for i in batch for j in i]

                if verbose:
                    print(time.time())
                    print(len(z),
                          sum(len(i[1]) > 0 for i in z),
                          sum(len(i[2]) > 0 for i in z),
                          sum(len(i[3]) > 0 for i in z))

                if shuffle:
                    np.shuffle(batch)
                    
                for article in batch:
                    yield article

    def processArgumentProbs(self, verbose=False):

        for article in self.iterArticles(verbose=verbose):
            for event in article:
                predicate, subjects, objects, indirectObjects = event
                self.predicateCounts[predicate] += 1.

                for index,arguments in enumerate((subjects, objects, indirectObjects)):
                    if tuple(arguments) not in self.argumentCounts[index+1][predicate]:
                        self.argumentCounts[index+1][predicate][tuple(arguments)] = 0.
                    self.argumentCounts[index+1][predicate][tuple(arguments)] += 1.

        #normalize
        if verbose:
            print('normalizing...')
            
        self.predicateProbs = countDictToProbList(self.predicateCounts)

        self.argumentProbs = [self.predicateProbs, {}, {}, {}]
                              
        for index,argument in enumerate(self.argumentCounts[1:]):
            for lhs in argument:
                self.argumentProbs[index+1][lhs] = countDictToProbList(argument[lhs])

    def setArgumentProbs(self, predicateProbs, subjectProbs, objectProbs, indirectObjectProbs):
        self.predicateProbs = predicateProbs 
        self.subjectProbs = subjectProbs 
        self.objectProbs = objectProbs
        self.indirectObjectProbs = indirectObjectProbs

        self.argumentProbs = [self.predicateProbs,
                              self.subjectProbs,
                              self.objectProbs,
                              self.indirectObjectProbs]

    def saveArgumentProbs(prefix):
        pass
    
    def iterEventBatches(self, shuffle=False, verbose=False):
        for article in self.iterArticles(shuffle=shuffle, verbose=verbose):
            ret_batch = []
            for e in article:
                ret = makeEvent(*e)
                ret_batch.append(ret)

            yield ret_batch

    def sampleEvent(self):
        if getattr(self, 'predicateDist', None) is None:
            self.predicateDist = rv_discrete(values=self.predicateProbs)
            
        #first sample a predicate
        #predicate = np.random.choice(self.predicateProbs[0],
        #                             p=self.predicateProbs[1])
        #choice = np.random.multinomial(1, self.predicateProbs[1])
        #predicate = self.predicateProbs[0][choice]
        predicate = self.predicateDist.rvs(1)
        
        #then sample all the arguments given the predicate
        arguments = [predicate]
        for i in range(1,4):
            if not len(self.arguments[i][predicate]):
                print(predicate, i)
                argument = tuple()
            elif len(self.arguments[i][predicate][0]) <= 1:
                argument = tuple()
            else:
                choice = np.random.choice(len(self.argumentProbs[i][predicate][0]),
                                          p=self.argumentProbs[i][predicate][1])
                argument = self.arguments[i][predicate][0][choice]
                #choice = np.random.multinomial(1, self.arguments[i][predicate][1])
                #predicate = self.arguments[i][predicate][0][choice]
        
            arguments.append(argument)

        #return arguments
        return makeEvent(*arguments)
        
    def iterEventContext(self, window=2, shuffle=False, verbose=False):
        for eventBatch in self.iterEventBatches(shuffle, verbose):
            for i in range(len(eventBatch)):
                for context in eventBatch[i-window:i+window+1]:
                    yield eventBatch[i] + context
                    
