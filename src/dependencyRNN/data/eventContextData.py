import collections
import os
import gzip
import json
import time

import numpy as np

from scipy.stats import rv_discrete

def countDictToProbList(counts):
    values = counts.values()
    total = sum(values)
    keys = counts.keys()
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

def makeEvent(predicate, subjects, objects, indirectObjects, padding=1):
    parent = [predicate]
    children = []
    relations = []
    mask = []
    maxLength = max([len(subjects), len(objects), len(indirectObjects)])
    padding = max(maxLength, padding)
    for index,arguments in enumerate((subjects, objects, indirectObjects)):
        children += list(arguments)
        children += [0]*(padding-len(arguments))
        relations += [index]*padding
        mask += ([1]*len(arguments) + [0]*(padding-len(arguments)))
    return parent, children, relations, mask

def padEvent(event, padding):
    parent,children,relations,mask = event
    orig = len(children) // 3

    diff = padding-orig
    children = children[:orig] + [0]*diff + children[orig:2*orig] + [0]*diff + children[2*orig:] + [0]*diff
    relations = relations[:orig] + [0]*diff + relations[orig:2*orig] + [0]*diff + relations[2*orig:] + [0]*diff
    mask = mask[:orig] + [0]*diff + mask[orig:2*orig] + [0]*diff + mask[2*orig:] + [0]*diff
    
    return parent, children, relations, mask
    
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
            np.random.shuffle(filenames)

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
                    np.random.shuffle(batch)
                    
                for article in batch:
                    yield article

    def processArgumentCounts(self, verbose=False):

        for article in self.iterArticles(verbose=verbose):
            for event in article:
                predicate, subjects, objects, indirectObjects = event
                self.predicateCounts[predicate] += 1.

                for index,arguments in enumerate((subjects, objects, indirectObjects)):
                    if tuple(arguments) not in self.argumentCounts[index+1][predicate]:
                        self.argumentCounts[index+1][predicate][tuple(arguments)] = 0.
                    self.argumentCounts[index+1][predicate][tuple(arguments)] += 1.

    @property
    def wordCounts(self):
        if getattr(self, '_wordCounts', None) is None:
            self._wordCounts = collections.defaultdict(int)
            for word in self.predicateCounts:
                self._wordCounts[word] += self.predicateCounts[word]

            for argumentCount in self.argumentCounts[1:]:
                for predicate in argumentCount:
                    for arguments in argumentCount[predicate]:
                        for argument in arguments:
                            self._wordCounts[argument] += argumentCount[predicate][tuple(arguments)]
        
        return self._wordCounts

    def setArgumentCounts(self, predicateCounts, subjectCounts, objectCounts, indirectObjectCounts):
        self.predicateCounts = predicateCounts 
        self.subjectCounts = subjectCounts 
        self.objectCounts = objectCounts
        self.indirectObjectCounts = indirectObjectCounts

        self.argumentCounts = [self.predicateCounts,
                              self.subjectCounts,
                              self.objectCounts,
                              self.indirectObjectCounts]

    def saveArgumentCounts(self, prefix):
        keys, values = self.predicateCounts.keys(), self.predicateCounts.values()

        with open(prefix + '.predicates.json', 'w') as f:
            json.dump([keys, values], f)

        for argumentName,arguments in (('subjects', self.subjectCounts),
                                       ('objects', self.objectCounts),
                                       ('indirectObjects', self.indirectObjectCounts)):
            keys = arguments.keys()
            values = []
            for key in arguments:
                values.append([arguments[key].keys(),
                              arguments[key].values()])

            with open(prefix + '.{}.json'.format(argumentName), 'w') as f:
                json.dump([keys, values], f)            

    @classmethod
    def loadArgumentCounts(cls, indir, prefix):
        print('loading predicates...')
        with open(prefix + '.predicates.json') as f:
            keys,values = json.load(f)
        argumentCounts = [dict(zip(keys,values)), {}, {}, {}]

        for i,argumentName in enumerate(('subjects', 'objects', 'indirectObjects')):
            print('loading {}'.format(argumentName))
            with open(prefix + '.{}.json'.format(argumentName)) as f:
                keys, values = json.load(f)

            for j in range(len(keys)):
                argumentCounts[i+1][keys[j]] = dict(zip(map(tuple,values[j][0]),
                                                        values[j][1]))
        
        ec = cls(indir)
        ec.setArgumentCounts(*argumentCounts)

        return ec

    def iterEvents(self, shuffle=False, verbose=False):
        for article in self.iterArticles(shuffle=shuffle, verbose=verbose):
            for e in article:
                yield e
        
    def iterEventBatches(self, shuffle=False, verbose=False):
        for article in self.iterArticles(shuffle=shuffle, verbose=verbose):
            ret_batch = []
            for e in article:
                ret = makeEvent(*e)
                ret_batch.append(ret)

            yield ret_batch

    def makeFlattenedArguments(self):
        self.flattenedArguments = [collections.defaultdict(list) for i in range(3)]
                              
        for i in range(1,4):
            print(i)
            for j in self.argumentCounts[i]:
                for k in self.argumentCounts[i][j]:
                    self.flattenedArguments[i-1][j] += ([k] * int(self.argumentCounts[i][j][k]))

    def sampleEvent(self, padding=1):
        if getattr(self, 'predicateDist', None) is None:
            print('making predicateDist...')
            self.predicateDist = rv_discrete(values=countDictToProbList(self.predicateCounts))
            
        #first sample a predicate
        predicate = self.predicateDist.rvs(size=1)[0]
        
        #then sample all the arguments given the predicate
        arguments = [predicate]
        for i in range(3):
            if getattr(self, 'flattenedArguments', None) is None:
                self.makeFlattenedArguments()

            choice = np.random.randint(len(self.flattenedArguments[i][predicate]))
            argument = self.flattenedArguments[i][predicate][choice]
            
            arguments.append(argument)

        #return arguments
        return makeEvent(*arguments, padding=padding)
        
    def iterEventContext(self, window=2, shuffle=False, verbose=False):
        for eventBatch in self.iterEventBatches(shuffle, verbose):
            for i in range(len(eventBatch)):
                for context in eventBatch[i-window:i]:
                    yield eventBatch[i] + context
                for context in eventBatch[i+1:i+window+1]:
                    yield eventBatch[i] + context

    #for each event and context and negative sample
    #add this datapoint to the max size buffer
    def iterEventBuffer(self, size=10000, window=2, shuffle=False, verbose=False):
        self.eventBuffer = {}
        for eventAndContext in self.iterEventContext(window, shuffle, verbose):
            maxLength = max((len(eventAndContext[1]) // 3, len(eventAndContext[5]) // 3))
            negativeSample = self.sampleEvent(maxLength)

            #there is a chance that we sampled an event that has some of its arguments
            #longer than all of the arguments in the event and context
            #in this case, we need to pad eventAndContext
            if len(negativeSample[1]) // 3 > maxLength:
                #print('padding from {} to {}'.format(maxLength, len(negativeSample[1]) // 3))
                maxLength = len(negativeSample[1]) // 3
                paddedEvent = padEvent(eventAndContext[:4], maxLength)
                paddedContext = padEvent(eventAndContext[4:], maxLength)
                eventAndContext = paddedEvent + paddedContext
            elif len(eventAndContext[1]) // 3 < maxLength:
                paddedEvent = padEvent(eventAndContext[:4], maxLength)
                eventAndContext = paddedEvent + eventAndContext[4:]
            elif len(eventAndContext[5]) // 3 < maxLength:                
                paddedContext = padEvent(eventAndContext[4:], maxLength)
                eventAndContext = eventAndContext[:4] + paddedContext
                
            datum = eventAndContext + negativeSample
                
            if maxLength not in self.eventBuffer:
                self.eventBuffer[maxLength] = [[] for i in range(12)]

            for i in range(len(datum)):
                self.eventBuffer[maxLength][i].extend(datum[i])

            if len(self.eventBuffer[maxLength][0]) >= size:
                yield self.eventBuffer[maxLength]
                del self.eventBuffer[maxLength]
