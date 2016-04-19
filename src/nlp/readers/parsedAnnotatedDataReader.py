import os
import json
import gzip
import collections

from nlp.utils import utils
from dependencyRNN.data.eventContextData import makeEvent

class Sentence:
    def __init__(self, words, title, index):
        self.words = words
        self.title = title
        self.index = index

class Event:
    def __init__(self, text, indices, predicateIndex, sentence):
        self.sentence = sentence
        self.text = text
        self.indices = indices
        self.predicateIndex = predicateIndex
        
    @property
    def title(self):
        return self.sentence.title
            
    @property
    def sentenceIndex(self):
        return self.sentence.index

    @property
    def predicate(self):
        return self.text[0]

class ParsedAnnotatedDataReader:
    def __init__(self, indir, vocabfile):
        self.indir = indir

        with gzip.open(vocabfile) as f:
            self.vocab = json.load(f)

    def iterEventBatches(self):
        for inputFile in sorted(os.listdir(self.indir)):
            if not inputFile.endswith('.gz'):
                continue
            print(inputFile)
            
            eventBatch = []
            with gzip.open(os.path.join(self.indir, inputFile)) as f:
                sentences = json.load(f)

            for index,sentence in enumerate(sentences):
                dependencies = sentence['dep']
                words = sentence['words']
                             
                for i in range(len(dependencies)):
                    s = Sentence(words[i],
                                 inputFile.replace('.sentences.json.gz', ''),
                                 index)
                    
                    c = collections.Counter(words[i])

                    events = utils.extractEvents(dependencies[i], words[i], True)

                    for event in events:
                        #print(event)
                        if event[0] not in self.vocab:
                            print('cant find {} for {}'.format(event[0], event))
                            continue
                        
                        indices = [self.vocab[event[0]], [], [], []]
                        flag = False
                        for j in range(1,4):
                            args = event[j]
                            for arg in args:
                                if arg not in self.vocab:
                                    print('cant find {} for {}'.format(arg, event))
                                    flag=True
                                    continue
                                    
                                indices[j].append(self.vocab[arg])

                        if flag:
                            continue
                        
                        if c[event[0]] > 1:
                            #raise NotImplementedError
                            predicateIndex = 0
                            print('{} = {}'.format(event[0], predicateIndex))
                        else:
                            predicateIndex = 0
                            
                        e = Event(event, makeEvent(*indices), predicateIndex, s)
                        eventBatch.append(e)

            yield eventBatch
                        
    def iterEventPairs(self):
        for eventBatch in self.iterEventBatches():
            print(len(eventBatch))
            for event in eventBatch:
                for context in eventBatch:
                    if event.text != context.text and abs(event.sentenceIndex-context.sentenceIndex) < 3:
                        yield event, context
