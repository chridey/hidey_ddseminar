import gzip
import json
import sys

import numpy as np

from dependencyRNN.rnn.eventContextRNN import EventContextRNN
from dependencyRNN.data.eventContextData import makeEvent
from nlp.utils import dependencyUtils
from nlp.utils.utils import extractEvents, getEventIndices

#load the model files
print('loading model...')
modelfile = sys.argv[1]
rnn = EventContextRNN.load(modelfile)

#load the vocab lookup
vocabfile = sys.argv[2]
print('loading vocab...')
with gzip.open(vocabfile) as f:
    vocab = json.load(f)

#read in train and maybe testfile and maybe tunefile
trainfile = sys.argv[3]
print('loading training...')
with gzip.open(trainfile) as f:
    train = json.load(f)
    
if len(sys.argv) > 4:
    testfile = sys.argv[4]
    print('loading testing...')
    with gzip.open(testfile) as f:
        test = json.load(f)
else:
    test = []

if len(sys.argv) > 5:
    tunefile = sys.argv[5]
    print('loading tuneing...')
    with gzip.open(tunefile) as f:
        tune = json.load(f)
else:
    tune = []

output = [[], [], []]

#for each dataset
for index,dataset in enumerate([train, tune, test]):
    #for each sentence in the data set
    for sentence in dataset:
        deps = sentence['origDependencies']
        altlexIndex = len(sentence['words'][0])
        words = sentence['words'][0] + sentence['words'][1] + sentence['words'][2]
        
        #extract all events in the sentence
        #print(deps, words)
        pre_events = extractEvents(deps, words, end=altlexIndex, triples=False)
        post_events = extractEvents(deps, words, start=altlexIndex+len(sentence['words'][1]), triples=False)

        #print(sentence['words'])
        pre_event_embedding = np.zeros(100).tolist()
        pre_context_embedding = np.zeros(100).tolist()            
        if len(pre_events):
            #find the events closest to either side of the connective        
            event = pre_events[-1]
            #print(event)
            event = getEventIndices(event, vocab)
            if event is not None:
                #determine the event and context embeddings for both sides
                e = makeEvent(*event)
                pre_event_embedding = rnn.event_states(*e).reshape((100,)).tolist()
                pre_context_embedding = rnn.context_states(*e).reshape((100,)).tolist()

        post_event_embedding = np.zeros(100).tolist()
        post_context_embedding = np.zeros(100).tolist()            
        if len(post_events):
            #find the events closest to either side of the connective        
            event = post_events[0]
            #print(event)
            event = getEventIndices(event, vocab)
            if event is not None:
                #determine the event and context embeddings for both sides
                e = makeEvent(*event)
                post_event_embedding = rnn.event_states(*e).reshape((100,)).tolist()
                post_context_embedding = rnn.context_states(*e).reshape((100,)).tolist()

        output[index].append([pre_event_embedding,
                              pre_context_embedding,
                              pre_event_embedding,
                              pre_context_embedding,
                              np.dot(pre_event_embedding,
                                     post_context_embedding),
                              np.dot(post_event_embedding,
                                     pre_context_embedding),
                              sentence['label']])
            

for name,dataset in zip(('train', 'test', 'dev'),
                        output):
    if len(dataset):
        with open(name, 'w') as f:
            json.dump(dataset, f)

