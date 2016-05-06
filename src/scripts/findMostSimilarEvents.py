import gzip
import json
import sys
import os

from collections import defaultdict

import numpy as np

from dependencyRNN.rnn.eventContextRNN import EventContextRNN
from dependencyRNN.data.eventContextData import makeEvent

modelfile = sys.argv[1]
vocabfile = sys.argv[2]
indir = sys.argv[3]
outfile = sys.argv[4]

#load the model files
print('loading model...')
rnn = EventContextRNN.load(modelfile)

#load the vocab file
print('loading vocab...')
with gzip.open(vocabfile) as f:
    vocab = json.load(f)

queries = [i.split() for i in '''dog barked
glass fell
glass broke
kid fell
obama spoke
congress passed law
he gave book her
hogwarts closing
astronauts launched space_shuttle
john bought milk'''.split('\n')]

embeddings = []
for i in queries:
    predicate = vocab[i[1]]
    subjects = [vocab[i[0]]]
    if len(i) > 2:
        objects = [vocab[i[2]]]
    else:
        objects = []
    if len(i) > 3:
        iobjects = [vocab[i[3]]]
    else:
        iobjects = []
        
    event = makeEvent(predicate, subjects, objects, iobjects)
    embedding = rnn.event_states(*event).reshape((100,)).tolist()
    embeddings.append(embedding)
embeddings = np.array(embeddings)

#for each event/context embedding file
#load the file
#dot product of desired events and all these embeddings
#keep the top n (argpartition)

num = 100

#go through contexts first
best_contexts = defaultdict(list)
for filename in os.listdir(indir):
    if 'contexts.values' in filename:
        print(filename)
        with open(os.path.join(indir, filename)) as f:
            contexts = np.array(json.load(f))
            #contexts is N x D
            
        with open(os.path.join(indir,
                               'contexts.keys' + filename[len('contexts.values'):])) as f:
            keys = json.load(f)

        #embeddings is Q X D
        dps = np.dot(embeddings, contexts.T)
        p = np.argpartition(dps, contexts.shape[0]-num, axis=1)
        #Q x N
        
        for ii,i in enumerate(p):
            for j in i[contexts.shape[0]-num:]:
                dp = dps[ii][j]
                rep = keys[j]
                best_contexts[ii].append((rep, dp))

with open(outfile + '.contexts', 'w') as f:
    json.dump(best_contexts, f)
    
#go through events next
best_events = defaultdict(list)
for filename in os.listdir(indir):
    if 'events.values' in filename:
        with open(os.path.join(indir, filename)) as f:
            events = np.array(json.load(f))

        with open(os.path.join(indir,
                               'events.keys' + filename[len('events.values'):])) as f:
            keys = json.load(f)

        dps = np.dot(embeddings, events.T)
        dps /= np.linalg.norm(embeddings, axis=1).reshape((embeddings.shape[0],1))
        dps /= np.linalg.norm(events, axis=1).reshape((1,events.shape[0]))
        p = np.argpartition(dps, events.shape[0]-num, axis=1)

        for ii,i in enumerate(p):
            for j in i[events.shape[0]-num:]:
                dp = dps[ii][j]
                rep = keys[j]
                best_events[ii].append((rep, dp))

with open(outfile + '.events', 'w') as f:
    json.dump(best_events, f)
    
