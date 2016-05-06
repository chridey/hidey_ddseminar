import collections
import sys
import gzip
import json

import numpy as np

from dependencyRNN.rnn.eventContextRNN import EventContextRNN
from dependencyRNN.data.causalEventContextData import CausalEventContextData
from dependencyRNN.data.eventContextData import makeEvent, EventContextData

model = sys.argv[1]
indir = sys.argv[2]
outfile = sys.argv[3]
flag = sys.argv[4]

#load event context rnn
print('compiling model...')
rnn = EventContextRNN.load(model)
if flag == '0':
    reader = CausalEventContextData(indir)
elif flag == '1':
    reader = EventContextData(indir)
    
event_embeddings = {}
context_embeddings = {}
event_counts = collections.defaultdict(int)

count = 0
for event in reader.iterEvents(verbose=True):
    predicate, subjects, objects, indirectObjects = event
    event_tuple = (predicate,
                   tuple(subjects),
                   tuple(objects),
                   tuple(indirectObjects))

    if event_tuple not in event_counts:
        e = makeEvent(*event)
        event_embedding = rnn.event_states(*e).reshape((100,))
        context_embedding = rnn.context_states(*e).reshape((100,))

        event_embeddings[event_tuple] = event_embedding.tolist()
        context_embeddings[event_tuple] = context_embedding.tolist()

    event_counts[event_tuple] += 1

    size = 1000000
    if len(event_embeddings) % 100000 == 0:
        print(len(event_embeddings))
        
    if len(event_embeddings) >= size:
        with open('events.keys.{}.'.format(count) + outfile, 'w') as f:
            json.dump(event_embeddings.keys(), f)

        with open('events.values.{}.'.format(count) + outfile, 'w') as f:
            json.dump(event_embeddings.values(), f)

        with open('contexts.keys.{}.'.format(count) + outfile, 'w') as f:
            json.dump(context_embeddings.keys(), f)

        with open('contexts.values.{}.'.format(count) + outfile, 'w') as f:
            json.dump(context_embeddings.values(), f)            

        event_embeddings = {}
        context_embeddings = {}

        count += 1

print('Total events', len(event_counts))
length = len(event_counts)
num_batches = length // size + 1
batch_size = length // num_batches

for i in range(num_batches):
    keys = event_counts.keys()[i*batch_size:(i+1)*batch_size]
    values = event_counts.values()[i*batch_size:(i+1)*batch_size]

    with open('counts.{}.'.format(i) + outfile, 'w') as f:
        json.dump([keys, values], f)

