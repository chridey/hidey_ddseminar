import collections
import sys
import gzip
import json

import numpy as np

from dependencyRNN.rnn.eventContextRNN import EventContextRNN
from dependencyRNN.data.causalEventContextData import CausalEventContextData
from dependencyRNN.data.eventContextData import makeEvent

model = sys.argv[1]
indir = sys.argv[2]
outfile = sys.argv[3]

#load event context rnn
print('compiling model...')
rnn = EventContextRNN.load(model)
reader = CausalEventContextData(indir)

event_embeddings = {}
context_embeddings = {}
event_counts = collections.defaultdict(int)

for event in reader.iterEvents(verbose=True):
    predicate, subjects, objects, indirectObjects = event
    
    e = makeEvent(*event)
    event_embedding = rnn.event_states(*e).reshape((100,))
    context_embedding = rnn.context_states(*e).reshape((100,))

    event_tuple = (predicate,
                   tuple(subjects),
                   tuple(objects),
                   tuple(indirectObjects))
    
    event_counts[event_tuple] += 1
    event_embeddings[event_tuple] = event_embedding.tolist()
    context_embeddings[event_tuple] = context_embedding.tolist()

print('Total events', len(event_counts))
length = len(context_embeddings)
num_batches = 4
batch_size = length // num_batches + 1

with open('events.keys.' + outfile, 'w') as f:
    json.dump(event_embeddings.keys(), f)

for i in range(num_batches):
    with open('events.values.{}.'.format(i) + outfile, 'w') as f:
        json.dump(event_embeddings.values()[i*batch_size:(i+1)*batch_size], f)

with open('contexts.keys.' + outfile, 'w') as f:
    json.dump(context_embeddings.keys(), f)

for i in range(num_batches):
    with open('contexts.values.{}.'.format(i) + outfile, 'w') as f:
        json.dump(context_embeddings.values()[i*batch_size:(i+1)*batch_size], f)

with open('counts.' + outfile, 'w') as f:
    json.dump([event_counts.keys(),
               event_counts.values()], f)
