import sys
import collections

import numpy as np

from dependencyRNN.rnn.eventContextRNN import EventContextRNN

from nlp.utils.extractLabels import extractLabels
from nlp.readers.parsedAnnotatedDataReader import ParsedAnnotatedDataReader

from sklearn.metrics import precision_recall_curve

model = sys.argv[1]
indir = sys.argv[2]
labelfile = sys.argv[3]
processedindir = sys.argv[4]
vocabfile = sys.argv[5]

#extract annotations
print ('extracting annotations...')
relations = extractLabels(indir, labelfile)

#read in parsed annotated data
print('loading reader...')
reader = ParsedAnnotatedDataReader(processedindir, vocabfile)

#load event context rnn
print('compiling model...')
rnn = EventContextRNN.load(model)

#for all pairs of events
#score the event pair (both ways), print the words, the scores, and the relationship (C/R/None)

metadata = collections.defaultdict(dict)
labels = []
scores = []
for event,context in reader.iterEventPairs():

    pair = ((event.sentenceIndex, event.predicate.replace('_(passive)', ''), event.predicateIndex),
            (context.sentenceIndex, context.predicate.replace('_(passive)', ''), context.predicateIndex))

    if pair in relations[event.title]:
        relation = relations[event.title][pair]
    else:
        relation = None

    eventState = rnn.event_states(*event.indices).reshape((rnn.d,))
    contextState = rnn.context_states(*context.indices).reshape((rnn.d,))

    PMI = np.dot(eventState, contextState)

    metadata[event.title][pair] = [event.title, event.sentenceIndex, context.sentenceIndex,
                                   event.text, context.text, PMI, relation]
    if relation is not None:
        print(event.title, event.sentenceIndex, context.sentenceIndex,
              event.text, context.text, PMI, relation)

    labels.append(relation is not None)
    scores.append(PMI)
        
#rank these by their PMI
for title in metadata:
    for pair,_ in sorted(metadata[title].items(), key=lambda x:x[1][5]):
        print(metadata[title][pair])

precision, recall, thresholds = precision_recall_curve(labels, scores)
for i in range(len(thresholds)):
    print(precision[i], recall[i], thresholds[i])

#also see what the most similar events are for each event
    

