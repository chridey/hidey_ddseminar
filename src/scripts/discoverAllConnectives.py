import json
import sys

from collections import defaultdict

import numpy as np

from dependencyRNN.data.dependencyData import DependencyData
from dependencyRNN.data.dependencyRNN import DependencyRNN

from nlp.readers.parsedWikipediaReader import ParsedWikipediaReader
from nlp.utils.dependencyUtils import makeDependencies
from nlp.utils.wordUtils import modal_auxiliary

#load pre-trained model
model = sys.argv[1]
rnn = DependencyRNN.load(model)
#load vocab
prefix = sys.argv[2]
with open(prefix + '.vocab') as f:
    vocab = json.load(f)
with open(prefix + '.relations') as f:
    relations = json.load(f)

wikipedia_dir = sys.argv[3]
outfile = sys.argv[4]

#go through all wikipedia
reader = ParsedWikipediaReader(wikipedia_dir)

dd = DependencyData()
dd.vocab = vocab
dd.relations = relations

predictions = defaultdict(lambda: defaultdict(float))
indices = defaultdict(list)

#for each article:
#for each sentence:
for total,(index,sentence) in enumerate(reader.iterSentences(index=True)):
    #for each content word in the sentence:
    #0) if not in vocab, set to random?
    #1) extract features
    #2) transform dependencies for rnn
    #3) predict

    words = sentence['words']
    dep = sentence['dep']
    pos = sentence['pos']
    for i in range(words):
        #limit to valid content words
        word = words[i].lower()
        if pos[i][:2] not in ('RB', 'NN', 'JJ', 'VB') or pos[i][:3] == 'NNP' or word in modal_auxiliary:
            continue
        
        #split words into 3 arrays
        splitWords = [words[:i], [words[i]], words[i+1:]]
        
        d = makeDependencies(splitWords, dep)
        label = None
        datum = dd.transform([(d,label)])

        prediction = model.classify(*datum[:-1])
        predictions[words[i]][prediction] += 1
        if prediction:
            indices[words[i]].append(index)

        if total % 10000 == 0:
            with open(outfile + '.predictions.{}'.format(total // 10000), 'w') as f:
                json.dump(f, predictions)
            with open(outfile + '.indices.{}'.format(total // 10000), 'w') as f:
                json.dump(f, indices)
            
#ALTERNATIVELY:
#(segment according to known phrases)


#save counts for each word
#save indices (file, article, sentence)
#write counts out periodically
