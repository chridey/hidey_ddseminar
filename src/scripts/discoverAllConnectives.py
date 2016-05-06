import json
import sys

from collections import defaultdict

import numpy as np

from dependencyRNN.data.dependencyData import DependencyData
from dependencyRNN.rnn.altlexDiscourseRNN import AltlexDiscourseRNN

from nlp.readers.parsedWikipediaReader import ParsedWikipediaReader
from nlp.utils.dependencyUtils import makeDependencies, tripleToList
from nlp.utils.wordUtils import modal_auxiliary

#load pre-trained model
print('loading model...')
model = sys.argv[1]
rnn = AltlexDiscourseRNN.load(model)

print('loading vocab...')
#load vocab
prefix = sys.argv[2]
with open(prefix + '.vocab') as f:
    vocab = json.load(f)
with open(prefix + '.relations') as f:
    relations = json.load(f)

wikipedia_dir = sys.argv[3]
outfile = sys.argv[4]

#go through all wikipedia
print('loading wikipedia...')
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
    lemmas = sentence['lemmas']
    for i in range(len(words)):
        deps = tripleToList(dep[i], len(words[i]))
        #print(deps)
        idxs = []
        rel_idxs = []
        p = []
        mask = []
        ws = []
        positions = []
        
        for j in range(len(words[i])):
            #limit to valid content words
            word = words[i][j].lower()
            if pos[i][j][:2] not in ('RB', 'NN', 'JJ', 'VB') or pos[i][j][:3] == 'NNP' or lemmas[i][j].lower() in modal_auxiliary:
                continue

            #split words into 3 arrays
            splitWords = [words[i][:j], [words[i][j]], words[i][j+1:]]

            try:
                d = makeDependencies(splitWords, deps)
            except Exception as e:
                continue
            label = None
            try:
                datum = dd.transform([(d,label)])
            except Exception as e:
                #print(e, word)
                continue

            #print(datum)
            idxs.append(datum[0][0])
            rel_idxs.append(datum[0][1])
            p.append(datum[0][2])
            mask.append(datum[0][3])                        
            ws.append(words[i][j])
            positions.append(index)

        if not len(idxs):
            continue

        try:
            ps = rnn.classify(idxs, p, rel_idxs, mask)
            for e,p in enumerate(ps):
                predictions[ws[e]][str(p)] += 1
                if p:
                    indices[ws[e]].append(positions[e])
        except Exception:
            continue
        
        stride = 100000
        if total % stride == 0:
            with open(outfile + '.predictions.{}'.format(total // stride), 'w') as f:
                json.dump(predictions, f)
            with open(outfile + '.indices.{}'.format(total // stride), 'w') as f:
                json.dump(indices, f)

with open(outfile + '.predictions.{}'.format(total // stride), 'w') as f:
    json.dump(predictions, f)
with open(outfile + '.indices.{}'.format(total // stride), 'w') as f:
    json.dump(indices, f)

            
#ALTERNATIVELY:
#(segment according to known phrases)


#save counts for each word
#save indices (file, article, sentence)
#write counts out periodically
