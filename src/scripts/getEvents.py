from __future__ import print_function

import os
import gzip
import json
import sys
import collections

from nlp.utils import dependencyUtils

def evalMinCount(word, wordCounts, minCount, vocab):
    if wordCounts is None:
        return True
    if word not in vocab:
        return False
    index = vocab[word]

    #print(word, wordCounts.get(str(index), None))
    
    if str(index) not in wordCounts or wordCounts[str(index)] < minCount:
        return False
    return True

in_dir = sys.argv[1]
inputFiles = os.listdir(in_dir)

if len(sys.argv) > 2:
    wordCountsFile = sys.argv[2]
    minCount = int(sys.argv[3])
    vocabFile = sys.argv[4]

    print('loading wordCounts')
    with open(wordCountsFile) as f:
        wordCounts = json.load(f)
    print('loading vocab')
    with gzip.open(vocabFile) as f:
        vocab = json.load(f)
else:
    wordCounts = None
    minCount = 0
    vocab = None
    
total = 0
count = 0
eventBatches = []
wordLookup = {}
for inputFile in sorted(inputFiles):
    inputFileFullPath = os.path.join(in_dir, inputFile)
    if inputFile.startswith('good') and inputFileFullPath.endswith('.json.gz'):
        print(inputFile, count, total, sum(len(eventBatch) for eventBatch in eventBatches))
        
        with gzip.open(inputFileFullPath) as f:
            articles = json.load(f)

        count += 1
        total += len(articles)

        for title in articles:
            #print(title)
            eventBatch = []
            for sentence in articles[title]:
                dep = sentence['dep']
                words = sentence['words']
                pos = sentence['pos']
                
                #print(words, dep)
                for i in range(len(dep)):
                    depTuples = dependencyUtils.tripleToList(dep[i], len(words[i]), True)
                    #print(zip(words[i],depTuples))
                    compounds = dependencyUtils.getCompounds(depTuples)
                    #print(compounds)
                    
                    es = dependencyUtils.getAllEventsAndArguments(depTuples)

                    for e in es:
                        if pos[i][e][:2] != 'VB':
                            continue
                        
                        if 'nsubjpass' in es[e]:
                            predicate = words[i][e].lower() + '_(passive)'
                            key = 'nsubjpass'
                        else:
                            predicate = words[i][e].lower()
                            key = 'nsubj'

                        #print(predicate)
                        if not evalMinCount(predicate,
                                            wordCounts,
                                            minCount,
                                            vocab):
                            continue

                        arguments = [[], [], []]
                        minCountGood=True
                        for index, argType in enumerate((key, 'dobj', 'iobj')):
                            for j in es[e].get(argType, []):
                                if j in compounds:
                                    arg = '_'.join(words[i][min(compounds[j] + [j]): max(compounds[j] + [j])+1]).lower()
                                else:
                                    arg = words[i][j].lower()

                                #print(argType)
                                if not evalMinCount(arg,
                                                    wordCounts,
                                                    minCount,
                                                    vocab):
                                    minCountGood=False
                                    break

                                arguments[index].append(arg)
                            if not minCountGood:
                                break
                            
                        if not minCountGood:
                            continue
                        #print(predicate, arguments)

                        if predicate not in wordLookup:
                            wordLookup[predicate] = len(wordLookup)
                        predicateIndex = wordLookup[predicate]
                        
                        argument_indices = [[], [], []]
                        for index, argType in enumerate(arguments):
                            for arg in sorted(argType):
                                if arg not in wordLookup:
                                    wordLookup[arg] = len(wordLookup)
                                
                                argument_indices[index].append(wordLookup[arg])

                        eventBatch.append((predicateIndex,
                                           tuple(argument_indices[0]), #subjects,
                                           tuple(argument_indices[1]), #directObjects,
                                           tuple(argument_indices[2]))) #indirectObjects))

            eventBatches.append(eventBatch)

        
        if (count-1) % 100 == 0:
            with gzip.open('events{}.json.gz'.format(count//100), 'w') as f:
                json.dump(eventBatches, f)

            eventBatches = []
            
#uniqueEvents = collections.Counter(events)
#print ('Total Events: {}'.format(len(events)))
#print ('Unique Events: {}'.format(len(uniqueEvents)))

with gzip.open('events{}.json.gz'.format(count//100), 'w') as f:
    json.dump(eventBatches, f)
                
with gzip.open('vocab.json.gz', 'w') as f:
    json.dump(wordLookup, f)

#for event in sorted(uniqueEvents.items(), key=lambda x:x[1]):
#    print(event, uniqueEvents[event])
