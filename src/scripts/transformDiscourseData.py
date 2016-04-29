#read in train and maybe testfile

#run dependency data transform

#read in word2vec model 

import gzip
import json
import sys

import numpy as np

from gensim.models.word2vec import Word2Vec

from dependencyRNN.data.dependencyData import DependencyData

def makeDependencies(datum):
    deps = datum['origDependencies']
    words = datum['words'][0] + datum['words'][1] + datum['words'][2]
    label = datum['label'] #> 0

    #make the root of the altlex the new root
    #for every edge on the path from the original root to the new root flip the direction
    altlexStart = len(datum['words'][0])
    altlexEnd = len(datum['words'][0]+datum['words'][1])
    altlexRoot = None

    #print()
    #print(datum['altlex'], altlexStart, altlexEnd, words[altlexStart:altlexEnd])
    #print(zip(words, deps))
    
    if len(datum['words'][1])==1:
        altlexRoot = altlexStart
    else:
        for i in range(altlexStart, altlexEnd):
            rel,gov = deps[i]
            #if the gov is the parent of the altlex root or there is none
            if gov in range(altlexStart, altlexEnd) and (altlexRoot is None or i == altlexRoot):
                altlexRoot = gov

    if altlexRoot is None:
        
        for i in range(altlexStart, altlexEnd):
            if deps[i][0] != 'det':
                altlexRoot = i
                break
        
    assert(altlexRoot is not None)

        
    parent = -1
    rel = 'root'
    child = altlexRoot
    while child != -1:
        dep, gov = deps[child]
        deps[child][0] = rel
        deps[child][1] = parent

        parent = child
        child = gov
        rel = dep + '_rev'

    #print(zip(words, deps))
    
    d = [('ROOT', None, None)]
    for i,j in enumerate(deps):
        if j is None:
            d.append((None,None,None))
        else:
            d.append((words[i].lower(), j[0], j[1]+1))

    return d,label

modelfile = sys.argv[1]

trainfile = sys.argv[2]
print('loading training...')
with gzip.open(trainfile) as f:
    train = json.load(f)
    
if len(sys.argv) > 2:
    testfile = sys.argv[3]
    print('loading testing...')
    with gzip.open(testfile) as f:
        test = json.load(f)
else:
    test = []

mod_train = []
for datum in train:
    d = makeDependencies(datum)
    mod_train.append(d)

mod_test = []
for datum in test:
    d = makeDependencies(datum)
    mod_test.append(d)

dd = DependencyData()
print('making training...')
transformed_train = dd.transform(mod_train, True)
print('making testing...')
transformed_test = dd.transform(mod_test, True)

print('loading model...')
model = Word2Vec.load(modelfile)

embeddings = dd.match_embeddings(model)

np.save(modelfile + '.embeddings', embeddings)
with open(modelfile + '.train', 'w') as f:
    json.dump(transformed_train, f)

with open(modelfile + '.vocab', 'w') as f:
    json.dump(dd.vocab, f)

with open(modelfile + '.relations', 'w') as f:
    json.dump(dd.relations, f)

if len(test):
    with open(modelfile + '.test', 'w') as f:
        json.dump(transformed_test, f)

