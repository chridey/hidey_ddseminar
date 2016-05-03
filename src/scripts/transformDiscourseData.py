#read in train and maybe testfile

#run dependency data transform

#read in word2vec model 

import gzip
import json
import sys

import numpy as np

from gensim.models.word2vec import Word2Vec

from dependencyRNN.data.dependencyData import DependencyData
from nlp.utils import dependencyUtils

def makeDependencies(datum):
    deps = datum['origDependencies']
    label = datum['label'] #> 0

    d = dependencyUtils.makeDependencies(datum['words'], deps)

    return d, label

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

