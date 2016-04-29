import argparse
import gzip
import json
import time

from sklearn.metrics import precision_score,recall_score,precision_recall_fscore_support
from sklearn.feature_extraction import DictVectorizer

import scipy.sparse as sp

import numpy as np

from dependencyRNN.rnn.altlexDiscourseRNN import AltlexDiscourseRNN
from dependencyRNN.data.altlexDiscourseData import AltlexDiscourseData
from nlp.utils import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('trainfile', 
                        help='file containing the causal or non-causal sentences, processed')
    parser.add_argument('relations')
    
    parser.add_argument('--testfile', 
                        help='file containing the causal or non-causal sentences, processed')

    parser.add_argument('--train_features')
    parser.add_argument('--test_features')    

    parser.add_argument('--balance',
                        type=int)

    parser.add_argument('--pairwise',
                        action='store_true')

    parser.add_argument('--embeddings')

    parser.add_argument('--dimension',
                        type = int,
                        default = 100)
    parser.add_argument('--size',
                        type = int)

    parser.add_argument('--num_epochs',
                        type = int,
                        default = 50)
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)

    parser.add_argument('--include')
    parser.add_argument('--ablate')    

    parser.add_argument('--save')
    parser.add_argument('--load')

    parser.add_argument('--verbose',
                        action= 'store_true')

    args = parser.parse_args()

    with open(args.relations) as f:
        relations = json.load(f)

    if args.embeddings:
        embeddings = np.load(args.embeddings)
        size = embeddings.shape[0]
    else:
        embeddings = None
        size = args.size

    if args.train_features:
        assert(args.test_features)
        dv = DictVectorizer()
        with open(args.train_features) as f:
            features = json.load(f)
        features = utils.createModifiedDataset(features, args.include, args.ablate)
        train_features = sp.csc_matrix(dv.fit_transform(features))
        
        with open(args.test_features) as f:
            features = json.load(f)
        features = utils.createModifiedDataset(features, args.include, args.ablate)
        test_features = sp.csc_matrix(dv.transform(features))

        print(train_features.shape)
        print(test_features.shape)
    else:
        train_features = None
        test_features = None
        
    if args.verbose:
        print(args.dimension, size, len(relations))

    iterator = AltlexDiscourseData(args.trainfile, args.balance, args.verbose,
                                   pairwise_constraint=args.pairwise,
                                   features=train_features)

    if args.testfile:
        testIterator = AltlexDiscourseData(args.testfile, verbose=args.verbose,
                                           features=test_features)

    adrnn = AltlexDiscourseRNN(args.dimension,
                               size,
                               len(relations),
                               embeddings=embeddings,
                               pairwise_constraint=args.pairwise,
                               nc = len(set(iterator.labels)),
                               nf = 0 if train_features is None else train_features.shape[1])
    
    for i in range(args.num_epochs):
        print('epoch {}'.format(i))
        for j,batch in enumerate(iterator.iterBatches(args.batch_size,
                                                      shuffle=True,
                                                      uniform=True,
                                                      verbose=args.verbose)):
            start = time.time()
            cost = adrnn.train(batch)
            
            print ("epoch: {} batch: {} cost: {} time: {}".format(i, j, cost, time.time()-start))

            if args.testfile:
                predictions = []
                for testBatch in testIterator.iterBatches():
                    predictions += adrnn.classify(*testBatch[:-1]).tolist()
                labels = testIterator.labels

                precision,recall,fscore,_ = (precision_recall_fscore_support(labels, predictions))
                print("Precision: {}".format(precision))
                print("Recall: {}".format(recall))
                print("Fscore: {}".format(fscore))
                
        adrnn.save('{}.{}'.format(args.save, i))

    
