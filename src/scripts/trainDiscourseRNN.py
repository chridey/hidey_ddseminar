import argparse
import gzip
import json
import time

from sklearn.metrics import precision_score,recall_score,precision_recall_fscore_support,accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib

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
    parser.add_argument('--tunefile', 
                        help='file containing the causal or non-causal sentences, processed')

    parser.add_argument('--train_features')
    parser.add_argument('--test_features')
    parser.add_argument('--tune_features')        

    parser.add_argument('--testwords')
    parser.add_argument('--tunewords')        

    parser.add_argument('--balance',
                        type=int)
    parser.add_argument('--combined',
                        action='store_true')

    parser.add_argument('--pairwise',
                        action='store_true')

    parser.add_argument('--embeddings')
    parser.add_argument('--beta')    
    parser.add_argument('--fixed_beta',
                        action='store_true')
    
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

    parser.add_argument('--alpha',
                        type=float,
                        default=0.001)
    parser.add_argument('--l1_ratio',
                        type=float,
                        default=0.15)
    
    parser.add_argument('--include')
    parser.add_argument('--ablate')    
    parser.add_argument('--int',
                        help='add intercept',
                        action='store_true')    
    parser.add_argument('--save')
    parser.add_argument('--load')

    parser.add_argument('--vectorizer')

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

    if args.beta:
        beta = np.load(args.beta)
    else:
        beta = None

    if args.train_features:
        assert(args.test_features)
        with open(args.train_features) as f:
            features = json.load(f)
        features = utils.createModifiedDataset(features, args.include, args.ablate,
                                               add_bias=args.int)
        if args.vectorizer:
            dv = joblib.load(args.vectorizer)
            train_features = sp.csc_matrix(dv.transform(features))
        else:
            dv = DictVectorizer()
            train_features = sp.csc_matrix(dv.fit_transform(features))
        
        with open(args.test_features) as f:
            features = json.load(f)
        features = utils.createModifiedDataset(features, args.include, args.ablate,
                                               add_bias=args.int)
        test_features = sp.csc_matrix(dv.transform(features))

        if args.tune_features:
            with open(args.tune_features) as f:
                features = json.load(f)
            features = utils.createModifiedDataset(features, args.include, args.ablate,
                                                   add_bias=args.int)
            tune_features = sp.csc_matrix(dv.transform(features))

        print(train_features.shape)
        print(test_features.shape)
        if args.tune_features:
            print(tune_features.shape)
    else:
        train_features = None
        test_features = None
        tune_features = None
        
    if args.verbose:
        print(args.dimension, size, len(relations))

    iterator = AltlexDiscourseData(args.trainfile, args.balance, args.verbose,
                                   pairwise_constraint=args.pairwise,
                                   features=train_features,
                                   combined=args.combined,
                                   rnn=args.ablate and 'rnn' not in args.ablate)

    if args.testfile:
        testIterator = AltlexDiscourseData(args.testfile, verbose=args.verbose,
                                           features=test_features,
                                           combined=args.combined,
                                           rnn=args.ablate and 'rnn' not in args.ablate)
    if args.tunefile:
        tuneIterator = AltlexDiscourseData(args.tunefile, verbose=args.verbose,
                                           balance=args.balance,
                                           features=tune_features,
                                           combined=args.combined,
                                           rnn=args.ablate and 'rnn' not in args.ablate)

    if args.load:
        adrnn = AltlexDiscourseRNN.load(args.load, beta=beta, pairwise_constraint=args.pairwise,
                                        fixed_beta=args.fixed_beta)
    else:
        adrnn = AltlexDiscourseRNN(args.dimension,
                                   size,
                                   len(relations),
                                   embeddings=embeddings,
                                   pairwise_constraint=args.pairwise,
                                   nc = len(set(iterator.labels)),
                                   nf = 0 if train_features is None else train_features.shape[1],
                                   lambda_w=args.alpha,
                                   lambda_e=args.alpha,
                                   lambda_f=args.alpha,
                                   rnn=args.ablate and 'rnn' not in args.ablate,
                                   l1_ratio=args.l1_ratio,
                                   beta=beta,
                                   fixed_beta=args.fixed_beta)

    best_fscore = 0
    for i in range(args.num_epochs):
        print('epoch {}'.format(i))
        for j,batch in enumerate(iterator.iterBatches(args.batch_size,
                                                      shuffle=True,
                                                      uniform=True,
                                                      verbose=args.verbose)):
            start = time.time()
            cost = adrnn.train(batch)
            
            print ("epoch: {} batch: {} cost: {} time: {}".format(i, j, cost, time.time()-start))

            if args.tunefile:
                predictions = []
                for tuneBatch in tuneIterator.iterBatches():
                    predictions += adrnn.classify(*tuneBatch[:-1]).tolist()
                labels = tuneIterator.labels

                precision,recall,fscore,_ = (precision_recall_fscore_support(labels, predictions))
                print("Precision: {}".format(precision))
                print("Recall: {}".format(recall))
                print("Fscore: {}".format(fscore))

                if fscore[1] > best_fscore:
                    print('saving best model')
                    adrnn.save('{}.{}.{}'.format(args.save, i,j))
                    best_fscore = fscore[1]
                    
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

if args.tunefile:
    predictions = []
    for tuneBatch in tuneIterator.iterBatches():
        predictions += adrnn.classify(*tuneBatch[:-1]).tolist()
    labels = tuneIterator.labels

    accuracy = accuracy_score(labels, predictions)
    precision,recall,fscore,_ = (precision_recall_fscore_support(labels, predictions))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Fscore: {}".format(fscore))

    if args.tunewords:
        #do error analysis
        with open(args.tunewords) as f:
            words = json.load(f)
        with open(args.tunewords + '.errors', 'w') as f:
            json.dump(zip(words, labels, predictions), f)

if args.testfile:
    predictions = []
    for testBatch in testIterator.iterBatches(verbose=False):
        predictions += adrnn.classify(*testBatch[:-1]).tolist()
    labels = testIterator.labels

    print(len(labels), len(predictions))
    accuracy = accuracy_score(labels, predictions)
    precision,recall,fscore,_ = (precision_recall_fscore_support(labels, predictions))
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Fscore: {}".format(fscore))

    if args.testwords:
        #do error analysis
        with open(args.testwords) as f:
            words = json.load(f)
        with open(args.testwords + '.errors', 'w') as f:
            json.dump(zip(words, labels, predictions), f)
