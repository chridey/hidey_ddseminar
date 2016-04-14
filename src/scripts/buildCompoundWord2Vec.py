import argparse

import numpy as np
from gensim.models.word2vec import Word2Vec

from nlp.readers.parsedWikipediaReader import ParsedWikipediaReader

parser = argparse.ArgumentParser(description='train a word2vec model on sentences in context')

parser.add_argument('indir', 
                    help='the directory containing the sentences and metadata in GZipped JSON format')
parser.add_argument('outfile', 
                    help='the name for the model file')
parser.add_argument('--n_components', 
                    type=int,
                    default=100)
parser.add_argument('--n_jobs', 
                    type=int,
                    default=1)
parser.add_argument('--load',
                    action='store_true')

args = parser.parse_args()

si = ParsedWikipediaReader(args.indir,
                           compounded=True)

outfilename = args.outfile

if args.load:
    modelName = "{}.model.word2vec".format(outfilename)
    print('loading {}'.format(modelName))
    model = Word2Vec.load(modelName)
else:
    model = Word2Vec(size=args.n_components, min_count=1, workers=args.n_jobs)
    print('building vocab...')
    model.build_vocab(si.iterPreprocessedData())

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes

try:
    for epoch in range(passes):
        model.alpha, model.min_alpha = alpha, alpha
        model.train(si.iterPreprocessedData(True))
        
        print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta
        
        #np.random.shuffle(si.iterPreprocessedData(True))
except KeyboardInterrupt:
    print('terminating on keyboard interrupt')

modelName = "{}.model.word2vec".format(outfilename)

model.save(modelName)
