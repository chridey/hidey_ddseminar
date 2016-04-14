import argparse
import gzip
import json

import numpy as np

from dependencyRNN.data.eventContextData import EventContextData
from dependencyRNN.rnn.eventContextRNN import EventContextRNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train an event2vec model on events in context')

    parser.add_argument('eventfile', 
                        help='the file containing the events')
    parser.add_argument('--dimension',
                        type=int,
                        default=100)
    parser.add_argument('--relations',
                        type=int,
                        default=4)
    parser.add_argument('--window',
                        type=int,
                        default=2)
    parser.add_argument('--batch_size',
                        type=int,
                        default=100)
    parser.add_argument('--embeddings')
    
    args = parser.parse_args()

    ec = EventContextData(args.eventfile)

    if args.embeddings:
        embeddings = np.loadz(args.embeddings)
        size = embeddings.shape[0]
    else:
        embeddings = None
        size = args.size

    er = EventContextRNN(args.dimension,
                         size,
                         args.relations,
                         embeddings=embeddings)

    eventDistribution = ec.eventProbs

    for i in range(args.num_epochs):
        print('epoch {}'.format(i))
        g = ec.iterEventContext(args.window, shuffle=True)
        while g:
            batch = [next(g) for i in range(args.batch_size)]
            start = time.time()
            cost = er.train(batch,
                            eventDistribution.keys(),
                            eventDistribution.values())
            
            print ("epoch: {} batch: {} cost: {} time: {}".format(i, j, cost, time.time()-start))
            
        d.save('{}.{}'.format(args.save, i))

    
