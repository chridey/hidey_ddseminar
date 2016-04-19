import os
import gzip
import json
import itertools

import numpy as np

from dependencyRNN.data.eventContextData import EventContextData

class CausalEventContextData(EventContextData):
    def iterArticles(self, shuffle=False, verbose=False):
        filenames = sorted(os.listdir(self.indir))
        if shuffle:
            np.random.shuffle(filenames)

        for filename in filenames:
            if verbose:
                print(filename)
            if filename.startswith('event') and filename.endswith('.json.gz'):
                with gzip.open(os.path.join(self.indir, filename)) as f:
                    batch = json.load(f)

                if shuffle:
                    np.random.shuffle(batch)

                for eventPair in batch:
                    if len(eventPair[0]) and len(eventPair[1]):
                        article = list(itertools.product(*eventPair))
                        for pair in article:
                            yield pair

    def iterEvents(self, shuffle=False, verbose=False):
        filenames = sorted(os.listdir(self.indir))
        if shuffle:
            np.random.shuffle(filenames)

        for filename in filenames:
            if verbose:
                print(filename)
            if filename.startswith('event') and filename.endswith('.json.gz'):
                with gzip.open(os.path.join(self.indir, filename)) as f:
                    batch = json.load(f)

                if shuffle:
                    np.random.shuffle(batch)

                for eventPair in batch:
                    for i in range(2):
                        if len(eventPair[i]):
                            for event in eventPair[i]:
                                yield event
