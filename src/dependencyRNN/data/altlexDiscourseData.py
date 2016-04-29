import json
import collections

import numpy as np

class AltlexDiscourseData:
    def __init__(self, filename, balance=None, verbose=False,
                 pairwise_constraint=False,
                 features=None):

        with open(filename) as f:
            data = json.load(f)

        self.features = features
        self.pairwise = pairwise_constraint

        #balance data
        if balance is not None:
            c = collections.Counter([i[-1] for i in data])

            if balance > 1:
                num = balance
            elif balance == 1:
                num = max(c.values())
            elif balance == 0:
                num = min(c.values())

            if pairwise_constraint:
                num = num // 2
                
            balanced_data = []
            balanced_indices = []
            for label in c:
                data_label = [i for i in data if i[-1] == label]

                length = len(data_label)
                if pairwise_constraint:
                    length = len(data_label) // 2

                if length >= num:
                    #sample without replacement
                    indices = np.random.choice(length, num, False)
                else:
                    #sample with replacement
                    indices = np.random.choice(length, num, True)

                if pairwise_constraint:
                    balanced_data.extend([data_label[2*i+j] for i in indices for j in range(2)])
                    indices = np.array([indices*2, indices*2+1]).reshape(2*num, order='F')
                else:
                    balanced_data.extend([data_label[i] for i in indices])
                balanced_indices.extend(indices.tolist())
                
            if self.features is not None:
                self.features = self.features[balanced_indices]
                        
            data = balanced_data
            
            '''
            true = [i for i in data if i[-1]]
            false = [i for i in data if not i[-1]]
            falseIndices = set(np.random.choice(len(false), len(true), False))
            data = true + [i for index,i in enumerate(false) if index in falseIndices]
            '''

        length = len(data)
        if pairwise_constraint:
            length = len(data) // 2

        shuffled_data = []
        indices = np.random.choice(length, length, False)
        for i in indices:
            if pairwise_constraint:
                for j in range(2):
                    shuffled_data.append(data[2*i+j])
            else:
                shuffled_data.append(data[i])

        #make buffers based on length
        self.buffer = collections.defaultdict(list)

        if self.features is not None:
            self.featureBuffer = collections.defaultdict(list)
            if pairwise_constraint:
                indices = np.array([indices*2, indices*2+1]).reshape(2*length, order='F')
            self.features = self.features[indices]
        
        stride = 1
        if pairwise_constraint:
            stride = 2

        for i in range(0, len(shuffled_data), stride):
            #if pairwise constraint is set, need to pad the shorter sentence
            if pairwise_constraint:
                minIndex,minDatum = min(enumerate(shuffled_data[i:i+2]),
                                        key=lambda x: len(x[1][0]))

                #print(minIndex, minDatum)

                maxLength = len(shuffled_data[i+(1-minIndex)][0])
                minLength = len(minDatum[0])

                #print(minLength, maxLength)

                newDatum = [[], [], [], [], minDatum[4]]
                for j in range(4):
                    newDatum[j] = [0]*(maxLength-minLength) + minDatum[j]
                #adjust parent indices to add the length of the padding
                for j in range(len(minDatum[3])):
                    newDatum[2][(maxLength-minLength)+j] += (maxLength-minLength)

                #print(newDatum)
                
                self.buffer[maxLength].append(newDatum)
                self.buffer[maxLength].append(shuffled_data[i+(1-minIndex)])
                if self.features is not None:
                    self.featureBuffer[maxLength].append(i+minIndex)
                    self.featureBuffer[maxLength].append(i+(1-minIndex))                    
            else:
                datum = shuffled_data[i]
                length = len(datum[0])
                self.buffer[length].append(datum)
                if self.features is not None:
                    self.featureBuffer[length].append(i)

        #for datum in shuffled_data:
        #    length = len(datum[0])
        #    self.buffer[length].append(datum)
            
        if verbose:
            if balance is not None:
                c = collections.Counter([i[-1] for i in data])
                for key in c:
                    print(key, c[key])
            for key in sorted(self.buffer):
                print (key, len(self.buffer[key]))

    def iterBatches(self, batch_size = 100, shuffle=False, verbose=False, uniform=False):
        if self.pairwise:
            assert(batch_size % 2 == 0)

        batches = []
        for key in self.buffer.keys():
            for i in range(len(self.buffer[key]) // batch_size + 1):
                batches.append((key,i))
                
        if shuffle:
            np.random.shuffle(batches)
            
        for key,i in batches:
            if verbose:
                print(key, i, len(self.buffer[key]))

            if uniform and len(self.buffer[key][i*batch_size:(i+1)*batch_size]) < batch_size:
                continue

            ret = [[], [], [], [], []]
            for datum in self.buffer[key][i*batch_size:(i+1)*batch_size]:
                ret[0].append(datum[0])
                ret[1].append(datum[2])
                ret[2].append(datum[1])
                ret[3].append(datum[3])
                ret[4].append(datum[4])

            if self.features is not None:
                indices = self.featureBuffer[key][i*batch_size:(i+1)*batch_size]
                yield ret[:4] + [self.features[indices]] + ret[4:]
            else:
                yield ret

    @property
    def labels(self):
        labels = []
        for key in self.buffer.keys():
            for datum in self.buffer[key]:
                labels.append(datum[-1])
        return labels
