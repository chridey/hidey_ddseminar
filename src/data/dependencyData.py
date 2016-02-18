import toposort

class DependencyData:
    #take in a list of dependency trees, topologically sort them,
    #and for each tree return a list of word indices, relation indices, parent node indices, and an answer index
    def __init__(self):
        self.vocab = {}
        self.relations = {}
        self.answers = {}

    #determine all the
    def scan_vocab(self, data):
        ret = []
        for datum1,datum2,label in data:
            params = []
            for datum in (datum1, datum2):
                #do topological sort here, also remove nodes that don't have parents (including ROOT)
                #order the nodes left to right with the root as the rightmost node
                indices = toposort.toposort_flatten({i:{j[2]} for i,j in enumerate(datum) if j[2] is not None})

                idxs, rel_idxs, p = [], [], []
                parentLookup = {}
                for index in indices:
                    if datum[index][2] is None:
                        parentLookup[index] = len(indices)-1
                        continue
                    if index not in parentLookup:
                        parentLookup[index] = len(indices)-len(parentLookup)-1

                    word, relation, parent = datum[index]

                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
                    if relation not in self.relations:
                        self.relations[relation] = len(self.relations)

                    idxs.insert(0, self.vocab[word])
                    rel_idxs.insert(0, self.relations[relation])
                    p.insert(0, parentLookup[parent])

                params.extend([idxs, rel_idxs, p])

            ret.append(params + [label])

        return ret    
