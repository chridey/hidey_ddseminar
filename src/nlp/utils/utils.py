from nlp.utils import dependencyUtils

#extract all events from a dependency parse
def extractEvents(dependencies, words, duplicate=False, start=0, end=float('inf'), triples=True):
    if triples:
        depTuples = dependencyUtils.tripleToList(dependencies, len(words), duplicate)
    else:
        depTuples = dependencies
        
    compounds = dependencyUtils.getCompounds(depTuples)
    
    es = dependencyUtils.getAllEventsAndArguments(depTuples)

    events = []
    for e in sorted(es):
        if e < start or e >= end:
            continue
        
        if 'nsubjpass' in es[e]:
            predicate = words[e].lower() + '_(passive)'
            key = 'nsubjpass'
        else:
            predicate = words[e].lower()
            key = 'nsubj'

        arguments = [[], [], []]
        for index, argType in enumerate((key, 'dobj', 'iobj')):
            for j in es[e].get(argType, []):
                if j in compounds:
                    arg = '_'.join(words[min(compounds[j] + [j]): max(compounds[j] + [j])+1]).lower()
                else:
                    arg = words[j].lower()
                arguments[index].append(arg)
                
        events.append([predicate] + arguments)

    return events

def getEventIndices(event, vocab, add=False):
    predicate = event[0]
    arguments = event[1:]
    
    if predicate not in vocab:
        if add:
            vocab[predicate] = len(vocab)
        else:
            return None
        
    predicateIndex = vocab[predicate]
                        
    argument_indices = [[], [], []]
    for index, argType in enumerate(arguments):
        for arg in sorted(argType):
            if arg not in vocab:
                if add:
                    vocab[arg] = len(vocab)
                else:
                    return None
                
            argument_indices[index].append(vocab[arg])

    return predicateIndex, tuple(argument_indices[0]), tuple(argument_indices[1]), tuple(argument_indices[2])

#make interaction features
#combine everything that matches pattern a with pattern b
def makeInteractionFeatures(features, pattern1, pattern2):
    new_features = {}
    for i in itertools.product(filter(lambda x:pattern1 in x, features.keys()),
                               filter(lambda x:pattern2 in x, features.keys())):
        new_features['_'.join(i)] = True
    return new_features

def filterFeatures(features, patterns=None, antipatterns=None):
    new_features = {}
    for feature in features:
        if (patterns is None or any(pattern in feature for pattern in patterns)) and (antipatterns is None or not any(antipattern in feature for antipattern in antipatterns)):
            new_features[feature] = features[feature]

    return new_features

def modifyFeatureSet(features, include=None, ablate=None, interaction=None, add_bias=False):
    if include:
        features = filterFeatures(features,
                                  include.split(','),
                                  None)
        
    if ablate:
        features = filterFeatures(features,
                                  None,
                                  ablate.split(','))
                
    if interaction:
        filtered_features = filterFeatures(features,
                                           interaction['include'],
                                           interaction['ablate'])
                                                                    
        interaction_features = makeInteractionFeatures(filtered_features,
                                                       interaction['first'],
                                                       interaction['second'])
        features.update(interaction_features)

    if add_bias:
        features['bias'] = 1

    return features

def createModifiedDataset(dataset, include=None, ablate=None, interaction=None, add_bias=False):
    ret = []
    for data in dataset:
        ret.append(modifyFeatureSet(data, include, ablate, interaction, add_bias))
    return ret
