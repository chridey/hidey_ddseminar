from nlp.utils import dependencyUtils

#extract all events from a dependency parse
def extractEvents(dependencies, words, duplicate=False):
    depTuples = dependencyUtils.tripleToList(dependencies, len(words), duplicate)

    compounds = dependencyUtils.getCompounds(depTuples)
    
    es = dependencyUtils.getAllEventsAndArguments(depTuples)

    events = []
    for e in es:
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
