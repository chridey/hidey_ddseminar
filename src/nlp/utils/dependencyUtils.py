import collections

def tripleToList(dependencies, length, multipleParents=False, ignoreOutOfBounds=False):
    '''take in a list of triples (gov, dep, rel)
    and return a list of doubles (rel, gov)
    where the dep is in sentence order'''
    ret = [None] * length
    for gov,dep,rel in dependencies:
        if dep >= length and ignoreOutOfBounds:
            continue
        if multipleParents:
            if ret[dep] is None:
                ret[dep] = []
            ret[dep].append((rel.lower(), gov))
        else:
            ret[dep] = [(rel.lower(), gov)]
    return ret

    #TODO: handle cases with more than one parent

def combineDependencies(*dependenciesList):
    offset = 0
    combined = []
    for dependencies in dependenciesList:
        for dependency in dependencies:
            if dependency is None:
                combined.append(None)
            else:
                appendee = []
                for parent in dependency:
                    appendee.append((parent[0], parent[1]+offset))
                combined.append(appendee)
        offset += len(dependencies)
    return combined

def splitDependencies(dependencies, connectiveIndices):
    '''split the dependency tree into two trees,
    removing any links that cross the connective
    and adding new root nodes as necessary
    '''
    start, end = connectiveIndices
    altlexIndices = set(range(start, end))
    newDependencies = {'prev': [None]*start,
                       'altlex': [None]*(end-start),
                       'curr': [None]*(len(dependencies)-end)}
    for dep,rel_gov in enumerate(dependencies):
        if rel_gov is None:
            continue
        rel,gov = rel_gov
        #if the relationship crosses a barrier, remove it
        if gov == -1:
            if dep < start:
                newDependencies['prev'][dep] = rel,gov
            elif dep >= end:
                newDependencies['curr'][dep-end] = rel,gov
            elif dep in altlexIndices:
                newDependencies['altlex'][dep-start] = rel,gov
        elif dep < start and gov < start:
            newDependencies['prev'][dep] = rel,gov
        elif dep >= end and gov >= end:
            newDependencies['curr'][dep-end] = rel,gov-end
        elif dep in altlexIndices and gov in altlexIndices:
            newDependencies['altlex'][dep-start] = rel,gov-start

    #finally, make sure all the new dependencies have a root node
    
    for section in newDependencies:
        if any(i == ('root', -1) for i in newDependencies[section]):
            continue
        root = 0
        for dependency in newDependencies[section]:
            if dependency is not None:
                #if dependency[1] >= len(newDependencies[section]):
                #    continue
                if dependency[-1] != -1 and newDependencies[section][dependency[1]] is None:
                    #if root < len(newDependencies[section]):
                    root = dependency[1]
                    newDependencies[section][root] = 'root',-1 #UNDO
        if len(newDependencies[section]) and newDependencies[section][root] is None:
            newDependencies[section][root] = 'root',-1
    return newDependencies

def getRoot(parse):
    for i in range(len(parse)):
        if parse[i] is not None and len(parse[i]) == 1 and parse[i][0][0] == 'root':
            return i
    return None

def iterDependencies(parse):
    '''iterate over the dependency structure from
    the leaves to the root
    (useful for dependency embeddings)'''
    pass

def getEventAndArguments(parse):
    '''take in a dependency parse and return the
    main event (usually a verb) and any corresponding
    (noun) arguments'''
    root = getRoot(parse)
    if root is None:
        return None,[]
    arguments = []
    for i in range(len(parse)):
        if parse[i] is not None and any(j[1] == root for j in parse[i]):
            arguments.append(i)
    return root,arguments

#modify a dependency parse so that compounds/names/mwes are combined
def getCompounds(parse):
    ret = collections.defaultdict(list)
    for i in range(len(parse)):
        if parse[i] is None:
            continue
        for j in parse[i]:
            if j[0] in ('compound', 'name', 'mwe'):
                ret[j[1]].append(i)
    return ret

def getAllEventsAndArguments(parse):
    '''given a dependency parse return a list of all
    events (verbs) and their nsubj,nsubjpass,dobj,and iobj'''

    #use subjects and objects to find the predicates
    #0 is rel, 1 is gov
    ret = collections.defaultdict(dict)
    for i in range(len(parse)):
        if parse[i] is None:
            print('Problem with parse: {}'.format(parse))
            continue
        for j in parse[i]:
            if j[0] in ('nsubj', 'nsubjpass', 'dobj', 'iobj'):
                if j[0] not in ret[j[1]]:
                    ret[j[1]][j[0]] = []

                #TODO: rerun, just changed this
                ret[j[1]][j[0]].append(i)

    return ret
                                                            
