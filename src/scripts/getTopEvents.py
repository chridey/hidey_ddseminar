import sys
import gzip
import json

infile = sys.argv[1]
vocabfile = sys.argv[2]
n = int(sys.argv[3])

with open(infile) as f:
    events = json.load(f)

with gzip.open(vocabfile) as f:
    vocab = json.load(f)
reverse_vocab = {j:i for i,j in vocab.items()}

queries = '''dog barked
glass fell
glass broke
kid fell
obama spoke
congress passed law
he gave book her
hogwarts closing
astronauts launched space_shuttle
john bought milk'''.split('\n')

for key in sorted(events):
    goodEvents = 0
    print(key, queries[int(key)])
    for event,PMI in sorted(events[key], key=lambda x:x[-1])[::-1]:
        if len(event[1]) != 1 or len(event[2]) > 1 or len(event[3]) > 1:
            continue

        predicate = reverse_vocab[event[0]]
        subjects = reverse_vocab[event[1][0]]
        objects = None
        if len(event[2]):
            objects = reverse_vocab[event[2][0]]
        iobjects = None
        if len(event[3]):
            iobjects = reverse_vocab[event[3][0]]

        print(subjects, predicate, objects, iobjects, PMI)
        goodEvents += 1

        if goodEvents > n:
            break
