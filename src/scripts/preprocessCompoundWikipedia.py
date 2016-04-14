import sys
import gzip
import json

from nlp.readers.parsedWikipediaReader import ParsedWikipediaReader

si = ParsedWikipediaReader(sys.argv[1],
                           compounded=True)

total = int(sys.argv[2])
curr = []
for words in si.iterData(start=int(sys.argv[3])):
    curr.append(words)
    if len(curr) == 100000:
        print('saving ', total)
        with gzip.open('preprocessed.{}.json.gz'.format(total), 'w') as f:
            json.dump(curr, f)
        total += 1
        curr = []
