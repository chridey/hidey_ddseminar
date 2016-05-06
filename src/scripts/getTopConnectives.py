import sys
import os
import json
import collections

indir = sys.argv[1]
n = int(sys.argv[2])
causalfile = sys.argv[3]

pos = collections.defaultdict(float)
neg = collections.defaultdict(float)

causal = set()
with open(causalfile) as f:
    j = json.load(f)
for phrase in j:
    causal.update(set(phrase))
    
for filename in os.listdir(indir):
    print(filename)
    with open(os.path.join(indir, filename)) as f:
        predictions = json.load(f)

    for word in predictions:
        if word.lower() in causal:
            continue
        pos[word.lower()] += predictions[word].get("1", 0)
        neg[word.lower()] += predictions[word].get("0", 0)        

s = sorted(pos.items(), key=lambda x:x[1])[-n:]

for word,count in s:
    print(word, count, neg[word])
