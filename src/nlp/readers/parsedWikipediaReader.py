import gzip
import json
import os

import numpy as np

from altlex.utils import dependencyUtils

class ParsedWikipediaReader:
    def __init__(self, indir, compounded=False):
        self.indir = indir
        self.compounded = compounded

    def iterPreprocessedData(self, shuffle=False):
        inputFiles = list(enumerate(sorted(os.listdir(self.indir))))
        print(len(inputFiles))
        if shuffle:
            np.random.shuffle(inputFiles)
        print(len(inputFiles))
        for index,inputFile in inputFiles:
            inputFileFullPath = os.path.join(self.indir, inputFile)
            print(index, inputFile)
            if inputFileFullPath.endswith('.json.gz'):
                with gzip.open(inputFileFullPath) as f:
                    sentences = json.load(f)

                if shuffle:
                    np.random.shuffle(sentences)
                    
                for sentence in sentences:
                    yield sentence

    def iterData(self, start=0, shuffle=False):
        inputFiles = list(enumerate(sorted(os.listdir(self.indir))))
        if shuffle:
            np.random.shuffle(inputFiles)
            
        for index,inputFile in inputFiles[start:]:
            inputFileFullPath = os.path.join(self.indir, inputFile)
            print(index, inputFile)
            if inputFileFullPath.endswith('.json.gz'):
                with gzip.open(inputFileFullPath) as f:
                    articles = json.load(f)

                if shuffle:
                    np.random.shuffle(articles)
                    
                for title in articles:
                    for sentence in articles[title]:
                        dep = sentence['dep']
                        words = sentence['words']

                        for i in range(len(dep)):
                            if self.compounded:
                                depTuples = dependencyUtils.tripleToList(dep[i], len(words[i]))
                                compounds = dependencyUtils.getCompounds(depTuples)
                                compoundsRange = {min(compounds[j] + [j]): max(compounds[j] + [j])
                                                  for j in compounds}
                                
                                final_words = []
                                j = 0
                                while j < len(words[i]):
                                    if j in compoundsRange:
                                        word = '_'.join(words[i][j:compoundsRange[j]+1]).lower()
                                        j += compoundsRange[j]+1-j
                                    else:
                                        word = words[i][j].lower()
                                        j += 1
                                        
                                    final_words.append(word)
                                #print(final_words)
                                yield final_words
                            else:
                                yield [j.lower() for j in words[i]]
