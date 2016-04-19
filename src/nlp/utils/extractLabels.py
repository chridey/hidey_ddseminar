import os
import sys
import collections
import json

import xml.etree.ElementTree as ET

def extractLabels(indir, labelfile):

    sentences = collections.defaultdict(dict)
    for filename in os.listdir(indir):
        with open(os.path.join(indir,filename)) as f:
            lines = f.read().splitlines()

        for line in lines:
            if line.startswith('<S3 '):
                xml = ET.fromstring(line)
                sentenceIndex = int(xml.attrib['id'])
                occurrence = collections.defaultdict(int)
                for wordIndex,item in enumerate(xml.text.split()):
                    word = '/'.join(item.split('/')[:-1])
                    sentences[filename][(sentenceIndex,wordIndex)] = word, occurrence[word]
                    occurrence[word] += 1

    #print(sentences.keys())

    tree = ET.parse(labelfile)
    root = tree.getroot()

    relations = collections.defaultdict(dict)
    for doc in root:
        #print(doc.tag, doc.attrib['id'], doc.text)
        #print(sorted(sentences[doc.attrib['id']].keys()))
        for relTuple in doc.text.splitlines():
            #print(relTuple)
            if not relTuple:
                continue
            rel, first, second = relTuple.split()
            sentenceIndex1, wordIndex1 = first.split('_')
            sentenceIndex2, wordIndex2 = second.split('_')
            word1, whichWord1 = sentences[doc.attrib['id']][(int(sentenceIndex1),int(wordIndex1))]
            word2, whichWord2 = sentences[doc.attrib['id']][(int(sentenceIndex2),int(wordIndex2))]        
            print(doc.attrib['id'], (sentenceIndex1, word1, whichWord1), (sentenceIndex2, word2, whichWord2))
            relations[doc.attrib['id']][((int(sentenceIndex1), word1, whichWord1),
                                         (int(sentenceIndex2), word2, whichWord2))] = rel


    return relations
