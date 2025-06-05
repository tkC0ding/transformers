import numpy as np
import unicodedata
import re

class preprocess:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0 : 'SOS', 1 : 'EOS'}
        self.word2count = {}
        self.n_words = 3
    
    def addSentence(self, sentence:str):
        words = sentence.split(' ')
        for word in words:
            self.addWord(word)
    
    def addWord(self, word:str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1



def unicode2ascii(s:str):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize(s:str):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return(s)

def string2index(language:preprocess, s:str):
    word_list = s.split()
    a = []
    for word in word_list:
        if word in language.word2index:
            a.append(language.word2index[word])
        else:
            language.word2index[word] = language.n_words
            language.n_words += 1
            a.append(language.word2index[word])
    return a

def readData(filename:str):
    return [ [normalize(i[0]), normalize(i[1])] for i in [line.strip('\n').split('\t') for line in open(filename)]]

a = readData('data/eng-fra.txt')