import numpy as np

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



