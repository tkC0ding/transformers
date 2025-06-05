from utils import readData
from utils import preprocess

input_lang = preprocess()
output_lang = preprocess()
FILENAME = 'data/eng-fra.txt'

data = readData(FILENAME)
for s in data:
    input_lang.addSentence(s[0])
    output_lang.addSentence(s[1])

