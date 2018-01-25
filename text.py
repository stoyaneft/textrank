from summarizer import sentence_similarity
import nltk

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sent1 = "The four houses are called Gryffindor, Hufflepuff, Ravenclaw, and Slytherin."
sent2 = "'HUFFLEPUFF!' shouted the hat again, and Susan scuttled off to sit next to Hannah."
print(tokenizer.tokenize(sent2))
