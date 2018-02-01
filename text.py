from nltk import pos_tag
from summarizer import symmetric_sentence_similarity

a = pos_tag("Cats are beautiful animals.")
b = pos_tag("Some gorgeous creatures are felines.")

print(symmetric_sentence_similarity(a, b))
