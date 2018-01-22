from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.cluster.util import cosine_distance
import nltk
import string

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def sentence_similarity(s1, s2):
    words1 = _get_words_from_sentence(s1)
    words2 = _get_words_from_sentence(s2)
    print(words1)
    all_words = list(set(words1 + words2))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}
    vector1 = _words_to_vector(words1, all_words, word_to_index)
    vector2 = _words_to_vector(words2, all_words, word_to_index)
    return 1 - cosine_distance(vector1, vector2)


def _get_words_from_sentence(sentence):
    return [word for word in word_tokenize(sentence.lower())
            if word not in STOP_WORDS]


def _words_to_vector(words, all_words, word_to_index):
    vector = [0] * len(all_words)
    for word in words:
        vector[word_to_index[word]] += 1
    return vector
