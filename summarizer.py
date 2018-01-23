from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.cluster.util import cosine_distance
from page_rank import page_rank
from textrank_util import file_to_sentences
import nltk
import string
import numpy as np

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def summarize(file_name, sentences_count):
    sentences = file_to_sentences(file_name)
    sentence_similarity(sentences[0], sentences[1])
    graph = create_sentences_similarity_graph(sentences)
    print('Calculating scores')
    scores = page_rank(graph)
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:sentences_count]
    print('top scores', sorted_scores)
    summary = [sentences[idx] for idx, _ in sorted(sorted_scores)]
    print("\n-----\n".join(summary))


def sentence_similarity(s1, s2):
    words1 = _get_words_from_sentence(s1)
    words2 = _get_words_from_sentence(s2)

    all_words = list(set(words1 + words2))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}

    vector1 = _words_to_vector(words1, all_words, word_to_index)
    vector2 = _words_to_vector(words2, all_words, word_to_index)

    return 1 - cosine_distance(vector1, vector2)


def create_sentences_similarity_graph(sentences):
    print("Creating sentences similarity graph")

    graph = np.zeros((len(sentences), len(sentences)))

    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences):
            if idx1 != idx2:
                graph[idx1][idx2] = sentence_similarity(sentence1, sentence2)

    for idx in range(len(sentences)):
        row_sum = graph[idx].sum()
        if (row_sum != 0):
            graph[idx] /= row_sum

    return graph


def _get_words_from_sentence(sentence):
    return [word for word in word_tokenize(sentence.lower())
            if word not in STOP_WORDS]


def _words_to_vector(words, all_words, word_to_index):
    vector = [0] * len(all_words)
    for word in words:
        vector[word_to_index[word]] += 1
    return vector
