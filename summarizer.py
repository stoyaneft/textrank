from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from page_rank import page_rank
from textrank_util import file_to_sentences
import nltk
import string
import numpy as np
from textrank_util import LOGGER_FORMAT
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig(format=LOGGER_FORMAT)

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def summarize(file_name, sentences_count):
    sentences = file_to_sentences(file_name)
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

    # LOGGER.info(str(s1))
    # LOGGER.info(str(s2))
    all_words = list(set(s1 + s2))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}

    vector1 = _words_to_vector(s1, all_words, word_to_index)
    vector2 = _words_to_vector(s2, all_words, word_to_index)

    if not any(vector1) or not any(vector2):
        LOGGER.debug("Empty sentences")
        # TODO remove sentences that contain only stopwords or punctuation
        # such sentences cause devision by 0 and give no information
        return 0
    return 1 - cosine_distance(vector1, vector2)


def create_sentences_similarity_graph(sentences):
    LOGGER.info("Creating sentences similarity graph")

    graph = np.zeros((len(sentences), len(sentences)))

    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences):
            if idx1 != idx2:
                graph[idx1][idx2] = sentence_similarity(sentence1, sentence2)

    LOGGER.info("Similarity graph created. Now normalizing")
    for idx in range(len(sentences)):
        row_sum = graph[idx].sum()
        if (row_sum != 0):
            graph[idx] /= row_sum

    LOGGER.info("Graph normalizing completed")
    LOGGER.debug("Graph: %s", str(graph))
    return graph


def _words_to_vector(words, all_words, word_to_index):
    vector = [0] * len(all_words)
    for word in words:
        vector[word_to_index[word]] += 1
    return vector
