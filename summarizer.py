from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from page_rank import page_rank
from textrank_util import file_to_sentences, tokenize_sentences
import nltk
from nltk import pos_tag_sents
import string
import numpy as np
from textrank_util import LOGGER_FORMAT
import logging
from nltk.stem import PorterStemmer

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig(format=LOGGER_FORMAT)
ps = PorterStemmer()

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))


def are_words_equal(word1, word2):
    pass


def summarize(file_name, sentences_count):
    LOGGER.info("Summarizing text")
    plain_sentences = file_to_sentences(file_name)
    sentences = tokenize_sentences(plain_sentences)
    sentences = pos_tag_sents(sentences)
    graph = create_sentences_similarity_graph(sentences)
    LOGGER.info('Calculating scores')
    scores = page_rank(graph)
    print(scores[219])
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:sentences_count]
    LOGGER.info('Top scores: %s', str(sorted_scores))
    summary = [plain_sentences[idx] for idx, _ in sorted(sorted_scores)]
    [LOGGER.info("Score: %f, Sentence: %s", score, plain_sentences[idx])
        for idx, score in sorted_scores]
    LOGGER.info("Summarizing completed")
    return summary


def fix_tag(tag):
    if ('NN' in tag):
        return 'NN'
    return tag


def stem_sentence(sentence):
    return [(ps.stem(word), fix_tag(tag)) for word, tag in sentence]


def sentence_similarity(s1, s2):
    s1 = stem_sentence(s1)
    s2 = stem_sentence(s2)
    all_words = list(set(s1 + s2))
    word_to_index = {word: idx for idx, word in enumerate(all_words)}

    vector1 = _words_to_vector(s1, all_words, word_to_index)
    vector2 = _words_to_vector(s2, all_words, word_to_index)

    if not any(vector1) or not any(vector2):
        # Sentences with only stop words and punctuation give no information
        return 0
    similarity = 1 - cosine_distance(vector1, vector2)
    if similarity != 0:
        LOGGER.debug("Sentence 1: %s", s1)
        LOGGER.debug("Sentence 2: %s", s2)
        LOGGER.debug("Similarity: %f", similarity)
    return similarity


def create_sentences_similarity_graph(sentences):
    LOGGER.info("Creating sentences similarity graph")

    graph = np.zeros((len(sentences), len(sentences)))

    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences):
            if idx1 < idx2:
                similarity = sentence_similarity(sentence1, sentence2)
                graph[idx1][idx2] = similarity
                graph[idx2][idx1] = similarity

    LOGGER.info("Similarity graph created. Now normalizing")
    # for idx in range(len(graph)):
    #     row_sum = graph[idx].sum()
    #     if (row_sum != 0):
    #         graph[idx] /= row_sum

    LOGGER.info("Graph normalizing completed")
    LOGGER.debug("Graph: %s", str(graph))
    return graph


def _words_to_vector(words, all_words, word_to_index):
    vector = [0] * len(all_words)
    for word in words:
        vector[word_to_index[word]] += 1
    return vector
