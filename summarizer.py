from page_rank import page_rank
from textrank_util import text_to_sentences, tokenize_sentences
import numpy as np
from textrank_util import LOGGER_FORMAT
from textrank_util import get_tagged_sentences
import logging
import string
from textrank_util import _should_skip_word_1
from math import sqrt

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig(format=LOGGER_FORMAT)


def are_words_equal(word1, word2):
    pass


def remove_unwanted_words(sentence):
    return list(filter(_should_skip_word_1, sentence))


def remove_punctuation(sentences):
    def remove_punctuation_internal(sentence):
        return list(filter(
            lambda tagged_word: not any(
                [character in tagged_word[0]
                 for character in list(string.punctuation)]),
            sentence))
    return [remove_punctuation_internal(sentence)
            for sentence in sentences]


def get_long_sentences(sentences, plain_sentences):
    idxs = [idx for idx, sentence in enumerate(sentences)
            if len(sentence) > 4]
    return [sentences[i] for i in idxs], [plain_sentences[i] for i in idxs]


def summarize(text, sentences_count=20):
    LOGGER.info("Summarizing text")
    plain_sentences = text_to_sentences(text)
    sentences = tokenize_sentences(plain_sentences)
    sentences = remove_punctuation(sentences)
    sentences = get_tagged_sentences(sentences)
    sentences, plain_sentences = get_long_sentences(sentences, plain_sentences)
    LOGGER.debug("All word tags: %s",
                 str(set([tag for sentence in sentences
                          for word, tag in sentence])))
    graph = create_sentences_similarity_graph(sentences)

    file = open('testfile.txt', 'w')

    for idx, row in enumerate(graph):
        file.write("Sentence 1: %s\n" % sentences[idx])
        file.write("Sentence 2: %s\n" % sentences[np.argmax(row)])
        file.write(str(np.max(row)) + '\n')
    # raise Exception
    file.close()

    LOGGER.info('Calculating scores')
    scores = page_rank(graph)
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:sentences_count]
    LOGGER.info('Top scores: %s', str(sorted_scores))
    summary = [plain_sentences[idx] for idx, _ in sorted(sorted_scores)]
    file = open('testfile2.txt', 'w')
    [file.write("\tRank: %d, Score: %f\nSentence: %s\n" %
                (len(sentences) - i, score, sentences[idx]))
        for i, (idx, score) in enumerate(sorted(enumerate(scores),
                                         key=lambda item: item[1]))]
    file.close()
    LOGGER.info("Summarizing completed")
    return summary


def sentence_similarity(s1, s2):
    all_words = {word: idx for idx, word in enumerate(list(set(s1 + s2)))}

    vector1 = _words_to_vector(s1, all_words, all_words)
    vector2 = _words_to_vector(s2, all_words, all_words)

    if not any(vector1) or not any(vector2):
        # Sentences with only stop words and punctuation give no information
        return 0
    similarity = cosine_similarity(vector1, vector2)
    if similarity != 0:
        LOGGER.debug("Sentence 1: %s", s1)
        LOGGER.debug("Sentence 2: %s", s2)
        LOGGER.debug("Similarity: %f", similarity)
    return similarity


def cosine_similarity(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is
    equal to (u.v / |u||v|).
    """
    return (np.dot(u, v) / (
                sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))


def create_sentences_similarity_graph(sentences):
    LOGGER.info("Creating sentences similarity graph")

    graph = np.zeros((len(sentences), len(sentences)))

    for idx1, s1 in enumerate(sentences):
        for idx2, s2 in enumerate(sentences):
            if idx1 < idx2:
                similarity = sentence_similarity(s1, s2)
                graph[idx1][idx2] = similarity
                graph[idx2][idx1] = similarity
    LOGGER.debug("Graph: %s", str(graph))
    return graph


def _words_to_vector(words, all_words, word_to_index):
    vector = [0] * len(all_words)
    for word in words:
        if word[0] in 'hufflepuffgryffindorravenclawslytherinhogwarts':
            vector[word_to_index[word]] += 3
        else:
            vector[word_to_index[word]] += 1
    return vector
