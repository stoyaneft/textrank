from nltk.cluster.util import cosine_distance
from page_rank import page_rank
from textrank_util import text_to_sentences, tokenize_sentences
import numpy as np
from textrank_util import LOGGER_FORMAT
from textrank_util import get_text_from_file
from textrank_util import get_tagged_sentences
import logging
import string
from textrank_util import _should_skip_word_1

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
                [word in tagged_word[0]
                 for word in list(string.punctuation)]),
            sentence))
    return [remove_punctuation_internal(sentence)
            for sentence in sentences]


def summarize_from_file(file_name, sentences_count=20):
    return summarize(get_text_from_file(file_name), sentences_count)


def summarize(text, sentences_count=20):
    LOGGER.info("Summarizing text")
    plain_sentences = text_to_sentences(text)
    sentences = tokenize_sentences(plain_sentences)
    sentences = get_tagged_sentences(sentences)
    sentences = remove_punctuation(sentences)
    LOGGER.debug("All word tags: %s",
                 str(set([tag for sentence in sentences
                          for word, tag in sentence])))
    # sentences = list(map(remove_unwanted_words, sentences))
    graph = create_sentences_similarity_graph(sentences)
    LOGGER.info('Calculating scores')
    scores = page_rank(graph)
    print(scores[219])
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:sentences_count]
    LOGGER.info('Top scores: %s', str(sorted_scores))
    summary = [plain_sentences[idx] for idx, _ in sorted(sorted_scores)]
    [LOGGER.info("Rank: %d, Score: %f, Sentence: %s",
                 len(sentences) - i, score, sentences[idx])
        for i, (idx, score) in enumerate(sorted(enumerate(scores),
                                         key=lambda item: item[1]))]
    LOGGER.info("Summarizing completed")
    return summary


def sentence_similarity(s1, s2):
    all_words = {word: idx for idx, word in enumerate(list(set(s1 + s2)))}

    vector1 = _words_to_vector(s1, all_words, all_words)
    vector2 = _words_to_vector(s2, all_words, all_words)

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

    for idx1, s1 in enumerate(sentences):
        for idx2, s2 in enumerate(sentences):
            if idx1 < idx2:
                similarity = sentence_similarity(s1, s2)
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
