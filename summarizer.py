from page_rank import page_rank
from textrank_util import text_to_sentences, tokenize_sentences
import numpy as np
from textrank_util import LOGGER_FORMAT
from textrank_util import get_tagged_sentences
import logging
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag_sents
from textrank_util import _should_skip_word_1
from math import sqrt

nltk.download("wordnet")

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


def summarize(text, sentences_count=20, cosine_similarity=True):
    LOGGER.info("Summarizing text")
    plain_sentences = text_to_sentences(text)
    sentences = tokenize_sentences(plain_sentences)
    sentences = remove_punctuation(sentences)
    sentences = get_tagged_sentences(sentences, cosine_similarity)
    sentences, plain_sentences = get_long_sentences(sentences, plain_sentences)
    LOGGER.debug("All word tags: %s",
                 str(set([tag for sentence in sentences
                          for word, tag in sentence])))
    graph = create_sentences_similarity_graph(sentences, cosine_similarity)

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


def symmetric_sentence_similarity(s1, s2):
    return (sentence_similarity(s1, s2) + sentence_similarity(s2, s1)) / 2


def cosine_sentence_similarity(s1, s2):
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


def sentence_similarity(s1, s2):
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in s1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in s2]

    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    for synset in synsets1:
        a = [synset.path_similarity(ss) for ss in synsets2]
        a = [ss for ss in a if ss]
        best_score = max(a) if len(a) else 0

        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score


def cosine_similarity(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is
    equal to (u.v / |u||v|).
    """
    return (np.dot(u, v) / (
                sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))


def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except Exception:
        return None


def create_sentences_similarity_graph(sentences, cosine_similarity):
    LOGGER.info("Creating sentences similarity graph")

    graph = np.zeros((len(sentences), len(sentences)))

    for idx1, s1 in enumerate(sentences):
        for idx2, s2 in enumerate(sentences):
            if idx1 < idx2:
                if cosine_similarity:
                    similarity = cosine_sentence_similarity(s1, s2)
                else:
                    similarity = symmetric_sentence_similarity(s1, s2)
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
