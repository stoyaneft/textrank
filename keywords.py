from nltk.stem import PorterStemmer
import nltk
from nltk import pos_tag_sents
import nltk.data
import numpy as np
from page_rank import page_rank
from textrank_util import tokenize_sentences, words_to_indexed_words, text_to_sentences
from textrank_util import LOGGER_FORMAT, get_text_from_file, sentence_to_words
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig(format=LOGGER_FORMAT)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()

TAG_CLASSES = ['NN', 'JJ']


def extract_keywords_from_file(file_name):
    return extract_keywords(get_text_from_file(file_name))


def are_neighbours(word1, word2, sentences):
    sentences_words = [ sentence_to_words(sentence) for sentence in sentences ]
    for sentence_words in sentences_words:
        for i in range(len(sentence_words) - 1):
            if set([word1, word2]) == set([sentence_words[i], sentence_words[i+1]]):
                return (sentence_words[i], sentence_words[i+1])
    return False


def match_pairs(ranked_words, sentences, keywords_count):
    matched_pairs = []
    paired_words = set()

    for k1 in ranked_words[:keywords_count]:
        for k2 in ranked_words[:keywords_count*2]:
            if k1 != k2 and are_neighbours(k1, k2, sentences) \
            and k1 not in paired_words and k2 not in paired_words:
                matched_pairs.append((k1, k2))
                paired_words.add(k1)
                paired_words.add(k2)

    return matched_pairs


def extract_keywords(text, keywords_count=10):
    LOGGER.info("Extracting keywords")
    sentences = text_to_sentences(text)
    tokenized_sentences = tokenize_sentences(sentences)
    LOGGER.info(tokenized_sentences)
    words_for_graph = _get_words_for_graph(tokenized_sentences)
    indexed_words = words_to_indexed_words(words_for_graph)
    graph = np.zeros((len(indexed_words), len(indexed_words)))
    for sentence in tokenized_sentences:
        for idx in range(len(sentence) - 1):
            # TODO try only filtered words (nouns and adjectives)
            word1 = ps.stem(sentence[idx])
            word2 = ps.stem(sentence[idx + 1])
            if word1 in indexed_words and word2 in indexed_words:
                graph[indexed_words[word1]][indexed_words[word2]] = 1
                graph[indexed_words[word2]][indexed_words[word1]] = 1

    scores = page_rank(graph)
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)
    ranked_words = [words_for_graph[idx] for idx, score in sorted_scores]
    print('top ranked', ranked_words[:keywords_count])

    matched_pairs = match_pairs(ranked_words, sentences, keywords_count)
    paired_words = list(sum(matched_pairs, ()))
    keywords = [' '.join(pair) for pair in matched_pairs] + list(filter(
        lambda word: word not in paired_words,
        ranked_words
    ))[:keywords_count - len(paired_words)]
    print('matched pairs', matched_pairs)
    LOGGER.info("Extracting keywords completed")
    LOGGER.debug(keywords)
    return keywords


def _get_words_for_graph(words):
    tagged_words = pos_tag_sents(words)
    tagged_words = [word for sent_words in tagged_words for word in sent_words]
    tagged_words = list(filter(
        lambda tagged_word: tagged_word[1] in TAG_CLASSES,
        tagged_words)
        )
    all_words = [ps.stem(tagged_word[0]) for tagged_word in tagged_words]
    return list(set(all_words))


extract_keywords_from_file('article.txt')
