from nltk.stem import PorterStemmer
import nltk
from nltk import pos_tag_sents
import nltk.data
import numpy as np
from page_rank import page_rank
from textrank_util import file_to_tokenized_sentences, words_to_indexed_words
from textrank_util import LOGGER_FORMAT
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig(format=LOGGER_FORMAT)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()

TAG_CLASSES = ['NN', 'JJ']


def extract_keywords(file_name, keywords_count=10):
    LOGGER.info("Extracting keywords")
    sentences = file_to_tokenized_sentences(file_name)
    LOGGER.info(sentences)
    words_for_graph = _get_words_for_graph(sentences)
    indexed_words = words_to_indexed_words(words_for_graph)
    graph = np.zeros((len(indexed_words), len(indexed_words)))
    for sentence in sentences:
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
                           reverse=True)[:keywords_count]
    keywords = [words_for_graph[idx] for idx, score in sorted_scores]
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
