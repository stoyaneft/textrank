from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag_sents
import nltk.data
import string
import numpy as np
from page_rank import page_rank
from textrank_util import file_to_sentences, words_to_indexed_words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('averaged_perceptron_tagger')

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))
ps = PorterStemmer()

TAG_CLASSES = ['NN', 'JJ']


def extract_keywords(file_name):
    sentences = file_to_sentences(file_name)
    sentences = [text_to_words(sentence) for sentence in sentences]
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

    # print(np.sum(graph))
    scores = page_rank(graph)
    print("Scores", len(scores))
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:10]
    keywords = [words_for_graph[idx] for idx, score in sorted_scores]
    return keywords
    return _get_words_for_graph(sentences)


def _get_words_for_graph(words):
    tagged_words = pos_tag_sents(words)
    tagged_words = [word for sent_words in tagged_words for word in sent_words]
    tagged_words = list(filter(
        lambda tagged_word: tagged_word[1] in TAG_CLASSES,
        tagged_words)
        )
    all_words = [ps.stem(tagged_word[0]) for tagged_word in tagged_words]
    return list(set(all_words))


def text_to_words(text):
    def should_skip_word(word):
        return word not in STOP_WORDS and \
            not any([punct in word for punct in list(string.punctuation)])
    return list(filter(should_skip_word,
                       word_tokenize(text.lower())))
