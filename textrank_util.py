import nltk.data
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag_sents
from nltk.corpus import stopwords
from nltk import pos_tag
import logging
import string
from nltk.stem import PorterStemmer

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

LOGGER_FORMAT = '%(name)s:%(levelname)s:%(message)s'

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
logging.basicConfig(format=LOGGER_FORMAT)


def file_to_sentences(file_name):
    text = get_text_from_file(file_name)
    return text_to_sentences(text)


def text_to_sentences(text):
    LOGGER.info("Extracting sentences")
    sentences = tokenizer.tokenize(text)

    LOGGER.info("Text tokenized. Sentences count: %d", len(sentences))
    return sentences


def tokenize_sentence(sentence):
    return pos_tag(word_tokenize(sentence))


def fix_tag(tag):
    if ('NN' in tag):
        return 'NN'
    elif 'VB' in tag:
        return 'VB'
    elif 'JJ' in tag:
        return 'JJ'
    elif 'RB' in tag:
        return 'RB'
    return tag


# DT - determiner (a, an, the ...)
# PRP - pronoun
# PRP$
# IN - preposition
# CC - conjunction
# CD - numeral
# SYM - symbol
# WP - that, what, who
# WP$ - whose
# WRB - how, why
# MD - can, would...
# PDT - all, both
# UH - uh, huh, amen
# RP - particles - if, about
# TO
# WDT - that, what, which
FUNCTIONAL_TAGS = ['DT', 'PRP', 'IN', 'CC', 'CD', 'RP', 'TO', 'PRP$', 'WDT'
                   'SYM', 'WP', 'WP$', 'WRB', 'MD', 'PDT', 'UH', 'EX'
                   ]


ps = PorterStemmer()


def is_not_function_word(tagged_word):
    word, tag = tagged_word
    return word.lower() not in STOP_WORDS and tag not in FUNCTIONAL_TAGS
print(STOP_WORDS)

def stem_sentence(sentence):
    return list([(ps.stem(word), fix_tag(tag))
                 for word, tag in filter(is_not_function_word, sentence)])


def tokenize_sentences(sentences):
    LOGGER.info("Clearing irrelevant sentences")

    sentences = [word_tokenize(sentence) for sentence in sentences]

    LOGGER.info("Extracted sentences count: %d", len(sentences))
    LOGGER.debug("Sentences: %s", str(sentences))

    return sentences


def get_stemmed_words(words):
    return list([ps.stem(word) for word, tag in filter(is_not_function_word, words)])


def get_tagged_words(sentences):
    tagged_sentences = pos_tag_sents(sentences)
    tagged_words = [word for sentence in tagged_sentences for word in sentence]
    return {word: fix_tag(key) for (word, key) in tagged_words}


def get_tagged_sentences(sentences):
    tagged_sentences = pos_tag_sents(sentences)
    # TODO stemming may not be a good idea
    return [stem_sentence(sentence) for sentence in tagged_sentences]


def sentence_to_words(sentence):
    return word_tokenize(sentence)


def _should_skip_word(tagged_word):
    word = tagged_word[0]
    return word not in STOP_WORDS and \
        not any([punct in word for punct in list(string.punctuation)])


# TODO unite with the other function
def _should_skip_word_1(tagger_word):
    word = tagger_word[0]
    tag = tagger_word[1]
    return word not in STOP_WORDS and \
        not any([punct in word for punct in list(string.punctuation)])


def filter_unwanted_words(sentence):
    return list(filter(_should_skip_word, sentence))


def get_text_from_file(file_name):
    fp = open(file_name, 'r')
    return fp.read()


def words_to_indexed_words(words):
    return {word: idx for idx, word in enumerate(words)}


def format_text(file_name):
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    fp = open(file_name)
    text = fp.read()
    text = text.replace(' .', '.')
    text = re.sub('\s+', ' ', text).strip()
    new_file_name = file_name.partition('.')[0] + "_formatted.txt"
    with open(new_file_name, "w") as text_file:
        print(text, file=text_file)
