import nltk.data
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
import string

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


def file_to_tokenized_sentences(file_name):
    text = get_text_from_file(file_name)
    return text_to_tokenized_sentences(text)


def text_to_tokenized_sentences(text):
    sentences = text_to_sentences(text)

    LOGGER.info("Clearing irrelevant sentences")

    sentences = [text_to_words(sentence) for sentence in sentences]
    # sentences = list(filter(lambda sentence: any(sentence), sentences))

    LOGGER.info("Extracted sentences count: %d", len(sentences))
    LOGGER.debug("Sentences: %s", str(sentences))

    return sentences



def text_to_words(text):
    def should_skip_word(word):
        return word not in STOP_WORDS and \
            not any([punct in word for punct in list(string.punctuation)])
    return list(filter(should_skip_word,
                       word_tokenize(text.lower())))


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
