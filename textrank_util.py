import nltk.data
import re

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def file_to_sentences(file_name):
    text = get_text_from_file(file_name)
    return text_to_sentences(text)


def text_to_sentences(text):
    print("Extracting sentences")
    sentences = tokenizer.tokenize(text)
    print("Extracted sentences count: %d" % len(sentences))
    return sentences


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
