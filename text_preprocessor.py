import nltk.data
import re


def text_to_sentences(file_name):
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    print("Extracting sentences from " + file_name)
    fp = open(file_name)
    text = fp.read()
    sentences = tokenizer.tokenize(text)
    print("Extracted sentences count: %d" % len(sentences))
    return sentences


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
