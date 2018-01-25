from summarizer import summarize
from keywords import extract_keywords
# from text_preprocessor import format_text
from textrank_util import get_text_from_file

SENTENCES_COUNT = 20
KEYWORDS_COUNT = 10


def main():
    # extract_keywords_test()
    summary_test("article2.txt")


def summary_test(filename, sentences_count=SENTENCES_COUNT):
    text = get_text_from_file(filename)
    summary = summarize(text, sentences_count)
    print("\n-----\n".join(summary))


def extract_keywords_test(filename, keywords_count):
    text = get_text_from_file(filename)
    result = extract_keywords(text, keywords_count)
    print(result)
    print(len(result))


if __name__ == "__main__":
    main()
