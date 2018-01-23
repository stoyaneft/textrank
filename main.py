from summarizer import summarize
from keywords import extract_keywords
# from text_preprocessor import format_text

SENTENCES_COUNT = 25


def main():
    extract_keywords_test()


def summary_test():
    print(summarize("chapter17.txt", SENTENCES_COUNT))


def extract_keywords_test():
    result = extract_keywords("article.txt")
    print(result)
    print(len(result))


if __name__ == "__main__":
    main()
