from summarizer import summarize
from keywords import extract_keywords
# from text_preprocessor import format_text

SENTENCES_COUNT = 60


def main():
    extract_keywords_test()
    # summary_test()


def summary_test():
    summarize("article.txt", SENTENCES_COUNT)


def extract_keywords_test():
    result = extract_keywords("article.txt")
    print(result)
    print(len(result))


if __name__ == "__main__":
    main()
