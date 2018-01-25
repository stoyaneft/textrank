from summarizer import summarize_from_file
from keywords import extract_keywords_from_file
# from text_preprocessor import format_text

SENTENCES_COUNT = 20


def main():
    extract_keywords_test()
    # summary_test()


def summary_test():
    summary = summarize_from_file("article.txt", SENTENCES_COUNT)
    print("\n-----\n".join(summary))


def extract_keywords_test():
    result = extract_keywords_from_file("article.txt")
    print(result)
    print(len(result))


if __name__ == "__main__":
    main()
