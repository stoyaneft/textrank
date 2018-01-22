from text_preprocessor import text_to_sentences


def main():
    text = ""
    sentences = text_to_sentences(text)
    [print(sentence) for sentence in sentences]


if __name__ == "__main__":
    main()
