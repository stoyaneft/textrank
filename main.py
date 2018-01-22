from text_preprocessor import text_to_sentences
from summarizer import sentence_similarity
from summarizer import create_sentences_similarity_graph
from page_rank import page_rank
# from text_preprocessor import format_text

SENTENCES_COUNT = 5

def main():
    # format_text("article")
    sentences = text_to_sentences("test.txt")
    sentence_similarity(sentences[0], sentences[1])
    graph = create_sentences_similarity_graph(sentences)
    scores = page_rank(graph)
    sorted_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:SENTENCES_COUNT]
    summary = [sentences[idx] for idx, _ in sorted(sorted_scores)]
    print("\n-----\n".join(summary))


if __name__ == "__main__":
    main()
