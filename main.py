from summarizer import summarize
from keywords import extract_keywords
# from text_preprocessor import format_text

SENTENCES_COUNT = 20

def get_summary(text):
    return '''
    It was Quirrell.
    "You!" gasped Harry.
    Quirrell smiled. His face wasn't twitching at all.
    "Me," he said calmly. "I wondered whether I'd be meeting you here,
    Potter."
    "But I thought -- Snape --"
    "Severus?" Quirrell laughed, and it wasn't his usual quivering treble,
    either, but cold and sharp. "Yes, Severus does seem the type, doesn't
    he? So useful to have him swooping around like an overgrown bat. Next to
    him, who would suspect p-p-poor, st-stuttering P-Professor Quirrell?"
    Harry couldn't take it in. This couldn't be true, it couldn't.
    "But Snape tried to kill me!"
    "No, no, no. I tried to kill you. Your friend Miss Granger accidentally
    knocked me over as she rushed to set fire to Snape at that Quidditch
    match. She broke my eye contact with you. Another few seconds and I'd
    have got you off that broom. I'd have managed it before then if Snape
    hadn't been muttering a countercurse, trying to save you."
    "Snape was trying to save me?"
    "Of course," said Quirrell coolly. "\Why do you think he wanted to
    referee your next match? He was trying to make sure I didn't do it
    again. Funny, really... he needn't have bothered. I couldn't do anything
    with Dumbledore watching. All the other teachers thought Snape was
    trying to stop Gryffindor from winning, he did make himself unpopular...
    and what a waste of time, when after all that, I'm going to kill you
    tonight."
    '''


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
