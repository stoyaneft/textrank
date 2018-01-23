from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import pos_tag_sents
import nltk.data
import string
import numpy as np
from scipy.sparse import csc_matrix
from page_rank import page_rank
from textrank_util import file_to_sentences, words_to_indexed_words

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('averaged_perceptron_tagger')

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english') + list(string.punctuation))
ps = PorterStemmer()

TAG_CLASSES = ['NN', 'JJ']


def extract_keywords(file_name):
    sentences = file_to_sentences(file_name)
    sentences = [text_to_words(sentence) for sentence in sentences]
    words_for_graph = _get_words_for_graph(sentences)
    indexed_words = words_to_indexed_words(words_for_graph)
    graph = np.zeros((len(indexed_words), len(indexed_words)))
    for sentence in sentences:
        print(sentence)
        for idx in range(len(sentence) - 1):
            # TODO try only filtered words (nouns and adjectives)
            word1 = ps.stem(sentence[idx])
            word2 = ps.stem(sentence[idx + 1])
            if word1 in indexed_words and word2 in indexed_words:
                print(word1, word2)
                graph[indexed_words[word1]][indexed_words[word2]] = 1

    print(np.sum(graph))
    scores = pageRank(graph)
    print(scores)
    sorted_scores = sorted(enumerate(scores),
                           key=lambda item: item[1],
                           reverse=True)[:10]
    keywords = [words_for_graph[idx] for idx, score in sorted_scores]
    return np.unique(scores)
    return _get_words_for_graph(sentences)


def pageRank(G, s=.85, maxerr=.0001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G: matrix representing state transitions
       Gij is a binary value representing a transition from state i to j.
    s: probability of following a transition. 1-s probability of teleporting
       to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged.
    """
    G = G.astype(int)
    G = np.array([[0,0,0,0,0,1,0],
                  [0,1,1,0,0,0,0],
                  [1,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,1],
                  [0,0,0,0,0,1,1],
                  [0,0,0,1,1,0,1]])

    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G, dtype=np.float)
    rsums = np.array(A.sum(1))[:, 0]
    ri, ci = A.nonzero()
    print(ri)
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums == 0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    print("TEST")
    while np.sum(np.abs(r-ro)) > maxerr:
        print(np.sum(np.abs(r-ro)))
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0, n):
            # inlinks of state i
            Ai = np.array(A[:, i].todense())[:, 0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot(Ai*s + Di*s + Ei*(1-s))

    # return normalized pagerank
    return r/float(sum(r))


def _get_words_for_graph(words):
    tagged_words = pos_tag_sents(words)
    tagged_words = [word for sent_words in tagged_words for word in sent_words]
    tagged_words = list(filter(
        lambda tagged_word: tagged_word[1] in TAG_CLASSES,
        tagged_words)
        )
    all_words = [ps.stem(tagged_word[0]) for tagged_word in tagged_words]
    return list(set(all_words))


def text_to_words(text):
    def should_skip_word(word):
        return word not in STOP_WORDS and \
            not any([punct in word for punct in list(string.punctuation)])
    return list(filter(should_skip_word,
                       word_tokenize(text.lower())))
