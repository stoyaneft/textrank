import numpy as np

beta = 0.85


def get_score(graph, vertex_idx, beta, scores):

    neighbours = [i for i in range(len(graph)) if graph[vertex_idx][i] != 0]
    score = 0
    for idx in neighbours:
        score += scores[idx] * graph[vertex_idx][idx] / graph[idx].sum()
    return beta * score + 1 - beta


# It is expected that A[i][i] = 0
def page_rank(graph, eps=0.0001, beta=0.85):
    iteration = 1
    # scores = np.random.rand(1, len(graph))[0]
    scores = np.ones(len(graph)) / len(graph)
    # print('init scores', scores)
    while True:
        new_scores = [get_score(graph, idx, beta, scores)
                      for idx in range(len(graph))]
        not_converged = [i for i in range(len(graph))
                         if not abs(scores[i] - new_scores[i]) < eps]
        print("Iteration: ", iteration)
        print("Not converged vertices: ", len(not_converged))
        if not any(not_converged) or iteration > 200:
            print('Iterations: ', iteration)
            return new_scores

        iteration += 1
        scores = new_scores


# results = page_rank(graph)
#
# print("Results:", results)
