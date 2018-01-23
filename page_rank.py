import numpy as np

# np.random.seed(0)
# graph = np.random.rand(5, 5)
# graph = [
#     [0, 0.71518937,  0.60276338,  0.54488318,  0.4236548],
#     [0.64589411,  0,         0.891773,    0.96366276,  0.38344152],
#     [0.79172504,  0.52889492,  0,         0.92559664,  0.07103606],
#     [0.0871293,   0.0202184,   0.83261985,  0,          0.87001215],
#     [0.97861834,  0.79915856,  0.46147936,  0.78052918,  0]
# ]
#
# graph[0][0] = graph[1][1] = graph[2][2] = graph[3][3] = graph[4][4] = 0
# print(graph)
beta = 0.85


def get_score(graph, vertex_idx, beta, scores):
    neighbours = [i for i in range(len(graph)) if graph[vertex_idx][i] != 0]
    weights_sum = sum(graph[vertex_idx])
    score = 0
    for (idx, neighbour) in enumerate(neighbours):
        score += scores[idx] * graph[vertex_idx][idx] / weights_sum

    return beta * score + 1 - beta

# It is expected that A[i][i] = 0
def page_rank(graph, eps=0.0001, beta=0.85):
    i = 0
    # scores = np.random.rand(1, len(graph))[0]
    scores = np.ones(len(graph)) / len(graph)
    # print('init scores', scores)
    while True:
        new_scores = [get_score(graph, idx, beta, scores) for idx in range(len(graph))]
        x = [abs(scores[i] - new_scores[i]) <= eps for i in range(len(graph))]
        if all(x) or i > 200:
            print('iterations', i)
            return new_scores

        i += 1
        scores = new_scores

# results = page_rank(graph)
#
# print("Results:", results)
