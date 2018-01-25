import numpy as np
import logging
from textrank_util import LOGGER_FORMAT

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.basicConfig(format=LOGGER_FORMAT)

beta = 0.85


def get_score(graph, vertex_idx, beta, scores):
    neighbours = [i for i in range(len(graph)) if graph[vertex_idx][i] != 0]
    score = 0
    for idx in neighbours:
        # LOGGER.debug(score)
        score += scores[idx] * graph[vertex_idx][idx] / graph[idx].sum()
    return beta * score + 1 - beta


# It is expected that A[i][i] = 0
def page_rank(graph, eps=0.0001, beta=0.85):
    iteration = 1
    # scores = np.random.rand(1, len(graph))[0]
    scores = np.ones(len(graph)) / len(graph)
    # print('init scores', scores)
    # LOGGER.debug("Graph: %s", str(graph))
    # LOGGER.debug("TEST: %f", graph[0].sum())
    test = np.apply_along_axis(lambda row: row.sum(), 1, graph)
    LOGGER.debug(test.argmin())
    LOGGER.debug(test.min())
    LOGGER.debug(test)
    while True:
        new_scores = [get_score(graph, idx, beta, scores)
                      for idx in range(len(graph))]
        score_change = [abs(scores[i] - new_scores[i])
                        for i in range(len(graph))]
        # LOGGER.debug("Score change: %s", str(score_change))
        not_converged_count = len(list(filter(lambda change: change > eps,
                                              score_change)))
        LOGGER.info("Iteration: %d", iteration)
        LOGGER.info("Not converged vertices: %d", not_converged_count)
        LOGGER.debug
        if not_converged_count == 0 or iteration > 200:
            LOGGER.info('Iterations: %d', iteration)
            return new_scores

        iteration += 1
        scores = new_scores


# results = page_rank(graph)
#
# print("Results:", results)
