import numpy as np
from deap import gp

from util.routing import GP_evolve_R
from util.sequencing import GP_evolve_S


def _average_rank(values):
    """Return average ranks (1-based) with tie handling."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)

    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_corr(x, y):
    """Compute Spearman correlation with safe fallbacks."""
    if len(x) != len(y) or len(x) <= 1:
        return 0.0

    xr = _average_rank(x)
    yr = _average_rank(y)
    x_std = np.std(xr)
    y_std = np.std(yr)

    if x_std == 0 or y_std == 0:
        return 0.0

    return float(np.corrcoef(xr, yr)[0, 1])


def _compute_decision_vector(routing_tree, sequencing_tree, decision_situations):
    decision_vector = []
    for routing_situation, sequencing_situation in decision_situations:
        selected_machine_index = GP_evolve_R(routing_tree, *routing_situation)
        decision_vector.append(selected_machine_index)
        job_position = GP_evolve_S(sequencing_situation, sequencing_tree)
        decision_vector.append(job_position)
    return decision_vector


def _iter_tree_subtrees(tree):
    """Yield subtree primitive trees rooted at every node index."""
    for node_idx in range(len(tree)):
        subtree_slice = tree.searchSubtree(node_idx)
        yield gp.PrimitiveTree(tree[subtree_slice])


def correlation(population, rd):
    """Compute node-wise Spearman correlation for each individual.

    For each node (as subtree root) in routing and sequencing trees, build a
    node-level decision vector and compare it with the individual's full
    decision vector using Spearman correlation.
    """
    decision_situations = rd["decision_situations"]

    for ind in population:
        node_corr = []

        for route_subtree in _iter_tree_subtrees(ind[0]):
            node_decision_vector = _compute_decision_vector(
                route_subtree, ind[1], decision_situations
            )
            if len(route_subtree) == len(ind[0]) or len(route_subtree) == 1:
                node_corr.append(0.0)
            else:
                node_corr.append(
                    _spearman_corr(ind.decision_vector, node_decision_vector)
                )

        ind.l_max = np.argmax(node_corr)

        node_corr = []

        for seq_subtree in _iter_tree_subtrees(ind[1]):
            node_decision_vector = _compute_decision_vector(
                ind[0], seq_subtree, decision_situations
            )
            if len(seq_subtree) == len(ind[1]) or len(seq_subtree) == 1:
                node_corr.append(0.0)
            else:
                node_corr.append(
                    _spearman_corr(ind.decision_vector, node_decision_vector)
                )

        ind.r_max = np.argmax(node_corr)
