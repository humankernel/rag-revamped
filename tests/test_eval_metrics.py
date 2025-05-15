import numpy as np

from tests.evals.metrics import (
    calc_precision,
    calc_recall,
    calc_mrr,
    calc_ndcg,
    calc_map,
)

preds = [[1, 2, 3, 4], [3, 2, 1, 4]]
truths = [[3], [1]]
cutoffs = [1, 3, 4]


def test_precision():
    """
    Precision@k: proportion of retrieved items in top-k that are relevant
    """
    precision = calc_precision(preds, truths, cutoffs)
    expected = [
        0.0,  # @1: no relevant items in top-1
        1 / 3,  # @3: 1 relevant out of 3 for each → (1/3 + 1/3)/2 = 1/3
        1 / 4,  # @4: 1 relevant out of 4 for each → (1/4 + 1/4)/2 = 0.25
    ]
    np.testing.assert_allclose(precision, expected, rtol=1e-5)


def test_recall():
    """
    Recall@k: proportion of relevant items that are retrieved
    """
    recall = calc_recall(preds, truths, cutoffs)
    expected = [
        0.0,  # @1: neither relevant item in top-1
        1.0,  # @3: both users found their item in top-3
        1.0,  # @4: same as above
    ]
    np.testing.assert_allclose(recall, expected, rtol=1e-5)


def test_mrr():
    """
    MRR@k: reciprocal rank of the first relevant item
    """
    mrr = calc_mrr(preds, truths, cutoffs)
    expected = [
        0.0,  # @1: no relevant in top-1
        1 / 3,  # @3: relevant at position 3 for both users → 1/3
        1 / 3,  # @4: same position
    ]
    np.testing.assert_allclose(mrr, expected, rtol=1e-5)


def test_ndcg():
    """
    NDCG@k: takes into account position of correct items (discounted gain)
    """
    ndcg = calc_ndcg(preds, truths, cutoffs)
    expected = [
        0.0,  # @1: no hits
        0.5,  # @3: hit at rank 3 → 1/log2(4) = 0.5, ideal DCG = 1
        0.5,  # @4: same as above
    ]
    np.testing.assert_allclose(ndcg, expected, rtol=1e-5)


def test_map():
    """
    MAP@k: average of precision values at ranks where relevant items occur
    """
    map_k = calc_map(preds, truths, cutoffs)
    expected = [
        0.0,  # @1: no relevant items in top-1
        1 / 3,  # @3: hit at rank 3 → precision = 1/3 → mean over 2 users = 1/3
        1 / 3,  # @4: same
    ]
    np.testing.assert_allclose(map_k, expected, rtol=1e-5)
