import numpy as np
import pytest

from birdshift.data import stream_order
from birdshift.metrics import accuracy, ece, nll, prediction_collapse, taxonomic_errors
from birdshift.stats import friedman_nemenyi, holm, mcnemar


def test_accuracy_nll():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    y = np.array([0, 1, 1])
    assert accuracy(logits, y) == pytest.approx(2 / 3)
    p_correct = np.exp(2) / (np.exp(2) + 1)
    assert nll(logits, y) == pytest.approx(-(2 * np.log(p_correct) + np.log(1 - p_correct)) / 3)


def test_ece_perfect_and_overconfident():
    # confidence 1.0 and always right -> ECE 0 ; confidence ~1 and always wrong -> ECE ~1
    logits = np.array([[50.0, 0.0]] * 10)
    assert ece(logits, np.zeros(10, int)) == pytest.approx(0, abs=1e-9)
    assert ece(logits, np.ones(10, int)) == pytest.approx(1, abs=1e-6)


def test_collapse():
    logits = np.zeros((10, 5))
    logits[:, 2] = 1
    c = prediction_collapse(logits)
    assert c["n_pred_classes"] == 1 and c["top_class_share"] == 1 and c["pred_hist_entropy"] == 0


def test_taxonomic():
    taxo = {"genus": np.array([0, 0, 1]), "family": np.array([0, 0, 0]), "order": np.array([0, 0, 0])}
    r = taxonomic_errors(np.array([1, 2, 0]), np.array([0, 0, 0]), taxo)
    assert r["n_errors"] == 2 and r["err_same_genus"] == 0.5 and r["err_same_family"] == 1.0


def test_mcnemar():
    a = np.array([1] * 50 + [0] * 50, bool)
    assert mcnemar(a, a)["p"] == 1.0
    b = np.ones(100, bool)  # b fixes 50 errors of a, never the reverse
    r = mcnemar(a, b)
    assert r["n01"] == 50 and r["n10"] == 0 and r["p"] < 1e-10


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_nemenyi_cd_matches_demsar_table():
    # Demsar 2006 Table 5(a): q_0.05 = 2.569 for k = 4
    scores = np.random.default_rng(0).normal(size=(20, 4))
    r = friedman_nemenyi(scores)
    assert r["cd"] == pytest.approx(2.569 * np.sqrt(4 * 5 / (6 * 20)), rel=1e-3)


def test_stream_orders():
    y = np.repeat(np.arange(10), 5)
    iid = stream_order(y, "iid", 0)
    assert sorted(iid) == list(range(50)) and not np.array_equal(iid, stream_order(y, "iid", 1))
    cs = stream_order(y, "class_sorted", 0)
    assert sorted(cs) == list(range(50))
    assert (np.diff(y[cs]) != 0).sum() == 9  # each class appears as one contiguous block
