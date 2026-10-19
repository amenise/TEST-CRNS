"""Evaluation metrics computed from stored logits (so every table can be regenerated without models)."""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, softmax


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(1) == y).mean())


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    return float(-log_softmax(logits.astype(np.float64), 1)[np.arange(len(y)), y].mean())


def ece(logits: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins (Guo et al. 2017)."""
    p = softmax(logits.astype(np.float64), 1)
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == y
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def entropy(logits: np.ndarray) -> np.ndarray:
    lp = log_softmax(logits.astype(np.float64), 1)
    return -(np.exp(lp) * lp).sum(1)


def per_class_accuracy(logits: np.ndarray, y: np.ndarray, num_classes: int = 100) -> np.ndarray:
    pred = logits.argmax(1)
    return np.array([(pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(num_classes)])


def prediction_collapse(logits: np.ndarray) -> dict:
    """How concentrated predictions are: #distinct predicted classes and normalised entropy of their histogram."""
    pred = logits.argmax(1)
    h = np.bincount(pred, minlength=logits.shape[1]) / len(pred)
    nz = h[h > 0]
    return dict(
        n_pred_classes=int((h > 0).sum()),
        pred_hist_entropy=float(-(nz * np.log(nz)).sum() / np.log(logits.shape[1])),
        top_class_share=float(h.max()),
    )


def taxonomic_errors(pred: np.ndarray, y: np.ndarray, taxo: dict) -> dict:
    """Share of errors whose predicted species shares genus / family / order with the true species.

    taxo: {"genus": array(C), "family": array(C), "order": array(C)} indexed by class id.
    """
    wrong = pred != y
    out = dict(n_errors=int(wrong.sum()))
    for lvl in ["genus", "family", "order"]:
        same = taxo[lvl][pred[wrong]] == taxo[lvl][y[wrong]]
        out[f"err_same_{lvl}"] = float(same.mean()) if wrong.any() else np.nan
    return out


def summary(logits: np.ndarray, y: np.ndarray) -> dict:
    return dict(acc=accuracy(logits, y), nll=nll(logits, y), ece=ece(logits, y), **prediction_collapse(logits))
