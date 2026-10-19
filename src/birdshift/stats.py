"""Paired statistics for comparing classifiers on the same test images.

- McNemar's test on per-image correctness of two methods (Dietterich 1998), exact binomial version when
  the number of discordant pairs is small, chi-square with continuity correction otherwise.
- Holm-Bonferroni step-down correction for multiple comparisons.
- Friedman test + Nemenyi critical difference over several conditions (Demsar 2006).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((~a & b).sum())  # a wrong, b right
    n10 = int((a & ~b).sum())  # a right, b wrong
    n = n01 + n10
    if n == 0:
        p = 1.0
    elif n < 25:
        p = float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        p = float(stats.chi2.sf(chi2, 1))
    return dict(n01=n01, n10=n10, p=p)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def friedman_nemenyi(scores: np.ndarray, alpha: float = 0.05) -> dict:
    """scores: (n_conditions, k_methods), higher is better. Returns mean ranks (1 = best), Friedman p, CD."""
    n, k = scores.shape
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, scores)
    p = float(stats.friedmanchisquare(*scores.T).pvalue) if k >= 3 else np.nan
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    return dict(mean_ranks=ranks.mean(0), friedman_p=p, cd=cd, n=n, k=k)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
