"""Analyses that explain *why* performance changes: feature drift, taxonomy of errors, photometry, per-species drops."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data


def taxonomy_arrays() -> dict:
    idx = data.index()
    g = idx.groupby("y")[["genus", "family", "order"]].first().reindex(range(data.NUM_CLASSES))
    return {lvl: pd.factorize(g[lvl])[0] for lvl in ["genus", "family", "order"]}


def chance_same_taxon(taxo: dict) -> dict:
    """Probability that a uniformly random *wrong* species shares the taxon of the true one (baseline for errors)."""
    out = {}
    c = len(taxo["genus"])
    for lvl, arr in taxo.items():
        same = (arr[:, None] == arr[None, :]).sum(1) - 1
        out[f"chance_same_{lvl}"] = float((same / (c - 1)).mean())
    return out


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two representations of the SAME images (n, d1), (n, d2)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def mmd_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Squared distance between feature means, normalised by the mean squared source feature norm."""
    return float(((x.mean(0) - y.mean(0)) ** 2).sum() / (x**2).sum(1).mean())


def class_conditional_shift(fs: np.ndarray, ys: np.ndarray, ft: np.ndarray, yt: np.ndarray) -> dict:
    """Decompose feature drift: distance between class centroids of source and target (same species), relative to
    the mean distance between centroids of different species in the source. >1 means a species moved further than
    the typical gap between two species (semantic/appearance shift)."""
    cs = np.stack([fs[ys == c].mean(0) for c in range(data.NUM_CLASSES)])
    ct = np.stack([ft[yt == c].mean(0) for c in range(data.NUM_CLASSES)])
    within = np.linalg.norm(cs - ct, axis=1)
    d = np.linalg.norm(cs[:, None] - cs[None], axis=2)
    between = d[np.triu_indices(len(cs), 1)].mean()
    return dict(centroid_shift_ratio=float(within.mean() / between), per_class_shift=within / between)


# ------------------------------------------------------------------ photometry (E2c)


def photometric_stats(imgs: np.ndarray) -> pd.DataFrame:
    """Per-image low-level statistics: brightness, contrast, saturation, sharpness, colour cast."""
    from scipy import ndimage

    x = imgs.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    sharp = np.array([ndimage.laplace(l).var() for l in lum])
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2) + 0.3 * np.sqrt(
        rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2
    )
    return pd.DataFrame(
        dict(
            brightness=lum.mean((1, 2)),
            contrast=lum.std((1, 2)),
            saturation=sat.mean((1, 2)),
            sharpness=sharp,
            colorfulness=colorfulness,
            blue_cast=(b - r).mean((1, 2)),
            green_cast=(g - 0.5 * (r + b)).mean((1, 2)),
        )
    )


def compare_distributions(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U per statistic with Cliff's delta effect size (|d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, else large; Romano et al. 2006)."""
    from .stats import holm

    rows = []
    for col in a.columns:
        u, p = stats.mannwhitneyu(a[col], b[col], alternative="two-sided")
        delta = 2 * u / (len(a) * len(b)) - 1
        rows.append(dict(stat=col, mean_a=a[col].mean(), mean_b=b[col].mean(), cliffs_delta=delta, p=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p"].values)
    df["effect"] = pd.cut(
        df["cliffs_delta"].abs(), [-1, 0.147, 0.33, 0.474, 2], labels=["negligible", "small", "medium", "large"]
    )
    return df


# ------------------------------------------------------------------ migration distance (per species)


def species_displacement(dom_a: str = "S", dom_b: str = "W") -> pd.DataFrame:
    """Per species: great-circle distance (km) between the median location of its observations in two seasons,
    computed from ALL metadata rows (pool + splits) of North America. Proxy for migratory displacement."""
    idx = data.index()
    idx = idx[idx["na"] & idx["lat"].notna()]
    rows = []
    for y, g in idx.groupby("y"):
        a, b = g[g["dom"] == dom_a], g[g["dom"] == dom_b]
        if len(a) < 5 or len(b) < 5:
            continue
        la, lo = np.radians(a["lat"].median()), np.radians(a["lon"].median())
        lb, lob = np.radians(b["lat"].median()), np.radians(b["lon"].median())
        dist = (
            6371
            * 2
            * np.arcsin(np.sqrt(np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin((lob - lo) / 2) ** 2))
        )
        rows.append(dict(y=y, name=g["name"].iloc[0], displacement_km=dist, dlat=b["lat"].median() - a["lat"].median()))
    return pd.DataFrame(rows)
