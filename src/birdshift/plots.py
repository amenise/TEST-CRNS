"""Figures. Every figure is saved as PNG + PDF in figures/<section>_<what>.{png,pdf}."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import FIGURES, ensure

_RUN_ID = time.strftime("%Y%m%d-%H%M%S")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
    "#1B9E77",
    "#E7298A",
    "#666666",
    "#A6761D",
]


def save(fig, name: str):
    """Save as PNG and PDF. If Windows refuses to overwrite an existing figure (file locked by a viewer or by the
    antivirus), write it to figures/_pending/ instead and warn, so that a figure run is never lost."""
    ensure(FIGURES)
    for ext in ("png", "pdf"):
        try:
            fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
        except PermissionError:
            pending = FIGURES / "_pending" / _RUN_ID  # fresh folder per run: never overwrites a file
            ensure(pending)
            fig.savefig(pending / f"{name}.{ext}", bbox_inches="tight")
            print(f"WARNING: figures/{name}.{ext} is locked; wrote figures/_pending/{_RUN_ID}/{name}.{ext}", flush=True)
    plt.close(fig)
    return FIGURES / f"{name}.png"


METHOD_ORDER = [
    "source",
    "bn",
    "tent",
    "pl",
    "eata",
    "sar",
    "t3a",
    "lame",
    "cotta",
    "rdumb",
    "anchored",
    "geo_tent",
    "geo_eata",
    "geo_anchored",
]


def method_color(m):
    base = m.split("-")[0].split("+")[0]
    return PALETTE[METHOD_ORDER.index(base) % len(PALETTE)] if base in METHOD_ORDER else "#333333"


def tta_figures(df, ranks, cd, bb):
    """Severity curves, gain heatmap, critical-difference diagram for the TTA benchmark."""
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    corr = df[df.group == "corruption"]
    # 1. accuracy vs severity (mean over corruptions and seeds)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    clean = df[df.group == "source_clean"].groupby("method").acc.mean()
    for m in methods:
        c = corr[corr.method == m].groupby("severity").acc.mean()
        xs, ys = [0] + list(c.index), [clean.get(m, np.nan)] + list(c.values)
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=3,
            lw=2.2 if m in ("source", "anchored") else 1.2,
            color=method_color(m),
            ls="--" if m == "source" else "-",
            label=m,
        )
    ax.set_xlabel("corruption severity (0 = clean S_test)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{bb}: accuracy under corruptions")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    save(fig, f"06_tta_{bb}_severity")
    # 2. gain over Source per corruption (severity 5) + seasonal domains
    piv = df.groupby(["method", "condition"]).acc.mean().unstack()
    keep = [c for c in piv.columns if c.endswith("_s5") or (c.endswith(":clean"))]
    gain = (piv[keep] - piv.loc["source", keep]).loc[[m for m in methods if m != "source"]]
    gain.columns = [c.replace("S_test:", "").replace(":clean", "").replace("_s5", " s5") for c in keep]
    fig, ax = plt.subplots(figsize=(0.55 * len(keep) + 1.5, 0.35 * len(gain) + 1))
    v = np.nanmax(np.abs(gain.values))
    im = ax.imshow(gain.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(gain.columns)), gain.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(gain)), gain.index)
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            ax.text(j, i, f"{100 * gain.values[i, j]:+.0f}", ha="center", va="center", fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="accuracy gain vs Source")
    ax.set_title(f"{bb}: gain over Source (points), seasonal domains and severity 5")
    save(fig, f"06_tta_{bb}_gain_heatmap")
    # 3. critical difference diagram (Demsar 2006)
    items = sorted(ranks.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(items) + 1))
    for i, (m, r) in enumerate(items):
        ax.plot([r], [i], "o", color=method_color(m))
        ax.text(r + 0.1, i, f"{m} ({r:.2f})", va="center", fontsize=7)
    best = items[0][1]
    ax.axvspan(best, best + cd, color="grey", alpha=0.15, label=f"CD = {cd:.2f}")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("mean rank over conditions (1 = best)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, f"06_tta_{bb}_cd_diagram")


def corruption_grid(img: np.ndarray, names, severities, corrupt_fn, name="03_corruption_examples"):
    fig, axes = plt.subplots(len(severities), len(names) + 1, figsize=(1.35 * (len(names) + 1), 1.4 * len(severities)))
    for r, s in enumerate(severities):
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(f"sev. {s}")
        for c, n in enumerate(names):
            axes[r, c + 1].imshow(corrupt_fn(img, n, s, 0))
            if r == 0:
                axes[r, c + 1].set_title(n.replace("_", "\n"), fontsize=7)
    axes[0, 0].set_title("clean", fontsize=7)
    for a in axes.ravel():
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
    return save(fig, name)
