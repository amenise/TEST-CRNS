"""Figures for the data, baseline, distribution-shift and geo-prior sections (TTA/CTTA figures come from analyze_*).

python -m birdshift.cli.make_figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import data, plots
from ..corruptions import TEST_CORRUPTIONS, VAL_CORRUPTIONS, corrupt
from ..paths import RESULTS, RUNS

BB = {
    "dinov2": "DINOv2 ViT-S/14 (LayerNorm)",
    "vit_s16": "ViT-S/16 (LayerNorm)",
    "resnet50": "ResNet-50 (BatchNorm)",
}
BB = {k: v for k, v in BB.items() if (RESULTS / f"shift_{k}.csv").exists()}  # backbones that were run


def fig_data():
    idx = data.index()
    na = idx[idx["na"] & idx["lat"].notna() & idx["dom"].isin(list("SAWP"))]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, d in enumerate("SAWP"):
        g = na[na.dom == d]
        ax[0].scatter(g.lon, g.lat, s=0.3, alpha=0.25, color=plots.PALETTE[i], label=f"{d}: {data.DOMAINS[d]}")
    ax[0].set_xlabel("longitude")
    ax[0].set_ylabel("latitude")
    ax[0].set_xlim(-170, -50)
    ax[0].legend(markerscale=15, fontsize=6, frameon=False)
    ax[0].set_title("observations (all metadata rows, North America)")
    ax[1].bar(
        range(1, 13),
        na.groupby("m").lat.median().reindex(range(1, 13)),
        color=[plots.PALETTE["SAWP".index(na[na.m == m].dom.iloc[0])] for m in range(1, 13)],
    )
    ax[1].set_xlabel("month")
    ax[1].set_ylabel("median latitude")
    ax[1].set_ylim(25, 40)
    ax[1].set_title("observations move south in winter")
    plots.save(fig, "01_data_map_latitude")


def fig_baseline_shift():
    fig, axes = plt.subplots(1, len(BB), figsize=(4.5 * len(BB), 3.3), sharey=True)
    for ax, bb in zip(axes, BB):
        d = pd.read_csv(RESULTS / f"shift_{bb}.csv")
        clean = d[(d.corruption == "clean") & (d.split == "S_test")].acc.iloc[0]
        for i, c in enumerate(TEST_CORRUPTIONS):
            g = d[d.corruption == c].sort_values("severity")
            ax.plot([0] + list(g.severity), [clean] + list(g.acc), marker="o", ms=3, color=plots.PALETTE[i], label=c)
        for s in ["A_test", "W_test", "P_test"]:
            a = d[(d.split == s) & (d.corruption == "clean")].acc.iloc[0]
            ax.axhline(a, color="k", lw=0.6, ls=":")
            ax.text(5.05, a, s[0], fontsize=7, va="center")
        ax.set_title(BB[bb])
        ax.set_xlabel("severity (0 = clean S_test)")
    axes[0].set_ylabel("accuracy (Source model)")
    handles, labels = axes[0].get_legend_handles_labels()
    # legend below the panels so that it never hides a curve
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.13)
    )
    fig.suptitle("Controlled corruptions vs seasonal shift (dotted: A, W, P test domains)", fontsize=9)
    plots.save(fig, "05_shift_severity_curves")

    fig, ax = plt.subplots(figsize=(5.5, 3))
    rows = []
    for bb in BB:
        d = pd.read_csv(RESULTS / f"shift_{bb}.csv")
        for s in ["S_test", "S_hold", "A_test", "W_test", "P_test"]:
            r = d[(d.split == s) & (d.corruption == "clean")].iloc[0]
            rows.append(dict(bb=bb, split=s, acc=r.acc, ece=r.ece))
    t = (
        pd.DataFrame(rows)
        .pivot(index="split", columns="bb", values="acc")
        .loc[["S_test", "S_hold", "A_test", "W_test", "P_test"]]
    )
    t.plot.bar(ax=ax, color=plots.PALETTE[:2], rot=0)
    ax.set_ylabel("accuracy")
    ax.set_title("Seasonal shift: Source accuracy per domain (S_hold = same-domain reference)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=6)
    plots.save(fig, "05_shift_seasonal_bars")


def fig_why():
    # photometry
    t = pd.read_csv(RESULTS / "photometry_tests.csv")
    piv = t.pivot(index="stat", columns="target", values="cliffs_delta")
    cols = ["S_hold:clean", "A_test:clean", "W_test:clean", "P_test:clean", "S_test:low_light_s3", "S_test:haze_s3"]
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(piv[cols].values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(cols)), [c.replace(":clean", "").replace("S_test:", "") for c in cols], rotation=30, ha="right"
    )
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(len(piv)):
        for j, c in enumerate(cols):
            ax.text(j, i, f"{piv[c].iloc[i]:+.2f}", ha="center", va="center", fontsize=6)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Cliff's delta vs S_test")
    ax.set_title("Low-level image statistics: seasons vs corruptions")
    plots.save(fig, "05_why_photometry")
    # taxonomy of errors
    fig, axes = plt.subplots(1, len(BB), figsize=(4.5 * len(BB), 3), sharey=True)
    for ax, bb in zip(axes, BB):
        t = pd.read_csv(RESULTS / f"shift_taxonomy_{bb}.csv")
        t["sev"] = t.condition.str[-1].where(t.condition != "clean", "0").astype(int)
        t["grp"] = np.where(t.condition == "clean", t.split.str[0] + " (season)", "corr. s" + t.sev.astype(str))
        g = t.groupby("grp")[["err_same_genus", "err_same_family", "err_same_order"]].mean()
        g = g.loc[
            [
                x
                for x in ["S (season)", "A (season)", "W (season)", "P (season)"] + [f"corr. s{s}" for s in range(1, 6)]
                if x in g.index
            ]
        ]
        g.plot.bar(ax=ax, color=plots.PALETTE[2:5], rot=45)
        for lvl, ls in zip(["genus", "family", "order"], [":", "--", "-."]):
            ax.axhline(t[f"chance_same_{lvl}"].iloc[0], color="grey", ls=ls, lw=0.8)
        ax.set_title(BB[bb])
        ax.legend(["same genus", "same family", "same order"], fontsize=6, frameon=False)
    axes[0].set_ylabel("share of errors (grey: chance)")
    fig.suptitle("Errors stay taxonomically coherent under the seasonal shift, not under corruptions", fontsize=9)
    plots.save(fig, "05_why_taxonomy")
    # layer-wise BN mismatch (ResNet) and block CKA
    l = pd.read_csv(RESULTS / "failure_layers_resnet50.csv")
    b = l[l.kind == "bn_mismatch"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    layers = list(dict.fromkeys(b.layer))
    for i, c in enumerate(b.condition.unique()):
        g = b[b.condition == c].set_index("layer").loc[layers]
        axes[0].plot(
            range(len(layers)),
            g.mean_shift,
            lw=1.6 if "clean" in c else 1,
            color=plots.PALETTE[i],
            ls="-" if "clean" in c else "--",
            label=c.replace(":clean", "").replace("S_test:", ""),
        )
    axes[0].set_xlabel("BatchNorm layer (input -> output)")
    axes[0].set_ylabel("|mu_target - mu_source| / sigma_source")
    axes[0].set_title("ResNet-50: BN statistics mismatch per layer")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    for bb, ls in zip(BB, ["-", "--", ":"]):
        l = pd.read_csv(RESULTS / f"failure_layers_{bb}.csv")
        c = l[l.kind == "block_cka"]
        for i, cond in enumerate(c.condition.unique()):
            g = c[c.condition == cond]
            axes[1].plot(
                np.linspace(0, 1, len(g)),
                g.cka,
                ls=ls,
                color=plots.PALETTE[i + 4],
                label=f"{cond.replace('S_test:', '')} ({bb})",
            )
    axes[1].set_xlabel("relative depth")
    axes[1].set_ylabel("linear CKA to clean (same images)")
    axes[1].set_title("Where corruptions alter the representation")
    axes[1].legend(fontsize=5, frameon=False, ncol=2)
    plots.save(fig, "05_why_layers")
    # per-species drop vs migratory displacement
    fig, axes = plt.subplots(1, len(BB), figsize=(4.5 * len(BB), 3), sharey=True)
    for ax, bb in zip(axes, BB):
        p = pd.read_csv(RESULTS / f"shift_species_{bb}.csv")
        p = p[p.target == "W"]
        ax.scatter(p.displacement_km, p["drop"], s=8, alpha=0.6, color=plots.PALETTE[0])
        ax.set_xscale("symlog", linthresh=50)
        ax.set_xlabel("displacement of median location S -> W (km)")
        ax.set_title(BB[bb])
    axes[0].set_ylabel("accuracy drop S -> W per species")
    plots.save(fig, "05_why_species_displacement")


def fig_geo():
    """Prior alone (top-5), fusion per backbone (top-1, one panel each), label-free error detection (AUROC)."""
    p = pd.read_csv(RESULTS / "geo_prior.csv")
    n = len(BB)
    fig, axes = plt.subplots(
        1, n + 2, figsize=(3.4 * (n + 2), 3.2), gridspec_kw=dict(width_ratios=[1.2] + [1] * n + [1.2])
    )
    q = p[p.split.str.endswith("_test")].pivot(index="split", columns="variant", values="top5")
    q.plot.bar(ax=axes[0], rot=0, color=plots.PALETTE[5:7])
    axes[0].set_title("prior alone (no image): top-5", fontsize=9)
    axes[0].axhline(0.05, color="grey", ls=":", lw=0.8, label="chance (5%)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    for i, bb in enumerate(BB):
        ax = axes[1 + i]
        f = pd.read_csv(RESULTS / f"geo_fusion_{bb}.csv")
        f["sev"] = f.condition.str[-1].where(f.condition != "clean", "0").astype(int)
        g = f[f.variant == "full"].groupby("sev")[["acc_visual", "acc_fused"]].mean()
        ax.plot(g.index, g.acc_visual, color="k", lw=1, ls=":", label="visual only")
        for v, ls in zip(["full", "blocked"], ["-", "--"]):
            g = f[f.variant == v].groupby("sev")[["acc_visual", "acc_fused"]].mean()
            ax.plot(g.index, g.acc_fused, color=plots.PALETTE[i], ls=ls, label=f"+ prior ({v})")
        ax.set_title(f"fusion, {BB[bb].split(' (')[0]}: top-1", fontsize=9)
        ax.set_xlabel("severity (0 = clean / seasonal)", fontsize=8)
        ax.set_ylim(0, 0.9)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
        e = pd.read_csv(RESULTS / f"geo_error_detection_{bb}.csv")
        e = e[e.variant == "blocked"].mean(numeric_only=True)
        width = 0.8 / n
        axes[-1].bar(
            np.arange(3) + width * i,
            [e.auroc_confidence, e.auroc_geo, e.auroc_combined],
            width,
            color=plots.PALETTE[i],
            label=bb,
        )
    axes[1].set_ylabel("accuracy")
    axes[-1].set_xticks(np.arange(3) + 0.4 - 0.4 / n, ["confidence", "geo\nplausibility", "combined"])
    axes[-1].set_ylim(0.5, 1)
    axes[-1].set_ylabel("AUROC (no label)")
    axes[-1].set_title("error detection (blocked prior)", fontsize=9)
    axes[-1].legend(fontsize=6, frameon=False)
    plots.save(fig, "08_geo_prior")


CTTA_METHODS = {
    "source": "Source",
    "tent": "TENT",
    "eata": "EATA",
    "cotta": "CoTTA",
    "anchored": "Anchored",
    "anchored-noteacher": "Anchored w/o teacher",
}
STAGE_LABEL = {"clean": "S", "low_light_s3": "low light", "motion_blur_s3": "motion blur"}


def fig_ctta_source():
    """Accuracy on the clean source held-out set along the brief sequence, key methods, one panel per backbone."""
    bbs = [b for b in ["dinov2", "vit_s16"] if (RUNS / "ctta" / b / "brief").exists()]
    fig, axes = plt.subplots(1, len(bbs), figsize=(4.6 * len(bbs), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bb in zip(axes, bbs):
        for m, label in CTTA_METHODS.items():
            f = RUNS / "ctta" / bb / "brief" / f"{m}__s0.npz"
            if not f.exists():
                continue
            z = np.load(f)
            doms = [str(d) for d in z["domains"]]
            r = z["R"][:, doms.index("S")]
            style = dict(ls="--") if m == "anchored-noteacher" else {}
            ax.plot(
                np.arange(len(r)), 100 * r, marker="o", ms=3, lw=1.4, color=plots.method_color(m), label=label, **style
            )
        stages = [str(x) for x in z["stages"]]
        ticks = ["start"] + ["W" if st.startswith("W_test") else STAGE_LABEL.get(st.split(":")[1], st) for st in stages]
        ax.set_xticks(np.arange(len(ticks)), ticks, rotation=45, ha="right", fontsize=7)
        ax.set_title(BB[bb])
        ax.set_ylim(0, 90)
    axes[0].set_ylabel("accuracy on clean source held-out set (%)")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    plots.save(fig, "07_ctta_brief_source_backbones")


def main():
    img = data.images("S_test")[3].copy()
    plots.corruption_grid(img, TEST_CORRUPTIONS + VAL_CORRUPTIONS, [1, 3, 5], corrupt)
    for f in [fig_data, fig_baseline_shift, fig_why, fig_geo, fig_ctta_source]:
        f()
        print("done", f.__name__, flush=True)


if __name__ == "__main__":
    main()
