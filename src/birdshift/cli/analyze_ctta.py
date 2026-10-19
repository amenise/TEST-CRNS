"""Aggregate continual TTA runs (runs/ctta) into tables and figures.

python -m birdshift.cli.analyze_ctta --backbone vit_s16
-> results/ctta_<bb>_summary.csv (mean +- std over seeds), results/ctta_<bb>_stages.csv (online accuracy per stage),
   figures/07_ctta_<bb>_<sequence>_{online,source,matrix}.png, figures/07_ctta_<bb>_detector.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plots
from ..paths import RESULTS, RUNS
from ..utils import write_csv

KEY = [
    "online_mean",
    "online_round1",
    "online_round2",
    "final_mean",
    "bwt",
    "mean_gain_after",
    "forgetting_source",
    "final_source",
    "n_detections",
    "n_recoveries",
    "seconds",
]
MATRIX_METHODS = ["source", "tent", "eata", "cotta", "anchored"]


def load(bb, seq):
    out = {}
    for f in sorted((RUNS / "ctta" / bb / seq).glob("*.npz")):
        m, s = f.stem.rsplit("__s", 1)
        z = np.load(f, allow_pickle=False)
        out.setdefault(m, []).append({k: z[k] for k in z.files})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    args = ap.parse_args()
    bb = args.backbone
    df = pd.read_csv(RESULTS / f"ctta_{bb}.csv")
    summ = df.groupby(["sequence", "method"])[KEY].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index().sort_values(["sequence", "online_mean_mean"], ascending=[True, False])
    write_csv(summ, RESULTS / f"ctta_{bb}_summary.csv")
    print(
        summ[
            [
                "sequence",
                "method",
                "online_mean_mean",
                "online_mean_std",
                "final_mean_mean",
                "bwt_mean",
                "forgetting_source_mean",
                "online_round1_mean",
                "online_round2_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    stage_rows = []
    for seq in df.sequence.unique():
        runs = load(bb, seq)
        if not runs:
            continue
        stages = [str(s) for s in next(iter(runs.values()))[0]["stages"]]
        labels = [s.replace("_test", "").replace(":clean", "").replace("S:", "").replace("S_test:", "") for s in stages]
        for m, rs in runs.items():
            on = np.stack([r["online"] for r in rs])
            for k in range(on.shape[1]):
                stage_rows.append(
                    dict(
                        sequence=seq,
                        method=m,
                        stage=k,
                        condition=stages[k],
                        online_mean=on[:, k].mean(),
                        online_std=on[:, k].std(),
                    )
                )
        # online accuracy per stage
        fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(stages)), 3.4))
        for m in [m for m in plots.METHOD_ORDER + sorted(runs) if m in runs]:
            if m in [l.get_label() for l in ax.get_lines()]:
                continue
            on = np.stack([r["online"] for r in runs[m]]).mean(0)
            ax.plot(
                range(len(on)),
                on,
                marker="o",
                ms=2.5,
                color=plots.method_color(m),
                label=m,
                ls="--" if m == "source" or "-no" in m or "+det" in m else "-",
                lw=2.2 if m in ("source", "anchored") else 1.1,
            )
        ax.set_xticks(range(len(stages)), labels, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("online accuracy")
        ax.set_title(f"{bb} - {seq}: online accuracy per stage")
        ax.legend(fontsize=5.5, ncol=3, frameon=False)
        plots.save(fig, f"07_ctta_{bb}_{seq}_online")
        # accuracy on the clean source hold set after each stage (forgetting)
        doms = [str(d) for d in next(iter(runs.values()))[0]["domains"]]
        if "S" in doms:
            si = doms.index("S")
            fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(stages)), 3.2))
            for m in runs:
                r = np.stack([x["R"][:, si] for x in runs[m]]).mean(0)
                ax.plot(
                    range(len(r)),
                    r,
                    color=plots.method_color(m),
                    label=m,
                    marker=".",
                    ls="--" if m == "source" or "-no" in m or "+det" in m else "-",
                )
            ax.set_xticks(range(len(stages) + 1), ["start"] + labels, rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("accuracy on S_hold (clean source)")
            ax.set_title(f"{bb} - {seq}: source accuracy along the stream")
            ax.legend(fontsize=5.5, ncol=3, frameon=False)
            plots.save(fig, f"07_ctta_{bb}_{seq}_source")
        # accuracy matrices
        ms = [m for m in MATRIX_METHODS if m in runs]
        fig, axes = plt.subplots(1, len(ms), figsize=(2.6 * len(ms), 0.22 * len(stages) + 1.8), squeeze=False)
        for ax, m in zip(axes[0], ms):
            R = np.stack([x["R"] for x in runs[m]]).mean(0)
            ax.imshow(R, cmap="viridis", vmin=0, vmax=max(0.85, R.max()), aspect="auto")
            for i in range(R.shape[0]):
                for j in range(R.shape[1]):
                    ax.text(j, i, f"{100 * R[i, j]:.0f}", ha="center", va="center", fontsize=5, color="w")
            ax.set_xticks(range(len(doms)), doms, rotation=45, fontsize=6)
            ax.set_yticks(range(len(stages) + 1), ["start"] + labels, fontsize=5)
            ax.set_title(m, fontsize=8)
            ax.grid(False)
        fig.suptitle(f"{bb} - {seq}: accuracy on held-out sets (columns) after each stage (rows)", fontsize=8)
        plots.save(fig, f"07_ctta_{bb}_{seq}_matrix")
    write_csv(pd.DataFrame(stage_rows), RESULTS / f"ctta_{bb}_stages.csv")

    # detector score trace on the brief sequence
    runs = load(bb, "brief")
    det = [m for m in runs if m.endswith("+detreset")]
    if det:
        r = runs[det[0]][0]
        thr = pd.read_json(RESULTS / f"detector_{bb}.json", typ="series")["threshold"]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(r["scores"], lw=0.8, color="#4C72B0")
        ax.axhline(thr, color="#C44E52", ls="--", lw=1, label="threshold (S_hold, 99th pct x 1.5)")
        n_b = int(np.ceil(1000 / 64))
        for k, s in enumerate([str(s) for s in r["stages"]]):
            ax.axvline(k * n_b, color="grey", lw=0.5)
            ax.text(
                k * n_b + 0.5, ax.get_ylim()[1] * 0.9, s.split(":")[-1].replace("clean", s[0]), fontsize=5, rotation=90
            )
        for k, b in r["detections"]:
            ax.plot(k * n_b + b, r["scores"][k * n_b + b], "v", color="#C44E52", ms=4)
        ax.set_xlabel("batch")
        ax.set_ylabel("shift score")
        ax.set_title(f"{bb}: label-free shift detector on the brief sequence ({det[0]}, seed 0)")
        ax.legend(fontsize=6, frameon=False)
        plots.save(fig, f"07_ctta_{bb}_detector")


if __name__ == "__main__":
    main()
