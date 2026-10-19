"""E12 headroom control: does TTA recover a mild corruption whose cost matches the seasonal drop?

Reads the stored per-image logits of the severity-1 runs (run_tta --conditions S_test:<c>_s1) and of the seasonal
runs, and compares, for each condition, the accuracy lost by the source model with the accuracy recovered by TTA.

python -m birdshift.cli.analyze_headroom [--backbone dinov2]
Output: results/headroom_<backbone>.csv
"""

import argparse

import numpy as np
import pandas as pd

from ..paths import RESULTS, RUNS
from ..utils import write_csv

CORRUPTIONS = ["gaussian_noise", "low_light", "defocus_blur", "motion_blur", "jpeg", "pixelate", "contrast", "haze"]
METHODS = ["tent", "eata", "sar", "geo_tent", "geo_eata"]


def acc(bb, method, split, cond, seed):
    d = np.load(RUNS / "tta" / bb / method / f"{split}__{cond}__s{seed}__bs64__iid.npz")
    return float((d["logits"].argmax(1) == d["y"]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    args = ap.parse_args()
    bb = args.backbone
    clean = np.mean([acc(bb, "source", "S_test", "clean", s) for s in args.seeds])
    conds = [("S_test", f"{c}_s1", "corruption") for c in CORRUPTIONS]
    conds += [(f"{d}_test", "clean", "season") for d in "AWP"]
    rows = []
    for split, cond, kind in conds:
        src = np.mean([acc(bb, "source", split, cond, s) for s in args.seeds])
        row = dict(split=split, condition=cond, kind=kind, source=src, drop=clean - src)
        for m in METHODS:
            row[f"gain_{m}"] = np.mean([acc(bb, m, split, cond, s) for s in args.seeds]) - src
        rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / f"headroom_{bb}.csv")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
