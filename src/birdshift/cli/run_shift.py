"""E2: how the frozen source model behaves under the seasonal shift and the controlled corruptions, and why.

python -m birdshift.cli.run_shift --backbone vit_s16
Uses cached features (cli/train_source.py) -> runs/shift/<backbone>/<split>__<cond>.npz (logits) and
results/shift_<backbone>.csv, results/shift_drift_<backbone>.csv, results/shift_taxonomy_<backbone>.csv,
results/shift_species_<backbone>.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .. import data
from ..analysis import (
    chance_same_taxon,
    class_conditional_shift,
    linear_cka,
    mmd_linear,
    species_displacement,
    taxonomy_arrays,
)
from ..corruptions import SEVERITIES, TEST_CORRUPTIONS
from ..metrics import per_class_accuracy, summary, taxonomic_errors
from ..models import load_source
from ..paths import RESULTS
from ..runner import run_file, save_run
from ..utils import write_csv
from .train_source import feature_file


def subsample_per_class(y, k, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False) for c in np.unique(y)]))


def conditions():
    # S_hold = same domain, different images: reference level of drift/accuracy due to sampling alone
    out = [(s, "clean") for s in ["S_test", "S_hold", "A_test", "W_test", "P_test"]]
    out += [("S_test", f"{c}_s{s}") for c in TEST_CORRUPTIONS for s in SEVERITIES]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    args = ap.parse_args()
    model = load_source(args.backbone)
    taxo = taxonomy_arrays()
    feats_src = np.load(feature_file(args.backbone, "S_test")).astype(np.float32)
    y_src = data.labels("S_test")
    rows, drift, tax = [], [], []
    logits_by = {}
    for split, cond in conditions():
        f = np.load(feature_file(args.backbone, split, cond)).astype(np.float32)
        y = data.labels(split)
        with torch.no_grad():
            lg = model.logits_from_features(torch.from_numpy(f).cuda()).cpu().numpy()
        logits_by[(split, cond)] = lg
        save_run(run_file("shift", args.backbone, f"{split}__{cond}.npz"), logits=lg.astype(np.float16), y=y)
        corr, sev = (cond.rsplit("_s", 1) + [0])[:2] if cond != "clean" else ("clean", 0)
        rows.append(dict(split=split, domain=split[0], corruption=corr, severity=int(sev), **summary(lg, y)))
        d = dict(split=split, condition=cond, mmd=mmd_linear(feats_src, f))
        if split == "S_test":  # same images -> paired representation similarity
            d["cka_to_clean"] = linear_cka(feats_src, f)
        # matched sample size: 5 images per class for every target (S_hold has 5), so that S_hold gives the level
        # of centroid shift expected from sampling noise alone
        sub = subsample_per_class(y, 5, seed=0)
        d["centroid_shift_ratio"] = class_conditional_shift(feats_src, y_src, f[sub], y[sub])["centroid_shift_ratio"]
        drift.append(d)
        tax.append(dict(split=split, condition=cond, **taxonomic_errors(lg.argmax(1), y, taxo)))
    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / f"shift_{args.backbone}.csv")
    write_csv(pd.DataFrame(drift), RESULTS / f"shift_drift_{args.backbone}.csv")
    t = pd.DataFrame(tax)
    for k, v in chance_same_taxon(taxo).items():
        t[k] = v
    write_csv(t, RESULTS / f"shift_taxonomy_{args.backbone}.csv")

    # per-species accuracy drop S -> W vs migratory displacement and vs class-conditional feature shift
    acc_s = per_class_accuracy(logits_by[("S_test", "clean")], y_src)
    sp = species_displacement("S", "W")
    rows = []
    for tgt in ["A", "W", "P"]:
        ft = np.load(feature_file(args.backbone, f"{tgt}_test")).astype(np.float32)
        yt = data.labels(f"{tgt}_test")
        acc_t = per_class_accuracy(logits_by[(f"{tgt}_test", "clean")], yt)
        shift = class_conditional_shift(feats_src, y_src, ft, yt)["per_class_shift"]
        for c in range(data.NUM_CLASSES):
            rows.append(
                dict(
                    target=tgt,
                    y=c,
                    name=data.classes()["classes"][c],
                    acc_S=acc_s[c],
                    acc_T=acc_t[c],
                    drop=acc_s[c] - acc_t[c],
                    feature_shift=shift[c],
                )
            )
    per = pd.DataFrame(rows).merge(sp[["y", "displacement_km", "dlat"]], on="y", how="left")
    write_csv(per, RESULTS / f"shift_species_{args.backbone}.csv")
    w = per[per.target == "W"].dropna()
    for col in ["displacement_km", "feature_shift"]:
        r, p = stats.spearmanr(w[col], w["drop"])
        print(f"Spearman(drop S->W, {col}) = {r:.3f} (p={p:.3g}, n={len(w)})")
    print(df[df.corruption == "clean"][["split", "acc", "nll", "ece"]].to_string(index=False))
    print(df.groupby("severity")["acc"].mean())


if __name__ == "__main__":
    main()
