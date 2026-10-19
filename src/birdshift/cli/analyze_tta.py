"""Aggregate the TTA benchmark (runs/tta) into tables, paired tests and figures.

python -m birdshift.cli.analyze_tta --backbone vit_s16
-> results/tta_<bb>_raw.csv            one row per (method, condition, seed)
   results/tta_<bb>_summary.csv        per method: mean +- std over seeds of grouped accuracies
   results/tta_<bb>_mcnemar.csv        per (method, condition), seed 0, vs Source; Holm over all tests
   results/tta_<bb>_friedman.json      mean ranks over conditions + Nemenyi critical difference
   figures/06_tta_<bb>_*.png
"""

import argparse
import json
import re

import numpy as np
import pandas as pd

from ..metrics import summary
from ..paths import RESULTS, RUNS
from ..stats import friedman_nemenyi, holm, mcnemar
from ..utils import write_csv, write_json

PAT = re.compile(r"(?P<split>[A-Z]_\w+?)__(?P<cond>.+)__s(?P<seed>\d+)__bs(?P<bs>\d+)__(?P<order>\w+)\.npz")


def load_runs(backbone, bs=64, order="iid"):
    rows, preds = [], {}
    for mdir in sorted((RUNS / "tta" / backbone).iterdir()):
        for f in sorted(mdir.glob("*.npz")):
            m = PAT.fullmatch(f.name)
            if not m or int(m["bs"]) != bs or m["order"] != order:
                continue
            z = np.load(f)
            lg, y = z["logits"].astype(np.float32), z["y"]
            cond = m["cond"]
            sev = int(cond[-1]) if cond != "clean" else 0
            corr = cond.rsplit("_s", 1)[0] if cond != "clean" else "clean"
            group = (
                "seasonal"
                if cond == "clean" and m["split"] != "S_test"
                else ("source_clean" if cond == "clean" else "corruption")
            )
            rows.append(
                dict(
                    method=mdir.name,
                    split=m["split"],
                    condition=f"{m['split']}:{cond}",
                    corruption=corr,
                    severity=sev,
                    group=group,
                    seed=int(m["seed"]),
                    seconds=float(z["seconds"]),
                    **summary(lg, y),
                )
            )
            preds[(mdir.name, f"{m['split']}:{cond}", int(m["seed"]))] = lg.argmax(1) == y
    return pd.DataFrame(rows), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    args = ap.parse_args()
    bb = args.backbone
    df, preds = load_runs(bb)
    write_csv(df, RESULTS / f"tta_{bb}_raw.csv")

    # grouped accuracies per seed, then mean +- std over seeds
    def grouped(d):
        out = dict(
            S_clean=d[d.group == "source_clean"].acc.mean(),
            seasonal_AWP=d[d.group == "seasonal"].acc.mean(),
            W=d[d.condition == "W_test:clean"].acc.mean(),
            corr_all=d[d.group == "corruption"].acc.mean(),
        )
        for s in sorted(d[d.group == "corruption"].severity.unique()):
            out[f"corr_s{s}"] = d[(d.group == "corruption") & (d.severity == s)].acc.mean()
        out["all"] = d.acc.mean()
        out["ece_all"] = d.ece.mean()
        out["min_pred_classes"] = d.n_pred_classes.min()
        return out

    per_seed = pd.DataFrame([dict(method=m, seed=s, **grouped(d)) for (m, s), d in df.groupby(["method", "seed"])])
    summ = per_seed.drop(columns="seed").groupby("method").agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.drop(columns=["seed_mean", "seed_std"], errors="ignore")
    src = summ.loc["source"] if "source" in summ.index else None
    if src is not None:
        for col in ["S_clean", "seasonal_AWP", "W", "corr_all", "all"]:
            summ[f"delta_{col}"] = summ[f"{col}_mean"] - src[f"{col}_mean"]
    write_csv(summ.sort_values("all_mean", ascending=False), RESULTS / f"tta_{bb}_summary.csv", index=True)
    print(
        summ[[c for c in summ.columns if c.endswith("_mean") or c.startswith("delta")]]
        .round(4)
        .sort_values("all_mean", ascending=False)
        .to_string()
    )

    # McNemar per condition vs Source (seed 0, paired on identical images), Holm over all tests
    tests = []
    for (m, cond, seed), corr in preds.items():
        if m == "source" or seed != 0 or ("source", cond, 0) not in preds:
            continue
        r = mcnemar(preds[("source", cond, 0)], corr)
        tests.append(dict(method=m, condition=cond, acc_source=preds[("source", cond, 0)].mean(), acc=corr.mean(), **r))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t["p"].values)
        t["significant"] = t["p_holm"] < 0.05
        t["direction"] = np.sign(t["acc"] - t["acc_source"]).astype(int)
        write_csv(t, RESULTS / f"tta_{bb}_mcnemar.csv")
        print(
            pd.DataFrame(
                [
                    dict(
                        method=m,
                        sig_better=int((g.significant & (g.direction > 0)).sum()),
                        sig_worse=int((g.significant & (g.direction < 0)).sum()),
                        n=len(g),
                    )
                    for m, g in t.groupby("method")
                ]
            ).to_string(index=False)
        )

    # Friedman + Nemenyi over conditions (accuracy averaged over seeds)
    piv = df.groupby(["condition", "method"]).acc.mean().unstack().dropna(axis=1)
    fr = friedman_nemenyi(piv.values)
    ranks = dict(zip(piv.columns, fr["mean_ranks"]))
    write_json(
        RESULTS / f"tta_{bb}_friedman.json",
        dict(mean_ranks=ranks, friedman_p=fr["friedman_p"], cd=fr["cd"], n_conditions=fr["n"], k_methods=fr["k"]),
    )
    print("Friedman p =", fr["friedman_p"], "CD =", round(fr["cd"], 3))
    print(json.dumps({k: round(v, 2) for k, v in sorted(ranks.items(), key=lambda kv: kv[1])}))

    from .. import plots

    plots.tta_figures(df, ranks, fr["cd"], bb)


if __name__ == "__main__":
    main()
