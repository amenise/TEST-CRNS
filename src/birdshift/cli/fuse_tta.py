"""E9: add the geo-temporal prior on top of every TTA run (post-hoc, CPU only, from the stored per-image logits).

For each run in runs/tta/<backbone>/<method>/<split>__<cond>__s<seed>__bs<bs>__<order>.npz the final prediction becomes
    log_softmax(TTA logits) + beta * llr(split)            (Bayes fusion, geo_prior.fuse)
with llr from runs/geo/<variant>/llr_<split>.npy and beta the value selected on S_val with the *source* model
(results/geo_beta_<backbone>_<variant>.json). Nothing is tuned on test data: beta and the TTA hyper-parameters are
reused as they are. Corruptions do not touch the metadata, so a corrupted image keeps the llr of its clean version.
T3A is excluded: its logits are cosine similarities on another scale, so beta (chosen on softmax logits) does not apply.

python -m birdshift.cli.fuse_tta [--backbones vit_s16 resnet50]
Outputs: results/tta_geo_fusion_<backbone>.csv (one row per run x prior variant: visual and fused accuracy, McNemar)
"""

import argparse

import numpy as np
import pandas as pd

from .. import geo_prior
from ..paths import RESULTS, RUNS
from ..stats import mcnemar
from ..utils import read_json, write_csv

VARIANTS = ["full", "blocked"]
EXCLUDED = {"t3a"}


def parse(name: str) -> dict:
    split, cond, seed, bs, order = name.split("__")
    return dict(split=split, condition=cond, seed=int(seed[1:]), batch_size=int(bs[2:]), order=order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["vit_s16", "resnet50"])
    args = ap.parse_args()
    for bb in args.backbones:
        betas = {v: read_json(RESULTS / f"geo_beta_{bb}_{v}.json")["selected_beta"] for v in VARIANTS}
        llr = {}
        rows = []
        for f in sorted((RUNS / "tta" / bb).glob("*/*.npz")):
            method = f.parent.name
            if method in EXCLUDED:
                continue
            meta = parse(f.stem)
            z = np.load(f)
            logits, y = z["logits"].astype(np.float64), z["y"]
            visual = logits.argmax(1) == y
            for v in VARIANTS:
                key = (v, meta["split"])
                if key not in llr:
                    llr[key] = np.load(RUNS / "geo" / v / f"llr_{meta['split']}.npy")
                fused = geo_prior.fuse(logits, llr[key], betas[v]).argmax(1) == y
                t = mcnemar(visual, fused)
                rows.append(
                    dict(
                        method=method,
                        **meta,
                        variant=v,
                        beta=betas[v],
                        acc_visual=float(visual.mean()),
                        acc_fused=float(fused.mean()),
                        gain=float(fused.mean() - visual.mean()),
                        mcnemar_p=t["p"],
                    )
                )
        df = pd.DataFrame(rows)
        write_csv(df, RESULTS / f"tta_geo_fusion_{bb}.csv")
        bench = df[(df.batch_size == 64) & (df.order == "iid")]
        bench = bench.assign(kind=np.where(bench.condition == "clean", "seasons+S", "corruptions"))
        print(f"== {bb}: {len(df)} rows ({df.method.nunique()} methods)")
        print(bench.groupby(["variant", "method", "kind"])[["acc_visual", "acc_fused"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
