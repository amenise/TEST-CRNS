"""Smoke test + timing of every TTA method on one condition (1000 images), to size experiments to the GPU.

python scripts/smoke_speed.py --backbone vit_s16 --batch-size 64  ->  results/speed_<backbone>_bs<B>.csv
"""

import argparse

import pandas as pd
import torch

from birdshift import tta
from birdshift.models import load_source
from birdshift.paths import RESULTS
from birdshift.runner import condition_tensor, evaluate_condition
from birdshift.utils import write_csv

HP = {
    "source": {},
    "bn": {"bn_prior": 16},
    "tent": {"lr": 1e-3},
    "pl": {"lr": 1e-3},
    "eata": {"lr": 1e-3},
    "sar": {"lr": 1e-3},
    "t3a": {"filter_k": 20},
    "lame": {"knn": 5},
    "cotta": {"lr": 1e-4, "ap": 0.5},
    "rdumb": {"lr": 1e-3},
    "anchored": {"lr": 1e-3},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--condition", default="W_test:clean")
    args = ap.parse_args()
    source = load_source(args.backbone)
    rows = []
    for name, hp in HP.items():
        if name == "eata":
            hp = {**hp, "fisher_images": condition_tensor("S_val:clean")}
        torch.cuda.reset_peak_memory_stats()
        m = tta.build(name, source, **hp)
        evaluate_condition(m, args.condition, batch_size=args.batch_size)  # warm-up
        r = evaluate_condition(m, args.condition, batch_size=args.batch_size)
        rows.append(
            dict(
                method=name,
                acc=r["acc"],
                seconds=r["seconds"],
                img_per_s=1000 / r["seconds"],
                peak_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
        )
        print(rows[-1], flush=True)
        del m
        torch.cuda.empty_cache()
    write_csv(pd.DataFrame(rows), RESULTS / f"speed_{args.backbone}_bs{args.batch_size}.csv")
