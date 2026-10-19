"""Sanity check: does gradient checkpointing change CoTTA's predictions?

CoTTA (all parameters adapted) needs gradient checkpointing to fit DINOv2 in 4 GB. Recomputation should give the
same gradients, hence the same predictions. This script streams the same condition with the same seed twice, with
checkpointing on and off (off spills to shared memory on a 4 GB card: slow but correct), and compares the outputs.

python scripts/check_cotta_checkpointing.py [--backbone dinov2] [--condition S_test:clean] [--seed 0]
Output: results/check_cotta_checkpointing_<backbone>.json
"""

import argparse

import numpy as np
import torch

from birdshift.experiment import build_method
from birdshift.models import load_source
from birdshift.paths import RESULTS
from birdshift.runner import evaluate_condition
from birdshift.utils import set_seed, write_json


def run(source, backbone, condition, seed, checkpointing):
    set_seed(seed)
    method, hp = build_method("cotta", source, backbone)
    method.model.backbone.set_grad_checkpointing(checkpointing)
    torch.cuda.reset_peak_memory_stats()
    r = evaluate_condition(method, condition, seed=seed)
    return r["logits"], r["y"], torch.cuda.max_memory_allocated() / 1e9, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2")
    ap.add_argument("--condition", default="S_test:clean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    source = load_source(args.backbone)
    out, logits = dict(backbone=args.backbone, condition=args.condition, seed=args.seed), {}
    for ckpt in (True, False):
        lg, y, peak, hp = run(source, args.backbone, args.condition, args.seed, ckpt)
        logits[ckpt] = lg
        out[f"checkpointing_{ckpt}"] = dict(acc=float((lg.argmax(1) == y).mean()), peak_gb=peak)
        out["hp"] = hp
        print(ckpt, out[f"checkpointing_{ckpt}"], flush=True)
    out["prediction_agreement"] = float((logits[True].argmax(1) == logits[False].argmax(1)).mean())
    out["max_abs_logit_diff"] = float(np.abs(logits[True] - logits[False]).max())
    print(out, flush=True)
    write_json(RESULTS / f"check_cotta_checkpointing_{args.backbone}.json", out)


if __name__ == "__main__":
    main()
