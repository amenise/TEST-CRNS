"""E4/E5: TTA benchmark - every method on independent streams (reset between conditions and seeds).

python -m birdshift.cli.run_tta --backbone vit_s16                                  # full benchmark from configs/tta.yaml
python -m birdshift.cli.run_tta --backbone resnet50 --methods tent bn --conditions W_test:clean --batch-size 16 --order class_sorted

Raw outputs: runs/tta/<backbone>/<method>/<split>__<cond>__s<seed>__bs<B>__<order>.npz
  (logits float16 in original image order, labels, stream order, wall time, method HPs)
Resumable: existing files are skipped.
"""

import argparse
import json
import time

import numpy as np

from .. import geo_prior  # noqa: F401  (documented dependency of geo-gated methods)
from ..corruptions import TEST_CORRUPTIONS
from ..experiment import build_method
from ..models import load_source
from ..runner import evaluate_condition, run_file, save_run
from ..utils import load_config


def benchmark_conditions(cfg):
    conds = list(cfg["benchmark"]["seasonal"])
    for c in TEST_CORRUPTIONS:
        for s in cfg["benchmark"]["corruption_severities"]:
            conds.append(f"S_test:{c}_s{s}")
    return conds


def geo_context(spec, variant="full"):
    """Per-image geo log-likelihood ratios (N, C) for geo-gated methods (built by cli/run_geo.py)."""
    import torch

    from ..paths import RUNS

    split = spec.split(":")[0]
    return {"geo_llr": torch.from_numpy(np.load(RUNS / "geo" / variant / f"llr_{split}.npy")).float().cuda()}


def out_path(backbone, method, spec, seed, bs, order):
    split, cond = spec.split(":")
    return run_file("tta", backbone, method, f"{split}__{cond}__s{seed}__bs{bs}__{order}.npz")


def run_benchmark(backbone, methods, conds, seeds, bs, order="iid", geo_variant="full", source=None, cfg=None):
    cfg = cfg or load_config("tta")
    source = source or load_source(backbone)
    has_bn = any(type(m).__name__ == "BatchNorm2d" for m in source.modules())
    for m in methods:
        if m == "bn" and not has_bn:
            print("skip bn: no BatchNorm in", backbone)
            continue
        todo = [(c, s) for c in conds for s in seeds if not out_path(backbone, m, c, s, bs, order).exists()]
        if not todo:
            continue
        method, hp = build_method(m, source, backbone, cfg)
        t0 = time.time()
        for spec, seed in todo:
            ctx = geo_context(spec, geo_variant) if m.startswith("geo_") else None
            r = evaluate_condition(method, spec, seed=seed, batch_size=bs, order=order, ctx=ctx)
            save_run(
                out_path(backbone, m, spec, seed, bs, order),
                logits=r["logits"].astype(np.float16),
                y=r["y"],
                order=r["order"],
                seconds=r["seconds"],
                hp=json.dumps(hp),
                n_recoveries=getattr(method, "n_recoveries", 0),
            )
            print(
                f"{m:10s} {spec:28s} seed={seed} bs={bs} {order} acc={r['acc']:.4f} ({r['seconds']:.1f}s)", flush=True
            )
        print(f"== {m}: {len(todo)} streams in {time.time() - t0:.0f}s", flush=True)
        del method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--order", default="iid", choices=["iid", "class_sorted"])
    ap.add_argument("--geo-variant", default="full")
    args = ap.parse_args()
    cfg = load_config("tta")
    default_methods = cfg["benchmark"].get("methods_override", {}).get(args.backbone, cfg["benchmark"]["methods"])
    run_benchmark(
        args.backbone,
        args.methods or default_methods,
        args.conditions or benchmark_conditions(cfg),
        args.seeds
        if args.seeds is not None
        else cfg["benchmark"].get("seeds_override", {}).get(args.backbone, cfg["seeds"]),
        args.batch_size or cfg["batch_size"],
        args.order,
        args.geo_variant,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
