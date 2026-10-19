"""E6: continual TTA on the brief's sequence, the ecological year and a long stream (configs/ctta.yaml).

python -m birdshift.cli.run_ctta --backbone vit_s16 [--sequences brief year] [--methods tent anchored] [--seeds 0 1 2]
Method names: any tta method, ablations anchored-noanchor|nofilter|noteacher, and "<m>+detreset" (reset to source
when the label-free shift detector fires).
Raw: runs/ctta/<backbone>/<sequence>/<method>__s<seed>.npz ; tables: results/ctta_<backbone>.csv (one row per run),
results/ctta_matrix_<backbone>.csv (long format accuracy matrices).
"""

import argparse

import numpy as np
import pandas as pd
import torch

from ..ctta import ShiftDetector, run_sequence, summarize
from ..experiment import build_method
from ..models import load_source
from ..paths import RESULTS, RUNS
from ..runner import condition_tensor, run_file, save_run
from ..utils import load_config, write_csv, write_json
from .train_source import feature_file


def domain_of(spec):
    split, cond = spec.split(":")
    return split[0] if cond == "clean" else cond.rsplit("_s", 1)[0]


def geo_ctx_fn(variant="full"):
    cache = {}

    def fn(spec, it):
        split = spec.split(":")[0]
        if split not in cache:
            cache[split] = torch.from_numpy(np.load(RUNS / "geo" / variant / f"llr_{split}.npy")).float().cuda()
        return {"geo_llr": cache[split][it.to(cache[split].device)]}

    return fn


def make_detector(source, backbone):
    f = np.load(feature_file(backbone, "S_train")).astype(np.float32)
    mu = torch.from_numpy(f.mean(0)).cuda()
    sd = torch.from_numpy(f.std(0) + 1e-6).cuda()
    det = ShiftDetector(source, mu, sd)
    det.calibrate(condition_tensor("S_hold:clean"))
    return det


SUMMARY_KEYS = [
    "online_mean", "online_round1", "online_round2", "final_mean", "source_model_mean", "bwt", "mean_gain_after",
    "forgetting_source", "final_source", "n_detections", "seconds", "n_recoveries",
]  # fmt: skip


def rebuild_csv(backbone, out=None):
    """Rebuild results/ctta_<backbone>.csv from the raw runs (each .npz stores its own summary)."""
    rows = []
    for f in sorted((RUNS / "ctta" / backbone).glob("*/*.npz")):
        method, seed = f.stem.rsplit("__s", 1)
        z = np.load(f)
        rows.append(
            dict(
                backbone=backbone,
                sequence=f.parent.name,
                method=method,
                seed=int(seed),
                **{k: float(z[k]) for k in SUMMARY_KEYS},
            )
        )
    written = write_csv(pd.DataFrame(rows), out or RESULTS / f"ctta_{backbone}.csv")
    print(f"rebuilt {written} from {len(rows)} runs" if written else f"rebuild of {len(rows)} runs NOT written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--sequences", nargs="*", default=["brief", "year"])
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--rebuild-csv", action="store_true", help="only rebuild the per-run CSV from runs/ctta/")
    ap.add_argument("--out", help="with --rebuild-csv: write to this path instead of results/")
    args = ap.parse_args()
    if args.rebuild_csv:
        rebuild_csv(args.backbone, args.out)
        return
    cfg = load_config("ctta")
    source = load_source(args.backbone)
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in source.modules())
    detector = None
    rows_path = RESULTS / f"ctta_{args.backbone}.csv"
    for seq in args.sequences:
        sc = cfg["sequences"][seq]
        stages = sc["stages"] * sc["rounds"]
        stage_domains = [domain_of(s) for s in stages]
        default_methods = cfg.get("methods_override", {}).get(args.backbone, cfg["methods"])
        methods = args.methods or (
            cfg["long_methods"]
            if seq == "brief_long"
            else default_methods + [f"{m}+detreset" for m in cfg["detector_methods"]]
        )
        default_seeds = cfg.get("seeds_override", {}).get(args.backbone, cfg["seeds"])
        seeds = (
            args.seeds
            if args.seeds is not None
            else (cfg.get("long_seeds", default_seeds[:2]) if seq == "brief_long" else default_seeds)
        )
        for m in methods:
            base = m.split("+")[0]
            if base == "bn" and not has_bn:
                continue
            todo = [s for s in seeds if not run_file("ctta", args.backbone, seq, f"{m}__s{s}.npz").exists()]
            if not todo:
                continue
            method, hp = build_method(base, source, args.backbone)
            use_det = m.endswith("+detreset")
            if use_det and detector is None:
                detector = make_detector(source, args.backbone)
                write_json(RESULTS / f"detector_{args.backbone}.json", dict(threshold=detector.threshold))
            ctx_fn = geo_ctx_fn() if base.startswith("geo_") else None
            for seed in todo:
                if use_det:
                    detector.ref, detector.n_in_segment = None, 0
                res = run_sequence(
                    method,
                    stages,
                    sc["domains"],
                    seed=seed,
                    batch_size=cfg["batch_size"],
                    detector=detector if use_det else None,
                    reset_on_shift=use_det,
                    ctx_fn=ctx_fn,
                )
                summ = summarize(res, stage_domains)
                curves = np.array([np.pad(c, (0, 16 - len(c)), constant_values=np.nan)[:16] for c in res["curves"]])
                save_run(
                    run_file("ctta", args.backbone, seq, f"{m}__s{seed}.npz"),
                    R=res["R"],
                    online=res["online"],
                    curves=curves,
                    domains=np.array(res["domains"]),
                    stages=np.array(stages),
                    detections=np.array(res["detections"]).reshape(-1, 2),
                    scores=res["scores"],
                    **summ,
                )
                row = dict(backbone=args.backbone, sequence=seq, method=m, seed=seed, **summ)
                print(
                    f"{seq:10s} {m:22s} s{seed} online={summ['online_mean']:.4f} final={summ['final_mean']:.4f} "
                    f"bwt={summ['bwt']:+.4f} src_forget={summ['forgetting_source']:+.4f} ({summ['seconds']:.0f}s)",
                    flush=True,
                )
                df = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
                if len(df):
                    df = df[~((df.sequence == seq) & (df.method == m) & (df.seed == seed))]
                write_csv(pd.concat([df, pd.DataFrame([row])]), rows_path)
            del method


if __name__ == "__main__":
    main()
