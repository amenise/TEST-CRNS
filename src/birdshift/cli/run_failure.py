"""E5: when and why does TTA fail?

python -m birdshift.cli.run_failure --backbone resnet50 [--parts batch noniid layers]
  batch   batch-size sweep (2, 8, 32, 64) on 3 representative conditions, seeds 0-1
  noniid  class-sorted (temporally correlated) streams, batch 64, all benchmark methods, seeds 0-1
  layers  where does the representation shift? ResNet: per-BatchNorm-layer mismatch between source running statistics
          and target statistics; ViT/ResNet: per-block feature drift (linear CKA to clean, same images)
Outputs: runs/tta (raw), results/failure_batch_<bb>.csv, results/failure_noniid_<bb>.csv, results/failure_layers_<bb>.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch

from ..analysis import linear_cka
from ..metrics import summary
from ..models import load_source
from ..paths import RESULTS
from ..runner import condition_tensor
from ..utils import load_config, write_csv
from .run_tta import out_path, run_benchmark

CONDS = ["W_test:clean", "S_test:gaussian_noise_s3", "S_test:motion_blur_s3"]
# reduced budget: batch-size sweep where it matters most (BatchNorm), 1 seed
BATCH_METHODS = ["source", "bn", "tent", "eata"]
BATCH_SIZES = [8, 32, 64]


def collect(backbone, methods, conds, seeds, bs, order):
    rows = []
    for m in methods:
        for c in conds:
            for s in seeds:
                p = out_path(backbone, m, c, s, bs, order)
                if p.exists():
                    z = np.load(p)
                    rows.append(
                        dict(
                            method=m,
                            condition=c,
                            seed=s,
                            batch_size=bs,
                            order=order,
                            **summary(z["logits"].astype(np.float32), z["y"]),
                        )
                    )
    return rows


@torch.no_grad()
def layer_analysis(backbone):
    model = load_source(backbone)
    bns = [(n, m) for n, m in model.backbone.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    blocks = [
        (n, m)
        for n, m in model.backbone.named_modules()
        if n.count(".") == 1 and (n.startswith("blocks.") or n.startswith("layer"))
    ]
    specs = [
        "S_hold:clean",
        "A_test:clean",
        "W_test:clean",
        "P_test:clean",
        "S_test:low_light_s3",
        "S_test:motion_blur_s3",
        "S_test:gaussian_noise_s3",
        "S_test:contrast_s3",
        "S_test:haze_s3",
    ]
    acts = {}

    def bn_hook(name):
        def h(mod, inp, out):
            x = inp[0].float()
            acts.setdefault(("bn", name), []).append((x.mean((0, 2, 3)), (x**2).mean((0, 2, 3)), x.shape[0]))

        return h

    def blk_hook(name):
        def h(mod, inp, out):
            o = out.float()
            acts.setdefault(("blk", name), []).append(o.mean(1) if o.dim() == 3 else o.mean((2, 3)))

        return h

    hooks = [m.register_forward_hook(bn_hook(n)) for n, m in bns] + [
        m.register_forward_hook(blk_hook(n)) for n, m in blocks
    ]
    feats = {}
    for spec in ["S_test:clean"] + specs:
        acts.clear()
        x = condition_tensor(spec, cache=False)
        for s in range(0, len(x), 100):
            with torch.autocast("cuda", dtype=torch.float16):
                model.features(x[s : s + 100].cuda())
        feats[spec] = {k: v for k, v in acts.items()}
    for h in hooks:
        h.remove()
    rows = []
    for spec in specs:
        for n, m in bns:
            lst = feats[spec][("bn", n)]
            tot = sum(b for *_, b in lst)
            mean = sum(a * b for a, _, b in lst) / tot
            var = sum(q * b for _, q, b in lst) / tot - mean**2
            sd_s = torch.sqrt(m.running_var + m.eps)
            rows.append(
                dict(
                    kind="bn_mismatch",
                    layer=n,
                    condition=spec,
                    mean_shift=float(((mean - m.running_mean).abs() / sd_s).mean()),
                    var_ratio_log=float(torch.log((var + m.eps) / (m.running_var + m.eps)).abs().mean()),
                )
            )
        if spec.startswith("S_test"):
            for n, _ in blocks:
                a = torch.cat(feats["S_test:clean"][("blk", n)]).cpu().numpy()
                b = torch.cat(feats[spec][("blk", n)]).cpu().numpy()
                rows.append(dict(kind="block_cka", layer=n, condition=spec, cka=linear_cka(a, b)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--parts", nargs="*", default=["layers", "batch", "noniid"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    args = ap.parse_args()
    seeds = args.seeds
    bb = args.backbone
    cfg = load_config("tta")
    source = load_source(bb)
    if "layers" in args.parts:
        df = layer_analysis(bb)
        write_csv(df, RESULTS / f"failure_layers_{bb}.csv")
        print(df.groupby(["kind", "condition"]).mean(numeric_only=True).round(3).to_string())
    if "batch" in args.parts:
        rows = []
        fc = cfg.get("failure", {})
        bm = [
            m
            for m in fc.get("batch_methods", BATCH_METHODS)
            if m != "bn" or any(isinstance(x, torch.nn.BatchNorm2d) for x in source.modules())
        ]
        for bs in fc.get("batch_sizes", BATCH_SIZES):
            run_benchmark(bb, bm, CONDS, seeds, bs, "iid", source=source, cfg=cfg)
            rows += collect(bb, bm, CONDS, seeds, bs, "iid")
        b = pd.DataFrame(rows)
        write_csv(b, RESULTS / f"failure_batch_{bb}.csv")
        plot_batch(b, bb)
    if "noniid" in args.parts:
        methods = cfg.get("failure", {}).get("noniid_methods", {}).get(bb)
        if methods is None:
            methods = [
                m
                for m in cfg["benchmark"].get("methods_override", {}).get(bb, cfg["benchmark"]["methods"])
                if not m.startswith("geo_")
            ]
        if not methods:
            print("noniid: skipped for", bb)
            return
        conds = CONDS + ["A_test:clean", "P_test:clean", "S_test:clean"]
        run_benchmark(bb, methods, conds, seeds, 64, "class_sorted", source=source, cfg=cfg)
        rows = collect(bb, methods, conds, seeds, 64, "class_sorted") + collect(bb, methods, conds, seeds, 64, "iid")
        n = pd.DataFrame(rows)
        write_csv(n, RESULTS / f"failure_noniid_{bb}.csv")
        plot_noniid(n, bb)


def plot_batch(b, bb):
    import matplotlib.pyplot as plt

    from .. import plots

    conds = list(b.condition.unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 3.2), sharey=False)
    for ax, c in zip(np.atleast_1d(axes), conds):
        g = b[b.condition == c].groupby(["method", "batch_size"]).acc.mean().unstack(0)
        for m in g.columns:
            ax.plot(
                g.index, g[m], marker="o", ms=3, color=plots.method_color(m), label=m, ls="--" if m == "source" else "-"
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.set_title(c, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{bb}: TTA vs batch size", fontsize=9)
    plots.save(fig, f"09_failure_batch_{bb}")


def plot_noniid(n, bb):
    import matplotlib.pyplot as plt

    from .. import plots

    g = n.groupby(["method", "order"]).acc.mean().unstack()
    g = g.loc[[m for m in plots.METHOD_ORDER if m in g.index]]
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["iid"], 0.4, label="i.i.d. stream", color=plots.PALETTE[0])
    ax.bar(x + 0.2, g["class_sorted"], 0.4, label="class-sorted stream", color=plots.PALETTE[3])
    ax.set_xticks(x, g.index, rotation=45, ha="right")
    ax.set_ylabel("accuracy (mean over 6 conditions)")
    ax.set_title(f"{bb}: temporally correlated (non-i.i.d.) streams")
    ax.legend(frameon=False, fontsize=7)
    plots.save(fig, f"09_failure_noniid_{bb}")


if __name__ == "__main__":
    main()
