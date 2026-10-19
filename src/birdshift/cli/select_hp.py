"""E3: leakage-free hyper-parameter selection on validation corruptions of S_val (configs/tta.yaml: hp_selection).

python -m birdshift.cli.select_hp --backbone vit_s16 [--methods tent eata ...]
Criterion: mean online accuracy over the validation conditions (clean S_val + 4 validation corruptions x sev 3,5).
Outputs: results/hp_<backbone>.json (best per method + full grid), results/hp_grid_<backbone>.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from ..experiment import build_method, grid_points, method_grid
from ..models import load_source
from ..paths import RESULTS
from ..runner import evaluate_condition
from ..utils import load_config, read_json, run_meta, write_csv, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--force", nargs="*", default=[], help="re-run these methods even if already selected")
    args = ap.parse_args()
    cfg = load_config("tta")
    # default: the base methods benchmarked for this backbone (geo-gated variants and ablations inherit their HPs)
    bench = cfg["benchmark"].get("methods_override", {}).get(args.backbone, cfg["benchmark"]["methods"])
    methods = args.methods or list(dict.fromkeys(m.replace("geo_", "").split("-")[0] for m in bench))
    source = load_source(args.backbone)
    out_json = RESULTS / f"hp_{args.backbone}.json"
    selected = read_json(out_json) if out_json.exists() else {}
    conds = cfg["hp_selection"]["conditions"]
    rows = []
    for m in methods:
        if m == "bn" and not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in source.modules()):
            continue  # no BatchNorm in ViTs: BN-adapt == Source
        if m in selected and m not in args.force:
            print(f"{m}: already selected {selected[m]['best']} (use --force {m} to redo)", flush=True)
            continue
        spec = method_grid(cfg, m, args.backbone)
        if "inherit" in spec:
            parent = spec["inherit"]
            selected[m] = dict(best=dict(selected[parent]["best"]), mean_acc=None, inherited_from=parent, grid=[])
            write_json(out_json, {**selected, "_meta": run_meta(backbone=args.backbone, conditions=conds)})
            print(f"{m}: inherits {parent} {selected[m]['best']}", flush=True)
            continue
        grid = []
        for hp in grid_points(cfg, m, args.backbone):
            t0 = time.time()
            method, used = build_method(m, source, args.backbone, cfg, hp=hp)
            accs = [
                evaluate_condition(method, c, seed=cfg["hp_selection"]["seed"], batch_size=cfg["batch_size"])["acc"]
                for c in conds
            ]
            res = dict(
                method=m,
                **hp,
                mean_acc=float(np.mean(accs)),
                seconds=time.time() - t0,
                **{c.split(":")[1]: a for c, a in zip(conds, accs)},
            )
            grid.append(res)
            rows.append(res)
            print(f"{m} {hp} mean_acc={res['mean_acc']:.4f} ({res['seconds']:.0f}s)", flush=True)
            del method
        best = max(grid, key=lambda r: r["mean_acc"])
        g = spec
        edges = [k for k, vals in g.items() if len(vals) > 2 and best[k] in (vals[0], vals[-1])]
        if edges:
            print(f"WARNING {m}: optimum on the grid edge for {edges}", flush=True)
        selected[m] = dict(best={k: best[k] for k in g}, mean_acc=best["mean_acc"], at_grid_edge=edges, grid=grid)
        write_json(out_json, {**selected, "_meta": run_meta(backbone=args.backbone, conditions=conds)})
    # full grid table rebuilt from the JSON (which is written after every method, so it survives interruptions)
    rows = [r for k, v in selected.items() if k != "_meta" for r in v.get("grid", [])]
    write_csv(pd.DataFrame(rows), RESULTS / f"hp_grid_{args.backbone}.csv")


if __name__ == "__main__":
    main()
