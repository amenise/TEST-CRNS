"""E7: geo-temporal prior - (a) prior alone, (b) fusion with the visual model under every shift, (c) label-free error
detection with the prior's plausibility, (d) per-image log-likelihood ratios for geo-gated TTA.

python -m birdshift.cli.run_geo --backbones vit_s16 resnet50
Outputs: runs/geo/<variant>/{llr,logprior}_<split>.npy ; results/geo_prior.csv ; results/geo_fusion_<backbone>.csv ;
results/geo_error_detection_<backbone>.csv
beta is selected on S_val only (source domain, clean) - never on a test domain.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .. import data, geo_prior
from ..corruptions import SEVERITIES, TEST_CORRUPTIONS
from ..metrics import accuracy, ece, nll
from ..models import load_source
from ..paths import RESULTS, RUNS, ensure
from ..utils import set_seed, write_csv, write_json
from .train_source import feature_file

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
VARIANTS = ["full", "blocked"]


def topk(logp, y, k):
    return float((np.argsort(-logp, 1)[:, :k] == y[:, None]).any(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["vit_s16", "resnet50"])
    ap.add_argument("--reuse-prior", action="store_true", help="keep the fitted prior in runs/geo/ (no refit)")
    args = ap.parse_args()
    set_seed(0)
    if not args.reuse_prior:  # refit the prior (otherwise keep runs/geo/* as they are)
        prior_rows = []
        for variant in VARIANTS:
            model, log_py, n_train = geo_prior.fit(variant)
            out = RUNS / "geo" / variant
            ensure(out)
            np.save(out / "log_py.npy", log_py)
            for split in geo_prior.EVAL_SPLITS:
                meta = data.split_meta(split)
                lp = geo_prior.log_prior(model, meta)
                np.save(out / f"logprior_{split}.npy", lp.astype(np.float32))
                np.save(out / f"llr_{split}.npy", (lp - log_py).astype(np.float32))
                y = data.labels(split)
                prior_rows.append(
                    dict(
                        variant=variant,
                        n_train_rows=n_train,
                        split=split,
                        top1=topk(lp, y, 1),
                        top5=topk(lp, y, 5),
                        top10=topk(lp, y, 10),
                        nll=float(-lp[np.arange(len(y)), y].mean()),
                    )
                )
                print(prior_rows[-1], flush=True)
        write_csv(pd.DataFrame(prior_rows), RESULTS / "geo_prior.csv")

    for bb in args.backbones:
        src = load_source(bb)

        def vis_logits(split, cond="clean", bb=bb, src=src):
            f = torch.from_numpy(np.load(feature_file(bb, split, cond)).astype(np.float32)).cuda()
            with torch.no_grad():
                return src.logits_from_features(f).cpu().numpy()

        fusion, det = [], []
        for variant in VARIANTS:
            gdir = RUNS / "geo" / variant
            # beta on S_val
            lv, yv = vis_logits("S_val"), data.labels("S_val")
            llr_v = np.load(gdir / "llr_S_val.npy")
            sel = {b: accuracy(geo_prior.fuse(lv, llr_v, b), yv) for b in BETAS}
            beta = max(sel, key=lambda b: (sel[b], -b))
            write_json(RESULTS / f"geo_beta_{bb}_{variant}.json", dict(selected_beta=beta, val_acc=sel))
            conds = [(s, "clean") for s in ["S_test", "A_test", "W_test", "P_test"]]
            conds += [("S_test", f"{c}_s{s}") for c in TEST_CORRUPTIONS for s in SEVERITIES]
            for split, cond in conds:
                lg, y = vis_logits(split, cond), data.labels(split)
                llr = np.load(gdir / f"llr_{split}.npy")
                fused = geo_prior.fuse(lg, llr, beta)
                fusion.append(
                    dict(
                        variant=variant,
                        beta=beta,
                        split=split,
                        condition=cond,
                        acc_visual=accuracy(lg, y),
                        acc_fused=accuracy(fused, y),
                        nll_visual=nll(lg, y),
                        nll_fused=nll(fused, y),
                        ece_visual=ece(lg, y),
                        ece_fused=ece(fused, y),
                    )
                )
                # label-free error detection: is the visual prediction wrong?
                pred = lg.argmax(1)
                wrong = (pred != y).astype(int)
                if 0 < wrong.sum() < len(wrong):
                    p = np.exp(lg - lg.max(1, keepdims=True))
                    msp = (p / p.sum(1, keepdims=True)).max(1)
                    plaus = llr[np.arange(len(y)), pred]
                    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
                    det.append(
                        dict(
                            variant=variant,
                            split=split,
                            condition=cond,
                            error_rate=wrong.mean(),
                            auroc_confidence=roc_auc_score(wrong, -msp),
                            auroc_geo=roc_auc_score(wrong, -plaus),
                            auroc_combined=roc_auc_score(wrong, -(z(np.log(msp)) + z(plaus))),
                        )
                    )
            print(bb, variant, "beta =", beta, flush=True)
        f = pd.DataFrame(fusion)
        write_csv(f, RESULTS / f"geo_fusion_{bb}.csv")
        write_csv(pd.DataFrame(det), RESULTS / f"geo_error_detection_{bb}.csv")
        print(f[f.condition == "clean"].round(3).to_string(index=False))
        print(f.groupby("variant")[["acc_visual", "acc_fused"]].mean())


if __name__ == "__main__":
    main()
