"""E12: controls requested by the review of the extended report (CPU only, from cached features and stored logits).

  bootstrap   95% bootstrap confidence intervals of source accuracy per domain and of the seasonal drops
  prior       geo-temporal prior trained on breeding-season (May-July) observations only, fused on A/W/P
  twosample   two-sample tests (MMD with permutation, classifier two-sample test) between S and A/W/P features
  species     Spearman correlation between per-species seasonal drop and migration displacement

python -m birdshift.cli.run_review_controls [--parts bootstrap prior twosample species]
Output: results/review_controls.json
"""

import argparse

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .. import data, geo_prior
from ..models import load_source
from ..paths import RESULTS, RUNS
from ..utils import read_json, set_seed, write_json
from .train_source import feature_file

BACKBONES = ["dinov2", "vit_s16", "resnet50"]
SPLITS = ["S_test", "S_hold", "A_test", "W_test", "P_test"]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def feats(bb, split, cond="clean"):
    return np.load(feature_file(bb, split, cond)).astype(np.float32)


def logits(model, f):
    with torch.no_grad():
        return model.logits_from_features(torch.from_numpy(f)).numpy()


def boot_ci(correct, b=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    reps = correct[rng.integers(0, n, size=(b, n))].mean(1)
    return reps


def part_bootstrap():
    out = {}
    for bb in BACKBONES:
        model = load_source(bb, device="cpu")
        corr = {s: (logits(model, feats(bb, s)).argmax(1) == data.labels(s)) for s in SPLITS}
        reps = {s: boot_ci(c, seed=i) for i, (s, c) in enumerate(corr.items())}
        res = {}
        for s in SPLITS:
            res[s] = dict(
                acc=float(corr[s].mean()), lo=float(np.percentile(reps[s], 2.5)), hi=float(np.percentile(reps[s], 97.5))
            )
        for t in ["S_hold", "A_test", "W_test", "P_test"]:
            d = reps["S_test"] - reps[t]
            res[f"drop_{t}"] = dict(
                drop=float(corr["S_test"].mean() - corr[t].mean()),
                lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)),
            )
        out[bb] = res
        print(bb, {k: round(v.get("acc", v.get("drop")), 4) for k, v in res.items()}, flush=True)
    return out


def part_prior():
    """Train the prior on May-July observations only and fuse it with DINOv2, beta chosen on S_val."""
    set_seed(0)
    model_p, log_py, n = geo_prior.fit("source_season", device="cpu")
    vis = load_source("dinov2", device="cpu")
    res = {"n_train_rows": int(n)}

    def llr(split, prior_model, lp_y):
        return geo_prior.log_prior(prior_model, data.split_meta(split), device="cpu") - lp_y

    lv, yv = logits(vis, feats("dinov2", "S_val")), data.labels("S_val")
    llr_v = llr("S_val", model_p, log_py)
    sel = {b: float((geo_prior.fuse(lv, llr_v, b).argmax(1) == yv).mean()) for b in BETAS}
    beta = max(sel, key=lambda b: (sel[b], -b))
    res["beta"] = beta
    for split in ["S_test", "A_test", "W_test", "P_test"]:
        lg, y = logits(vis, feats("dinov2", split)), data.labels(split)
        row = {"visual": float((lg.argmax(1) == y).mean())}
        row["source_season_prior"] = float(
            (geo_prior.fuse(lg, llr(split, model_p, log_py), beta).argmax(1) == y).mean()
        )
        for v in ["full", "blocked"]:
            b = read_json(RESULTS / f"geo_beta_dinov2_{v}.json")["selected_beta"]
            ll = np.load(RUNS / "geo" / v / f"llr_{split}.npy")
            row[f"{v}_prior"] = float((geo_prior.fuse(lg, ll, b).argmax(1) == y).mean())
        res[split] = row
        print("prior", split, {k: round(v, 4) for k, v in row.items()}, flush=True)
    return res


def mmd_rbf(x, y, n_perm=500, seed=0):
    z = np.concatenate([x, y])
    sq = (z**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    gamma = 1.0 / np.median(d2[d2 > 0])
    k = np.exp(-gamma * d2)
    n = len(x)

    def stat(idx):
        a, b = idx[:n], idx[n:]
        return k[np.ix_(a, a)].mean() + k[np.ix_(b, b)].mean() - 2 * k[np.ix_(a, b)].mean()

    rng = np.random.default_rng(seed)
    base = np.arange(len(z))
    obs = stat(base)
    null = np.array([stat(rng.permutation(len(z))) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + n_perm))


def c2st(x, y, seed=0):
    z = np.concatenate([x, y])
    lab = np.r_[np.zeros(len(x)), np.ones(len(y))]
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    scores = np.zeros(len(z))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(z, lab):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(z[tr], lab[tr])
        scores[te] = clf.predict_proba(z[te])[:, 1]
    return float(roc_auc_score(lab, scores))


def part_twosample():
    out = {}
    for bb in ["dinov2", "vit_s16"]:
        ref = feats(bb, "S_test")
        targets = {
            "S_hold": feats(bb, "S_hold"),
            "A_test": feats(bb, "A_test"),
            "W_test": feats(bb, "W_test"),
            "P_test": feats(bb, "P_test"),
            "gaussian_noise_s1": feats(bb, "S_test", "gaussian_noise_s1"),
            "haze_s1": feats(bb, "S_test", "haze_s1"),
        }
        res = {}
        for name, t in targets.items():
            m, p = mmd_rbf(ref, t)
            res[name] = dict(mmd2=m, p_perm=p, c2st_auc=c2st(ref, t))
            print(bb, name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)
        out[bb] = res
    return out


def part_species():
    out = {}
    for bb in ["dinov2", "vit_s16"]:
        import pandas as pd

        sp = pd.read_csv(RESULTS / f"shift_species_{bb}.csv")
        w = sp[sp.target == "W"].dropna()  # same selection as run_shift
        r, p = stats.spearmanr(w["displacement_km"], w["drop"])
        out[bb] = dict(spearman_rho=float(r), p=float(p), n_species=int(len(w)))
        print(bb, out[bb], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["bootstrap", "prior", "twosample", "species"])
    args = ap.parse_args()
    path = RESULTS / "review_controls.json"
    res = read_json(path) if path.exists() else {}
    for part in args.parts:
        res[part] = globals()[f"part_{part}"]()
    write_json(path, res)


if __name__ == "__main__":
    main()
