"""E1: source models = frozen pretrained backbone + L2-regularised multinomial logistic regression (linear probe)
+ temperature scaling. Also caches features of every evaluation condition (used by analyses and post-hoc methods).

python -m birdshift.cli.train_source --backbone vit_s16

Protocol
- training data: S_train (+ horizontal flips as augmentation: features of flipped images are added);
- the L2 strength lambda is selected on S_val accuracy (ties -> lower NLL); S_test/S_hold and other domains unused;
- the final head is the one trained on S_train with the selected lambda (S_val is kept for calibration);
- temperature T minimises NLL on S_val (Guo et al. 2017).
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import data
from ..corruptions import SEVERITIES, TEST_CORRUPTIONS
from ..metrics import accuracy, ece, nll
from ..models import Classifier, extract_features, head_path
from ..paths import FEATURES, RESULTS, ensure
from ..utils import run_meta, set_seed, write_json

LAMBDAS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
EVAL_SPLITS = ["S_val", "S_hold", "S_test", "A_hold", "A_test", "W_hold", "W_test", "P_hold", "P_test"]


def feature_file(backbone, split, cond="clean"):
    return FEATURES / backbone / f"{split}__{cond}.npy"


def cache_features(model, backbone, split, cond="clean", corruption=None, severity=None, flip=False):
    p = feature_file(backbone, split, cond + ("_flip" if flip else ""))
    if not p.exists():
        ensure(p.parent)
        imgs = data.condition_images(split, corruption, severity)
        np.save(p, extract_features(model, imgs, flip=flip).astype(np.float16))
    return np.load(p).astype(np.float32)


def fit_logreg(x, y, lam, num_classes=100, device="cuda", iters=300):
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    w = torch.zeros(num_classes, x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(num_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w.T + b, y) + lam * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_temperature(logits, y):
    lt = torch.as_tensor(logits, dtype=torch.float64).detach()
    yt = torch.tensor(y)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


@torch.no_grad()
def recalibrate_bn(model, imgs, batch_size=128):
    """Re-estimate BatchNorm running statistics on the source training images (cumulative average, AdaBN-style).

    Why: the pretrained running statistics describe ImageNet, not our source domain S. A linear head trained on
    ImageNet-normalised features breaks as soon as any batch statistics of bird images are used, even on clean
    in-distribution data (sanity check: ResNet-50 S_test 57.7% -> 38.8% with test-batch statistics). In the TTA
    literature the source model's statistics are those of its source training data, so we make that true.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if not bns:
        return 0
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over all batches
        m.train()
    perm = np.random.default_rng(0).permutation(len(imgs))
    for s in range(0, len(perm), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(imgs[np.sort(perm[s : s + batch_size])])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            model.features(x)
    model.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--no-corrupted", action="store_true", help="skip caching features of corrupted conditions")
    args = ap.parse_args()
    set_seed(0)
    t0 = time.time()
    model = Classifier(args.backbone).cuda().eval()
    n_bn = recalibrate_bn(model, data.images("S_train"))
    print(f"recalibrated {n_bn} BatchNorm layers on S_train", flush=True)

    xtr = np.concatenate(
        [cache_features(model, args.backbone, "S_train"), cache_features(model, args.backbone, "S_train", flip=True)]
    )
    ytr = np.concatenate([data.labels("S_train")] * 2)
    xval, yval = cache_features(model, args.backbone, "S_val"), data.labels("S_val")
    t_feat = time.time() - t0

    sweep = {}
    best = None
    for lam in LAMBDAS:
        w, b = fit_logreg(xtr, ytr, lam)
        lv = (torch.from_numpy(xval).cuda() @ w.T + b).cpu().numpy()
        sweep[lam] = dict(val_acc=accuracy(lv, yval), val_nll=nll(lv, yval))
        print(f"lambda={lam:g} val_acc={sweep[lam]['val_acc']:.4f} val_nll={sweep[lam]['val_nll']:.4f}", flush=True)
        key = (sweep[lam]["val_acc"], -sweep[lam]["val_nll"])
        if best is None or key > best[0]:
            best = (key, lam, w, b, lv)
    _, lam, w, b, lv = best
    temperature = fit_temperature(lv, yval)
    model.head.weight.data.copy_(w)
    model.head.bias.data.copy_(b)
    model.temperature.fill_(temperature)
    ensure(head_path(args.backbone).parent)
    bn_stats = {k: v for k, v in model.backbone.state_dict().items() if "running_" in k or "num_batches" in k}
    torch.save(
        dict(head=model.head.state_dict(), temperature=temperature, lam=lam, bn_stats=bn_stats),
        head_path(args.backbone),
    )

    summary = {}
    for split in EVAL_SPLITS:
        f = cache_features(model, args.backbone, split)
        lg = model.logits_from_features(torch.from_numpy(f).cuda()).detach().cpu().numpy()
        y = data.labels(split)
        summary[split] = dict(acc=accuracy(lg, y), nll=nll(lg, y), ece=ece(lg, y))
        print(split, summary[split], flush=True)
    if not args.no_corrupted:
        for split in ["S_test", "S_hold"]:
            for c in TEST_CORRUPTIONS:
                for s in SEVERITIES:
                    cache_features(model, args.backbone, split, data.condition_name(c, s), c, s)
    write_json(
        RESULTS / f"source_{args.backbone}.json",
        dict(
            backbone=args.backbone,
            selected_lambda=lam,
            temperature=temperature,
            sweep=sweep,
            eval=summary,
            n_train=len(ytr),
            feature_seconds=t_feat,
            total_seconds=time.time() - t0,
            meta=run_meta(),
        ),
    )


if __name__ == "__main__":
    main()
