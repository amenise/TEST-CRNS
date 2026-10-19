"""E10: would a stronger frozen backbone help? Same linear-probe protocol as train_source, on clean domains only.

Candidates (all frozen, features of 224x224 images):
  vit_s16  our current backbone (AugReg IN-21k -> IN-1k), features already cached by train_source
  dinov2   ViT-S/14 DINOv2 (self-supervised, LVD-142M; timm `vit_small_patch14_dinov2.lvd142m`), same size as vit_s16
  bioclip  BioCLIP ViT-B/16 image encoder (Stevens et al., CVPR 2024; trained on the TreeOfLife-10M images,
           open_clip `hf-hub:imageomics/bioclip`)
Protocol (identical to train_source): S_train + horizontal flips, L2 logistic regression with lambda selected on S_val
accuracy, temperature on S_val; evaluation on S/A/W/P test sets (10 images per species each). No TTA.

python -m birdshift.cli.probe_backbones [--backbones vit_s16 dinov2 bioclip]
Output: results/backbone_probe.csv
"""

import argparse

import numpy as np
import torch

from .. import data
from ..metrics import accuracy, ece, nll
from ..paths import RESULTS, ensure
from ..utils import set_seed, write_csv
from .train_source import LAMBDAS, feature_file, fit_logreg, fit_temperature

TEST_SPLITS = ["S_test", "A_test", "W_test", "P_test"]


def load_encoder(name):
    """Return (encode, mean, std): encode maps normalised float NCHW images to features."""
    import truststore

    truststore.inject_into_ssl()  # model weights come from the HF hub; HTTPS is intercepted on this network
    if name == "dinov2":
        import timm

        m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        cfg = m.pretrained_cfg
        return m.cuda().eval(), cfg["mean"], cfg["std"]
    if name == "bioclip":
        import open_clip

        m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        enc = m.visual.cuda().eval()
        return enc, open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
    raise ValueError(name)


@torch.no_grad()
def extract(encoder, mean, std, imgs, flip=False, batch_size=64):
    mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    out = []
    for s in range(0, len(imgs), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(imgs[s : s + batch_size])).cuda().permute(0, 3, 1, 2).float() / 255
        if flip:
            x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(encoder((x - mean) / std).float().cpu())
    return torch.cat(out).numpy()


def features(name, split, flip=False, enc=None):
    p = feature_file(name, split, "clean" + ("_flip" if flip else ""))
    if not p.exists():
        ensure(p.parent)
        np.save(p, extract(*enc, data.images(split), flip=flip).astype(np.float16))
    return np.load(p).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="*", default=["vit_s16", "dinov2", "bioclip"])
    args = ap.parse_args()
    set_seed(0)
    rows = []
    for name in args.backbones:
        enc = None if name == "vit_s16" else load_encoder(name)
        xtr = np.concatenate([features(name, "S_train", enc=enc), features(name, "S_train", flip=True, enc=enc)])
        ytr = np.concatenate([data.labels("S_train")] * 2)
        xval, yval = features(name, "S_val", enc=enc), data.labels("S_val")
        best = None
        for lam in LAMBDAS:
            w, b = fit_logreg(xtr, ytr, lam)
            lv = (torch.from_numpy(xval).cuda() @ w.T + b).cpu().numpy()
            key = (accuracy(lv, yval), -nll(lv, yval))
            if best is None or key > best[0]:
                best = (key, lam, w, b, lv)
        (val_acc, _), lam, w, b, lv = best
        t = fit_temperature(lv, yval)
        print(f"== {name}: dim={xtr.shape[1]} lambda={lam:g} T={t:.3f} S_val={val_acc:.3f}", flush=True)
        for split in TEST_SPLITS:
            lg = (torch.from_numpy(features(name, split, enc=enc)).cuda() @ w.T + b).cpu().numpy() / t
            y = data.labels(split)
            rows.append(dict(backbone=name, dim=xtr.shape[1], lam=lam, temperature=t, val_acc=val_acc, split=split,
                             acc=accuracy(lg, y), nll=nll(lg, y), ece=ece(lg, y)))  # fmt: skip
            print(f"   {split}: acc={rows[-1]['acc']:.3f} ece={rows[-1]['ece']:.3f}", flush=True)
        del enc
        torch.cuda.empty_cache()
    import pandas as pd

    write_csv(pd.DataFrame(rows), RESULTS / "backbone_probe.csv")


if __name__ == "__main__":
    main()
