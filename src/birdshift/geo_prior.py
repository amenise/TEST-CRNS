"""Geo-temporal species prior p(y | lat, lon, day-of-year) from metadata only (no image is ever used).

Model: residual MLP on a sinusoidal encoding of (lon, lat, day of year), in the spirit of SINR (Cole et al. 2023)
and of the spatio-temporal prior of Mac Aodha et al. (2019). Because our label space is closed (100 species) and
every training row is a confirmed observation, we fit the discriminative softmax p(y | loc, t) with cross-entropy
(the "full" presence-only setting reduces to this when all observations of all candidate species are available).

Training rows: metadata of S_train + the unassigned North-American pool (A/W/P months, no image). Rows of any
val/hold/test split are excluded. Variant "blocked" additionally removes training rows that fall in a
0.1 deg x 0.1 deg cell containing any evaluation row, to measure how much the prior relies on near-duplicate
locations (spatial autocorrelation / same observer).

Fusion with a visual classifier (Bayes rule, assuming image and context independent given the species):
    log p(y | x, loc, t) = log p(y | x) + beta * [log p(y | loc, t) - log p(y)] + const
The second term, the log-likelihood ratio `llr`, is what geo-gated TTA uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data

EVAL_SPLITS = ["S_val", "S_hold", "S_test", "A_hold", "A_test", "W_hold", "W_test", "P_hold", "P_test"]


def encode(df: pd.DataFrame) -> np.ndarray:
    lat = df["lat"].to_numpy(np.float64) / 90.0
    lon = df["lon"].to_numpy(np.float64) / 180.0
    doy = pd.to_datetime(df["date"], utc=True).dt.dayofyear.to_numpy(np.float64) / 365.25
    feats = []
    for k in (1, 2, 4, 8):
        feats += [
            np.sin(np.pi * k * lon),
            np.cos(np.pi * k * lon),
            np.sin(np.pi * k * lat / 2),
            np.cos(np.pi * k * lat / 2),
        ]
    feats += [np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), np.sin(4 * np.pi * doy), np.cos(4 * np.pi * doy)]
    return np.stack(feats, 1).astype(np.float32)


class ResBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d), nn.ReLU())

    def forward(self, x):
        return x + self.net(x)


class GeoPrior(nn.Module):
    def __init__(self, d_in, num_classes=100, hidden=256, blocks=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            *[ResBlock(hidden, dropout) for _ in range(blocks)],
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def training_rows(variant: str = "full") -> pd.DataFrame:
    idx = data.index()
    rows = idx[((idx["split"] == "S_train") | ((idx["split"] == "") & idx["na"])) & idx["lat"].notna()].copy()
    if variant == "source_season":  # control: only breeding-season (May-July) observations, no target-season labels
        rows = rows[rows["dom"] == "S"]
    if variant == "blocked":
        ev = idx[idx["split"].isin(EVAL_SPLITS)]
        cells = set(zip((ev["lat"] * 10).round().astype(int), (ev["lon"] * 10).round().astype(int)))
        key = list(zip((rows["lat"] * 10).round().astype(int), (rows["lon"] * 10).round().astype(int)))
        rows = rows[[k not in cells for k in key]]
    return rows


def fit(variant="full", epochs=60, seed=0, device="cuda"):
    torch.manual_seed(seed)
    rows = training_rows(variant)
    x = torch.from_numpy(encode(rows)).to(device)
    y = torch.from_numpy(rows["y"].to_numpy()).to(device)
    model = GeoPrior(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g).to(device)
        for s in range(0, len(y), 512):
            b = perm[s : s + 512]
            loss = F.cross_entropy(model(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    log_py = np.log(np.bincount(rows["y"], minlength=data.NUM_CLASSES) / len(rows))
    return model, log_py, len(rows)


@torch.no_grad()
def log_prior(model, df, device="cuda") -> np.ndarray:
    return F.log_softmax(model(torch.from_numpy(encode(df)).to(device)), 1).cpu().numpy()


def fuse(visual_logits: np.ndarray, llr: np.ndarray, beta: float) -> np.ndarray:
    from scipy.special import log_softmax

    return log_softmax(visual_logits.astype(np.float64), 1) + beta * llr
