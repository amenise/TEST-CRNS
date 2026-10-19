"""Geo-gated TTA (ours): ecological context as a reliability signal that visual corruption cannot alter.

ctx["geo_llr"] is a (B, C) tensor of log-likelihood ratios log p_geo(y | lat, lon, month) - log p_geo(y)
(see birdshift.geo_prior). A test sample is used for adaptation only if the visually predicted species is at least
as plausible at that place and time as on average, i.e. geo_llr[i, argmax_i] >= tau (hp `geo_tau`, default 0).
The gate is combined with the method's own selection; predictions themselves stay purely visual (geo fusion of the
outputs is evaluated separately and post hoc from stored logits).
"""

from __future__ import annotations

import torch

from .anchored import Anchored
from .base import softmax_entropy
from .eata import EATA
from .simple import TENT


def geo_gate(logits, ctx, tau):
    if ctx is None or "geo_llr" not in ctx:
        return torch.ones(len(logits), dtype=torch.bool, device=logits.device)
    llr = ctx["geo_llr"]
    return llr.gather(1, logits.argmax(1, keepdim=True)).squeeze(1) >= tau


class GeoTENT(TENT):
    name = "geo_tent"

    def loss(self, logits, x, ctx):
        g = geo_gate(logits, ctx, self.hp.get("geo_tau", 0.0))
        self.n_selected = int(g.sum())
        return softmax_entropy(logits)[g].mean() if g.any() else None


class GeoEATA(EATA):
    name = "geo_eata"

    def sample_mask(self, logits, ctx):
        return geo_gate(logits, ctx, self.hp.get("geo_tau", 0.0))


class GeoAnchored(Anchored):
    name = "geo_anchored"

    def sample_mask(self, logits, ctx):
        return geo_gate(logits, ctx, self.hp.get("geo_tau", 0.0))
