"""Source, BN-adapt, TENT and pseudo-labelling."""

from __future__ import annotations

import torch.nn.functional as F

from .base import GradientTTA, TTAMethod, softmax_entropy


class Source(TTAMethod):
    """No adaptation: frozen source model with source normalisation statistics."""

    name = "source"


class BNAdapt(TTAMethod):
    """Test-time BatchNorm statistics adaptation (Schneider et al. 2020; AdaBN, Li et al. 2017).

    Normalisation statistics = N/(N+n) source + n/(N+n) current batch (Eq. 3 of Schneider et al.),
    hp `bn_prior` = N. No parameter is learned. Identical to Source for LayerNorm models.
    """

    name = "bn"
    bn_prior = 16.0


class TENT(GradientTTA):
    """TENT (Wang et al. 2021, Sec. 3): minimise the Shannon entropy of predictions w.r.t. the affine parameters of
    normalisation layers, with test-batch normalisation statistics (BN) - one gradient step per batch, the returned
    prediction is the one computed before the update (as in the official implementation)."""

    name = "tent"

    def loss(self, logits, x, ctx):
        return softmax_entropy(logits).mean()


class PseudoLabel(GradientTTA):
    """Hard pseudo-labelling (Lee 2013) applied at test time: cross-entropy to the arg-max prediction on samples whose
    confidence exceeds `threshold`, updating normalisation affine parameters (same parameter set as TENT)."""

    name = "pl"

    def loss(self, logits, x, ctx):
        p = logits.softmax(1)
        conf, yhat = p.max(1)
        keep = conf >= self.hp.get("threshold", 0.9)
        if not keep.any():
            return None
        return F.cross_entropy(logits[keep], yhat[keep])
