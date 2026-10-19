"""SAR: sharpness-aware and reliable entropy minimisation (Niu et al. 2023)."""

from __future__ import annotations

import math

import torch

from ..data import NUM_CLASSES
from .base import GradientTTA, softmax_entropy


class SAR(GradientTTA):
    """SAR (Niu et al. 2023, Sec. 3.2, Eq. 2-5 and Algorithm 1):
    - reliable entropy minimisation: only samples with E(x) < E0 = 0.4 ln C (Eq. 2);
    - sharpness-aware minimisation (SAM, rho = 0.05) of that entropy, the second (perturbed) pass re-filters
      samples with E < E0;
    - model recovery: if the moving average (0.9) of the second-pass entropy falls below 0.2, the model is reset.
    Following the official code, the top blocks of ViTs (blocks.9-11 and the final norm) and layer4 of ResNets are
    not adapted.
    hp: lr, rho (0.05), e0_factor (0.4), reset_ema (0.2)
    """

    name = "sar"

    def setup(self):
        self.exclude = ("blocks.9", "blocks.10", "blocks.11", "norm", "layer4")
        super().setup()
        self.e0 = self.hp.get("e0_factor", 0.4) * math.log(NUM_CLASSES)

    def reset_state(self):
        super().reset_state()
        self.ema = None
        self.n_recoveries = getattr(self, "n_recoveries", 0)

    def forward(self, x, ctx=None):
        rho = self.hp.get("rho", 0.05)
        with self.autocast():
            logits = self.model(x).float()
        ent = softmax_entropy(logits)
        keep = ent < self.e0
        if keep.any():
            self.optimizer.zero_grad(set_to_none=True)
            ent[keep].mean().backward()
            # SAM first step: theta + e(theta), e = rho * g / ||g||
            grads = [p.grad for p in self.params if p.grad is not None]
            norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) + 1e-12
            eps = []
            with torch.no_grad():
                for p in self.params:
                    e = None if p.grad is None else p.grad * (rho / norm)
                    if e is not None:
                        p.add_(e)
                    eps.append(e)
            self.optimizer.zero_grad(set_to_none=True)
            with self.autocast():
                ent2 = softmax_entropy(self.model(x).float())[keep]
            keep2 = ent2 < self.e0
            loss2_value = ent2.detach().mean().item()
            if keep2.any():
                ent2[keep2].mean().backward()
            with torch.no_grad():  # SAM second step: back to theta, then descend with the perturbed gradient
                for p, e in zip(self.params, eps):
                    if e is not None:
                        p.sub_(e)
            if keep2.any():
                self.optimizer.step()
            if not math.isnan(loss2_value):
                self.ema = loss2_value if self.ema is None else 0.9 * self.ema + 0.1 * loss2_value
            if self.ema is not None and self.ema < self.hp.get("reset_ema", 0.2):
                steps, rec = self.steps, self.n_recoveries + 1
                self.reset()
                self.steps, self.n_recoveries = steps, rec
        self.steps += 1
        return logits.detach()
