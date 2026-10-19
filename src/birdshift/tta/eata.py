"""EATA (Niu et al. 2022) and RDumb (Press et al. 2023)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..data import NUM_CLASSES
from .base import GradientTTA, softmax_entropy


class EATA(GradientTTA):
    """Efficient Anti-forgetting Test-time Adaptation (Niu et al. 2022).

    - Reliable samples: entropy E(x) < E0 with E0 = 0.4 ln C (Eq. 3 and Sec. 5 implementation details);
      weight S_ent(x) = exp(E0 - E(x)) (implemented as 1/exp(E - E0), official code).
    - Non-redundant samples: |cos(m_{t-1}, p(x))| < epsilon, where m is a moving average (0.9) of the mean
      prediction of selected samples (Eq. 4-5).
    - Anti-forgetting: Fisher-weighted L2 to source parameters, beta * sum F (theta - theta0)^2 (Eq. 7-8),
      Fisher estimated on unlabelled source data with pseudo-labels (`fisher_images`).
    hp: lr, e0_factor (0.4), d_margin (epsilon, 0.05), fisher_alpha (beta, 2000), fisher_images (uint8 tensor or None)
    """

    name = "eata"
    use_fisher = True

    def setup(self):
        super().setup()
        self.e0 = self.hp.get("e0_factor", 0.4) * math.log(NUM_CLASSES)
        self.d_margin = self.hp.get("d_margin", 0.05)
        self.fisher = None
        imgs = self.hp.get("fisher_images")
        if self.use_fisher and imgs is not None and self.hp.get("fisher_alpha", 2000.0) > 0:
            self.fisher = self._compute_fisher(imgs)

    def _compute_fisher(self, imgs, batch_size=32):
        """Diagonal Fisher of the cross-entropy to the model's own pseudo-labels (official EATA code)."""
        fisher = [torch.zeros_like(p) for p in self.params]
        n = 0
        for s in range(0, len(imgs), batch_size):
            x = imgs[s : s + batch_size].cuda(non_blocking=True)
            self.model.zero_grad(set_to_none=True)
            with self.autocast():
                logits = self.model(x).float()
            F.cross_entropy(logits, logits.argmax(1)).backward()
            for f, p in zip(fisher, self.params):
                f += p.grad.detach() ** 2
            n += 1
        self.model.zero_grad(set_to_none=True)
        return [(f / n, p.detach().clone()) for f, p in zip(fisher, self.params)]

    def reset_state(self):
        super().reset_state()
        self.model_probs = None

    def _update_probs(self, new_probs):
        if new_probs.shape[0] == 0:
            return
        mean = new_probs.mean(0)
        self.model_probs = mean if self.model_probs is None else 0.9 * self.model_probs + 0.1 * mean

    def sample_mask(self, logits, ctx):
        """Extra gate on top of EATA's filters (used by geo-gated variants). All-true by default."""
        return torch.ones(len(logits), dtype=torch.bool, device=logits.device)

    def loss(self, logits, x, ctx):
        ent = softmax_entropy(logits)
        keep = (ent < self.e0) & self.sample_mask(logits, ctx)
        probs = logits.softmax(1)
        if self.model_probs is not None:
            cos = F.cosine_similarity(self.model_probs.unsqueeze(0), probs, dim=1)
            keep2 = keep & (cos.abs() < self.d_margin)
        else:
            keep2 = keep
        self._update_probs(probs[keep2].detach())
        self.n_selected = int(keep2.sum())
        if self.n_selected == 0:
            return None
        # official code: entropys.mul(1/exp(E - E0)).mean(0) over the selected samples
        loss = (ent[keep2] * torch.exp(-(ent[keep2].detach() - self.e0))).mean()
        if self.use_fisher and self.fisher is not None:
            beta = self.hp.get("fisher_alpha", 2000.0)
            loss = loss + beta * sum((f * (p - p0) ** 2).sum() for (f, p0), p in zip(self.fisher, self.params))
        return loss


class RDumb(EATA):
    """RDumb (Press et al. 2023): ETA loss (EATA without the Fisher regulariser, E0 = 0.4 ln C, epsilon = 0.05)
    and a full reset to the source model every T steps (T = 1000 in the paper; hp `reset_every`)."""

    name = "rdumb"
    use_fisher = False

    def forward(self, x, ctx=None):
        if self.steps > 0 and self.steps % self.hp.get("reset_every", 1000) == 0:
            steps = self.steps
            self.reset()
            self.steps = steps
        return super().forward(x, ctx)
