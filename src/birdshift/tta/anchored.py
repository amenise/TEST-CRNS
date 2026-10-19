"""Anchored TTA (ours): a simple combination of known anti-forgetting ingredients for continual TTA.

Loss on each test batch (student parameters = normalisation affine parameters, as in TENT):

    L = sum_i w_i CE(p_student(x_i), argmax p_teacher(x_i)) / #{w_i > 0}    reliable teacher pseudo-labelling
        + lambda_kl * mean_i KL(p_source(x_i) || p_student(x_i))             prediction anchor (LwF-like, Li & Hoiem 2017)
        + lambda_l2 * ||theta - theta_0||^2                                  parameter anchor (L2-SP / EWC-like)

  w_i = 1[E_i < E0] exp(E0 - E_i), E_i = entropy of the teacher, E0 = 0.4 ln C   (EATA reliability weight)
  teacher = EMA of the student (alpha), as in mean-teacher CTTA (CoTTA, RMT); prediction = teacher output.

Ablations (hp flags): use_teacher=False (pseudo-labels and predictions from the student itself),
use_filter=False (w_i = 1), use_anchor=False (lambda_kl = lambda_l2 = 0).
Hard (arg-max) pseudo-labels are used on purpose: with soft labels the cross-entropy to a teacher that equals the
student at initialisation has exactly zero gradient (p_student - p_teacher = 0), so adaptation would never start
(found by tests/test_tta.py).
Hypothesis: the anchor bounds drift on long streams (forgetting, collapse) while the reliability filter and the
slow teacher limit error accumulation from wrong pseudo-labels.
"""

from __future__ import annotations

import copy
import math

import torch

from ..data import NUM_CLASSES
from .base import GradientTTA, softmax_entropy
from .cotta import ema_update


class Anchored(GradientTTA):
    name = "anchored"

    def setup(self):
        super().setup()
        h = self.hp
        self.use_teacher = h.get("use_teacher", True)
        self.use_filter = h.get("use_filter", True)
        self.use_anchor = h.get("use_anchor", True)
        self.e0 = h.get("e0_factor", 0.4) * math.log(NUM_CLASSES)
        self.teacher = copy.deepcopy(self.model).requires_grad_(False)
        self.source = copy.deepcopy(self.model).requires_grad_(False)

    def reset_state(self):
        super().reset_state()
        self.teacher.load_state_dict(self.init_state)
        self.source.load_state_dict(self.init_state)
        self.params0 = [p.detach().clone() for p in self.params]

    def sample_mask(self, logits, ctx):
        return torch.ones(len(logits), dtype=torch.bool, device=logits.device)

    @torch.no_grad()
    def predict(self, x, ctx=None):
        with self.autocast():
            return (self.teacher if self.use_teacher else self.model)(x).float()

    def forward(self, x, ctx=None):
        h = self.hp
        with self.autocast():
            s = self.model(x).float()
            with torch.no_grad():
                t = self.teacher(x).float() if self.use_teacher else s.detach()
                src = self.source(x).float() if self.use_anchor else None
        ent = softmax_entropy(t)
        if self.use_filter:
            w = (ent < self.e0).float() * torch.exp(self.e0 - ent)
        else:
            w = torch.ones_like(ent)
        w = w * self.sample_mask(t, ctx).float()
        ce = torch.nn.functional.cross_entropy(s, t.argmax(1), reduction="none")
        loss = (w * ce).sum() / (w > 0).sum().clamp_min(1)
        if self.use_anchor:
            kl = (src.softmax(1) * (src.log_softmax(1) - s.log_softmax(1))).sum(1).mean()
            l2 = sum(((p - p0) ** 2).sum() for p, p0 in zip(self.params, self.params0))
            loss = loss + h.get("lambda_kl", 1.0) * kl + h.get("lambda_l2", 1.0) * l2
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.use_teacher:
            ema_update(self.teacher, self.model, h.get("alpha", 0.99))
        self.steps += 1
        return t if self.use_teacher else s.detach()
