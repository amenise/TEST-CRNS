"""T3A: test-time classifier adjustment (Iwasawa & Matsuo 2021).

Follows the authors' public DomainBed implementation (`T3A` algorithm), checked against the paper (Sec. 3, eqs. 3-5;
PDF from the NeurIPS proceedings). One difference: like the official code, we L2-normalise each prototype, while
eq. 5 uses the plain mean of the normalised supports.
- supports are initialised with the linear head's weight vectors (one per class), labelled by the head's own
  prediction, with their prediction entropy;
- each test feature z is added to the supports with its pseudo-label argmax(head(z)) and entropy;
- per class only the `filter_k` lowest-entropy supports are kept (filter_k = -1 keeps all);
- class prototypes = sum of L2-normalised supports of that class, normalised; logits = z @ prototypes.
The backbone is never updated. Logits live on a cosine-like scale, so NLL/ECE of T3A are not comparable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import TTAMethod, softmax_entropy


class T3A(TTAMethod):
    name = "t3a"

    def reset_state(self):
        w = self.model.head.weight.detach().float()
        with torch.no_grad():
            p = self.model.head(w) / self.model.temperature
        self.supports = w.clone()
        self.labels = F.one_hot(p.argmax(1), w.shape[0]).float()
        self.ent = softmax_entropy(p)

    def _select(self):
        k = self.hp.get("filter_k", 20)
        if k < 0:
            return
        yhat = self.labels.argmax(1)
        keep = []
        for c in yhat.unique():
            idx = torch.nonzero(yhat == c).squeeze(1)
            keep.append(idx[self.ent[idx].argsort()[:k]])
        keep = torch.cat(keep)
        self.supports, self.labels, self.ent = self.supports[keep], self.labels[keep], self.ent[keep]

    def _logits(self, z):
        s = F.normalize(self.supports, dim=1)
        w = s.T @ self.labels
        return z @ F.normalize(w, dim=0)

    @torch.no_grad()
    def predict(self, x, ctx=None):
        with self.autocast():
            z = self.model.features(x).float()
        return self._logits(z)

    @torch.no_grad()
    def forward(self, x, ctx=None):
        with self.autocast():
            z = self.model.features(x).float()
        p = self.model.logits_from_features(z)
        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, F.one_hot(p.argmax(1), p.shape[1]).float()])
        self.ent = torch.cat([self.ent, softmax_entropy(p)])
        self._select()
        self.steps += 1
        return self._logits(z)
