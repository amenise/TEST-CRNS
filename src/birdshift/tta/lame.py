"""LAME: Laplacian Adjusted Maximum-likelihood Estimation (Boudiaf et al. 2022)."""

from __future__ import annotations

import torch

from .base import TTAMethod


def knn_affinity(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Binary kNN affinity W_ij = 1 if j is among the k nearest neighbours of i (official LAME kNN kernel)."""
    n = feats.shape[0]
    dist = torch.cdist(feats, feats)
    idx = dist.topk(min(k + 1, n), dim=-1, largest=False).indices[:, 1:]
    w = torch.zeros(n, n, device=feats.device)
    w.scatter_(-1, idx, 1.0)
    return w


def laplacian_optimization(unary: torch.Tensor, kernel: torch.Tensor, lam: float = 1.0, max_steps: int = 100):
    """Concave-convex procedure of LAME: Y <- softmax(-unary + lam * W Y) until the energy converges."""
    y = (-unary).softmax(-1)
    old = float("inf")
    for i in range(max_steps):
        pairwise = lam * kernel @ y
        y = (-unary + pairwise).softmax(-1)
        e = (unary * y - lam * pairwise * y + y * torch.log(y.clamp_min(1e-20))).sum().item()
        if i > 1 and abs(e - old) <= 1e-8 * abs(old):
            break
        old = e
    return y


class LAME(TTAMethod):
    """LAME (Boudiaf et al. 2022): the model is NOT adapted; the batch's output probabilities are corrected by
    maximising likelihood under a Laplacian regulariser that encourages neighbouring samples (kNN in feature space,
    k = `knn`) to share labels. Designed to be robust to non-i.i.d. streams. Features are L2-normalised."""

    name = "lame"

    @torch.no_grad()
    def forward(self, x, ctx=None):
        with self.autocast():
            z = self.model.features(x).float()
        p = self.model.logits_from_features(z).softmax(1)
        z = torch.nn.functional.normalize(z, dim=1)
        y = laplacian_optimization(-torch.log(p + 1e-10), knn_affinity(z, self.hp.get("knn", 5)))
        self.steps += 1
        return torch.log(y.clamp_min(1e-20))

    predict = forward
