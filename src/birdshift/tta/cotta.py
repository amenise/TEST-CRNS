"""CoTTA: continual test-time adaptation (Wang et al. 2022)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from .base import GradientTTA


def augment(x: torch.Tensor) -> torch.Tensor:
    """CoTTA-style stochastic augmentation on a uint8 (B,H,W,3) batch -> float (B,3,H,W) in [0,255]:
    colour jitter (brightness/contrast +-20%), random affine (rotation +-15 deg, translation +-1/16,
    scale 0.9-1.1), horizontal flip (p=0.5), small Gaussian noise. Mirrors the transforms of the official code."""
    x = x.permute(0, 3, 1, 2).float()
    b = x.shape[0]
    dev = x.device
    mean = x.mean((1, 2, 3), keepdim=True)
    c = 1 + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.4
    br = 1 + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.4
    x = ((x - mean) * c + mean) * br
    ang = (torch.rand(b, device=dev) - 0.5) * 2 * math.radians(15)
    sc = 0.9 + 0.2 * torch.rand(b, device=dev)
    tr = (torch.rand(b, 2, device=dev) - 0.5) * 2 / 16 * 2  # normalised coords span 2
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = torch.cos(ang) / sc
    theta[:, 0, 1] = -torch.sin(ang) / sc
    theta[:, 1, 0] = torch.sin(ang) / sc
    theta[:, 1, 1] = torch.cos(ang) / sc
    theta[:, :, 2] = tr
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    flip = torch.rand(b, device=dev) < 0.5
    x[flip] = x[flip].flip(3)
    x = x + torch.randn_like(x) * 0.005 * 255
    return x.clamp(0, 255)


@torch.no_grad()
def ema_update(teacher, student, alpha):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(alpha).add_(ps.detach(), alpha=1 - alpha)


class CoTTA(GradientTTA):
    """CoTTA (Wang et al. 2022):
    - weight-averaged teacher (EMA of the student, alpha = 0.999), prediction = teacher output;
    - augmentation-averaged pseudo-labels: if the source (anchor) model's mean confidence on the batch is below
      `ap`, the teacher prediction is averaged over `n_aug` augmentations (32 in the paper; 8 here for compute,
      documented deviation);
    - student trained with cross-entropy to the teacher's soft pseudo-labels, ALL parameters updated;
    - stochastic restoration: each weight is reset to its source value with probability `rst` (0.01) after each step.
    hp: lr, ap (confidence threshold), n_aug, rst, alpha
    """

    name = "cotta"
    adapt_all = True

    def setup(self):
        super().setup()
        self.teacher = copy.deepcopy(self.model).requires_grad_(False)
        self.anchor = copy.deepcopy(self.model).requires_grad_(False)

    def reset_state(self):
        super().reset_state()
        self.teacher.load_state_dict(self.init_state)
        self.anchor.load_state_dict(self.init_state)
        self.source_params = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def predict(self, x, ctx=None):
        with self.autocast():
            return self.teacher(x).float()

    def forward(self, x, ctx=None):
        with self.autocast():
            logits = self.model(x).float()
            with torch.no_grad():
                anchor_conf = self.anchor(x).float().softmax(1).max(1).values
                t = self.teacher(x).float()
                if anchor_conf.mean() < self.hp.get("ap", 0.5):
                    t = torch.stack([self.teacher(augment(x)).float() for _ in range(self.hp.get("n_aug", 8))]).mean(0)
        loss = -(t.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        ema_update(self.teacher, self.model, self.hp.get("alpha", 0.999))
        rst = self.hp.get("rst", 0.01)
        with torch.no_grad():
            for p, p0 in zip(self.params, self.source_params):
                mask = (torch.rand_like(p) < rst).to(p.dtype)
                p.mul_(1 - mask).add_(p0 * mask)
        self.steps += 1
        return t
