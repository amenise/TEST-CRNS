"""Common machinery for test-time adaptation (TTA) methods.

Interface (all methods):
    m = Method(source_model, **hp)   # deep-copies the source model; the source model is never modified
    logits = m.forward(x, ctx)       # predict AND adapt on the batch (online TTA)
    logits = m.predict(x, ctx)       # predict WITHOUT any state change (used to measure forgetting)
    m.reset()                        # back to the exact source state (between independent streams)
x is a uint8 (B,H,W,3) CUDA tensor; ctx is an optional dict (e.g. geo-temporal prior for geo-gated methods).

BatchNorm handling: every BatchNorm2d is replaced by `AdaptiveBN`, which normalises with a mixture of source
running statistics and current batch statistics, mu = N/(N+n) mu_s + n/(N+n) mu_b (Schneider et al. 2020,
Eq. 3 with pseudo batch size N). N = 0 is the "test-time batch statistics" used by TENT/EATA/SAR/CoTTA;
N = inf is the frozen source model. LayerNorm models (ViT) have no running statistics.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import Classifier


class AdaptiveBN(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, prior: float):
        super().__init__()
        self.weight, self.bias, self.eps = bn.weight, bn.bias, bn.eps
        self.register_buffer("source_mean", bn.running_mean.detach().clone())
        self.register_buffer("source_var", bn.running_var.detach().clone())
        self.prior = prior  # pseudo batch size N; math.inf -> source statistics only

    def forward(self, x):
        if math.isinf(self.prior):
            return F.batch_norm(x, self.source_mean, self.source_var, self.weight, self.bias, False, 0.0, self.eps)
        if self.prior == 0:
            # pure test-batch statistics = train-mode BN without running-stat update (TENT); back-propagates through
            # the batch statistics. cuDNN's training-mode BN kernels are pathologically slow on this GPU
            # (RTX 2050, cuDNN 9: 11 s vs 0.46 s per ResNet-50 batch of 64),
            # so PyTorch's native kernel is used; outputs agree to fp16 precision.
            with torch.backends.cudnn.flags(enabled=False):
                return F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, self.eps)
        if not torch.is_grad_enabled() or not (x.requires_grad or self.weight.requires_grad):
            with torch.no_grad():
                xb = x.float()
                w = self.prior / (self.prior + x.shape[0])
                mean = w * self.source_mean + (1 - w) * xb.mean((0, 2, 3))
                var = w * self.source_var + (1 - w) * xb.var((0, 2, 3), unbiased=False)
            return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0.0, self.eps)
        # explicit normalisation so that gradients flow through mixed statistics (not used by the benchmark methods)
        xb = x.float()
        mean_b = xb.mean((0, 2, 3))
        var_b = xb.var((0, 2, 3), unbiased=False)
        w = self.prior / (self.prior + x.shape[0])
        mean = w * self.source_mean + (1 - w) * mean_b
        var = w * self.source_var + (1 - w) * var_b
        y = (xb - mean[None, :, None, None]) * torch.rsqrt(var + self.eps)[None, :, None, None]
        return (y * self.weight[None, :, None, None] + self.bias[None, :, None, None]).to(x.dtype)


def replace_bn(module: nn.Module, prior: float) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, AdaptiveBN(child, prior))
            n += 1
        else:
            n += replace_bn(child, prior)
    return n


def norm_modules(model: nn.Module):
    return [
        (n, m) for n, m in model.backbone.named_modules() if isinstance(m, (AdaptiveBN, nn.LayerNorm, nn.GroupNorm))
    ]


def norm_params(model: nn.Module, exclude=()):
    """Affine parameters of normalisation layers (TENT, Sec. 3), optionally excluding name prefixes (SAR)."""
    params = []
    for n, m in norm_modules(model):
        if any(n.startswith(e) for e in exclude):
            continue
        params += [p for p in (m.weight, m.bias) if p is not None]
    return params


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class TTAMethod:
    """Base class = the unadapted source model (also used for 'Source')."""

    name = "source"
    bn_prior = math.inf  # source statistics
    adapt_all = False  # CoTTA adapts every parameter

    def __init__(self, source: Classifier, **hp):
        self.hp = hp
        self.model = copy.deepcopy(source).eval()
        self.has_bn = replace_bn(self.model, hp.get("bn_prior", self.bn_prior)) > 0
        self.model.requires_grad_(False)
        self.setup()
        self.init_state = copy.deepcopy(self.model.state_dict())
        self.reset()

    # hooks -------------------------------------------------------------
    def setup(self):
        """Choose trainable parameters / create auxiliary models. Called once."""

    def reset_state(self):
        """Reset method-specific state (optimizer, memories). Called by reset()."""

    # API ---------------------------------------------------------------
    def reset(self):
        self.model.load_state_dict(self.init_state)
        self.steps = 0
        self.reset_state()

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16)

    @torch.no_grad()
    def predict(self, x, ctx=None):
        with self.autocast():
            return self.model(x).float()

    def forward(self, x, ctx=None):
        return self.predict(x, ctx)


class GradientTTA(TTAMethod):
    """Methods that update normalisation affine parameters with SGD (momentum 0.9) as in TENT's ImageNet setup."""

    bn_prior = 0.0
    exclude = ()

    def setup(self):
        self.params = norm_params(self.model, self.exclude) if not self.adapt_all else list(self.model.parameters())
        for p in self.params:
            p.requires_grad_(True)
        if (self.has_bn or self.adapt_all) and hasattr(self.model.backbone, "set_grad_checkpointing"):
            # ResNet-50 back-propagation at batch 64, and CoTTA (all parameters, 3 model copies) on DINOv2 (4.1 GB
            # peak, measured), do not fit in 4 GB otherwise; recomputation sees the same batch (and the saved RNG
            # state), so batch statistics and results are unchanged
            self.model.backbone.set_grad_checkpointing(True)

    def make_optimizer(self):
        return torch.optim.SGD(self.params, lr=self.hp["lr"], momentum=0.9)

    def reset_state(self):
        self.optimizer = self.make_optimizer()

    def loss(self, logits, x, ctx):
        raise NotImplementedError

    def forward(self, x, ctx=None):
        with self.autocast():
            logits = self.model(x).float()
        loss = self.loss(logits, x, ctx)
        if loss is not None and loss.requires_grad:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.steps += 1
        return logits.detach()
