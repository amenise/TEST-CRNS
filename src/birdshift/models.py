"""Source models: pretrained timm backbone (frozen during source training) + linear head + temperature.

Model output = head(features) / T, i.e. temperature-calibrated logits. TTA methods operate on this output,
so entropy-based thresholds (E0 = 0.4 ln C in EATA/SAR) see calibrated probabilities.

Input convention: uint8 tensors (B, H, W, 3) on the target device; normalisation happens inside the model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:  # HTTPS is intercepted on the development machine; use the OS certificate store for weight downloads
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
import timm

from .data import NUM_CLASSES
from .paths import RUNS

BACKBONES = {
    # ViT-S/16, LayerNorm, AugReg ImageNet-21k pre-training then ImageNet-1k fine-tuning (Steiner et al. 2022)
    "vit_s16": dict(timm="vit_small_patch16_224.augreg_in21k_ft_in1k", norm="LayerNorm"),
    # ResNet-50, BatchNorm, original torchvision ImageNet-1k weights (He et al. 2016)
    "resnet50": dict(timm="resnet50.tv_in1k", norm="BatchNorm"),
    # ViT-B/16, optional larger LayerNorm model
    "vit_b16": dict(timm="vit_base_patch16_224.augreg_in21k_ft_in1k", norm="LayerNorm"),
    # ViT-S/14 DINOv2 (Oquab et al. 2024), self-supervised on LVD-142M (iNaturalist not used to build it), LayerNorm;
    # run at 224x224 (16x16 patches) like the other backbones. Chosen after results/backbone_probe.csv.
    "dinov2": dict(timm="vit_small_patch14_dinov2.lvd142m", norm="LayerNorm", kwargs=dict(img_size=224)),
}


class Classifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.name = backbone
        spec = BACKBONES[backbone]
        self.backbone = timm.create_model(spec["timm"], pretrained=pretrained, num_classes=0, **spec.get("kwargs", {}))
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1) * 255)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        self.register_buffer("temperature", torch.ones(()))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.permute(0, 3, 1, 2).float()
        return (x - self.mean) / self.std

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocess(x))

    def logits_from_features(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f.float()) / self.temperature

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.features(x)
        out = self.logits_from_features(f)
        return (out, f) if return_features else out


def head_path(backbone: str):
    return RUNS / "source" / backbone / "head.pt"


def load_source(backbone: str, device: str = "cuda") -> Classifier:
    """Backbone with pretrained weights + trained linear head and temperature (cli/train_source.py)."""
    model = Classifier(backbone)
    state = torch.load(head_path(backbone), map_location="cpu")
    model.head.load_state_dict(state["head"])
    model.temperature.fill_(state["temperature"])
    if state.get("bn_stats"):  # BatchNorm statistics re-estimated on S_train (cli/train_source.py)
        missing = model.backbone.load_state_dict(state["bn_stats"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
    return model.to(device).eval()


@torch.no_grad()
def extract_features(
    model: Classifier, imgs: np.ndarray, batch_size: int = 128, flip: bool = False, device: str = "cuda"
) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(imgs), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(imgs[s : s + batch_size])).to(device)
        if flip:
            x = x.flip(2)  # horizontal flip of (B, H, W, C)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = model.features(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def to_device(batch: np.ndarray, device: str = "cuda") -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch)).to(device, non_blocking=True)
