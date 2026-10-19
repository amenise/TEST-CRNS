"""Streaming evaluation shared by the HP selection, the TTA benchmark and the continual-TTA experiments.

A *condition* is written "<split>:<cond>", e.g. "W_test:clean" or "S_test:motion_blur_s3".
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .paths import RUNS

_GPU_CACHE: dict = {}


def parse_condition(spec: str):
    split, cond = spec.split(":")
    if cond == "clean":
        return split, None, None
    name, sev = cond.rsplit("_s", 1)
    return split, name, int(sev)


def condition_tensor(spec: str, cache=True) -> torch.Tensor:
    """All images of a condition as a uint8 tensor in pinned CPU memory (1000 images = 150 MB).

    Images are NOT kept on the GPU: with 4 GB of VRAM, CoTTA (3 model copies, ~3.3 GB peak) plus cached images
    exceeded the card and Windows silently spilled to shared system memory (one HP configuration took 1908 s instead
    of ~60 s). Batches are moved to the GPU when used (`to_gpu`)."""
    if spec in _GPU_CACHE:
        return _GPU_CACHE[spec]
    split, c, s = parse_condition(spec)
    t = torch.from_numpy(np.array(data.condition_images(split, c, s))).pin_memory()
    if cache:
        while _GPU_CACHE and sum(v.numel() for v in _GPU_CACHE.values()) + t.numel() > 2e9:  # 2 GB of host RAM
            _GPU_CACHE.pop(next(iter(_GPU_CACHE)))
        _GPU_CACHE[spec] = t
    return t


def to_gpu(x: torch.Tensor, idx) -> torch.Tensor:
    """x[idx] on the GPU; idx: numpy array or CPU LongTensor."""
    if not torch.is_tensor(idx):
        idx = torch.from_numpy(np.asarray(idx))
    return x[idx].cuda(non_blocking=True)


def condition_labels(spec: str) -> np.ndarray:
    return data.labels(spec.split(":")[0])


def run_stream(
    method, x: torch.Tensor, order: np.ndarray, batch_size: int, ctx: dict | None = None, adapt: bool = True
) -> np.ndarray:
    """Feed x[order] in batches; return logits re-indexed to the original image order (N, C) float32."""
    out = torch.empty(len(order), 100, device="cpu")
    for idx in data.batches(order, batch_size):
        it = torch.from_numpy(idx)
        xb = to_gpu(x, it)
        bctx = None if ctx is None else {k: v[it.to(v.device)] for k, v in ctx.items()}
        logits = method.forward(xb, bctx) if adapt else method.predict(xb, bctx)
        out[it] = logits.float().cpu()
    return out.numpy()


def evaluate_condition(
    method,
    spec: str,
    seed: int = 0,
    batch_size: int = 64,
    order: str = "iid",
    ctx: dict | None = None,
    reset: bool = True,
) -> dict:
    """Independent stream: reset -> adapt online on the whole condition -> online predictions."""
    torch.manual_seed(seed)
    y = condition_labels(spec)
    perm = data.stream_order(y, order, seed)
    if reset:
        method.reset()
    t0 = time.time()
    logits = run_stream(method, condition_tensor(spec), perm, batch_size, ctx)
    torch.cuda.synchronize()
    return dict(logits=logits, y=y, order=perm, seconds=time.time() - t0, acc=float((logits.argmax(1) == y).mean()))


def run_file(*parts) -> Path:
    return RUNS.joinpath(*parts)


def save_run(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
