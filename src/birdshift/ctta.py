"""Continual test-time adaptation: one model, a sequence of domains, no reset (unless the method resets itself).

For a sequence of stages k = 1..K (each stage = one domain streamed once, online predictions recorded):
  online[k]       online accuracy while adapting on stage k (the usual online TTA metric)
  curve[k]        per-batch online accuracy (adaptation speed / stability)
  R[k, d]         accuracy on the *held-out* set of domain d after stage k, measured with predict() (no update);
                  row 0 = before any adaptation (source model). Domains d are the distinct domains of the sequence.
Derived (Lopez-Paz & Ranzato 2017 / De Lange et al. 2021 style, adapted to unsupervised TTA):
  gain_after[d]   R[first stage of d, d] - R[0, d]       benefit of having adapted to d
  BWT             mean_d (R[K, d] - R[first stage of d, d])  backward transfer (negative = forgetting)
  forgetting_src  R[0, source] - min_k R[k, source]      worst degradation on the clean source domain
Held-out sets (*_hold, 5 images/class) are never streamed, so R measures generalisation, not memorisation.

Shift detector (ours, label-free): z-scored features of a FROZEN source model; the score of a batch is the squared
distance between its mean and the running mean of the current segment (normalised by dimension); a shift is declared
when the score exceeds a threshold calibrated on random S_hold batches (99th percentile x margin).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import data
from .runner import condition_labels, condition_tensor, to_gpu


class ShiftDetector:
    def __init__(self, source_model, mu, sd, threshold=None, ema=0.5):
        self.src, self.mu, self.sd = source_model, mu, sd
        self.threshold, self.ema = threshold, ema
        self.ref = None
        self.n_in_segment = 0

    @torch.no_grad()
    def batch_mean(self, x):
        with torch.autocast("cuda", dtype=torch.float16):
            f = self.src.features(x).float()
        return ((f - self.mu) / self.sd).mean(0)

    def score(self, m):
        return float(((m - self.ref) ** 2).mean()) if self.ref is not None else 0.0

    def update(self, x) -> tuple[bool, float]:
        m = self.batch_mean(x)
        s = self.score(m)
        shift = self.ref is not None and self.n_in_segment >= 2 and s > self.threshold
        if shift or self.ref is None:
            self.ref, self.n_in_segment = m, 1
        else:
            self.ref = self.ema * self.ref + (1 - self.ema) * m if self.n_in_segment > 1 else (self.ref + m) / 2
            self.n_in_segment += 1
        return shift, s

    def calibrate(self, imgs: torch.Tensor, batch_size=64, n_streams=20, q=99, margin=1.5, seed=0):
        rng = np.random.default_rng(seed)
        scores = []
        self.threshold = float("inf")  # no shift can be declared while calibrating
        for _ in range(n_streams):
            self.ref, self.n_in_segment = None, 0
            perm = rng.permutation(len(imgs))
            for idx in data.batches(perm, batch_size):
                if len(idx) < batch_size:
                    continue
                _, s = self.update(to_gpu(imgs, idx))
                if self.n_in_segment > 2:
                    scores.append(s)
        self.threshold = float(np.percentile(scores, q) * margin)
        self.ref, self.n_in_segment = None, 0
        return self.threshold


def run_sequence(
    method,
    stages: list[str],
    eval_sets: dict[str, str],
    seed=0,
    batch_size=64,
    detector: ShiftDetector | None = None,
    reset_on_shift=False,
    ctx_fn=None,
) -> dict:
    """stages: condition specs streamed in order, e.g. ["S_test:clean", "S_test:low_light_s3", ...]
    eval_sets: domain name -> held-out condition spec, e.g. {"S": "S_hold:clean", ...}; stage domain = domain_of(spec)."""
    torch.manual_seed(seed)
    method.reset()
    domains = list(eval_sets)

    def evaluate():
        out = {}
        for d, spec in eval_sets.items():
            x, y = condition_tensor(spec), condition_labels(spec)
            preds = []
            for idx in data.batches(np.arange(len(y)), batch_size):
                it = torch.from_numpy(idx)
                ctx = ctx_fn(spec, it) if ctx_fn else None
                preds.append(method.predict(to_gpu(x, it), ctx).argmax(1).cpu().numpy())
            out[d] = float((np.concatenate(preds) == y).mean())
        return out

    R = [evaluate()]
    online, curves, logits_all, detections, scores = [], [], [], [], []
    t0 = time.time()
    for k, spec in enumerate(stages):
        x, y = condition_tensor(spec), condition_labels(spec)
        perm = data.stream_order(y, "iid", seed * 1000 + k)
        correct, out = [], np.zeros((len(y), 100), np.float16)
        for b, idx in enumerate(data.batches(perm, batch_size)):
            it = torch.from_numpy(idx)
            xb = to_gpu(x, it)
            if detector is not None:
                shift, s = detector.update(xb)
                scores.append(s)
                if shift:
                    detections.append((k, b))
                    if reset_on_shift:
                        steps = method.steps
                        method.reset()
                        method.steps = steps
            ctx = ctx_fn(spec, it) if ctx_fn else None
            lg = method.forward(xb, ctx).float().cpu().numpy()
            out[idx] = lg
            correct.append((lg.argmax(1) == y[idx]).mean())
        online.append(float((out.argmax(1) == y).mean()))
        curves.append(correct)
        logits_all.append(out)
        R.append(evaluate())
    Rm = np.array([[r[d] for d in domains] for r in R])
    return dict(
        stages=stages,
        domains=domains,
        R=Rm,
        online=np.array(online),
        curves=curves,
        detections=detections,
        scores=np.array(scores),
        seconds=time.time() - t0,
        n_recoveries=getattr(method, "n_recoveries", 0),
    )


def summarize(res: dict, stage_domains: list[str], source_domain: str = "S") -> dict:
    R, doms = res["R"], res["domains"]
    first = {}
    for k, d in enumerate(stage_domains):
        first.setdefault(d, k + 1)
    gain = {d: R[first[d], doms.index(d)] - R[0, doms.index(d)] for d in first}
    bwt = float(np.mean([R[-1, doms.index(d)] - R[first[d], doms.index(d)] for d in first]))
    s = doms.index(source_domain) if source_domain in doms else None
    K = len(stage_domains)
    half = K // 2
    return dict(
        online_mean=float(res["online"].mean()),
        online_round1=float(res["online"][:half].mean()),
        online_round2=float(res["online"][half:].mean()),
        final_mean=float(R[-1].mean()),
        source_model_mean=float(R[0].mean()),
        bwt=bwt,
        mean_gain_after=float(np.mean(list(gain.values()))),
        forgetting_source=float(R[0, s] - R[1:, s].min()) if s is not None else np.nan,
        final_source=float(R[-1, s]) if s is not None else np.nan,
        n_detections=len(res["detections"]),
        seconds=res["seconds"],
        n_recoveries=res["n_recoveries"],
    )
