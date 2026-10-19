"""E8: does the label-free shift detector notice each corruption (and each season)?

Same detector and calibration as the continual-TTA runs (`run_ctta.make_detector`: frozen source model, z-scored
S_train features, threshold = 99th percentile of S_hold batch scores x 1.5). Each stream is 16 batches of clean S_test
followed by 16 batches of the target condition, in a seeded random order. Recorded per target and stream order:
  detected      the detector fires at least once on the target batches
  at_first      it fires on the very first target batch (zero delay)
  delay         index of the first target batch that fires (NaN if never)
  false_alarms  alarms during the clean part (should be 0)
The frozen-model features of every condition are computed once, then streams are replayed from them.

python -m birdshift.cli.run_detector [--backbone vit_s16] [--seeds 0 1 2]
Output: results/detector_corruptions_<backbone>.csv (one row per target x seed)
"""

import argparse

import numpy as np
import pandas as pd
import torch

from .. import data
from ..corruptions import SEVERITIES, TEST_CORRUPTIONS
from ..ctta import ShiftDetector
from ..models import load_source
from ..paths import RESULTS
from ..runner import condition_tensor, to_gpu
from ..utils import set_seed, write_csv, write_json
from .run_ctta import make_detector

BATCH = 64


class FeatureDetector(ShiftDetector):
    """The calibrated detector, fed with precomputed frozen-model features instead of images."""

    def batch_mean(self, f):
        return ((f - self.mu) / self.sd).mean(0)


@torch.no_grad()
def features(model, spec):
    x = condition_tensor(spec, cache=False)
    out = []
    for idx in data.batches(np.arange(len(x)), 256):
        with torch.autocast("cuda", dtype=torch.float16):  # same precision as ShiftDetector.batch_mean
            out.append(model.features(to_gpu(x, idx)).float())
    return torch.cat(out)


def replay(det, clean, target, seed):
    rng = np.random.default_rng(seed)
    det.ref, det.n_in_segment = None, 0
    fired = []
    for part in (clean, target):
        perm = rng.permutation(len(part))
        for idx in data.batches(perm, BATCH):
            if len(idx) == BATCH:
                fired.append(det.update(part[torch.from_numpy(idx)])[0])
    n_clean = len(clean) // BATCH
    tgt = fired[n_clean:]
    first = next((i for i, f in enumerate(tgt) if f), None)
    return dict(
        detected=first is not None,
        at_first=bool(tgt[0]),
        delay=np.nan if first is None else first,
        false_alarms=int(sum(fired[:n_clean])),
        n_alarms_target=int(sum(tgt)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_s16")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    set_seed(0)
    model = load_source(args.backbone)
    det = make_detector(model, args.backbone)
    fdet = FeatureDetector(model, det.mu, det.sd, threshold=det.threshold)
    targets = [("no shift (S_hold)", "S_hold:clean", "none", 0)]
    targets += [(f"season {d}", f"{d}_test:clean", "season", 0) for d in "AWP"]
    targets += [(c, f"S_test:{c}_s{s}", c, s) for c in TEST_CORRUPTIONS for s in SEVERITIES]
    clean = features(model, "S_test:clean")
    rows = []
    for name, spec, kind, sev in targets:
        tgt = features(model, spec)
        for seed in args.seeds:
            rows.append(
                dict(target=name, condition=spec, kind=kind, severity=sev, seed=seed, **replay(fdet, clean, tgt, seed))
            )
        r = pd.DataFrame(rows[-len(args.seeds) :])
        print(
            f"{spec:28s} detected {r.detected.mean():.0%}  first-batch {r.at_first.mean():.0%}  false alarms {r.false_alarms.sum()}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    write_csv(df, RESULTS / f"detector_corruptions_{args.backbone}.csv")
    write_json(
        RESULTS / f"detector_corruptions_{args.backbone}_meta.json",
        dict(threshold=det.threshold, batch=BATCH, seeds=args.seeds),
    )


if __name__ == "__main__":
    main()
