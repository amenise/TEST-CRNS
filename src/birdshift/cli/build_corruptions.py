"""E2b: build frozen corrupted copies of evaluation splits (uint8 .npy), deterministic per image.

python -m birdshift.cli.build_corruptions                 # test corruptions on S_test + S_hold, validation corruptions on S_val
Seeds: stable_seed(split, corruption, severity, image index), so every method sees identical images (paired tests).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import data
from ..corruptions import SEVERITIES, TEST_CORRUPTIONS, VAL_CORRUPTIONS, corrupt
from ..paths import CORRUPTED, ensure
from ..utils import stable_seed

JOBS = [("S_test", TEST_CORRUPTIONS), ("S_hold", TEST_CORRUPTIONS), ("S_val", VAL_CORRUPTIONS)]


def build_one(args):
    split, name, sev = args
    out = CORRUPTED / split / f"{name}_s{sev}.npy"
    if out.exists():
        return str(out), "exists"
    imgs = data.images(split)
    res = np.empty_like(np.asarray(imgs))
    for i in range(len(imgs)):
        res[i] = corrupt(np.asarray(imgs[i]), name, sev, stable_seed(split, name, sev, i))
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, res)
    tmp.replace(out)
    return str(out), "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    tasks = []
    for split, names in JOBS:
        ensure(CORRUPTED / split)
        data.images(split)  # make sure the clean cache exists before forking workers
        tasks += [(split, n, s) for n in names for s in SEVERITIES]
    with ProcessPoolExecutor(args.workers) as ex:
        for path, status in ex.map(build_one, tasks):
            print(status, path, flush=True)


if __name__ == "__main__":
    main()
