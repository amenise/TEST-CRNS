"""Data access: metadata index, image arrays per split, corrupted conditions, and test streams.

Layout of data/raw:
  cache/index.parquet       one row per photo; rows of a split are in the same order as its images/labels
  cache/<split>_y.npy       int64 labels 0..99
  jpg/<split>/{i:05d}.jpg   224x224 RGB, i-th row of that split
Images are cached once as uint8 arrays in data/processed/images/<split>.npy (memory-mapped afterwards).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from .paths import CORRUPTED, DATA_RAW, IMAGES, ensure

SPLITS = ["S_train", "S_val", "S_hold", "S_test", "A_hold", "A_test", "W_hold", "W_test", "P_hold", "P_test"]
DOMAINS = {
    "S": "breeding (May-Jul, source)",
    "A": "autumn migration (Aug-Oct)",
    "W": "winter (Nov-Feb)",
    "P": "spring migration (Mar-Apr)",
}
NUM_CLASSES = 100


@lru_cache(maxsize=1)
def index() -> pd.DataFrame:
    df = pd.read_parquet(DATA_RAW / "cache" / "index.parquet")
    df["split"] = df["split"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def classes() -> dict:
    with open(DATA_RAW / "cache" / "classes.json", encoding="utf-8") as f:
        return json.load(f)


def split_meta(split: str) -> pd.DataFrame:
    """Metadata rows of a split, in image order (row i <-> jpg/<split>/{i:05d}.jpg)."""
    return index()[index()["split"] == split].reset_index(drop=True)


def labels(split: str) -> np.ndarray:
    return np.load(DATA_RAW / "cache" / f"{split}_y.npy")


def jpg_paths(split: str) -> list:
    return sorted((DATA_RAW / "jpg" / split).glob("*.jpg"))


def cache_split(split: str) -> np.ndarray:
    """Decode all JPEGs of a split once into a uint8 (N,224,224,3) array."""
    ensure(IMAGES)
    out = IMAGES / f"{split}.npy"
    if not out.exists():
        paths = jpg_paths(split)
        arr = np.lib.format.open_memmap(
            out.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(len(paths), 224, 224, 3)
        )
        for i, p in enumerate(paths):
            assert p.stem == f"{i:05d}", f"unexpected file order: {p}"
            with Image.open(p) as im:
                arr[i] = np.asarray(im.convert("RGB"))
        arr.flush()
        del arr
        out.with_suffix(".tmp.npy").replace(out)
    return np.load(out, mmap_mode="r")


def images(split: str) -> np.ndarray:
    return cache_split(split)


def condition_name(corruption: str | None, severity: int | None) -> str:
    return "clean" if corruption in (None, "clean") else f"{corruption}_s{severity}"


def condition_images(split: str, corruption: str | None = None, severity: int | None = None) -> np.ndarray:
    """Images of a split under a frozen corruption (built by cli/build_corruptions.py)."""
    if corruption in (None, "clean"):
        return images(split)
    p = CORRUPTED / split / f"{condition_name(corruption, severity)}.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python -m birdshift.cli.build_corruptions`")
    return np.load(p, mmap_mode="r")


def stream_order(y: np.ndarray, order: str = "iid", seed: int = 0) -> np.ndarray:
    """Permutation defining the order in which a test set is streamed.

    iid          uniformly shuffled
    class_sorted grouped by class (temporally correlated / non-i.i.d. stream, cf. LAME, NOTE),
                 class order and within-class order shuffled by the seed
    """
    rng = np.random.default_rng(seed)
    if order == "iid":
        return rng.permutation(len(y))
    if order == "class_sorted":
        cls = rng.permutation(np.unique(y))
        return np.concatenate([rng.permutation(np.flatnonzero(y == c)) for c in cls])
    raise ValueError(order)


def batches(idx: np.ndarray, batch_size: int):
    for s in range(0, len(idx), batch_size):
        yield idx[s : s + batch_size]
