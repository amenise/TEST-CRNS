import numpy as np
import pytest

from birdshift import data
from birdshift.paths import DATA_RAW

pytestmark = pytest.mark.skipif(not (DATA_RAW / "cache" / "index.parquet").exists(), reason="data/raw not unzipped")

EXPECTED_SIZES = {
    "S_train": 6595,
    "S_val": 500,
    **{f"{d}_hold": 500 for d in "SAWP"},
    **{f"{d}_test": 1000 for d in "SAWP"},
}


@pytest.mark.parametrize("split", data.SPLITS)
def test_images_metadata_and_labels_are_aligned(split):
    """Image i of a split, metadata row i and label i describe the same photo."""
    meta, y, jpgs = data.split_meta(split), data.labels(split), data.jpg_paths(split)
    assert len(meta) == len(y) == len(jpgs) == EXPECTED_SIZES[split]
    assert [p.stem for p in jpgs] == [f"{i:05d}" for i in range(len(jpgs))]
    np.testing.assert_array_equal(meta["y"].to_numpy(), y)
    assert (meta["dom"] == split[0]).all()  # S_*, A_*, W_*, P_* come from their own season


def test_splits_are_disjoint():
    idx = data.index()
    used = idx[idx["split"] != ""]
    assert used["file"].is_unique  # no photo in two splits


@pytest.mark.parametrize("split", [s for s in data.SPLITS if s.endswith("_test")])
def test_test_sets_are_balanced(split):
    assert (np.bincount(data.labels(split), minlength=data.NUM_CLASSES) == 10).all()
