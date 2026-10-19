"""E0: cache images as uint8 arrays and run data integrity checks -> results/data_checks.json.

python -m birdshift.cli.prepare_data
"""

import numpy as np

from .. import data
from ..paths import RESULTS
from ..utils import write_json


def main():
    idx = data.index()
    cl = data.classes()
    checks = {}
    for split in data.SPLITS:
        imgs = data.cache_split(split)
        y = data.labels(split)
        meta = data.split_meta(split)
        counts = np.bincount(y, minlength=data.NUM_CLASSES)
        checks[split] = dict(
            n=len(y),
            images_shape=list(imgs.shape),
            dtype=str(imgs.dtype),
            labels_match_parquet_order=bool(np.array_equal(meta["y"].values, y)),
            label_min=int(y.min()),
            label_max=int(y.max()),
            n_classes=int((counts > 0).sum()),
            per_class_min=int(counts.min()),
            per_class_max=int(counts.max()),
            domain=sorted(meta["dom"].unique().tolist()),
            months=sorted(meta["m"].unique().tolist()),
            all_north_america=bool(meta["na"].all()),
            missing_gps=int(meta["lat"].isna().sum()),
            pixel_mean=float(imgs[: min(500, len(imgs))].mean()),
        )
        print(split, checks[split])
    used = idx[idx["split"] != ""]
    checks["global"] = dict(
        n_rows=len(idx),
        n_pool=int((idx["split"] == "").sum()),
        files_unique=bool(idx["file"].is_unique),
        no_file_shared_between_splits=bool(used.groupby("file")["split"].nunique().max() == 1),
        y_to_name_matches_classes_json=bool(
            all(used.groupby("y")["name"].first().reindex(range(100)).values == np.array(cl["classes"]))
        ),
    )
    print(checks["global"])
    write_json(RESULTS / "data_checks.json", checks)


if __name__ == "__main__":
    main()
