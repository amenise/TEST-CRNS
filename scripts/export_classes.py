"""Write the 100 classes (species) with their taxonomy and image counts per split to results/classes.csv.

Usage: python scripts/export_classes.py
"""

from birdshift import data
from birdshift.paths import RESULTS
from birdshift.utils import write_csv


def main():
    idx = data.index()
    taxo = idx.drop_duplicates("y").sort_values("y")[["y", "name", "sci", "genus", "family", "order"]]
    used = idx[idx["split"] != ""]
    counts = used.pivot_table(index="y", columns="split", values="file", aggfunc="count", fill_value=0)
    counts = counts[[s for s in data.SPLITS if s in counts.columns]]
    pool = idx[(idx["split"] == "") & idx["na"]].groupby("y").size().rename("metadata_pool")
    out = taxo.merge(counts, left_on="y", right_index=True).merge(pool, left_on="y", right_index=True, how="left")
    out["metadata_pool"] = out["metadata_pool"].fillna(0).astype(int)
    write_csv(out, RESULTS / "classes.csv")
    print(
        f"results/classes.csv: {len(out)} species, {out['order'].nunique()} orders, {out['family'].nunique()} families"
    )


if __name__ == "__main__":
    main()
