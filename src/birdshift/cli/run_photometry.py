"""E2c: is the seasonal shift photometric (like the corruptions) or semantic (plumage, habitat)?

python -m birdshift.cli.run_photometry
Low-level image statistics of S_test vs A/W/P_test (Mann-Whitney U, Cliff's delta, Holm), plus S_test vs S_hold as a
same-domain reference and S_test vs low_light_s3 as a photometric-shift reference.
-> results/photometry_tests.csv, results/photometry_stats.csv
"""

import pandas as pd

from .. import data
from ..analysis import compare_distributions, photometric_stats
from ..paths import RESULTS
from ..utils import write_csv


def main():
    stats = {}
    for spec in [
        "S_test:clean",
        "S_hold:clean",
        "A_test:clean",
        "W_test:clean",
        "P_test:clean",
        "S_test:low_light_s3",
        "S_test:haze_s3",
    ]:
        split, cond = spec.split(":")
        c, s = (None, None) if cond == "clean" else (cond.rsplit("_s", 1)[0], int(cond[-1]))
        stats[spec] = photometric_stats(data.condition_images(split, c, s))
        print("computed", spec, flush=True)
    write_csv(pd.concat([d.assign(condition=k) for k, d in stats.items()]), RESULTS / "photometry_stats.csv")
    rows = []
    for tgt in [
        "S_hold:clean",
        "A_test:clean",
        "W_test:clean",
        "P_test:clean",
        "S_test:low_light_s3",
        "S_test:haze_s3",
    ]:
        rows.append(compare_distributions(stats["S_test:clean"], stats[tgt]).assign(target=tgt))
    df = pd.concat(rows)
    write_csv(df, RESULTS / "photometry_tests.csv")
    print(df.pivot(index="stat", columns="target", values="cliffs_delta").round(3))
    print(df[df.p_holm < 0.05][["target", "stat", "cliffs_delta", "p_holm", "effect"]].to_string(index=False))


if __name__ == "__main__":
    main()
