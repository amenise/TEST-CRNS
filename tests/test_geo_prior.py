import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax

from birdshift import geo_prior
from birdshift.paths import DATA_RAW

needs_data = pytest.mark.skipif(not (DATA_RAW / "cache" / "index.parquet").exists(), reason="data/raw not unzipped")


def test_encode_is_periodic_in_longitude_and_day_of_year():
    df = pd.DataFrame(
        {
            "lat": [40.0, 40.0, 40.0, 40.0],
            "lon": [-180.0, 180.0, 10.0, 10.0],
            "date": ["2020-06-01", "2020-06-01", "2021-01-01", "2020-12-31"],
        }
    )
    f = geo_prior.encode(df)
    assert f.shape == (4, 20) and f.dtype == np.float32
    np.testing.assert_allclose(f[0], f[1], atol=1e-6)  # -180 deg and +180 deg are the same meridian
    assert np.abs(f[2] - f[3]).max() < 0.05  # 31 Dec is next to 1 Jan


def test_fuse_is_bayes_rule():
    visual = np.array([[2.0, 1.0, 0.0]])
    llr = np.array([[-3.0, 1.0, 0.0]])  # context says class 1 is plausible here, not 0
    np.testing.assert_allclose(geo_prior.fuse(visual, llr, 0.0), log_softmax(visual, 1))
    fused = geo_prior.fuse(visual, llr, 1.0)
    np.testing.assert_allclose(fused, log_softmax(visual, 1) + llr)
    assert fused.argmax() == 1  # the prior overturns the visual top-1
    # a context term that is the same for every class cannot change the decision
    assert geo_prior.fuse(visual, np.full_like(llr, 5.0), 2.0).argmax() == visual.argmax()


@needs_data
def test_training_rows_never_contain_evaluation_rows():
    for variant in ("full", "blocked"):
        rows = geo_prior.training_rows(variant)
        assert not rows["split"].isin(geo_prior.EVAL_SPLITS).any()
        assert rows["lat"].notna().all()


@needs_data
def test_blocked_prior_shares_no_cell_with_evaluation_rows():
    """The spatial-leakage control behind the 'blocked' numbers of the report: no training row may fall in a
    0.1 x 0.1 deg cell that contains an evaluation image."""
    from birdshift import data

    idx = data.index()
    ev = idx[idx["split"].isin(geo_prior.EVAL_SPLITS)]
    ev_cells = set(zip((ev["lat"] * 10).round().astype(int), (ev["lon"] * 10).round().astype(int)))
    full, blocked = geo_prior.training_rows("full"), geo_prior.training_rows("blocked")
    tr_cells = set(zip((blocked["lat"] * 10).round().astype(int), (blocked["lon"] * 10).round().astype(int)))
    assert not tr_cells & ev_cells
    assert set(blocked.index) <= set(full.index)
    assert (len(full), len(blocked)) == (22906, 9325)  # row counts quoted in the reports
