import numpy as np
import pytest
import torch

from birdshift.ctta import ShiftDetector, summarize


def test_summarize_on_hand_computed_matrix():
    # domains S, W ; stages S, W, S, W ; row 0 = source model before any adaptation
    R = np.array(
        [
            [0.80, 0.70],
            [0.82, 0.70],  # after stage 1 (S)
            [0.80, 0.75],  # after stage 2 (W)
            [0.78, 0.74],  # after stage 3 (S)
            [0.79, 0.76],  # after stage 4 (W)
        ]
    )
    res = dict(
        R=R, domains=["S", "W"], online=np.array([0.5, 0.6, 0.7, 0.8]), detections=[(1, 0)], seconds=1.0, n_recoveries=0
    )
    s = summarize(res, ["S", "W", "S", "W"])
    assert s["mean_gain_after"] == pytest.approx(np.mean([0.82 - 0.80, 0.75 - 0.70]))
    assert s["bwt"] == pytest.approx(np.mean([0.79 - 0.82, 0.76 - 0.75]))  # first visit of S is stage 1, of W stage 2
    assert s["forgetting_source"] == pytest.approx(0.80 - 0.78)  # worst drop on S over stages 1..4
    assert s["final_source"] == pytest.approx(0.79)
    assert s["final_mean"] == pytest.approx((0.79 + 0.76) / 2)
    assert s["source_model_mean"] == pytest.approx(0.75)
    assert (s["online_mean"], s["online_round1"], s["online_round2"]) == pytest.approx((0.65, 0.55, 0.75))
    assert s["n_detections"] == 1


def test_no_forgetting_gives_zero_bwt():
    R = np.full((5, 2), 0.7)
    res = dict(R=R, domains=["S", "W"], online=np.full(4, 0.7), detections=[], seconds=0.0, n_recoveries=0)
    s = summarize(res, ["S", "W", "S", "W"])
    assert s["bwt"] == 0 and s["forgetting_source"] == 0 and s["mean_gain_after"] == 0


class _FakeDetector(ShiftDetector):
    """ShiftDetector whose batch statistic is given directly (no backbone needed)."""

    def batch_mean(self, x):
        return x


def test_detector_fires_on_the_shift_batch_only():
    det = _FakeDetector(None, None, None, threshold=1.0)
    stream = [torch.zeros(4)] * 5 + [torch.full((4,), 3.0)] * 5  # mean jumps by 3 at batch 5
    fired = [det.update(m)[0] for m in stream]
    assert fired == [False] * 5 + [True] + [False] * 4  # one alarm, at the jump, then a new segment


def test_detector_needs_two_batches_before_firing():
    det = _FakeDetector(None, None, None, threshold=1.0)
    assert det.update(torch.zeros(4))[0] is False
    assert det.update(torch.full((4,), 3.0))[0] is False  # segment too short to declare a shift
