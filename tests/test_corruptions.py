import numpy as np
import pytest

from birdshift.corruptions import FUNCS, SEVERITIES, TEST_CORRUPTIONS, VAL_CORRUPTIONS, corrupt


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    base = np.zeros((224, 224, 3), np.float32)
    base[:, :112] = 0.8  # sharp edge
    base += rng.uniform(0, 0.15, base.shape)  # texture
    return (np.clip(base, 0, 1) * 255).astype(np.uint8)


def test_split_is_disjoint():
    assert not set(TEST_CORRUPTIONS) & set(VAL_CORRUPTIONS)
    assert set(TEST_CORRUPTIONS) | set(VAL_CORRUPTIONS) == set(FUNCS)


@pytest.mark.parametrize("name", TEST_CORRUPTIONS + VAL_CORRUPTIONS)
def test_deterministic_shape_dtype(img, name):
    for s in SEVERITIES:
        a = corrupt(img, name, s, seed=123)
        b = corrupt(img, name, s, seed=123)
        assert a.shape == img.shape and a.dtype == np.uint8
        assert np.array_equal(a, b), f"{name} s{s} not deterministic"


@pytest.mark.parametrize("name", TEST_CORRUPTIONS)
def test_severity_increases_distortion(img, name):
    d = [np.abs(corrupt(img, name, s, seed=7).astype(float) - img).mean() for s in (1, 5)]
    assert d[1] > d[0] > 0, f"{name}: {d}"


def test_low_light_darkens(img):
    assert corrupt(img, "low_light", 5, 0).mean() < corrupt(img, "low_light", 1, 0).mean() < img.mean()


def test_noise_depends_on_seed(img):
    assert not np.array_equal(corrupt(img, "gaussian_noise", 3, 1), corrupt(img, "gaussian_noise", 3, 2))
