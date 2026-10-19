"""Controlled visual shifts, 5 severities each, deterministic per (image, corruption, severity).

Test corruptions (reported):   gaussian_noise, low_light, defocus_blur, motion_blur, jpeg, pixelate, contrast, haze
Validation corruptions (only for hyper-parameter selection, never reported as test):
                               speckle_noise, gaussian_blur, spatter, saturate
The split test/validation follows ImageNet-C (Hendrycks & Dietterich 2019), which provides speckle noise,
gaussian blur, spatter and saturate as held-out validation corruptions.

Severity parameters are adapted from ImageNet-C to 224x224 crops. Two corruptions are physically motivated
rather than taken from ImageNet-C:
  low_light  under-exposure in linear space (gamma 2.2) + Poisson shot noise + Gaussian read noise
             (standard sensor noise model), no auto-exposure correction;
  haze       atmospheric scattering model I = J t + A (1 - t) with airlight A and a transmission t that
             decreases towards the top of the image (distance proxy), no depth map available.
All functions take/return uint8 HxWx3 arrays.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage

TEST_CORRUPTIONS = [
    "gaussian_noise",
    "low_light",
    "defocus_blur",
    "motion_blur",
    "jpeg",
    "pixelate",
    "contrast",
    "haze",
]
VAL_CORRUPTIONS = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
SEVERITIES = [1, 2, 3, 4, 5]

PARAMS = {
    "gaussian_noise": [0.06, 0.10, 0.15, 0.22, 0.32],  # sigma on [0,1]
    "low_light": [(0.45, 0.002), (0.30, 0.003), (0.20, 0.004), (0.12, 0.005), (0.07, 0.006)],  # (exposure, read sigma)
    "defocus_blur": [1.5, 2.0, 3.0, 4.0, 5.5],  # disk radius (px)
    "motion_blur": [5, 9, 13, 17, 23],  # kernel length (px)
    "jpeg": [30, 18, 12, 8, 5],  # quality
    "pixelate": [0.6, 0.45, 0.35, 0.27, 0.2],  # downscale factor
    "contrast": [0.5, 0.38, 0.27, 0.17, 0.1],  # contrast factor
    "haze": [0.75, 0.6, 0.47, 0.36, 0.27],  # mean transmission t
    "speckle_noise": [0.15, 0.20, 0.30, 0.40, 0.55],
    "gaussian_blur": [1.0, 1.7, 2.5, 3.3, 4.5],
    "spatter": [0.10, 0.17, 0.25, 0.33, 0.42],  # area fraction covered
    "saturate": [(0.3, 0.0), (0.1, 0.0), (2.0, 0.0), (5.0, 0.1), (20.0, 0.2)],  # (S scale, S shift)
}
PEAK_PHOTONS = 300.0  # photons at full scale for low_light


def _f(x):
    return x.astype(np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def _conv(x, k):
    return np.stack([ndimage.convolve(x[..., c], k, mode="reflect") for c in range(3)], -1)


def gaussian_noise(img, s, rng):
    x = _f(img)
    return _u8(x + rng.normal(0, PARAMS["gaussian_noise"][s - 1], x.shape))


def low_light(img, s, rng):
    exposure, read = PARAMS["low_light"][s - 1]
    lin = _f(img) ** 2.2 * exposure
    noisy = rng.poisson(lin * PEAK_PHOTONS) / PEAK_PHOTONS + rng.normal(0, read, lin.shape)
    return _u8(np.clip(noisy, 0, 1) ** (1 / 2.2))


def _disk(r):
    a = np.arange(-int(np.ceil(r)), int(np.ceil(r)) + 1)
    xx, yy = np.meshgrid(a, a)
    k = (xx**2 + yy**2 <= r**2).astype(np.float32)
    k = ndimage.gaussian_filter(k, 0.5)  # anti-aliasing, as in ImageNet-C
    return k / k.sum()


def defocus_blur(img, s, rng):
    return _u8(_conv(_f(img), _disk(PARAMS["defocus_blur"][s - 1])))


def _line_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = ndimage.rotate(k, angle, reshape=False, order=1)
    return k / k.sum()


def motion_blur(img, s, rng):
    return _u8(_conv(_f(img), _line_kernel(PARAMS["motion_blur"][s - 1], rng.uniform(-45, 45))))


def jpeg(img, s, rng):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=PARAMS["jpeg"][s - 1])
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB")).copy()


def pixelate(img, s, rng):
    h, w = img.shape[:2]
    f = PARAMS["pixelate"][s - 1]
    im = Image.fromarray(img).resize((int(w * f), int(h * f)), Image.BOX)
    return np.asarray(im.resize((w, h), Image.NEAREST)).copy()


def contrast(img, s, rng):
    x = _f(img)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * PARAMS["contrast"][s - 1] + m)


def haze(img, s, rng):
    x = _f(img)
    h = x.shape[0]
    t0 = PARAMS["haze"][s - 1]
    t = np.clip(t0 * (0.8 + 0.4 * np.linspace(0, 1, h)), 0.05, 1)[:, None, None]  # hazier at the top
    airlight = 0.85 + rng.uniform(-0.03, 0.03)
    return _u8(x * t + airlight * (1 - t))


def speckle_noise(img, s, rng):
    x = _f(img)
    return _u8(x + x * rng.normal(0, PARAMS["speckle_noise"][s - 1], x.shape))


def gaussian_blur(img, s, rng):
    x = _f(img)
    return _u8(ndimage.gaussian_filter(x, sigma=(PARAMS["gaussian_blur"][s - 1],) * 2 + (0,)))


def spatter(img, s, rng):
    """Mud/water spatter: smoothed random field thresholded to cover a given area fraction."""
    x = _f(img)
    field = ndimage.gaussian_filter(rng.normal(size=x.shape[:2]), 4)
    thr = np.quantile(field, 1 - PARAMS["spatter"][s - 1])
    mask = ndimage.gaussian_filter((field > thr).astype(np.float32), 1.2)[..., None]
    mud = np.array([0.40, 0.30, 0.20], np.float32) * rng.uniform(0.8, 1.2)
    return _u8(x * (1 - 0.85 * mask) + mud * 0.85 * mask)


def saturate(img, s, rng):
    scale, shift = PARAMS["saturate"][s - 1]
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32) / 255.0
    hsv[..., 1] = np.clip(hsv[..., 1] * scale + shift, 0, 1)
    return np.asarray(Image.fromarray(_u8(hsv), "HSV").convert("RGB")).copy()


FUNCS = {
    f.__name__: f
    for f in [
        gaussian_noise,
        low_light,
        defocus_blur,
        motion_blur,
        jpeg,
        pixelate,
        contrast,
        haze,
        speckle_noise,
        gaussian_blur,
        spatter,
        saturate,
    ]
}


def corrupt(img: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    assert severity in SEVERITIES, severity
    out = FUNCS[name](img, severity, np.random.default_rng(seed))
    assert out.shape == img.shape and out.dtype == np.uint8
    return out
