"""TTA invariants on small randomly initialised backbones (no download): reset restores the exact source state,
predict() never changes state, adaptation actually changes parameters, and the source model is never modified."""

import pytest
import torch

from birdshift import models, tta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

models.BACKBONES.setdefault("tiny_vit_test", dict(timm="vit_tiny_patch16_224", norm="LayerNorm"))
models.BACKBONES.setdefault("tiny_resnet_test", dict(timm="resnet18", norm="BatchNorm"))

METHOD_HP = {
    "source": {},
    "bn": {"bn_prior": 16},
    "tent": {"lr": 1e-2},
    "pl": {"lr": 1e-2, "threshold": 0.0},
    "eata": {"lr": 1e-2, "d_margin": 1.0, "e0_factor": 1.0},
    "sar": {"lr": 1e-2, "e0_factor": 1.0, "reset_ema": -1.0},
    "t3a": {"filter_k": 5},
    "lame": {"knn": 3},
    "cotta": {"lr": 1e-3, "ap": 1.1, "n_aug": 2},
    "rdumb": {"lr": 1e-2, "d_margin": 1.0, "e0_factor": 1.0, "reset_every": 2},
    "anchored": {"lr": 1e-2, "e0_factor": 1.0},
    "anchored-noteacher": {"lr": 1e-2, "e0_factor": 1.0},
}
STATEFUL = {"tent", "pl", "eata", "sar", "cotta", "rdumb", "anchored", "anchored-noteacher"}


def make(backbone):
    torch.manual_seed(0)
    m = models.Classifier(backbone, num_classes=100, pretrained=False).cuda().eval()
    m.temperature.fill_(0.05)  # confident predictions so that reliability filters select samples
    return m


@pytest.fixture(scope="module", params=["tiny_vit_test", "tiny_resnet_test"])
def source(request):
    return make(request.param)


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (8, 224, 224, 3), generator=g, dtype=torch.uint8).cuda()


def state(method):
    return {k: v.detach().clone() for k, v in method.model.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("name", list(METHOD_HP))
def test_reset_predict_and_source_untouched(source, name):
    src_before = {k: v.clone() for k, v in source.state_dict().items()}
    m = tta.build(name, source, **METHOD_HP[name])
    s0 = state(m)
    x1, x2 = batch(1), batch(2)
    p0 = m.predict(x1)
    assert same(s0, state(m)), "predict() modified the model"
    for _ in range(3):
        out = m.forward(x2)
    assert out.shape == (8, 100) and torch.isfinite(out).all()
    if name in STATEFUL and name != "rdumb":
        assert not same(s0, state(m)), f"{name} did not adapt any parameter"
    m.reset()
    assert same(s0, state(m)), "reset() did not restore the source state"
    assert torch.allclose(m.predict(x1), p0, atol=1e-4), "predictions after reset differ"
    assert same(src_before, source.state_dict()), "the source model was modified"


def test_rdumb_resets_periodically(source):
    m = tta.build("rdumb", source, **METHOD_HP["rdumb"])
    s0 = state(m)
    m.forward(batch(1))
    m.forward(batch(2))  # steps = 2
    assert not same(s0, state(m))
    m.forward(batch(3))  # reset happens before the 3rd step, then one update
    m.reset()
    assert same(s0, state(m))


def test_bn_prior_interpolates():
    src = make("tiny_resnet_test")
    x = batch(5)
    p_src = tta.build("source", src).predict(x)
    p_inf = tta.build("bn", src, bn_prior=float("inf")).predict(x)
    p_0 = tta.build("bn", src, bn_prior=0.0).predict(x)
    assert torch.allclose(p_src, p_inf, atol=1e-4)
    assert not torch.allclose(p_src, p_0, atol=1e-3)
