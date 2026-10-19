"""Helpers to build TTA methods from the YAML config and the selected hyper-parameters."""

from __future__ import annotations

import itertools

import torch

from . import tta
from .paths import RESULTS
from .runner import condition_tensor
from .utils import load_config, read_json


def method_grid(cfg: dict, method: str, backbone: str | None = None) -> dict:
    base = method.split("-")[0].replace("geo_", "")
    over = (cfg.get("grid_overrides") or {}).get(backbone, {})
    return dict(over.get(base, cfg["grids"].get(base, {})))


def grid_points(cfg: dict, method: str, backbone: str | None = None):
    grid = method_grid(cfg, method, backbone)
    grid.pop("inherit", None)
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def fixed_hp(cfg: dict, method: str) -> dict:
    base = method.split("-")[0].replace("geo_", "")
    return dict(cfg.get("fixed", {}).get(base, {}))


def selected_hp(backbone: str, method: str) -> dict:
    """HPs chosen by cli/select_hp.py (geo-gated variants and ablations inherit those of their base method)."""
    sel = read_json(RESULTS / f"hp_{backbone}.json")
    base = method.split("-")[0].replace("geo_", "")
    return dict(sel[base]["best"]) if base in sel else {}


def build_method(method: str, source, backbone: str, cfg: dict | None = None, hp: dict | None = None):
    cfg = cfg or load_config("tta")
    hp = {**fixed_hp(cfg, method), **(selected_hp(backbone, method) if hp is None else hp)}
    if method.split("-")[0].replace("geo_", "") == "eata" and hp.get("fisher_alpha", 0) > 0:
        hp["fisher_images"] = condition_tensor("S_val:clean")  # unlabelled source data for the Fisher estimate
    torch.manual_seed(0)
    return tta.build(method, source, **hp), {k: v for k, v in hp.items() if k != "fisher_images"}
