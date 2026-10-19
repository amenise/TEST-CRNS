"""Seeding, config loading, run bookkeeping."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from .paths import CONFIGS, ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_config(name: str | Path) -> dict:
    p = Path(name)
    if not p.suffix:
        p = CONFIGS / f"{name}.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


def run_meta(**extra) -> dict:
    import platform

    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_hash(), python=platform.python_version(), **extra)
    try:
        import torch

        meta.update(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    except ImportError:
        pass
    return meta


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def write_csv(df, path: Path, index: bool = False, retries: int = 20, wait: float = 0.5) -> Path | None:
    """df.to_csv(path), robust to Windows file locks (a CSV open in Excel or an editor, or briefly held by the
    antivirus right after a write). Retries for ~10 s, then writes `<name>.pending.csv`; if that is locked too,
    warns and returns None instead of raising, so a long run is never lost because of a summary table (the raw
    outputs in runs/ remain the source of truth)."""
    import time

    path = Path(path)
    for target in (path, path.with_name(path.stem + ".pending.csv")):
        for _ in range(retries):
            try:
                df.to_csv(target, index=index)
                if target != path:
                    print(f"WARNING: {path.name} is locked; wrote {target.name} instead", flush=True)
                return target
            except PermissionError:
                time.sleep(wait)
    print(f"WARNING: could not write {path.name} (locked); rebuild it later from runs/", flush=True)
    return None


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false")
