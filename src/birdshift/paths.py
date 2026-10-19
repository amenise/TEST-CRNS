"""Project paths. Everything is relative to the repository root so runs are relocatable."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES = DATA_PROCESSED / "images"  # <split>.npy, uint8 (N, 224, 224, 3)
CORRUPTED = DATA_PROCESSED / "corrupted"  # <split>/<corruption>_s<k>.npy
FEATURES = DATA_PROCESSED / "features"  # <backbone>/<condition>.npz
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def ensure(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
