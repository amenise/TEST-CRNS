# Adaptive Bird Recognition under Distribution Shifts

Mini research project (technical test): how does a bird recognition model behave when the
data distribution changes, can it adapt **without labels** (test-time adaptation, TTA), and can it adapt **continually**
to successive shifts without catastrophic forgetting?

- Data: 100 North-American species from iNaturalist 2021 with GPS + date. Domains are seasons: **S** breeding (May–Jul,
  source), **A** autumn migration, **W** winter, **P** spring migration. There are also 8 controlled corruptions × 5
  severities.
- Models: frozen **DINOv2 ViT-S/14** (main model, LayerNorm), ViT-S/16 (LayerNorm, IN-21k) and ResNet-50 (BatchNorm,
  IN-1k) + linear probe + temperature scaling. BioCLIP was probed and discarded: it was pre-trained on the iNat21 train
  split that our images come from.
- Nine published TTA methods compared with the source model: BN-adapt, TENT, PL, EATA, SAR, T3A, LAME, CoTTA and RDumb.
  Our own methods: **Anchored TTA** and geo-gated variants (geo-TENT, geo-EATA, geo-Anchored) that use a
  **geo-temporal species prior** to decide which test images to adapt on.

## Setup (Windows, 4 GB GPU tested)

```powershell
winget install astral-sh.uv          # Python manager
uv sync                              # Python 3.11 + torch (CUDA 12.8) + deps from uv.lock
# if HTTPS is intercepted on your network: $env:UV_SYSTEM_CERTS='1'
# torch needs the Microsoft Visual C++ Redistributable (x64)
```

Put the data archive in `data/bird_shift_data.zip` and unzip it to `data/raw/` (`cache/`, `jpg/`, `runs/`, `patterns.txt`).

## Key results (DINOv2 unless stated; tables in `results/tables.md`)

| Question | Finding |
|---|---|
| Q1 shift | Seasonal shift: −1.8 to −4.4 pts (ViT-S/16: −3 to −6), **semantic** (photometry, BN statistics and class centroids unchanged; errors between related species). Corruptions: 77.9 → 45.7 % (severity 1 → 5). The winter drop is within sampling noise (95 % CI [−1.6, 5.2]) |
| Q2 TTA | Entropy methods (TENT, EATA, RDumb, SAR) give **+3 to +4 pts on corruptions** (ResNet-50: up to +15) but **≤ +0.4 pt on seasons** (ViT-S/16: ≤ +0.8). Same conclusion on all three backbones |
| Q3 continual | Stability depends on the backbone: TENT collapses on ViT-S/16 (1 %) but not on DINOv2 (73.6 %); CoTTA the opposite. **EATA is the only published method stable everywhere**, including 20 passes (80,000 images): 74.7 % (DINOv2), 68.8 % (ViT-S/16). **Geo-gating stops TENT's collapse** on ViT-S/16 (67.9 % vs 1.0 %). 3 seeds |
| Q4 context | A geo-temporal prior (metadata only) gives **+5.7 to +10.2 pts**, more under heavier corruption; winter 80.0 → 85.7 %. Fused with every TTA run, it improves all 480 runs; geo-EATA + prior reaches **72.9 %** over 20 conditions (DINOv2 alone: 61.2 %; ViT-S/16: 56.7 → 69.6 %). A prior trained on summer observations only keeps most of the gain |

## Reproduce everything

The GPU experiments (steps 6–9 below) run sequentially via `powershell -File scripts/run_pipeline.ps1` for ViT-S/16 and ResNet-50 (about 2.5 h on an RTX 2050 with the reduced budget in `configs/`), and via `bash scripts/run_pipeline_dinov2.sh` for DINOv2 (about 3 h, after `train_source --backbone dinov2`). Tables and notebooks are then regenerated with `python scripts/make_tables.py`, `python scripts/build_notebooks.py` and `jupyter nbconvert --execute --inplace notebooks/*.ipynb`.

Each step is resumable: existing outputs are skipped. Times are for an RTX 2050.

| # | Command | Produces | Time |
|---|---|---|---|
| 0 | `python -m birdshift.cli.prepare_data` | `data/processed/images/`, `results/data_checks.json` | 2 min |
| 1 | `python -m birdshift.cli.build_corruptions` | `data/processed/corrupted/` (frozen, 9.8 GB) | 15 min |
| 2 | `python -m birdshift.cli.train_source --backbone vit_s16` (then `resnet50`) | `runs/source/<bb>/head.pt`, cached features, `results/source_<bb>.json` | 10 + 15 min |
| 3 | `python -m birdshift.cli.run_shift --backbone <bb>` | `results/shift_*_<bb>.csv` | 1 min |
| 4 | `python -m birdshift.cli.run_photometry` | `results/photometry_*.csv` | 2 min |
| 5 | `python -m birdshift.cli.run_geo` | `runs/geo/`, `results/geo_*.csv` | 5 min |
| 6 | `python -m birdshift.cli.select_hp --backbone <bb>` | `results/hp_<bb>.json`, `results/hp_grid_<bb>.csv` | 1.5–4 h |
| 7 | `python -m birdshift.cli.run_tta --backbone <bb>` then `python -m birdshift.cli.analyze_tta --backbone <bb>` | `runs/tta/`, `results/tta_<bb>_*.csv`, `figures/06_*` | 1–3 h |
| 8 | `python -m birdshift.cli.run_failure --backbone <bb>` | `results/failure_*_<bb>.csv` | 1–2 h |
| 9 | `python -m birdshift.cli.run_ctta --backbone <bb> --sequences brief year brief_long` then `python -m birdshift.cli.analyze_ctta --backbone <bb>` | `runs/ctta/`, `results/ctta_*`, `figures/07_*` | 2–5 h |
| 10 | `python -m birdshift.cli.make_figures` | all remaining `figures/` | 1 min |
| 11 | `python -m birdshift.cli.run_detector --backbone <bb>` | `results/detector_corruptions_<bb>.csv` (shift detector on 8 corruptions × 5 severities and the seasons) | 3 min |
| 12 | `python scripts/export_classes.py` | `results/classes.csv` (100 species, taxonomy, images per split) | 5 s |
| 13 | `python -m birdshift.cli.probe_backbones` | `results/backbone_probe.csv` (ViT-S/16, DINOv2, BioCLIP linear probes) | 10 min |
| 14 | `python -m birdshift.cli.fuse_tta --backbones <bb>` | `results/tta_geo_fusion_<bb>.csv` (prior fused with every TTA run, CPU only) | 1 min |
| 15 | `python scripts/check_cotta_checkpointing.py` | `results/check_cotta_checkpointing_dinov2.json` (CoTTA with vs without gradient checkpointing) | 5 min |
| 16 | `python -m birdshift.cli.run_review_controls` | `results/review_controls.json` (bootstrap CIs, summer-only prior, MMD/C2ST two-sample tests, species correlation; CPU) | 10 min |
| 17 | `python -m birdshift.cli.run_tta --backbone dinov2 --methods source tent eata sar geo_tent geo_eata --conditions S_test:<c>_s1 ...` then `python -m birdshift.cli.analyze_headroom` | `results/headroom_dinov2.csv` (severity-1 headroom control) | 15 min |

Tests: `python -m pytest`. They cover:

- corruption determinism, metrics and statistics (McNemar, Holm, Nemenyi CD);
- TTA invariants: reset, predict without side effects, source model untouched;
- continual-TTA metrics (BWT and source forgetting on a hand-computed matrix) and the shift detector;
- the geo-temporal prior: Bayes fusion, encoding periodicity, and the spatial-leakage control of the `blocked` variant;
- data alignment (image i = metadata row i = label i), disjoint splits and balanced test sets. These tests skip when
  `data/raw/` is absent.

Lint and format: `ruff check src tests scripts` and `ruff format src tests scripts` (configuration in
`pyproject.toml`).

## Layout

```
src/birdshift/        importable package
  data.py corruptions.py models.py runner.py experiment.py metrics.py stats.py analysis.py geo_prior.py ctta.py plots.py
  tta/                one module per method family, common interface (base.TTAMethod)
  cli/                entry points listed above
configs/              tta.yaml (HP grids, benchmark), ctta.yaml (sequences)
notebooks/            analysis notebooks (read results/, import src/)
runs/                 raw per-image outputs (gitignored)
results/              aggregated tables
figures/              PNG + PDF
tests/                pytest
```

## Licence

The code is released under the MIT licence (`LICENSE`). The photographs and observations come from
iNaturalist 2021; each photograph keeps the Creative Commons licence chosen by its author. The data are not included in
this repository.
