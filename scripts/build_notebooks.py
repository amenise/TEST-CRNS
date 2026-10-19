"""Generate the analysis notebooks (notebooks/0X_*.ipynb). They only read results/ and figures/ (no GPU, no model).

python scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
R, F = ROOT / "results", ROOT / "figures"
pd.set_option("display.precision", 3); pd.set_option("display.width", 200)
def fig(name):
    p = F / f"{name}.png"
    display(Image(str(p))) if p.exists() else print("missing figure:", p.name)
def csv(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else print("missing table:", p.name)"""

NOTEBOOKS = {
    "01_data_exploration": [
        (
            "md",
            "# 01 — Exploration et contrôle des données\nSource : `results/data_checks.json` (`python -m birdshift.cli.prepare_data`), `data/raw/cache/index.parquet`. Détails : `docs/01_data.md`.",
        ),
        ("code", SETUP),
        (
            "code",
            "checks = json.load(open(R / 'data_checks.json'))\npd.DataFrame({k: v for k, v in checks.items() if k != 'global'}).T[['n','n_classes','per_class_min','per_class_max','labels_match_parquet_order','domain','months','missing_gps']]",
        ),
        ("code", "checks['global']"),
        (
            "code",
            "sys.path.insert(0, str(ROOT / 'src'))\nfrom birdshift import data\nidx = data.index()\npd.crosstab(idx['dom'], idx['m'])",
        ),
        (
            "md",
            "Les domaines sont définis par le mois ; `X` = hors Amérique du Nord (jamais utilisé). Le pool (`split` vide) n'a pas d'image mais garde GPS + date : il sert au prior géo-temporel.",
        ),
        ("code", "fig('01_data_map_latitude')"),
        (
            "code",
            "import matplotlib.pyplot as plt\nfig_, ax = plt.subplots(4, 6, figsize=(12, 8))\nfor r, s in enumerate(['S_test','A_test','W_test','P_test']):\n    imgs = data.images(s)\n    for c in range(6):\n        ax[r, c].imshow(imgs[c * 37]); ax[r, c].axis('off')\n    ax[r, 0].set_title(s, loc='left')\nplt.tight_layout()",
        ),
        ("code", "fig('03_corruption_examples')"),
    ],
    "02_baseline": [
        (
            "md",
            "# 02 — Modèles source (Part II)\nBackbone pré-entraîné gelé + régression logistique L2 (λ choisi sur S_val) + temperature scaling. Source : `results/source_<backbone>.json` (`cli/train_source.py`).",
        ),
        ("code", SETUP),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    s = json.load(open(R / f'source_{bb}.json'))\n    display(Markdown(f\"**{bb}** — λ = {s['selected_lambda']}, T = {s['temperature']:.3f}\"))\n    display(pd.DataFrame(s['sweep']).T)\n    display(pd.DataFrame(s['eval']).T)",
        ),
        (
            "md",
            "ResNet-50 : les statistiques BN ont été ré-estimées sur S_train avant la sonde linéaire (voir `docs/03_experiments_log.md`) ; sinon toute statistique de batch d'oiseaux casse la tête linéaire, même sans shift.",
        ),
        ("code", "fig('05_shift_seasonal_bars')"),
    ],
    "03_distribution_shift": [
        (
            "md",
            "# 03 — Shift de distribution (Part III)\nSources : `results/shift_*.csv`, `results/photometry_tests.csv`, `results/failure_layers_*.csv`.",
        ),
        ("code", SETUP),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    d = csv(f'shift_{bb}.csv')\n    display(Markdown(f'**{bb}**'))\n    display(d[d.corruption=='clean'][['split','acc','nll','ece']])\n    display(d[d.corruption!='clean'].pivot(index='corruption', columns='severity', values='acc'))",
        ),
        ("code", "fig('05_shift_severity_curves')"),
        ("md", "## Pourquoi ? (1) photométrie"),
        ("code", "t = csv('photometry_tests.csv'); t.pivot(index='stat', columns='target', values='cliffs_delta')"),
        ("code", "fig('05_why_photometry')"),
        ("md", "## Pourquoi ? (2) dérive des représentations (référence S_hold = même domaine)"),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    display(Markdown(f'**{bb}**')); display(csv(f'shift_drift_{bb}.csv'))",
        ),
        ("code", "fig('05_why_layers')"),
        ("md", "## Pourquoi ? (3) structure taxonomique des erreurs"),
        ("code", "fig('05_why_taxonomy')"),
        ("md", "## Pourquoi ? (4) chute par espèce vs déplacement migratoire"),
        (
            "code",
            "from scipy import stats\nfor bb in ['vit_s16', 'resnet50']:\n    p = csv(f'shift_species_{bb}.csv'); w = p[p.target=='W'].dropna()\n    print(bb, stats.spearmanr(w.displacement_km, w['drop']))",
        ),
        ("code", "fig('05_why_species_displacement')"),
    ],
    "04_tta_benchmark": [
        (
            "md",
            "# 04 — Benchmark de TTA (Part IV)\nSources : `results/tta_<bb>_{summary,mcnemar,friedman}`, `results/hp_<bb>.json`.",
        ),
        ("code", SETUP),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    h = json.load(open(R / f'hp_{bb}.json')) if (R / f'hp_{bb}.json').exists() else {}\n    display(Markdown(f'**HP sélectionnés ({bb})**'))\n    display(pd.DataFrame({k: v.get('best') for k, v in h.items() if k != '_meta'}).T)",
        ),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    s = csv(f'tta_{bb}_summary.csv')\n    if s is not None:\n        display(Markdown(f'**{bb}**'))\n        display(s[['method'] + [c for c in s.columns if c.endswith('_mean') or c.startswith('delta')]].set_index('method'))",
        ),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    t = csv(f'tta_{bb}_mcnemar.csv')\n    if t is not None:\n        display(t.groupby('method').apply(lambda g: pd.Series(dict(sig_better=(g.significant & (g.direction>0)).sum(), sig_worse=(g.significant & (g.direction<0)).sum(), n=len(g)))))\n        print(json.load(open(R / f'tta_{bb}_friedman.json')))",
        ),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    for n in ['severity', 'gain_heatmap', 'cd_diagram']:\n        fig(f'06_tta_{bb}_{n}')",
        ),
    ],
    "05_failure_analysis": [
        ("md", "# 05 — Quand et pourquoi la TTA échoue\nSources : `results/failure_{batch,noniid,layers}_<bb>.csv`."),
        ("code", SETUP),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    b = csv(f'failure_batch_{bb}.csv')\n    if b is not None:\n        display(Markdown(f'**{bb} — taille de batch**'))\n        display(b.groupby(['method', 'condition', 'batch_size']).acc.mean().unstack())",
        ),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    b = csv(f'failure_noniid_{bb}.csv')\n    if b is not None:\n        display(Markdown(f'**{bb} — flux i.i.d. vs triés par classe**'))\n        display(b.groupby(['method', 'order']).agg(acc=('acc','mean'), n_pred_classes=('n_pred_classes','mean')).unstack())",
        ),
        (
            "code",
            "for n in ['09_failure_batch', '09_failure_noniid']:\n    for bb in ['vit_s16', 'resnet50']:\n        fig(f'{n}_{bb}')",
        ),
    ],
    "06_continual_tta": [
        (
            "md",
            "# 06 — TTA continue et oubli (Part V)\nSources : `results/ctta_<bb>_summary.csv`, `results/ctta_<bb>_stages.csv`, `runs/ctta/`.",
        ),
        ("code", SETUP),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    s = csv(f'ctta_{bb}_summary.csv')\n    if s is not None:\n        display(Markdown(f'**{bb}**'))\n        display(s[['sequence','method','online_mean_mean','online_mean_std','final_mean_mean','bwt_mean','forgetting_source_mean','online_round1_mean','online_round2_mean']])",
        ),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    for seq in ['brief', 'year', 'brief_long']:\n        for n in ['online', 'source', 'matrix']:\n            fig(f'07_ctta_{bb}_{seq}_{n}')\n    fig(f'07_ctta_{bb}_detector')",
        ),
    ],
    "07_geo_temporal_prior": [
        (
            "md",
            "# 07 — Contexte écologique : prior géo-temporel\nSources : `results/geo_prior.csv`, `results/geo_fusion_<bb>.csv`, `results/geo_error_detection_<bb>.csv`, `results/geo_beta_*.json`.",
        ),
        ("code", SETUP),
        ("code", "csv('geo_prior.csv')"),
        (
            "code",
            "for bb in ['vit_s16', 'resnet50']:\n    f = csv(f'geo_fusion_{bb}.csv')\n    display(Markdown(f'**{bb}**'))\n    display(f[f.condition=='clean'][['variant','beta','split','acc_visual','acc_fused','nll_visual','nll_fused','ece_visual','ece_fused']])\n    e = csv(f'geo_error_detection_{bb}.csv'); e['kind'] = np.where(e.condition=='clean', 'clean', 'corrupted')\n    display(e.groupby(['variant','kind'])[['error_rate','auroc_confidence','auroc_geo','auroc_combined']].mean())",
        ),
        ("code", "fig('08_geo_prior')"),
    ],
}


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, out / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main()
