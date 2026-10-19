"""Generate results/tables.md: every key table as markdown, straight from results/*.csv|json.
The documentation copies numbers from this file only (no hand transcription from logs).

python scripts/make_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
BBS = ["dinov2", "vit_s16", "resnet50"]  # sections are skipped for backbones whose results are missing
out = []


def h(t):
    out.append(f"\n## {t}\n")


def tab(df, floatfmt=".3f", index=True):
    out.append(df.to_markdown(floatfmt=floatfmt, index=index))
    out.append("")


def exists(name):
    return (R / name).exists()


h("Modèles source (results/source_<bb>.json)")
rows = []
for bb in BBS:
    if exists(f"source_{bb}.json"):
        s = json.load(open(R / f"source_{bb}.json"))
        rows.append(
            dict(
                backbone=bb,
                lam=s["selected_lambda"],
                T=s["temperature"],
                **{f"acc_{k}": v["acc"] for k, v in s["eval"].items() if k.endswith("test") or k == "S_val"},
                **{f"ece_{k}": v["ece"] for k, v in s["eval"].items() if k == "S_test"},
            )
        )
tab(pd.DataFrame(rows).set_index("backbone"))

h("Shift saisonnier et corruptions, modèle Source (results/shift_<bb>.csv)")
for bb in BBS:
    if not exists(f"shift_{bb}.csv"):
        continue
    d = pd.read_csv(R / f"shift_{bb}.csv")
    out.append(f"**{bb}** — domaines propres")
    tab(d[d.corruption == "clean"][["split", "acc", "nll", "ece"]], index=False)
    out.append(f"**{bb}** — corruptions (précision)")
    p = d[d.corruption != "clean"].pivot(index="corruption", columns="severity", values="acc")
    p.loc["mean"] = p.mean()
    tab(p)

h("Dérive des représentations (results/shift_drift_<bb>.csv)")
for bb in BBS:
    if exists(f"shift_drift_{bb}.csv"):
        d = pd.read_csv(R / f"shift_drift_{bb}.csv")
        d = d[(d.condition == "clean") | d.condition.str.endswith(("s3", "s5"))]
        out.append(f"**{bb}**")
        tab(d, index=False)

h("Photométrie, δ de Cliff vs S_test (results/photometry_tests.csv)")
if exists("photometry_tests.csv"):
    t = pd.read_csv(R / "photometry_tests.csv")
    tab(t.pivot(index="stat", columns="target", values="cliffs_delta"))
    tab(t[t.p_holm < 0.05][["target", "stat", "cliffs_delta", "p_holm", "effect"]], index=False)

h("Erreurs taxonomiques (results/shift_taxonomy_<bb>.csv)")
for bb in BBS:
    if exists(f"shift_taxonomy_{bb}.csv"):
        t = pd.read_csv(R / f"shift_taxonomy_{bb}.csv")
        t["group"] = np.where(t.condition == "clean", t.split, "corr_s" + t.condition.str[-1])
        g = t.groupby("group")[["err_same_genus", "err_same_family", "err_same_order"]].mean()
        out.append(
            f"**{bb}** (hasard : genre {t.chance_same_genus.iloc[0]:.3f}, famille {t.chance_same_family.iloc[0]:.3f}, ordre {t.chance_same_order.iloc[0]:.3f})"
        )
        tab(g)

h("Désaccord BN par couche, ResNet-50 (results/failure_layers_resnet50.csv)")
if exists("failure_layers_resnet50.csv"):
    l = pd.read_csv(R / "failure_layers_resnet50.csv")
    tab(l[l.kind == "bn_mismatch"].groupby("condition")[["mean_shift", "var_ratio_log"]].mean())
    for bb in BBS:
        if exists(f"failure_layers_{bb}.csv"):
            l = pd.read_csv(R / f"failure_layers_{bb}.csv")
            c = l[l.kind == "block_cka"]
            out.append(f"**{bb}** — CKA par bloc (min / dernier bloc)")
            tab(c.groupby("condition").cka.agg(["min", "last"]))

h("Prior géo-temporel (results/geo_*.csv)")
if exists("geo_prior.csv"):
    tab(pd.read_csv(R / "geo_prior.csv"), index=False)
for bb in BBS:
    if exists(f"geo_fusion_{bb}.csv"):
        f = pd.read_csv(R / f"geo_fusion_{bb}.csv")
        f["sev"] = f.condition.str[-1].where(f.condition != "clean", "0").astype(int)
        out.append(f"**{bb}** — fusion, domaines propres")
        tab(
            f[f.condition == "clean"][
                [
                    "variant",
                    "beta",
                    "split",
                    "acc_visual",
                    "acc_fused",
                    "nll_visual",
                    "nll_fused",
                    "ece_visual",
                    "ece_fused",
                ]
            ],
            index=False,
        )
        out.append(f"**{bb}** — fusion par sévérité (moyenne sur les corruptions ; 0 = 4 domaines propres)")
        tab(
            f.groupby(["variant", "sev"])[["acc_visual", "acc_fused"]]
            .mean()
            .assign(gain=lambda x: x.acc_fused - x.acc_visual)
        )
        e = pd.read_csv(R / f"geo_error_detection_{bb}.csv")
        e["kind"] = np.where(e.condition == "clean", "clean", "corrupted")
        out.append(f"**{bb}** — détection d'erreurs sans label (AUROC)")
        tab(e.groupby(["variant", "kind"])[["error_rate", "auroc_confidence", "auroc_geo", "auroc_combined"]].mean())

h("Hyper-paramètres sélectionnés (results/hp_<bb>.json)")
for bb in BBS:
    if exists(f"hp_{bb}.json"):
        s = json.load(open(R / f"hp_{bb}.json"))
        tab(
            pd.DataFrame(
                [
                    dict(
                        method=k,
                        best=json.dumps(v.get("best")),
                        val_mean_acc=v.get("mean_acc"),
                        edge=",".join(v.get("at_grid_edge", []) or []),
                        inherited=v.get("inherited_from", ""),
                    )
                    for k, v in s.items()
                    if k != "_meta"
                ]
            ).set_index("method")
        )

h("Benchmark TTA (results/tta_<bb>_summary.csv) — moyenne ± écart-type sur les graines")
for bb in BBS:
    if exists(f"tta_{bb}_summary.csv"):
        s = pd.read_csv(R / f"tta_{bb}_summary.csv").set_index("method")
        cols = ["S_clean", "seasonal_AWP", "W", "corr_s1", "corr_s3", "corr_s5", "corr_all", "all"]
        disp = pd.DataFrame(
            {
                c: s[f"{c}_mean"].map("{:.3f}".format) + " ± " + s[f"{c}_std"].fillna(0).map("{:.3f}".format)
                for c in cols
                if f"{c}_mean" in s
            }
        )
        disp["Δall"] = s["delta_all"].map("{:+.3f}".format)
        disp["ECE"] = s["ece_all_mean"].map("{:.3f}".format)
        out.append(f"**{bb}**")
        tab(disp)
    if exists(f"tta_{bb}_mcnemar.csv"):
        t = pd.read_csv(R / f"tta_{bb}_mcnemar.csv")
        g = t.groupby("method").apply(
            lambda g: pd.Series(
                dict(
                    better=int((g.significant & (g.direction > 0)).sum()),
                    worse=int((g.significant & (g.direction < 0)).sum()),
                    n=len(g),
                )
            ),
            include_groups=False,
        )
        fr = json.load(open(R / f"tta_{bb}_friedman.json"))
        g["mean_rank"] = pd.Series(fr["mean_ranks"])
        out.append(
            f"**{bb}** — McNemar vs Source (seed 0, Holm), rang moyen (Friedman p = {fr['friedman_p']:.2g}, CD Nemenyi = {fr['cd']:.2f}, {fr['n_conditions']} conditions)"
        )
        tab(g.sort_values("mean_rank"))

h("TTA continue (results/ctta_<bb>_summary.csv)")
for bb in BBS:
    if exists(f"ctta_{bb}_summary.csv"):
        s = pd.read_csv(R / f"ctta_{bb}_summary.csv")
        for seq in s.sequence.unique():
            q = s[s.sequence == seq].set_index("method")
            disp = pd.DataFrame(
                {
                    "online": q.online_mean_mean.map("{:.3f}".format)
                    + " ± "
                    + q.online_mean_std.fillna(0).map("{:.3f}".format),
                    "round1": q.online_round1_mean.map("{:.3f}".format),
                    "round2": q.online_round2_mean.map("{:.3f}".format),
                    "final (hold)": q.final_mean_mean.map("{:.3f}".format),
                    "BWT": q.bwt_mean.map("{:+.3f}".format),
                    "gain_after": q.mean_gain_after_mean.map("{:+.3f}".format),
                    "forget_S": q.forgetting_source_mean.map("{:+.3f}".format),
                    "detections": q.n_detections_mean.map("{:.1f}".format),
                }
            )
            out.append(f"**{bb} — {seq}**")
            tab(disp)

h("Analyse des échecs (results/failure_*_<bb>.csv)")
for bb in BBS:
    if exists(f"failure_batch_{bb}.csv"):
        b = pd.read_csv(R / f"failure_batch_{bb}.csv")
        out.append(
            f"**{bb}** — précision vs taille de batch (moyenne sur {b.condition.nunique()} conditions, {b.seed.nunique()} graine(s))"
        )
        tab(b.groupby(["method", "batch_size"]).acc.mean().unstack())
    if exists(f"failure_noniid_{bb}.csv"):
        n = pd.read_csv(R / f"failure_noniid_{bb}.csv")
        out.append(
            f"**{bb}** — flux i.i.d. vs triés par classe ({n.condition.nunique()} conditions, {n.seed.nunique()} graine(s))"
        )
        tab(n.groupby(["method", "order"]).agg(acc=("acc", "mean"), n_pred=("n_pred_classes", "mean")).unstack())

h("TTA + prior géo-temporel, fusion a posteriori (results/tta_geo_fusion_<bb>.csv)")
for bb in BBS:
    if exists(f"tta_geo_fusion_{bb}.csv"):
        fu = pd.read_csv(R / f"tta_geo_fusion_{bb}.csv")
        fu = fu[(fu.batch_size == 64) & (fu.order == "iid")].copy()
        fu["groupe"] = np.select(
            [
                (fu.split == "S_test") & (fu.condition == "clean"),
                fu.condition == "clean",
                fu.condition.str.endswith("s3"),
            ],
            ["S", "A/W/P", "corr. sév. 3"],
            "corr. sév. 5",
        )
        for v in ["blocked", "full"]:
            q = fu[fu.variant == v]
            out.append(
                f"**{bb}** — prior {v} (β = {q.beta.iloc[0]}), précision visuelle → fusionnée, moyenne sur graines ; "
                f"{(q.gain > 0).mean():.0%} des runs améliorés, {(q.mcnemar_p < 0.05).mean():.0%} significatifs (McNemar)"
            )
            m = q.groupby(["method", "groupe"])[["acc_visual", "acc_fused"]].mean()
            disp = (m.acc_visual.map("{:.3f}".format) + " → " + m.acc_fused.map("{:.3f}".format)).unstack("groupe")
            disp["toutes (fusionné)"] = q.groupby("method").acc_fused.mean().map("{:.3f}".format)
            tab(disp.loc[q.groupby("method").acc_fused.mean().sort_values(ascending=False).index])

h("Détecteur de shift sans label, toutes conditions (results/detector_corruptions_<bb>.csv)")
for bb in BBS:
    if exists(f"detector_corruptions_{bb}.csv"):
        dc = pd.read_csv(R / f"detector_corruptions_{bb}.csv")
        out.append(
            f"**{bb}** — part des flux détectés (16 batchs propres puis 16 batchs cibles, {dc.seed.nunique()} ordres) ; "
            f"fausses alarmes sur la partie propre : {int(dc.false_alarms.sum())} sur {len(dc)} flux"
        )
        cor = dc[~dc.kind.isin(["none", "season"])]
        tab(cor.pivot_table(index="kind", columns="severity", values="detected", aggfunc="mean"), floatfmt=".2f")
        tab(dc[dc.kind.isin(["none", "season"])].groupby("target")[["detected", "at_first"]].mean(), floatfmt=".2f")

(R / "tables.md").write_text(
    "# Tables générées (scripts/make_tables.py) — ne pas éditer\n" + "\n".join(out), encoding="utf-8"
)
print("wrote", R / "tables.md")
