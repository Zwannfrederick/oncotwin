"""OncoTwin v1 — Deterministik makale figürleri (Figures 1–4) üreticisi.

Arşivlenmiş JSON/NPZ çıktılarından doğrudan okur, Tablo 1–4 sayılarıyla assert
parity kontrolü yürütür ve PNG/PDF/SVG çıktıları bayt düzeyinde aynı üretir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_OUT_DIR = Path("docs/manuscript/figures")

PALETTE = {
    "navy": "#1F4E79",
    "teal": "#0E7C7B",
    "amber": "#C97A10",
    "crimson": "#B83227",
    "purple": "#6B4C9A",
    "slate": "#475569",
    "gray": "#94A3B8",
    "grid": "#E2E8F0",
    "band": "#F8FAFC",
}


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 9.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "svg.hashsalt": "oncotwin-v1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _to_float(val: Any) -> float:
    return float("nan") if val is None else float(val)


def _wilson_ci_pct(x: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = x / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return 100.0 * (center - half), 100.0 * (center + half)


def _patient_mean_bootstrap_ci(
    values: np.ndarray, seed: int = 2026, n_boot: int = 2000
) -> tuple[float, float, float]:
    """Deterministik hasta-düzeyi ortalama ve %95 yüzdelik bootstrap GA (yalnız arşiv vektörü)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    assert arr.size > 0, "Boş veya tamamen NaN hasta vektörü"
    mean_val = float(np.mean(arr))
    if float(np.ptp(arr)) == 0.0:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    lo, hi = np.percentile(np.mean(arr[idx], axis=1), [2.5, 97.5])
    return mean_val, float(lo), float(hi)


def _augrc_curve_from_arrays(
    scores: np.ndarray,
    losses: np.ndarray,
    available_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """rank_conformal._trapezoid_augrc_from_arrays ile birebir aynı bağ-duyarlı G(c)=c*R(c)."""
    sc = np.asarray(scores, dtype=np.float64)
    lo = np.asarray(losses, dtype=np.float64)
    avail = np.isfinite(sc) & np.isfinite(lo)
    if available_mask is not None:
        avail = avail & np.asarray(available_mask, dtype=bool)
    s_av, l_av = sc[avail], lo[avail]
    n = int(s_av.size)
    assert n > 0, "AUGRC eğrisi için geçerli hasta yok"

    order = np.argsort(s_av, kind="mergesort")
    s_ord, l_ord = s_av[order], l_av[order]

    covs, grs = [0.0], [0.0]
    area = cum_loss = prev_cov = prev_gr = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(s_ord[end] - s_ord[start]) <= tie_tol:
            end += 1
        cum_loss += float(np.sum(l_ord[start:end]))
        cov, gr = end / float(n), cum_loss / float(n)
        area += 0.5 * (prev_gr + gr) * (cov - prev_cov)
        covs.append(cov)
        grs.append(gr)
        prev_cov, prev_gr, start = cov, gr, end
    return np.asarray(covs, dtype=np.float64), np.asarray(grs, dtype=np.float64), float(area)


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, title: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        ax.set_position([round(x0, 8), round(y0, 8), round(w, 8), round(h, 8)])
    hashes: dict[str, str] = {}
    specs = (
        ("png", {"dpi": 300, "metadata": {"Software": None}}),
        ("pdf", {"metadata": {"CreationDate": None, "Producer": None, "Creator": None, "Title": title}}),
        ("svg", {"metadata": {"Date": None, "Creator": None, "Title": title}}),
    )
    for ext, kwargs in specs:
        target = out_dir / f"{stem}.{ext}"
        fig.savefig(target, format=ext, **kwargs)
        hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    plt.close(fig)
    return hashes


def _load_archives(data_dir: Path) -> dict[str, Any]:
    return {
        "s3": json.loads((data_dir / "sprint03_v2_results.json").read_text(encoding="utf-8")),
        "s4": json.loads((data_dir / "sprint04_v2_results.json").read_text(encoding="utf-8")),
        "s4_19": json.loads((data_dir / "sprint04_v2_dev_opus19_results.json").read_text(encoding="utf-8")),
        "s5": json.loads((data_dir / "sprint05_results.json").read_text(encoding="utf-8")),
        "z4": np.load(data_dir / "sprint04_v2_vectors.npz"),
        "z5": np.load(data_dir / "sprint05_vectors.npz"),
    }


def verify_figure_table_parity(arch: dict[str, Any]) -> None:
    """Figürlerde çizilen tüm kritik değerlerin Tablo 1–4 ve arşivle birebir eşleşmesini doğrular."""
    s3, s5, z5 = arch["s3"], arch["s5"], arch["z5"]

    cv_k5 = s3["cv_5fold_3seeds"]["metrics"]["by_k"]["5"]
    assert round(cv_k5["Random50"]["rmse"], 2) == 45.11
    assert round(cv_k5["StaticDOpt"]["rmse"], 2) == 42.74
    assert round(cv_k5["AdaptiveOED"]["rmse"], 2) == 42.98
    d_cv = s3["cv_5fold_3seeds"]["comparisons"]["panel"]["k5_StaticDOpt_vs_Random50"]["rmse_diff"]
    assert round(d_cv["diff"], 2) == -2.37 and round(d_cv["ci_low"], 2) == -2.84 and round(d_cv["ci_high"], 2) == -1.89
    assert round(s3["wave_ood"]["metrics"]["by_k"]["5"]["StaticDOpt"]["rmse"], 2) == 50.81
    d_w = s3["wave_ood"]["comparisons"]["panel"]["k5_StaticDOpt_vs_Random50"]["rmse_diff"]
    assert round(d_w["diff"], 2) == -1.81 and round(d_w["ci_low"], 2) == -2.94 and round(d_w["ci_high"], 2) == -0.65
    assert s3["wave_ood"]["full_panel_report"]["k5_StaticDOpt"]["n_full_panel_patients"] == 0

    a1_id = s5["track_a_m4"]["regimes"]["id_cv_5fold_3seed"]["experiment_a1_fixed_panel"]["k5"]["comparisons"]
    assert round(a1_id["M4_vs_M3"]["rmse_diff"]["diff"], 3) == 0.011
    assert round(a1_id["M4_vs_M4_perm_null"]["rmse_diff"]["diff"], 3) == 0.017
    assert s5["track_a_m4"]["drug_family_graph"]["n_perm_draws"] == 20
    c2_cmp = s5["track_c_fimm"]["experiment_c2_within_fimm_replication"]["comparisons"]
    assert abs(c2_cmp["M4_Static_k5_vs_M3_Static_k5"]["rmse_diff"]["diff"]) < 0.01

    tb_id = s5["track_b_shortlist"]["regimes"]["id_60_20_20"]["arms"]["StaticDOpt_k5"]["methods"]
    tb_w = s5["track_b_shortlist"]["regimes"]["wave_ood"]["arms"]["StaticDOpt_k5"]["methods"]
    assert round(tb_id["RankConformal"]["test"]["shortlist_mean_size_obs"], 2) == 6.00
    assert round(tb_id["RankConformal"]["test"]["shortlist_contains_true_top1_rate"] * 91) == 85
    assert round(tb_w["RankConformal"]["test"]["shortlist_contains_true_top1_rate"] * 148) == 137
    assert round(tb_w["RelRankConformal"]["test"]["shortlist_contains_true_top1_rate"] * 148) == 141
    assert round(tb_w["RelRankConformal"]["test"]["shortlist_mean_size_obs"], 2) == 9.13
    for reg, exp_ok, exp_n in (("id_60_20_20", 5, 9), ("wave_ood", 19, 25)):
        d1 = z5[f"track_b/{reg}/StaticDOpt_k5/Top1GapConformal/patient_definite_top1"]
        ok = z5[f"track_b/{reg}/StaticDOpt_k5/Top1GapConformal/patient_shortlist_contains_top1"]
        sel = np.isfinite(d1) & (d1 == 1.0)
        assert int(sel.sum()) == exp_n and int(np.nansum(ok[sel])) == exp_ok

    c1 = s5["track_c_fimm"]["experiment_c1_frozen_transfer"]["cohorts"]["index_109_patients"]
    assert round(c1["arms"]["C1_M3_StaticDOpt_k5"]["ndcg_at_3"], 3) == 0.683
    assert round(c1["comparisons"]["C1_M3_k5_vs_M1_k0"]["ndcg3_diff"]["diff"], 3) == 0.032
    assert round(c1["comparisons"]["C1_M3_Static_k5_vs_Random_k5"]["ndcg3_diff"]["diff"], 3) == 0.033
    c2 = s5["track_c_fimm"]["experiment_c2_within_fimm_replication"]
    assert round(c2["arms"]["M3_StaticDOpt_k5"]["rmse"], 2) == 3.83
    assert round(c2["comparisons"]["M3_Static_k5_vs_M1_k0"]["rmse_diff"]["diff"], 2) == -1.00
    assert round(c2["comparisons"]["M3_Static_k5_vs_Random20x_k5"]["rmse_diff"]["diff"], 2) == -0.46


def make_figure1(arch: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Figure 1 — Beat AML bütçe eğrileri (k ∈ {0, 1, 2, 3, 5, 8, 12}, ID CV ve Waves 3+4)."""
    s3 = arch["s3"]
    ks, all_ticks = [1, 2, 3, 5, 8, 12], [0, 1, 2, 3, 5, 8, 12]
    regimes = [
        ("cv_5fold_3seeds", "Beat AML ID 5-Fold CV × 3 (n = 427)", "ID CV (n = 427)", "A", "C"),
        ("wave_ood", "Beat AML Temporal Waves 3+4 — Exploratory (n = 146)", "Waves 3+4 (n = 146)", "B", "D"),
    ]
    policies = [
        ("Random50", "Random50 + M3", PALETTE["amber"], "^", "--", 1.7),
        ("StaticDOpt", "StaticDOpt + M3", PALETTE["navy"], "o", "-", 2.2),
        ("AdaptiveOED", "AdaptiveOED + M3", PALETTE["teal"], "s", "-.", 1.7),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.9), layout="constrained", sharex=True)
    for col, (reg_key, title_top, short_lbl, lbl_top, lbl_bot) in enumerate(regimes):
        node = s3[reg_key]
        k0, by_k = node["metrics"]["k0_refs"], node["metrics"]["by_k"]
        pair_k5 = node["comparisons"]["panel"]["k5_StaticDOpt_vs_Random50"]["rmse_diff"]

        for row, (metric, ylabel, plbl) in enumerate(
            [("rmse", "Target RMSE (AUC)  ↓", lbl_top), ("ndcg_at_3", "Within-Patient NDCG@3  ↑", lbl_bot)]
        ):
            ax = axes[row, col]
            ax.grid(True, linestyle=":", alpha=0.8)
            ax.axvline(5, color="#CBD5E1", linestyle=":", linewidth=1.2, zorder=1)
            m0_v, m1_v = _to_float(k0["M0"][metric]), _to_float(k0["M1"][metric])
            ax.axhline(m0_v, color=PALETTE["gray"], linestyle=":", linewidth=1.1, zorder=2)
            ax.axhline(m1_v, color=PALETTE["slate"], linestyle="--", linewidth=1.1, zorder=2)
            ax.scatter([0], [m0_v], marker="D", s=38, facecolors="white", edgecolors=PALETTE["gray"], linewidths=1.4, zorder=5, label="M0 (k=0) baseline")
            ax.scatter([0], [m1_v], marker="X", s=46, color=PALETTE["slate"], zorder=5, label="M1 (k=0) omics baseline")
            for pol, label, color, marker, ls, lw in policies:
                ys = [_to_float(by_k[str(k)][pol][metric]) for k in ks]
                ax.plot(ks, ys, color=color, linestyle=ls, linewidth=lw, marker=marker, markersize=5.2, label=label, zorder=4)
            ax.set_xticks(all_ticks)
            ax.set_xlim(-0.6, 12.6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Measured probe budget k (archive-available with backfill)")
            ax.set_title(f"{plbl}. {title_top}" if row == 0 else f"{plbl}. {short_lbl} — NDCG@3", loc="left")
            if row == 0:
                box_txt = f"k = 5 StaticDOpt − Random50\nΔRMSE = {pair_k5['diff']:+.2f} [{pair_k5['ci_low']:+.2f}, {pair_k5['ci_high']:+.2f}] AUC"
                ax.text(0.97, 0.94, box_txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
                        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": PALETTE["navy"], "alpha": 0.94})

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Beat AML 2.0 Probe-Budget Curves (k ∈ {0, 1, 2, 3, 5, 8, 12}; Archive-Available with Backfill)", fontsize=11.0, fontweight="bold")
    return _save_figure(fig, out_dir, "fig1_beataml_budget_curves", "OncoTwin v1 Figure 1: Beat AML Budget Curves")


def make_figure2(arch: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Figure 2 — Kapı 3 M4 sabit-panel forest grafiği (Beat AML tam ölçek + yakınlaştırma, FIMM C2 ayrı sDSS)."""
    s5 = arch["s5"]
    ba_regimes = [
        ("id_cv_5fold_3seed", "Beat AML ID CV × 3\n(n = 427, Primary)"),
        ("id_60_20_20", "Beat AML ID 60/20/20\n(n = 91)"),
        ("wave_ood_s3", "Beat AML Waves 3+4 S3\n(n = 146, Exploratory)"),
    ]
    comp_specs = [
        ("M4_vs_M3", PALETTE["navy"], "o", 0.22, True),
        ("M4_vs_M4_perm_null", PALETTE["amber"], "s", 0.0, True),
        ("M4_forced_pos_vs_M4_perm_forced", PALETTE["slate"], "D", -0.22, False),
    ]

    fig = plt.figure(figsize=(12.2, 4.9), layout="constrained")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.38, 1.05])
    ax_full, ax_zoom, ax_fimm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    y_base = np.arange(len(ba_regimes))[::-1].astype(float)

    for ax, is_zoom in ((ax_full, False), (ax_zoom, True)):
        ax.grid(True, axis="x", linestyle=":", alpha=0.8)
        ax.axvline(0.0, color="#334155", linewidth=1.0, zorder=2)
        if not is_zoom:
            ax.axhspan(y_base[0] - 0.38, y_base[0] + 0.38, color="#FEF2F2", alpha=0.45, zorder=0)
            ax.axvline(-0.50, color=PALETTE["crimson"], linestyle="--", linewidth=1.3, zorder=3)
            ax.text(-0.485, 0.08, "Gate 3 RMSE criterion\n(≤ −0.50 AUC, primary ID CV)", color=PALETTE["crimson"],
                    fontsize=7.3, ha="left", va="bottom",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["crimson"], "alpha": 0.9})
        for idx, (reg_key, _) in enumerate(ba_regimes):
            comps = s5["track_a_m4"]["regimes"][reg_key]["experiment_a1_fixed_panel"]["k5"]["comparisons"]
            for c_key, color, marker, dy, filled in comp_specs:
                d = comps[c_key]["rmse_diff"]
                val, lo, hi = _to_float(d["diff"]), _to_float(d["ci_low"]), _to_float(d["ci_high"])
                y = y_base[idx] + dy
                ax.errorbar(val, y, xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]], fmt=marker,
                            color=color, markerfacecolor=color if filled else "white", markersize=5.2, elinewidth=1.4, capsize=2.6, zorder=4)
                if is_zoom:
                    fmt_s = f"{val:+.3f} [{lo:+.3f}, {hi:+.3f}]" if reg_key == "id_cv_5fold_3seed" else f"{val:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                    ax.text(0.044, y, fmt_s, va="center", ha="left", fontsize=7.1, color=color)

    ax_full.set_yticks(y_base)
    ax_full.set_yticklabels([r[1] for r in ba_regimes])
    ax_full.set_ylim(-0.55, len(ba_regimes) - 0.45)
    ax_full.set_xlim(-0.58, 0.10)
    ax_full.set_xlabel("Paired ΔRMSE (AUC; negative favors M4)")
    ax_full.set_title("A. Beat AML — Gate 3 Scale (AUC)", loc="left")

    ax_zoom.set_yticks(y_base)
    ax_zoom.tick_params(axis="y", labelleft=False)
    ax_zoom.set_ylim(-0.55, len(ba_regimes) - 0.45)
    ax_zoom.set_xlim(-0.072, 0.148)
    ax_zoom.set_xlabel("Paired ΔRMSE (AUC, zoomed)")
    ax_zoom.set_title("B. Beat AML — Detail Near Zero (AUC)", loc="left")

    c2_comps = s5["track_c_fimm"]["experiment_c2_within_fimm_replication"]["comparisons"]
    ax_fimm.grid(True, axis="x", linestyle=":", alpha=0.8)
    ax_fimm.axvline(0.0, color="#334155", linewidth=1.0, zorder=2)
    for c_key, color, marker, y in (
        ("M4_Static_k5_vs_M3_Static_k5", PALETTE["navy"], "o", 1.0),
        ("M4_Static_k5_vs_M4_perm_null_k5", PALETTE["amber"], "s", 0.0),
    ):
        d = c2_comps[c_key]["rmse_diff"]
        val, lo, hi = _to_float(d["diff"]), _to_float(d["ci_low"]), _to_float(d["ci_high"])
        ax_fimm.errorbar(val, y, xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]], fmt=marker,
                         color=color, markerfacecolor=color, markersize=5.6, elinewidth=1.5, capsize=2.8, zorder=4)
        ax_fimm.text(val, y + 0.18, f"{val:+.3f} [{lo:+.3f}, {hi:+.3f}]", ha="center", va="bottom", fontsize=7.2, color=color)

    ax_fimm.text(0.0, -0.85, "Forced ρ>0 arm:\nnot run in FIMM C2", ha="center", va="center", fontsize=7.6,
                 style="italic", color=PALETTE["slate"],
                 bbox={"boxstyle": "round,pad=0.3", "facecolor": PALETTE["band"], "edgecolor": "#CBD5E1"})
    ax_fimm.set_yticks([1.0, 0.0, -0.85])
    ax_fimm.set_yticklabels(["M4 − M3", "M4 − Perm\n(10 draws)", "Forced ρ>0"])
    ax_fimm.set_ylim(-1.35, 1.55)
    ax_fimm.set_xlim(-0.022, 0.014)
    ax_fimm.set_xticks([-0.02, -0.01, 0.0, 0.01])
    ax_fimm.set_xlabel("Paired ΔRMSE (sDSS; separate scale)")
    ax_fimm.set_title("C. FIMM C2 CV (n = 109, sDSS)", loc="left")

    legend_handles = [
        Line2D([0], [0], marker="o", color=PALETTE["navy"], markerfacecolor=PALETTE["navy"], linestyle="none", markersize=6, label="M4 − M3 (primary)"),
        Line2D([0], [0], marker="s", color=PALETTE["amber"], markerfacecolor=PALETTE["amber"], linestyle="none", markersize=6, label="M4 − Perm null (20× Beat AML / 10× FIMM)"),
        Line2D([0], [0], marker="D", color=PALETTE["slate"], markerfacecolor="white", linestyle="none", markersize=6, label="M4 forced ρ>0 − Perm forced (secondary, Beat AML only)"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Gate 3 Fixed-Panel M4 Pathway Covariance Evaluation (k = 5, Patient-Paired 95% Bootstrap CIs)", fontsize=10.8, fontweight="bold")
    return _save_figure(fig, out_dir, "fig2_m4_gate3_forest", "OncoTwin v1 Figure 2: Gate 3 M4 Forest Plot")


def make_figure3(arch: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Figure 3 — Kısa liste ödünleşimi (A/B) ve genelleştirilmiş risk–kapsam AUGRC eğrileri (C/D)."""
    s4, s4_19, s5, z4, z5 = arch["s4"], arch["s4_19"], arch["s5"], arch["z4"], arch["z5"]
    methods = [
        ("EqualWidthAUC", PALETTE["slate"], "s"),
        ("RelRankConformal", PALETTE["amber"], "^"),
        ("RankConformal", PALETTE["navy"], "o"),
        ("Top1GapConformal", PALETTE["purple"], "D"),
    ]
    regimes = [
        ("id_60_20_20", "A. Shortlists — ID 60/20/20 (n_cal=90, n_test=91)", "C. Generalized Risk–Coverage — ID (n = 91)", 91, "90% nominal target"),
        ("wave_ood", "B. Shortlists — Waves 3+4 Exploratory (n_cal=70, n_test=148)", "D. Generalized Risk–Coverage — Waves 3+4 (n = 148)", 148, "90% visual reference (shifted)"),
    ]
    # Data-coordinate box anchors (x_txt, y_txt, ha, va) in wide-open whitespace with crisp leader lines
    box_pos = {
        "id_60_20_20": {
            "EqualWidthAUC": (18.8, 95.8, "center", "top"),
            "RankConformal": (9.2, 100.8, "left", "center"),
            "RelRankConformal": (12.2, 91.4, "center", "center"),
            "Top1GapConformal": (12.2, 84.4, "center", "center"),
        },
        "wave_ood": {
            "EqualWidthAUC": (19.0, 96.2, "center", "top"),
            "RelRankConformal": (13.6, 99.0, "center", "center"),
            "RankConformal": (12.4, 89.5, "center", "center"),
            "Top1GapConformal": (10.6, 83.2, "center", "center"),
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4), layout="constrained")
    for col, (reg_key, top_title, bot_title, n_test, nom_label) in enumerate(regimes):
        ax_top = axes[0, col]
        ax_top.grid(True, linestyle=":", alpha=0.8)
        ax_top.axhline(90.0, color=PALETTE["crimson"], linestyle="--", linewidth=1.1, zorder=2, label=nom_label)
        reg_b = s5["track_b_shortlist"]["regimes"][reg_key]["arms"]["StaticDOpt_k5"]["methods"]

        for m_key, color, marker in methods:
            t = reg_b[m_key]["test"]
            sz_vec = z5[f"track_b/{reg_key}/StaticDOpt_k5/{m_key}/patient_shortlist_size_obs"]
            mean_sz, x_lo, x_hi = _patient_mean_bootstrap_ci(sz_vec, seed=2026, n_boot=2000)
            assert abs(mean_sz - float(t["shortlist_mean_size_obs"])) <= 1e-12
            x_hits = int(round(float(t["shortlist_contains_true_top1_rate"]) * n_test))
            pct = 100.0 * x_hits / float(n_test)
            y_lo, y_hi = _wilson_ci_pct(x_hits, n_test)
            ax_top.errorbar(mean_sz, pct, xerr=[[max(0.0, mean_sz - x_lo)], [max(0.0, x_hi - mean_sz)]],
                            yerr=[[max(0.0, pct - y_lo)], [max(0.0, y_hi - pct)]],
                            fmt=marker, color=color, markersize=6.8, elinewidth=1.4, capsize=3.0, zorder=5, label=m_key)
            tx, ty, ha, va = box_pos[reg_key][m_key]
            ax_top.annotate(
                f"{m_key}\n{x_hits}/{n_test} ({pct:.1f}%) · {100.0 * float(t['shortlist_mean_fraction_obs']):.1f}% obs\nMean shortlist Regret@1: {float(t['shortlist_mean_regret1']):.1f}",
                xy=(mean_sz, pct), xytext=(tx, ty), textcoords="data", ha=ha, va=va, fontsize=6.9, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.75, "alpha": 0.85, "shrinkA": 2, "shrinkB": 5},
                bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": color, "alpha": 0.94, "linewidth": 0.7},
                zorder=6,
            )

        d1 = z5[f"track_b/{reg_key}/StaticDOpt_k5/Top1GapConformal/patient_definite_top1"]
        ok = z5[f"track_b/{reg_key}/StaticDOpt_k5/Top1GapConformal/patient_shortlist_contains_top1"]
        sel = np.isfinite(d1) & (d1 == 1.0)
        s_n, s_ok = int(sel.sum()), int(np.nansum(ok[sel]))
        s_lo, s_hi = _wilson_ci_pct(s_ok, s_n)
        ax_top.text(0.97, 0.03,
                    f"Singleton (size = 1) true-best accuracy:\nTop1GapConformal: {s_ok}/{s_n} ({100.0 * s_ok / s_n:.1f}%, Wilson {s_lo:.1f}–{s_hi:.1f}%)\nOther 3 methods: 0 singletons (N/A)",
                    transform=ax_top.transAxes, ha="right", va="bottom", fontsize=7.0,
                    bbox={"boxstyle": "round,pad=0.30", "facecolor": "#FAF5FF", "edgecolor": PALETTE["purple"], "alpha": 0.95})
        ax_top.set_xlim(2.5, 23.2)
        ax_top.set_ylim(73.5, 104.2)
        ax_top.set_xlabel("Mean shortlist size (measured targets; 95% patient bootstrap CI)")
        if col == 0:
            ax_top.set_ylabel("Patients with ≥1 true best measured target (%)\n(Wilson 95% CI)")
        ax_top.set_title(top_title, loc="left")
        ax_top.legend(loc="lower left", fontsize=7.0, frameon=True, facecolor="white", framealpha=0.92)

        ax_bot = axes[1, col]
        ax_bot.grid(True, linestyle=":", alpha=0.8)
        vm = z4[f"{reg_key}/archive_available/StaticDOpt_k5/valid_mask"].astype(bool)
        losses = z4[f"{reg_key}/archive_available/StaticDOpt_k5/patient_regret1"]
        c_om, g_om, area_om = _augrc_curve_from_arrays(z4[f"{reg_key}/test_omics_pca_dist"], losses, vm)
        c_m3, g_m3, area_m3 = _augrc_curve_from_arrays(z4[f"{reg_key}/archive_available/StaticDOpt_k5/m3_post_scale"], losses, vm)
        c_or, g_or, area_or = _augrc_curve_from_arrays(losses, losses, vm)

        ag = s5["track_b_shortlist"]["regimes"][reg_key]["augrc_sensitivity"]["StaticDOpt_k5"]
        ab = s4["regimes"][reg_key]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["abstention"]
        assert abs(area_om - float(ag["omics_pca_dist"]["augrc"])) <= 1e-10
        assert abs(area_m3 - float(ag["m3_post_scale"]["augrc"])) <= 1e-10
        assert abs(area_or - float(ag["m3_post_scale"]["oracle_augrc"])) <= 1e-10
        mean_loss = float(g_m3[-1])
        rand_area = 0.5 * mean_loss
        assert abs(rand_area - float(ag["m3_post_scale"]["random_augrc"])) <= 1e-10

        ax_bot.plot(c_om, g_om, color=PALETTE["crimson"], lw=2.0,
                    label=f"omics_pca_dist (AUGRC {area_om:.2f} | AURC {ab['omics_pca_dist']['risk_coverage_regret1']['aurc']:.2f})")
        ax_bot.plot(c_m3, g_m3, color=PALETTE["teal"], lw=2.0,
                    label=f"m3_post_scale (AUGRC {area_m3:.2f} | AURC {ab['m3_post_scale']['risk_coverage_regret1']['aurc']:.2f})")
        ax_bot.plot([0.0, 1.0], [0.0, mean_loss], color="#334155", ls="--", lw=1.3,
                    label=f"Random c·mean loss (AUGRC {rand_area:.2f} | AURC {mean_loss:.2f})")
        ax_bot.plot(c_or, g_or, color=PALETTE["gray"], ls=":", lw=1.4, label=f"Oracle lower bound (AUGRC {area_or:.2f})")

        d19 = s4_19["regimes"][reg_key]["archive_available"]["arms"]["StaticDOpt_k5"]["final_fit_threshold_eval"]
        c_19 = float(d19["n_accepted"]) / float(d19["n_available"])
        ax_bot.scatter([c_19], [c_19 * float(d19["accepted_mean_regret1"])], marker="*", s=110, color=PALETTE["navy"],
                       edgecolors="white", linewidths=0.6, zorder=6,
                       label=f"Addendum 19 exploratory point ({d19['n_accepted']}/{d19['n_available']}, c={c_19:.2f})")

        ap = s4["regimes"][reg_key]["comparisons"]["archive_available"]["aurc_pairs"]
        r_cmp, m_cmp = ap["StaticDOpt_k5_omics_vs_random"], ap["StaticDOpt_k5_omics_vs_m3_post_scale"]
        ax_bot.text(0.97, 0.05,
                    f"Archived paired ΔAURC (95% CI):\nomics − random: {r_cmp['diff']:+.2f} [{r_cmp['ci_low']:+.2f}, {r_cmp['ci_high']:+.2f}]\nomics − m3_post: {m_cmp['diff']:+.2f} [{m_cmp['ci_low']:+.2f}, {m_cmp['ci_high']:+.2f}]",
                    transform=ax_bot.transAxes, ha="right", va="bottom", fontsize=7.0,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92})
        ax_bot.set_xlim(0.0, 1.02)
        ax_bot.set_ylim(0.0, mean_loss * 1.16)
        ax_bot.set_xlabel("Accepted patient fraction c")
        if col == 0:
            ax_bot.set_ylabel("Generalized risk G(c) = c · R(c) (AUC Regret@1)\nArea under curve = AUGRC")
        ax_bot.set_title(bot_title, loc="left")
        ax_bot.legend(loc="upper left", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.90)

    fig.suptitle("Beat AML Conformal Shortlist Tradeoff and Selective Risk (StaticDOpt + M3, k = 5)", fontsize=11.0, fontweight="bold")
    return _save_figure(fig, out_dir, "fig3_shortlist_and_selective_risk", "OncoTwin v1 Figure 3: Shortlists and AUGRC")


def _plot_paired_diff_panel(
    ax: plt.Axes, items: list[tuple[str, dict[str, Any], str]], xlabel: str, title: str, decimals: int = 3
) -> None:
    ax.grid(True, axis="x", linestyle=":", alpha=0.8)
    ax.axvline(0.0, color="#334155", linestyle="--", linewidth=1.0, zorder=2)
    ys = np.array([1.0, 0.0], dtype=float)
    x_vals: list[float] = [0.0]
    for y, (_, node, color) in zip(ys, items):
        val, lo, hi = _to_float(node["diff"]), _to_float(node["ci_low"]), _to_float(node["ci_high"])
        x_vals.extend([lo, hi])
        ax.errorbar(val, y, xerr=[[max(0.0, val - lo)], [max(0.0, hi - val)]],
                    fmt="o", color=color, markersize=5.4, elinewidth=1.5, capsize=2.8, zorder=4)
        ax.text(val, y + 0.24, f"{val:+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]",
                ha="center", va="bottom", fontsize=7.1, color=color)
    span = max(x_vals) - min(x_vals)
    pad = max(0.32 * span, 0.012)
    ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
    ax.set_ylim(-0.45, 1.78)
    ax.set_yticks(ys)
    ax.set_yticklabels([it[0] for it in items], fontsize=7.6)
    ax.set_xlabel(xlabel, fontsize=7.8)
    ax.set_title(title, loc="left", fontsize=8.8)


def make_figure4(arch: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Figure 4 — FIMM C1 (donmuş Beat AML aktarımı) ve C2 (FIMM içi 5-kat CV × 3)."""
    s5 = arch["s5"]
    c1 = s5["track_c_fimm"]["experiment_c1_frozen_transfer"]["cohorts"]["index_109_patients"]
    c2 = s5["track_c_fimm"]["experiment_c2_within_fimm_replication"]

    # 4-row × 3-col grid: bar panels span 2 rows each; paired-Δ panels get full dedicated rows (zero squashing)
    fig = plt.figure(figsize=(13.0, 8.2), layout="constrained")
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1.15, 1.15, 1.25])

    c1_arms = [
        ("C1_M0_k0", "M0\nk=0", PALETTE["gray"]),
        ("C1_M1_rank_k0", "M1\nk=0", PALETTE["slate"]),
        ("C1_M3_Random20x_k5", "M3 Rand\nk=5", PALETTE["amber"]),
        ("C1_M3_StaticDOpt_k2", "M3 DOpt\nk=2", "#5B8DB8"),
        ("C1_M3_StaticDOpt_k5", "M3 DOpt\nk=5", PALETTE["navy"]),
        ("C1_M3_StaticDOpt_k8", "M3 DOpt\nk=8", "#5B8DB8"),
    ]
    xs1 = np.arange(len(c1_arms))

    for col, (metric, fmt_s, ylim, ylabel, title, plot_kind, ref_idx) in enumerate([
        ("ndcg_at_3", "{:.3f}", (0.54, 0.72), "Within-Patient NDCG@3  ↑", "A. C1 Frozen Transfer — NDCG@3\n(n = 109, 64 Beat AML probes)", "dot", 1),
        ("regret_at_1_sdss", "{:.2f}", (0.0, 13.0), "Regret@1 (sDSS)  ↓", "B. C1 Frozen Transfer — Regret@1\n(109 evaluated; 98 complete prefixes; 11 backfilled)", "bar", None),
    ]):
        ax = fig.add_subplot(gs[0:2, col])
        vals = [_to_float(c1["arms"][k][metric]) for k, _, _ in c1_arms]
        colors = [c for _, _, c in c1_arms]
        ax.grid(True, axis="y", linestyle=":", alpha=0.8)
        dy = (ylim[1] - ylim[0]) * 0.032
        if plot_kind == "bar":
            bars = ax.bar(xs1, vals, color=colors, width=0.62, edgecolor="#1E293B", linewidth=0.5, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2.0, v + dy * 0.78, fmt_s.format(v), ha="center", va="bottom", fontsize=7.3)
        else:
            if ref_idx is not None:
                ax.axhline(vals[ref_idx], color=PALETTE["slate"], linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)
            ax.vlines(xs1, ylim[0], vals, color="#CBD5E1", linewidth=1.2, linestyles=":", zorder=2)
            ax.scatter(xs1, vals, s=105, c=colors, edgecolors="#1E293B", linewidths=0.8, zorder=4)
            for x, v in zip(xs1, vals):
                ax.text(x, v + dy, fmt_s.format(v), ha="center", va="bottom", fontsize=7.3)
            ax.set_xlim(-0.6, len(c1_arms) - 0.4)
        ax.set_xticks(xs1)
        ax.set_xticklabels([lbl for _, lbl, _ in c1_arms], fontsize=7.6)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    c1_cmp = c1["comparisons"]
    _plot_paired_diff_panel(
        fig.add_subplot(gs[0, 2]),
        [("vs M1 (k=0)", c1_cmp["C1_M3_k5_vs_M1_k0"]["ndcg3_diff"], PALETTE["navy"]),
         ("vs Rand20x (k=5)", c1_cmp["C1_M3_Static_k5_vs_Random_k5"]["ndcg3_diff"], PALETTE["amber"])],
        xlabel="ΔNDCG@3 (right favors DOpt k=5)", title="C. C1 Paired ΔNDCG@3 (k=5)", decimals=3,
    )
    _plot_paired_diff_panel(
        fig.add_subplot(gs[1, 2]),
        [("vs M1 (k=0)", c1_cmp["C1_M3_k5_vs_M1_k0"]["regret1_sdss_diff"], PALETTE["navy"]),
         ("vs Rand20x (k=5)", c1_cmp["C1_M3_Static_k5_vs_Random_k5"]["regret1_sdss_diff"], PALETTE["amber"])],
        xlabel="ΔRegret@1 sDSS (left favors DOpt k=5)", title="D. C1 Paired ΔRegret@1 (crosses 0)", decimals=2,
    )

    c2_arms = [
        ("M0_k0", "M0\nk=0", PALETTE["gray"]),
        ("M0_k5", "M0\nk=5", "#B6C2CF"),
        ("M1_k0", "M1\nk=0", PALETTE["slate"]),
        ("M2a_StaticDOpt_k5", "M2a DOpt\nk=5", PALETTE["purple"]),
        ("M3_Random20x_k5", "M3 Rand\nk=5", PALETTE["amber"]),
        ("M3_StaticDOpt_k5", "M3 DOpt\nk=5", PALETTE["navy"]),
    ]
    xs2 = np.arange(len(c2_arms))

    for col, (metric, fmt_s, ylim, ylabel, title, plot_kind, ref_idx) in enumerate([
        ("rmse", "{:.2f}", (0.0, 6.1), "Target RMSE (sDSS)  ↓", "E. C2 Within-FIMM 5-Fold CV × 3 — RMSE\n(n = 109, fold-rebuilt probe pool)", "bar", None),
        ("ndcg_at_3", "{:.3f}", (0.69, 0.88), "Within-Patient NDCG@3  ↑", "F. C2 Within-FIMM 5-Fold CV × 3 — NDCG@3\n(Random20x run for M3 k=5 only)", "dot", 2),
    ]):
        ax = fig.add_subplot(gs[2:4, col])
        vals = [_to_float(c2["arms"][k][metric]) for k, _, _ in c2_arms]
        colors = [c for _, _, c in c2_arms]
        ax.grid(True, axis="y", linestyle=":", alpha=0.8)
        dy = (ylim[1] - ylim[0]) * 0.032
        if plot_kind == "bar":
            bars = ax.bar(xs2, vals, color=colors, width=0.62, edgecolor="#1E293B", linewidth=0.5, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2.0, v + dy * 0.78, fmt_s.format(v), ha="center", va="bottom", fontsize=7.3)
        else:
            if ref_idx is not None:
                ax.axhline(vals[ref_idx], color=PALETTE["slate"], linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)
            ax.vlines(xs2, ylim[0], vals, color="#CBD5E1", linewidth=1.2, linestyles=":", zorder=2)
            ax.scatter(xs2, vals, s=105, c=colors, edgecolors="#1E293B", linewidths=0.8, zorder=4)
            for x, v in zip(xs2, vals):
                ax.text(x, v + dy, fmt_s.format(v), ha="center", va="bottom", fontsize=7.3)
            ax.set_xlim(-0.6, len(c2_arms) - 0.4)
        ax.set_xticks(xs2)
        ax.set_xticklabels([lbl for _, lbl, _ in c2_arms], fontsize=7.6)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    c2_cmp = c2["comparisons"]
    _plot_paired_diff_panel(
        fig.add_subplot(gs[2, 2]),
        [("vs M1 (k=0)", c2_cmp["M3_Static_k5_vs_M1_k0"]["rmse_diff"], PALETTE["navy"]),
         ("vs Rand20x (k=5)", c2_cmp["M3_Static_k5_vs_Random20x_k5"]["rmse_diff"], PALETTE["amber"])],
        xlabel="ΔRMSE sDSS (left favors DOpt k=5)", title="G. C2 Paired ΔRMSE (k=5)", decimals=2,
    )
    _plot_paired_diff_panel(
        fig.add_subplot(gs[3, 2]),
        [("vs M1 (k=0)", c2_cmp["M3_Static_k5_vs_M1_k0"]["ndcg3_diff"], PALETTE["navy"]),
         ("vs Rand20x (k=5)", c2_cmp["M3_Static_k5_vs_Random20x_k5"]["ndcg3_diff"], PALETTE["amber"])],
        xlabel="ΔNDCG@3 (right favors DOpt k=5)", title="H. C2 Paired ΔNDCG@3 (k=5)", decimals=3,
    )

    fig.suptitle(
        "FIMM/Malani External Evaluation (109 Index Patients, 23 Shared Targets): C1 Frozen Transfer vs. C2 Within-FIMM CV",
        fontsize=10.8, fontweight="bold",
    )
    return _save_figure(fig, out_dir, "fig4_fimm_c1_c2", "OncoTwin v1 Figure 4: FIMM C1 and C2")


def generate_all_figures(data_dir: Path = DEFAULT_DATA_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, str]:
    """Dört makale figürünü deterministik biçimde üretir ve SHA-256 sözlüğü döndürür."""
    _configure_style()
    arch = _load_archives(data_dir)
    verify_figure_table_parity(arch)
    all_hashes: dict[str, str] = {}
    all_hashes.update(make_figure1(arch, out_dir))
    all_hashes.update(make_figure2(arch, out_dir))
    all_hashes.update(make_figure3(arch, out_dir))
    all_hashes.update(make_figure4(arch, out_dir))
    return all_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoTwin v1 deterministik makale figür üreticisi.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(generate_all_figures(data_dir=args.data_dir, out_dir=args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
