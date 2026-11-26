"""Sprint 5 Markdown rapor üreticisi (M4 + Kapı 3, Sıra-Kalibreli Kısa Liste + AUGRC, FIMM C1/C2)."""

from __future__ import annotations

import math
from typing import Any


def _f(v: float | None, dec: int = 2, pct: bool = False) -> str:
    if v is None or not math.isfinite(float(v)):
        return "hesaplanamadı"
    return f"%{float(v) * 100.0:.{dec}f}" if pct else f"{float(v):.{dec}f}"


def _ci(d: dict[str, Any], dec: int = 2, pct: bool = False) -> str:
    if d.get("diff") is None or not math.isfinite(float(d["diff"])):
        return "hesaplanamadı"
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return (
        f"{d['diff'] * scale:+.{dec}f}{unit} "
        f"[{d['ci_low'] * scale:+.{dec}f}, {d['ci_high'] * scale:+.{dec}f}] "
        f"(p={d['p_two_sided']:.3f}, n={d['n_paired']})"
    )


def render_sprint05_md(res: dict[str, Any]) -> str:
    """Sprint 5 sonuç sözlüğünü kapsamlı Türkçe Markdown raporuna dönüştürür."""
    ta = res["track_a_m4"]
    tb = res["track_b_shortlist"]
    tc = res["track_c_fimm"]
    g3 = ta["gate3"]

    lines = [
        "# Sprint 5 — `M4` Yolak Kovaryansı (`Kapı 3`), Sıra-Kalibreli Konformal Kısa Liste ve FIMM/Malani Dış Tekrarı",
        "",
        f"- **Toplam Süre:** `{res['elapsed_seconds']:.1f} s` (`Kulvar A: {ta['elapsed_seconds']:.1f} s`, `Kulvar B: {tb['elapsed_seconds']:.1f} s`, `Kulvar C: {tc['elapsed_seconds']:.1f} s`)",
        "- **Protokol:** [PROTOCOL_SPRINT05.md](../PROTOCOL_SPRINT05.md) deney koşusundan önce dondurulmuştur; tüm `sprint01..04` (`v1`, `v2`, `DEV-OPUS-19`) artefaktları korunmuştur.",
        f"- **Ön Kayıtlı Kapı 3 (`AD-001`) Hükmü:** **`{g3['decision']}`**",
        "- **Bilimsel Sınırlar:** `Waves3+4` sonuçları yalnız keşifsel OOD stres testidir (yöntem/eşik seçimine girmez); FIMM `C1` (donmuş aktarım) ve `C2` (FIMM içi yöntem tekrarı) kesin biçimde ayrılmıştır; Beat AML konformal aralığı FIMM'e taşınmamıştır.",
        "",
        "---",
        "",
        "## 1. Kulvar A — `M4` Yolak / Hedef-Ailesi Yapılı Kovaryans & Kapı 3 (`AD-001`)",
        "",
        f"- **İlaç–Aile Grafı (`beataml_drug_families.xlsx`):** `{ta['drug_family_graph']['n_annotated_drugs']}/{ta['drug_family_graph']['n_drugs_total']}` anotasyonlu ilaç, `{ta['drug_family_graph']['n_families']}` aile, `{ta['drug_family_graph']['n_incidence_edges']}` üyelik kenarı; derece & aile-boyutu korumalı permütasyon kontrolü `{ta['drug_family_graph']['n_perm_draws']}` sabit çekiliş ortalaması.",
        "",
        "### 1.1 Ön Kayıtlı Kapı 3 Değerlendirmesi (Deney A1, Sabit `StaticDOpt(M3)` Paneli, `k=5`)",
        "",
        "| Kapı 3 Koşulu | Gözlenen Değer | Durum |",
        "|---|---|---|",
        f"| 1. ID 5-Fold CV (`n=427`): `ΔRMSE(M4 − M3) ≤ −0.50`, GA`<0`, sıralama bozulmasız | `{_ci(g3['observed_k5_a1']['id_cv_m4_minus_m3_rmse'])}` | `{'PASS' if g3['conditions']['1_id_cv_beats_m3_ge_050_no_ranking_deg'] else 'FAIL'}` |",
        f"| 2. ID 5-Fold CV (`n=427`): `ΔRMSE(M4 − M4_perm_null) ≤ −0.50`, GA`<0` | `{_ci(g3['observed_k5_a1']['id_cv_m4_minus_perm_rmse'])}` | `{'PASS' if g3['conditions']['2_id_cv_beats_degree_preserving_perm_null_ge_050'] else 'FAIL'}` |",
        f"| 3. Keşifsel `Waves3+4` (`n=146`): `ΔRMSE(M4 − M3) < 0` ve `ΔRMSE(M4 − perm) < 0` | `vs M3: {_ci(g3['observed_k5_a1']['wave_ood_m4_minus_m3_rmse'])}` ; `vs Perm: {_ci(g3['observed_k5_a1']['wave_ood_m4_minus_perm_rmse'])}` | `{'PASS' if g3['conditions']['3_exploratory_wave_ood_consistent_direction'] else 'FAIL'}` |",
        "",
        "### 1.2 Deney A1 (Birincil — Sabit `StaticDOpt(M3)` Paneli, Aynı Hastalar ve Hedefler)",
        "",
        "| Rejim | `k` | `n` | `M3` RMSE / NDCG3 / Reg1 | `M4` (iç-CV $\\rho\\ge 0$) | `M4_forced` ($\\rho>0$) | `M4_perm_null` (20×) | `%ρ>0` | `ΔRMSE (M4 − M3)` | `ΔRMSE (M4 − Perm)` | `ΔRMSE (Forced − PermF)` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    regime_labels = (
        ("id_cv_5fold_3seed", "ID CV 5x3 (Birincil)"),
        ("id_60_20_20", "ID 60/20/20 (Holdout)"),
        ("wave_ood_s3", "W3+4 S3 (Keşifsel OOD)"),
        ("wave_ood_s4", "W3+4 S4 (Keşifsel OOD)"),
    )
    for rkey, rlab in regime_labels:
        rnode = ta["regimes"][rkey]["experiment_a1_fixed_panel"]
        for k in (5, 2, 8):
            kn = rnode[f"k{k}"]
            ms = kn["models"]
            cs = kn["comparisons"]
            def _tri(m: dict[str, Any]) -> str:
                return f"{_f(m['rmse'])} / {_f(m['ndcg_at_3'], 3)} / {_f(m['regret_at_1'], 1)}"
            lines.append(
                f"| {rlab} | {k} | {ms['M3']['n_eval']} | {_tri(ms['M3'])} | {_tri(ms['M4'])} | "
                f"{_tri(ms['M4_forced_pos'])} | {_tri(ms['M4_perm_null'])} | "
                f"{_f(kn['selected_rho_m4_pos_fraction'], 0, True)} | "
                f"{_ci(cs['M4_vs_M3']['rmse_diff'])} | {_ci(cs['M4_vs_M4_perm_null']['rmse_diff'])} | "
                f"{_ci(cs['M4_forced_pos_vs_M4_perm_forced']['rmse_diff'])} |"
            )

    lines.extend([
        "",
        "### 1.3 Deney A2 (İkincil — `M4` Kovaryansıyla Yeniden Seçilmiş `StaticDOpt(M4)` Paneli, `k=5`)",
        "",
        "| Rejim | Top-5 Panel Jaccard (`M4` vs `M3`) | `M4_reselected` RMSE / NDCG3 / Reg1 | `ΔRMSE (M4_resel − M3)` | `ΔRMSE (M4_resel − M4_fixed)` |",
        "|---|---:|---:|---:|---:|",
    ])
    for rkey, rlab in regime_labels:
        reg_t = ta["regimes"][rkey]
        a2 = reg_t["experiment_a2_reselected_panel"]["k5"]
        mr = a2["M4_reselected_panel"]
        lines.append(
            f"| {rlab} | {_f(reg_t['mean_panel_jaccard_k5'], 3)} | "
            f"{_f(mr['rmse'])} / {_f(mr['ndcg_at_3'], 3)} / {_f(mr['regret_at_1'], 1)} | "
            f"{_ci(a2['M4_reselected_vs_M3']['rmse_diff'])} | {_ci(a2['M4_reselected_vs_M4_fixed']['rmse_diff'])} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Kulvar B — Hasta-Düzeyi Sıra-Kalibreli Konformal Kısa Liste ve `AUGRC` (Nominal %90, $\\alpha=0.10$)",
        "",
        f"> {tb['note']}",
        "",
    ])
    for rn, rlab in (("id_60_20_20", "ID 60/20/20 (Birincil)"), ("wave_ood", "Waves3+4 (Keşifsel OOD Stres Testi)")):
        rnode = tb["regimes"][rn]
        lines.extend([
            f"### 2.{1 if rn == 'id_60_20_20' else 2} {rlab} (`n_cal={rnode['n_cal']}`, `n_test={rnode['n_test']}`)",
            "",
            "| Kol | Yöntem | Kalibreli Eşik | Gerçek #1 Listede | Ort. Boyut (Med.) | Gözl. Oran | Tekil Oran (`|S|=1`) | En İyi AUC Kaybı (`Best Reg1`) | Ort. AUC Kaybı (`Mean Reg1`) | En Kötü AUC Kaybı (`Worst Reg1`) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm_name, anode in rnode["arms"].items():
            for mlab in ("EqualWidthAUC", "RankConformal", "RelRankConformal", "Top1GapConformal"):
                mn = anode["methods"][mlab]
                cal, te = mn["calibration"], mn["test"]
                if mlab == "EqualWidthAUC":
                    th = f"2q={_f(cal['width_sim'], 1)}"
                elif mlab == "RankConformal":
                    th = f"k̂={cal['k_hat']}"
                elif mlab == "RelRankConformal":
                    th = f"q_rel={_f(cal['q_rel'], 3)}"
                else:
                    th = f"m̂={_f(cal['m_hat'], 1)}"
                lines.append(
                    f"| `{arm_name}` | `{mlab}` | {th} | {_f(te['shortlist_contains_true_top1_rate'], 1, True)} | "
                    f"{_f(te['shortlist_mean_size_obs'], 2)} ({_f(te['shortlist_median_size_obs'], 1)}) | "
                    f"{_f(te['shortlist_mean_fraction_obs'], 1, True)} | {_f(te['definite_top1_rate_obs'], 1, True)} | "
                    f"{_f(te['shortlist_best_regret1'], 2)} | {_f(te['shortlist_mean_regret1'], 2)} | {_f(te['shortlist_worst_regret1'], 2)} |"
                )
        # Ana kol StaticDOpt_k5 eşleştirilmiş farklar
        c_main = rnode["arms"]["StaticDOpt_k5"]["comparisons"]
        lines.extend([
            "",
            f"**`StaticDOpt_k5` ({rlab}) Eşleştirilmiş Kısa Liste Farkları (Bootstrap B=1000):**",
            "",
            f"- `RankConformal − EqualWidthAUC`: Boyut `{_ci(c_main['RankConformal_vs_EqualWidthAUC']['size_diff'])}`, Gerçek #1 `{_ci(c_main['RankConformal_vs_EqualWidthAUC']['contains_top1_diff'], 1, True)}`, Ort. Regret `{_ci(c_main['RankConformal_vs_EqualWidthAUC']['mean_regret1_diff'])}`, En İyi Regret `{_ci(c_main['RankConformal_vs_EqualWidthAUC']['best_regret1_diff'])}`",
            f"- `Top1GapConformal − EqualWidthAUC`: Boyut `{_ci(c_main['Top1GapConformal_vs_EqualWidthAUC']['size_diff'])}`, Gerçek #1 `{_ci(c_main['Top1GapConformal_vs_EqualWidthAUC']['contains_top1_diff'], 1, True)}`, Tekil Oran `{_ci(c_main['Top1GapConformal_vs_EqualWidthAUC']['definite_top1_diff'], 1, True)}`, Ort. Regret `{_ci(c_main['Top1GapConformal_vs_EqualWidthAUC']['mean_regret1_diff'])}`",
            f"- `Top1GapConformal − RankConformal`: Boyut `{_ci(c_main['Top1GapConformal_vs_RankConformal']['size_diff'])}`, Gerçek #1 `{_ci(c_main['Top1GapConformal_vs_RankConformal']['contains_top1_diff'], 1, True)}`, Ort. Regret `{_ci(c_main['Top1GapConformal_vs_RankConformal']['mean_regret1_diff'])}`",
            "",
        ])

    lines.extend([
        "### 2.3 Ek Seçici Risk Duyarlılığı: `AUGRC` (Traub et al., NeurIPS 2024, `arXiv:2407.01032`) vs `AURC`",
        "",
        "| Rejim | Kol | `omics` AUGRC / Rand (`E-AUGRC`) | `m3_post` AUGRC / Rand (`E-AUGRC`) | `omics` v2 AURC / Rand | `m3_post` v2 AURC / Rand |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for rn, rlab in (("id_60_20_20", "ID 60/20/20"), ("wave_ood", "W3+4 OOD")):
        for arm_name, an in tb["regimes"][rn]["augrc_sensitivity"].items():
            om, m3 = an["omics_pca_dist"], an["m3_post_scale"]
            lines.append(
                f"| {rlab} | `{arm_name}` | {_f(om['augrc'])} / {_f(om['random_augrc'])} ({_f(om['e_augrc'])}) | "
                f"{_f(m3['augrc'])} / {_f(m3['random_augrc'])} ({_f(m3['e_augrc'])}) | "
                f"{_f(om['v2_aurc_regret1'])} / {_f(om['v2_random_aurc'])} | "
                f"{_f(m3['v2_aurc_regret1'])} / {_f(m3['v2_random_aurc'])} |"
            )

    c1 = tc["experiment_c1_frozen_transfer"]
    c2 = tc["experiment_c2_within_fimm_replication"]
    lines.extend([
        "",
        "---",
        "",
        "## 3. Kulvar C — FIMM / Malani Dış Sıralama Tekrarı (`Zenodo 7370747`)",
        "",
        f"- **Ön-Modelleme Katalog & QC Denetimi (`sprint05_fimm_catalog_audit.json`):** `132` eşleşmiş AML örneği / `109` benzersiz hasta (`File_7 = 167` RNA [`163 AML + 4 CD34+`]), `{tc['common_targets_count']}/24` ortak hedef (`Gilteritinib` yok), `{tc['common_probes_count']}` ortak Beat AML probu, `{tc['common_genes_count']}` ortak Ensembl gen; yön uyumu `y_aligned = -sDSS`; Beat AML konformal aralığı **taşınmamıştır** (`conformal_intervals_transferred = False`).",
        f"- **Beat AML Eğitiminde Yeniden Seçilen `StaticDOpt` Ortak Katalog Top-8:** `{', '.join(c1['reselected_beataml_static_dopt_top8'])}`",
        "",
        f"### 3.1 Deney C1 — {c1['label']}",
        "",
    ])
    for cname, clab in (("index_109_patients", "Birincil: 109 Benzersiz İndeks Hasta"), ("all_132_specimens_clustered", "Duyarlılık: Tüm 132 Örnek (109 Hasta Kümeli)")):
        cn = c1["cohorts"][cname]
        lines.extend([
            f"#### {clab} (`n_spec={cn['n_evaluable_specimens']}/{cn['n_specimens']}`)",
            "",
            "| Kol | `n_units` | `NDCG@3` ↑ | `Recall@3` ↑ | `Within-Patient Spearman` ↑ | `Regret@1 (sDSS)` ↓ | `Regret@3 (sDSS)` ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for aname, av in cn["arms"].items():
            lines.append(
                f"| `{aname}` | {av['n_eval_units']} | {_f(av['ndcg_at_3'], 3)} | {_f(av['recall_at_3'], 3)} | "
                f"{_f(av['within_patient_spearman'], 3)} | {_f(av['regret_at_1_sdss'], 2)} | {_f(av['regret_at_3_sdss'], 2)} |"
            )
        lines.append("")
        for ck, cv in cn["comparisons"].items():
            lines.append(
                f"- `{ck}`: `ΔNDCG@3 = {_ci(cv['ndcg3_diff'], 3)}`, `ΔSpearman = {_ci(cv['spearman_diff'], 3)}`, `ΔRegret@1 (sDSS) = {_ci(cv['regret1_sdss_diff'], 2)}`"
            )
        lines.append("")

    lines.extend([
        f"### 3.2 Deney C2 — {c2['label']}",
        "",
        f"- **Değerlendirilen Hasta Sayısı:** `n_eval = {c2['n_eval']}/{c2['n_patients_total']}` (tüm istatistikler, gen filtreleri ve `StaticDOpt` paneli her katta yalnızca o katın FIMM eğitim hastalarından öğrenilmiştir; `-sDSS` ölçeği).",
        "",
        "| Kol | `RMSE (sDSS)` ↓ | `Residual R²` ↑ | `NDCG@3` ↑ | `Recall@3` ↑ | `Regret@1 (sDSS)` ↓ | `Within-Pat Res Spearman` ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for aname, av in c2["arms"].items():
        lines.append(
            f"| `{aname}` | {_f(av['rmse'], 2)} | {_f(av['residual_r2'], 3)} | {_f(av['ndcg_at_3'], 3)} | "
            f"{_f(av['recall_at_3'], 3)} | {_f(av['regret_at_1'], 2)} | {_f(av['within_patient_residual_spearman'], 3)} |"
        )
    lines.append("")
    for ck, cv in c2["comparisons"].items():
        lines.append(
            f"- `{ck}`: `ΔRMSE (sDSS) = {_ci(cv['rmse_diff'], 2)}`, `ΔNDCG@3 = {_ci(cv['ndcg3_diff'], 3)}`, `ΔRegret@1 (sDSS) = {_ci(cv['regret1_diff'], 2)}`"
        )
    return "\n".join(lines) + "\n"
