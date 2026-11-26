"""Sprint 3 Markdown rapor biçimlendiricisi."""

from __future__ import annotations

from typing import Any

POLICIES = ("Random50", "StaticDiversity", "StaticDOpt", "AdaptiveOED")
ABLATION_K_BUDGETS = (3, 5, 8)
ABLATION_NAMES = (
    "M3_full",
    "M3_no_omics",
    "M3_perm_within_patient",
    "M3_perm_across_patients",
    "M3_perm_train_cov",
)


def fmt_ci(ci: dict[str, Any], digits: int = 3) -> str:
    return f"{ci['diff']:+.{digits}f} [{ci['ci_low']:+.{digits}f}, {ci['ci_high']:+.{digits}f}]"


def _render_policy_table(metrics_dict: dict[str, Any]) -> list[str]:
    m1 = metrics_dict["k0_refs"]["M1"]
    m0 = metrics_dict["k0_refs"]["M0"]
    rows = [
        "| $k$ | Politika / Referans | `n_eval` | `RMSE` ↓ | `Res R²` ↑ | $\\rho^{\\text{res}}_{\\text{pat}}$ ↑ | `NDCG@3` ↑ | `Recall@3` ↑ | `Regret@1` ↓ | `Regret@3` ↓ | Ek Bilgi |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| 0 | `M0(0)` (ref) | {m0['n_eval']} | {m0['rmse']:.2f} | {m0['residual_r2']:+.3f} | {m0['within_patient_residual_spearman']:+.3f} | {m0['ndcg_at_3']:.3f} | {m0['recall_at_3']:.3f} | {m0['regret_at_1']:.1f} | {m0['regret_at_3']:.1f} | Panel seçimi yok |",
        f"| 0 | **`M1` (omik ref)** | {m1['n_eval']} | {m1['rmse']:.2f} | {m1['residual_r2']:+.3f} | {m1['within_patient_residual_spearman']:+.3f} | {m1['ndcg_at_3']:.3f} | {m1['recall_at_3']:.3f} | {m1['regret_at_1']:.1f} | {m1['regret_at_3']:.1f} | Panel seçimi yok |",
    ]
    for k_str in ("1", "2", "3", "5", "8", "12"):
        pdict = metrics_dict["by_k"][k_str]
        for pol in POLICIES:
            r = pdict[pol]
            extra = "—"
            if "top_k_direct_hit_rate" in r:
                extra = f"direct={r['top_k_direct_hit_rate']*100:.0f}% (tam={r['full_panel_patient_rate']*100:.0f}%)"
            elif "n_unique_panels" in r:
                extra = f"uniq={r['n_unique_panels']:.0f}, Jacc={r['mean_pairwise_jaccard']:.2f}"
            rows.append(
                f"| {k_str} | **{pol}** | {r['n_eval']} | {r['rmse']:.2f} | {r['residual_r2']:+.3f} | "
                f"{r['within_patient_residual_spearman']:+.3f} | {r['ndcg_at_3']:.3f} | {r['recall_at_3']:.3f} | "
                f"{r['regret_at_1']:.1f} | {r['regret_at_3']:.1f} | {extra} |"
            )
    return rows


def render_sprint03_md(res: dict[str, Any]) -> str:
    cv_m = res["cv_5fold_3seeds"]["metrics"]
    cv_c = res["cv_5fold_3seeds"]["comparisons"]
    wv_m = res["wave_ood"]["metrics"]
    wv_c = res["wave_ood"]["comparisons"]
    g2 = res["gate2_decision"]
    abl_txt = "GEÇTİ" if g2["ablation_sanity_passed"] else f"BAŞARISIZ ({g2['ablation_failures']})"

    lines = [
        "# Sprint 3 — Toplu Panel Seçimi, `M3` Mekanizma Ablasyonları ve Kapı 2",
        "",
        f"Tüm deneyler dondurulmuş [PROTOCOL_SPRINT03.md](../PROTOCOL_SPRINT03.md) uyarınca **{res['elapsed_seconds']} sn** içinde CPU üzerinde çalıştırılmıştır.",
        "",
        "---",
        "",
        "## 1. Birincil ID Rejimi — `Patient-Disjoint 5-Fold CV` (3 Seed, Eşit Bütçe $k \\in \\{1..12\\}$)",
        "",
        *_render_policy_table(cv_m),
        "",
        "### Eşleştirilmiş Hasta Bootstrap Farkları (CV — %95 GA)",
        "",
        "| Karşılaştırma | `ΔRMSE` (negatif = iyi) | `ΔRecall@3` (pozitif = iyi) | `ΔNDCG@3` (pozitif = iyi) | `ΔRegret@1` (negatif = iyi) | `Δ(ρ^res)` (pozitif = iyi) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in (1, 2, 3, 5, 8, 12):
        for name in (
            f"k{k}_StaticDiversity_vs_Random50",
            f"k{k}_StaticDOpt_vs_Random50",
            f"k{k}_AdaptiveOED_vs_Random50",
            f"k{k}_AdaptiveOED_vs_StaticDOpt",
        ):
            c = cv_c["panel"][name]
            lines.append(
                f"| `{name}` | {fmt_ci(c['rmse_diff'], 2)} | {fmt_ci(c['recall3_diff'])} | "
                f"{fmt_ci(c['ndcg3_diff'])} | {fmt_ci(c['regret1_diff'], 1)} | {fmt_ci(c['res_spearman_diff'])} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Zamansal OOD Rejimi — `Clean Waves1+2 → Waves3+4`",
            "",
            *_render_policy_table(wv_m),
            "",
            "### Zamansal OOD Eşleştirilmiş Bootstrap Farkları (%95 GA)",
            "",
            "| Karşılaştırma | `ΔRMSE` (negatif = iyi) | `ΔRecall@3` | `ΔNDCG@3` | `ΔRegret@1` (negatif = iyi) | `Δ(ρ^res)` |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for k in (1, 2, 3, 5, 8, 12):
        for name in (f"k{k}_StaticDOpt_vs_Random50", f"k{k}_AdaptiveOED_vs_Random50", f"k{k}_AdaptiveOED_vs_StaticDOpt"):
            c = wv_c["panel"][name]
            lines.append(
                f"| `{name}` | {fmt_ci(c['rmse_diff'], 2)} | {fmt_ci(c['recall3_diff'])} | "
                f"{fmt_ci(c['ndcg3_diff'])} | {fmt_ci(c['regret1_diff'], 1)} | {fmt_ci(c['res_spearman_diff'])} |"
            )

    lines.extend(
        [
            "",
            "### Zamansal OOD `Random 50x` Yüzdelik Kıyaslaması & Tam-Panel (Sıfır Geri-Dolgu) Alt-Grubu",
            "",
            "| $k$ | `Random50` Medyan RMSE [P10–P90] | `DOpt` Random Panellerin % Kaçını Geçiyor (RMSE / NDCG@3) | `Adaptive` RMSE Geçiş % | Tam-Panel (`DOpt`) Hasta Sayısı (RMSE: DOpt vs Rand50) |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for k in (1, 2, 3, 5, 8, 12):
        rp = res["wave_ood"]["random50_percentiles"][str(k)]
        fp = res["wave_ood"]["full_panel_subgroups_dopt"].get(str(k))
        fp_txt = f"{fp['n_full_panel_patients']}/{fp['n_total_eval']} ({fp['dopt_rmse']:.2f} vs {fp['random_rmse']:.2f})" if fp else "—"
        lines.append(
            f"| {k} | {rp['random_rmse_median']:.2f} [{rp['random_rmse_p10_best']:.2f}–{rp['random_rmse_p90_worst']:.2f}] | "
            f"%{rp['dopt_beats_random_rmse_pct']:.0f} / %{rp['dopt_beats_random_ndcg3_pct']:.0f} | "
            f"%{rp['adaptive_beats_random_rmse_pct']:.0f} | {fp_txt} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. `M3` Mekanizma Ablasyonları ($k \\in \\{3, 5, 8\\}$)",
            "",
            "| Rejim | Panel | $k$ | Model Koşulu | `RMSE` ↓ | `Res R²` ↑ | $\\rho^{\\text{res}}_{\\text{pat}}$ ↑ | `Recall@3` ↑ | `Regret@1` ↓ | `ΔRMSE (M3_full - Koşul)` [%95 GA] |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for reg_name, m_node, c_node in [("CV (3 Seed)", cv_m, cv_c), ("Waves3+4 OOD", wv_m, wv_c)]:
        for ps in ("StaticDOpt", "Random50"):
            for k in ABLATION_K_BUDGETS:
                for abl in ABLATION_NAMES:
                    r = m_node["ablations_by_k"][str(k)][ps][abl]
                    diff_txt = "referans" if abl == "M3_full" else fmt_ci(c_node["ablations"][f"k{k}_{ps}_M3_full_vs_{abl}"]["rmse_diff"], 2)
                    lines.append(
                        f"| {reg_name} | {ps} | {k} | `{abl}` | {r['rmse']:.2f} | {r['residual_r2']:+.3f} | "
                        f"{r['within_patient_residual_spearman']:+.3f} | {r['recall_at_3']:.3f} | {r['regret_at_1']:.1f} | {diff_txt} |"
                    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. `Waves1+2` Eğitiminden Öğrenilen İlk-12 Statik Mini-Panel",
            "",
            "| Sıra | `Static-D-Optimal` İnhibitörü | Aile / Sınıf | `Static-Diversity` İnhibitörü | Aile / Sınıf |",
            "|---:|---|---|---|---|",
        ]
    )
    for d_row, div_row in zip(
        res["learned_panels_waves12"]["static_d_optimal_top12"],
        res["learned_panels_waves12"]["static_diversity_top12"],
    ):
        lines.append(
            f"| {d_row['rank']} | **{d_row['inhibitor']}** | {d_row['family']} | {div_row['inhibitor']} | {div_row['family']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Ardışık Bütçe Bilgi Değeri (VoI — `Static-D-Optimal` & `Adaptive-OED`, CV)",
            "",
            "| Adım | `Static-D-Optimal` `ΔRMSE` [%95 GA] | `Adaptive-OED` `ΔRMSE` [%95 GA] |",
            "|---|---:|---:|",
        ]
    )
    for k_prev, k_next in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 12)):
        c_do = cv_c["voi"][f"StaticDOpt_k{k_prev}_to_k{k_next}"]["rmse_diff"]
        c_ad = cv_c["voi"][f"AdaptiveOED_k{k_prev}_to_k{k_next}"]["rmse_diff"]
        lines.append(f"| `k={k_prev} → k={k_next}` | {fmt_ci(c_do, 2)} | {fmt_ci(c_ad, 2)} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Önceden Kayıtlı Kapı 2 Kararı",
            "",
            f"- **Kapı 2 Sonucu:** **`{g2['status']}`**",
            f"- **`M3` Ablasyon Geçerliliği:** {abl_txt}",
            f"- **`Random 50x`'i Geçen Politikalar ($k \\in {{3,5,8}}$):** `{g2['passing_policies']}`",
            f"- **Sabit vs Adaptif Kararı:** **`{g2['adaptive_vs_static']}`**",
            f"- **Doygunluk Eşiği ($k^*$):** **`k = {g2['k_star_saturation']}`**",
            "",
        ]
    )
    return "\n".join(lines)


def render_sprint03_v2_md(res: dict[str, Any]) -> str:
    """v2 ek bölümleri: metrik notları, W3+4 tam-panel raporu, keşifsel ortak havuz ve v2 ablasyonları."""
    notes = res["metric_notes"]
    lines = [
        "",
        "---",
        "",
        "## v2 Ekleri (Opus DEV-OPUS-01..11 — keşifsel bölümler v1 kararlarını değiştirmez)",
        "",
        "- Bu rapor `sprint03_v2_results.json` / `sprint03_v2_vectors.npz` çıktısıdır; v1 (`sprint03_results.json`) korunmuştur.",
        "- Kapı 2 kararı dondurulmuş protokol kuralıyla üretilir (W3+4 ablasyonu artık protokol §5.1'deki gibi %95 GA ile).",
        "- `ADAPTIVE_WINS` etiketi v1 asimetrik kuralıdır; simetrik Holm yeniden analizi için `sprint-03-v2-reanalysis.md`'ye bakın.",
        f"- **RMSE:** {notes['rmse']}",
        f"- **Güven aralıkları:** {notes['confidence_intervals']}",
        f"- **Sprint 2 sızıntı GA'sı:** {notes['sprint02_leakage_ci']}",
        "",
        "### v2.1 W3+4 Tam Panel / Doğrudan İsabet / Backfill (Sol §3.2/2)",
        "",
        "| Hücre | Tam panel | Doğrudan isabet | Backfill slot oranı | Politika RMSE (tam panel) | Random50 RMSE (tam panel) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell, e in res["wave_ood"]["full_panel_report"].items():
        perf = (f"{e['policy_rmse']:.2f} | {e['random_rmse']:.2f}" if e["status"] == "hesaplandı"
                else "hesaplanamadı | hesaplanamadı")
        lines.append(f"| `{cell}` | {e['n_full_panel_patients']}/{e['n_total_eval']} | "
                     f"{e['top_k_direct_hit_rate'] * 100:.1f}% | {e['backfilled_slot_rate'] * 100:.1f}% | {perf} |")

    com = res["wave_ood_common_pool_exploratory"]
    lines += [
        "",
        "### v2.2 Keşifsel: W3+4 Ortak Prob Havuzu Duyarlılığı",
        "",
        f"- {com['definition']} Seçim havuzu: {com['n_selection_pool_probes']} prob; çıkarılanlar: "
        + ", ".join(f"`{d}`" for d in com["excluded_train_pool_drugs"]) + ".",
        "- Modeller tam eğitim havuzuyla fit edilir; yalnız seçim havuzu daraltılır. Test AUC değerleri kullanılmaz.",
        "",
        "| $k$ | Politika | `n_eval` | `RMSE` | `NDCG@3` | ΔRMSE vs Random50 [%95 GA] | Tam panel |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for k_str, pdict in com["metrics"]["by_k"].items():
        for pol in POLICIES:
            r = pdict[pol]
            comp = com["comparisons"]["panel"].get(f"k{k_str}_{pol}_vs_Random50")
            full = com["full_panel_report"].get(f"k{k_str}_{pol}")
            full_txt = f"{full['n_full_panel_patients']}/{full['n_total_eval']}" if full else "—"
            comp_txt = fmt_ci(comp["rmse_diff"], 2) if comp else "—"
            lines.append(
                f"| {k_str} | `{pol}` | {r['n_eval']} | {r['rmse']:.2f} | {r['ndcg_at_3']:.3f} | {comp_txt} | {full_txt} |"
            )

    lines += [
        "",
        "### v2.3 Keşifsel Ablasyon Duyarlılığı (OPUS-002, Sol §3.2/4)",
        "",
        "- `M3_offset_only`: M1 öncülü + prob artıklarından eğitimde seçilmiş daraltmayla tek hasta ofseti (tüm hedeflere eşit).",
        "- `M3_perm_offset_fixed`: ofset sabit, eğitim ölçeğiyle standardize artıklar 20 sabit çekilişle karıştırılır (duyarlılık özeti; resmî p-değeri değildir).",
        "",
        "| Rejim | $k$ | M3_full (DOpt) | offset-only (DOpt) | Ofset payı (DOpt / Rand) | Full − offset-only ΔRMSE [%95 GA] | perm-offset-fixed RMSE ort ± sd [min, max] |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for regime, label in (("cv_5fold_3seeds", "CV"), ("wave_ood", "W3+4")):
        full_metrics = res[regime]["metrics"]["ablations_by_k"]
        for k_str, a in res["ablations_v2_exploratory"][regime].items():
            d = a["perm_offset_fixed_draws"]["rmse"]
            sh = a["offset_share_of_M3_rmse_gain"]
            lines.append(
                f"| {label} | {k_str} | {full_metrics[k_str]['StaticDOpt']['M3_full']['rmse']:.2f} | "
                f"{a['metrics']['StaticDOpt/M3_offset_only']['rmse']:.2f} | "
                f"{sh['StaticDOpt'] * 100:.0f}% / {sh['Random50'] * 100:.0f}% | "
                f"{fmt_ci(a['comparisons']['StaticDOpt_M3_full_vs_M3_offset_only']['rmse_diff'], 2)} | "
                f"{d['mean']:.2f} ± {d['sd']:.2f} [{d['min']:.2f}, {d['max']:.2f}] |"
            )
    lines.append("")
    return "\n".join(lines)
