"""Opus — docs/manuscript/oncotwin_v1_preprint.md sayılarının arşiv çıktılarına karşı doğrulanması (salt okunur).

Her iddia makaledeki yazılış hassasiyetinde (ondalık basamak) kaynağına yuvarlanarak karşılaştırılır.
Kullanım: PYTHONPATH=. python3 audits/opus-2026-11-26-manuscript/verify_manuscript_numbers.py [--with-fimm]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

P = Path("data/processed")
J = {n: json.loads((P / f"{n}.json").read_text()) for n in (
    "sprint01_audit_summary", "sprint02_results", "sprint03_results", "sprint03_v2_results", "sprint03_v2_reanalysis",
    "sprint04_v2_results", "sprint04_v2_dev_opus19_results", "sprint05_results", "sprint05_fimm_catalog_audit")}
J["opus028"] = json.loads(Path("audits/sol-sprint06/opus028_beataml_probe_scale.json").read_text())
Z5 = np.load(P / "sprint05_vectors.npz")
RESULTS: list[tuple[str, str, str, object]] = []


def g(src: str, path: str):
    node = J[src]
    for part in path.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def chk(label: str, claimed: str, actual, scale: float = 1.0) -> None:
    """claimed: makaledeki metin ('−7.29', '93.4', '91'); yuvarlama hassasiyeti metinden okunur."""
    c = claimed.replace("−", "-").replace("+", "").replace(",", "")
    dp = len(c.split(".")[1]) if "." in c else 0
    a = float(actual) * scale
    ok = round(a, dp) == round(float(c), dp) or abs(a - float(c)) <= 0.5 * 10 ** (-dp) + 1e-9
    RESULTS.append(("OK" if ok else "FAIL", label, claimed, f"{a:.6g}"))


def ci(label: str, claimed: tuple[str, str, str], node: dict, scale: float = 1.0) -> None:
    for name, key, cl in zip(("diff", "lo", "hi"), ("diff", "ci_low", "ci_high"), claimed):
        if cl is not None:
            chk(f"{label}.{name}", cl, node[key], scale)


def wilson(x: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# ---------- Kohort ve paydalar (Özet, §2.1–2.2) ----------
s1 = "sprint01_audit_summary"
chk("S1 specimens", "942", g(s1, "leakage_and_units/n_specimens"))
chk("S1 patients", "805", g(s1, "leakage_and_units/n_patients"))
chk("S1 multi-specimen patients", "108", g(s1, "leakage_and_units/specimens_per_patient/patients_with_multiple_specimens"))
co_w = g(s1, "leakage_and_units/cross_wave_patient_overlap")
chk("S1 W12&W34 direct (r2)", "13", co_w["waves12_and_waves34"])
chk("S1 Both&W12 (r2)", "4", co_w["both_and_waves12"])
chk("S1 Both&W34 (r2)", "1", co_w["both_and_waves34"])
chk("S1 cross-wave patients", "18", g(s1, "leakage_and_units/cross_wave_patient_overlap/total_patients_in_multiple_cohort_labels"))
chk("S1 drug+RNA specimens", "520", g(s1, "modality_overlap/drug_and_norm_exp_specimens"))
chk("S1 drug+RNA patients", "479", g(s1, "modality_overlap/drug_and_norm_exp_patients"))
fam = g("sprint05_results", "track_a_m4/drug_family_graph")
chk("families annotated drugs", "117", fam["n_annotated_drugs"])
chk("families total drugs", "166", fam["n_drugs_total"])
chk("families", "86", fam["n_families"])
chk("family edges", "395", fam["n_incidence_edges"])
chk("perm draws", "20", fam["n_perm_draws"])
fa = J["sprint05_fimm_catalog_audit"]
chk("FIMM File_0 specimens", "252", fa["sample_counts"]["file0_total_aml_specimens"])
chk("FIMM File_0 patients", "186", fa["sample_counts"]["file0_unique_aml_patients"])
chk("FIMM File_3.1 columns", "181", fa["sample_counts"]["file3_sdss_columns_total"])
chk("FIMM compounds", "515", fa["sample_counts"]["file3_sdss_compounds"])
chk("FIMM File_7 columns", "167", fa["sample_counts"]["file7_rna_columns_total"])
chk("FIMM matched specimens", "132", fa["sample_counts"]["matched_aml_rna_sdss_specimens"])
chk("FIMM matched patients", "109", fa["sample_counts"]["matched_aml_rna_sdss_unique_patients"])
co = fa["compound_overlap"]
chk("matched compounds", "102", co["matched_unique_compounds"])
chk("shared targets", "23", co["common_targets_count"])
chk("shared probes", "64", co["common_probes_from_all471_pool_count"])
hc = co["target_high_conc_comparison"]
chk("targets with differing max conc", "11", hc["fimm_lower_max_conc"] + hc["fimm_higher_max_conc"])
ratios = {t["beataml_name"]: t["high_conc_ratio_fimm_over_beataml"] for t in co["target_details"]}
chk("max-conc ratio min (0.01x)", "0.01", min(v for v in ratios.values() if v))
chk("max-conc ratio max (10x)", "10", max(v for v in ratios.values() if v))
tram = [v for k, v in ratios.items() if k.startswith("Trametinib")]
rapa = [v for k, v in ratios.items() if k.startswith("Rapamycin")]
chk("Trametinib 40x lower", "40", 1 / tram[0])
chk("Rapamycin 100x lower", "100", 1 / rapa[0])
tc = g("sprint05_results", "track_c_fimm")
chk("common genes", "14688", tc["common_genes_count"])
s4 = "sprint04_v2_results"
for rn, cnt in (("id_60_20_20", ("284", "94", "93", "90", "91")), ("wave_ood", ("213", "71", "165", "70", "148"))):
    c = g(s4, f"regimes/{rn}/counts")
    for key, cl in zip(("train_tune", "calibration", "test", "archive_eval_cal", "archive_eval_test"), cnt):
        chk(f"S4 {rn} {key}", cl, c[key])
chk("S2 CV n_eval", "448", g("sprint02_results", "cv_5fold_3seeds/metrics/5/M3/n_eval"))
chk("S3v2 CV n_eval", "427", g("sprint03_v2_results", "cv_5fold_3seeds/metrics/by_k/5/StaticDOpt/n_eval"))
chk("S3v2 W34 n_eval", "146", g("sprint03_v2_results", "wave_ood/metrics/by_k/5/StaticDOpt/n_eval"))

# ---------- §3.1 ve Tablo 1 ----------
s2 = "sprint02_results"
ci("leakage clean_wave_perm", ("+0.013", None, None), {"diff": g(s2, "permutation_leakage_check/clean_wave_perm/per_drug_spearman")})
chk("leakage CI lo", "−0.052", g(s2, "permutation_leakage_check/clean_wave_perm/per_drug_spearman_ci/0"))
chk("leakage CI hi", "+0.079", g(s2, "permutation_leakage_check/clean_wave_perm/per_drug_spearman_ci/1"))
for arm, k, rmse, ndcg in (("M0", 0, "55.85", "0.785"), ("M1", 0, "49.07", "0.855"), ("M3", 5, "44.78", "0.869")):
    chk(f"T1 S2 {arm} RMSE", rmse, g(s2, f"cv_5fold_3seeds/metrics/{k}/{arm}/rmse"))
    chk(f"T1 S2 {arm} NDCG", ndcg, g(s2, f"cv_5fold_3seeds/metrics/{k}/{arm}/ndcg_at_3"))
ci("T1 S2 M3-M1", ("−4.29", "−4.98", "−3.60"), g(s2, "cv_5fold_3seeds/paired_bootstrap/k5_M3_vs_M1/rmse_diff"))
s3 = "sprint03_v2_results"
for reg, arm, vals in (("cv_5fold_3seeds", "Random50", ("45.11", "0.871", "24.0")),
                       ("cv_5fold_3seeds", "StaticDOpt", ("42.74", "0.876", "23.8")),
                       ("cv_5fold_3seeds", "AdaptiveOED", ("42.98", "0.881", "21.6")),
                       ("wave_ood", "StaticDOpt", ("50.81", "0.871", "28.9"))):
    m = g(s3, f"{reg}/metrics/by_k/5/{arm}")
    chk(f"T1 S3 {reg} {arm} RMSE", vals[0], m["rmse"])
    chk(f"T1 S3 {reg} {arm} NDCG", vals[1], m["ndcg_at_3"])
    chk(f"T1 S3 {reg} {arm} Reg1", vals[2], m["regret_at_1"])
ci("T1 S3 CV Static-Random", ("−2.37", "−2.84", "−1.89"), g(s3, "cv_5fold_3seeds/comparisons/panel/k5_StaticDOpt_vs_Random50/rmse_diff"))
ci("T1 S3 CV Adaptive-Static", ("+0.24", "−0.33", "+0.81"), g(s3, "cv_5fold_3seeds/comparisons/panel/k5_AdaptiveOED_vs_StaticDOpt/rmse_diff"))
ci("T1 S3 W34 Static-Random", ("−1.81", "−2.94", "−0.65"), g(s3, "wave_ood/comparisons/panel/k5_StaticDOpt_vs_Random50/rmse_diff"))
for rn, vals, cis in (("id_60_20_20", ("39.84", "0.887", "13.7"), ("−7.29", "−9.45", "−5.42")),
                      ("wave_ood", ("51.99", "0.873", "24.6"), ("−4.18", "−5.86", "−2.66"))):
    m = g(s4, f"regimes/{rn}/scenarios/archive_available/arms/StaticDOpt_k5/test_metrics")
    chk(f"T1 S4 {rn} RMSE", vals[0], m["rmse"])
    chk(f"T1 S4 {rn} NDCG", vals[1], m["ndcg_at_3"])
    chk(f"T1 S4 {rn} Reg1", vals[2], m["regret_at_1"])
    ci(f"T1 S4 {rn} Static-M1", cis, g(s4, f"regimes/{rn}/comparisons/archive_available/policy_pairs/StaticDOpt_k5_vs_M1_k0/rmse_diff"))
rea = J["sprint03_v2_reanalysis"]
RESULTS.append(("OK" if rea["cv_5fold_3seeds"]["label"] == rea["wave_ood"]["label"] == "INSUFFICIENT_EVIDENCE" else "FAIL",
                "Holm labels", "INSUFFICIENT_EVIDENCE x2", f'{rea["cv_5fold_3seeds"]["label"]}/{rea["wave_ood"]["label"]}'))
chk("W34 k8 Adaptive RMSE", "49.61", g(s3, "wave_ood/metrics/by_k/8/AdaptiveOED/rmse"))
chk("W34 k8 Static RMSE", "50.56", g(s3, "wave_ood/metrics/by_k/8/StaticDOpt/rmse"))
chk("W34 k8 Adaptive NDCG", "0.868", g(s3, "wave_ood/metrics/by_k/8/AdaptiveOED/ndcg_at_3"))
chk("W34 k8 Static NDCG", "0.879", g(s3, "wave_ood/metrics/by_k/8/StaticDOpt/ndcg_at_3"))
for rn, cl in (("id_60_20_20", "19"), ("wave_ood", "1")):
    chk(f"locked k5 full panel {rn}", cl, g(s4, f"regimes/{rn}/scenarios/locked_train_catalog/arms/StaticDOpt_k5/availability/n_complete_test"))
chk("locked k5 ID target-eligible", "93", g(s4, "regimes/id_60_20_20/scenarios/locked_train_catalog/arms/StaticDOpt_k5/availability/n_target_eligible_test"))
chk("locked k5 W34 target-eligible", "163", g(s4, "regimes/wave_ood/scenarios/locked_train_catalog/arms/StaticDOpt_k5/availability/n_target_eligible_test"))
ci("offset-only ID", ("−3.70", "−4.48", "−2.96"), g(s3, "ablations_v2_exploratory/cv_5fold_3seeds/5/comparisons/StaticDOpt_M3_full_vs_M3_offset_only/rmse_diff"))
ci("offset-only W34", ("−2.46", "−3.89", "−1.04"), g(s3, "ablations_v2_exploratory/wave_ood/5/comparisons/StaticDOpt_M3_full_vs_M3_offset_only/rmse_diff"))

# ---------- §3.2 Tablo 2 ve A2 ----------
A = "sprint05_results"
rhos = g(A, "track_a_m4/regimes/id_cv_5fold_3seed/experiment_a1_fixed_panel/k5/selected_rho_m4_values")
chk("rho>0 folds (k5)", "13", sum(r > 0 for r in rhos))
t2 = (("id_cv_5fold_3seed", "427", ("42.91", "42.92", "42.90"), ("+0.011", "−0.010", "+0.033"), "0.294", ("+0.017", "−0.005", "+0.039"), "0.128"),
      ("id_60_20_20", "91", ("39.84", "39.82", "39.84"), ("−0.02", "−0.06", "+0.01"), "0.262", ("−0.02", "−0.05", "+0.00"), "0.128"),
      ("wave_ood_s3", "146", ("50.64", "50.62", "50.63"), ("−0.02", "−0.05", "+0.01"), "0.156", ("−0.01", "−0.04", "+0.02"), "0.438"))
for reg, n, rm, d1, p1, d2, p2 in t2:
    node = g(A, f"track_a_m4/regimes/{reg}/experiment_a1_fixed_panel/k5")
    chk(f"T2 {reg} n", n, node["models"]["M3"]["n_eval"])
    for arm, cl in zip(("M3", "M4", "M4_perm_null"), rm):
        chk(f"T2 {reg} {arm} RMSE", cl, node["models"][arm]["rmse"])
    ci(f"T2 {reg} M4-M3", d1, node["comparisons"]["M4_vs_M3"]["rmse_diff"])
    chk(f"T2 {reg} M4-M3 p", p1, node["comparisons"]["M4_vs_M3"]["rmse_diff"]["p_two_sided"])
    ci(f"T2 {reg} M4-perm", d2, node["comparisons"]["M4_vs_M4_perm_null"]["rmse_diff"])
    chk(f"T2 {reg} M4-perm p", p2, node["comparisons"]["M4_vs_M4_perm_null"]["rmse_diff"]["p_two_sided"])
c2 = g(A, "track_c_fimm/experiment_c2_within_fimm_replication")
for arm, cl in (("M3_StaticDOpt_k5", "3.83"), ("M4_StaticDOpt_k5", "3.83"), ("M4_perm_null_k5", "3.83")):
    chk(f"T2 C2 {arm} RMSE", cl, c2["arms"][arm]["rmse"])
ci("T2 C2 M4-M3", ("−0.00", "−0.01", "+0.01"), c2["comparisons"]["M4_Static_k5_vs_M3_Static_k5"]["rmse_diff"])
chk("T2 C2 M4-M3 p", "0.444", c2["comparisons"]["M4_Static_k5_vs_M3_Static_k5"]["rmse_diff"]["p_two_sided"])
ci("T2 C2 M4-perm", ("−0.00", "−0.01", "+0.00"), c2["comparisons"]["M4_Static_k5_vs_M4_perm_null_k5"]["rmse_diff"])
chk("T2 C2 M4-perm p", "0.196", c2["comparisons"]["M4_Static_k5_vs_M4_perm_null_k5"]["rmse_diff"]["p_two_sided"])
a2 = g(A, "track_a_m4/regimes/id_cv_5fold_3seed/experiment_a2_reselected_panel/k5")
ci("A2 ID M4resel-M3", ("−0.10", "−0.27", "+0.08"), a2["M4_reselected_vs_M3"]["rmse_diff"])
chk("A2 ID M3 Reg1", "23.3", a2["M3_reference_panel"]["regret_at_1"])
chk("A2 ID M4 Reg1", "24.9", a2["M4_reselected_panel"]["regret_at_1"])
a2w = g(A, "track_a_m4/regimes/wave_ood_s3/experiment_a2_reselected_panel/k5")
ci("A2 W34 M4resel-M3", ("+0.16", "+0.01", "+0.35"), a2w["M4_reselected_vs_M3"]["rmse_diff"])
chk("A2 W34 p", "0.044", a2w["M4_reselected_vs_M3"]["rmse_diff"]["p_two_sided"])

# ---------- §3.3 Tablo 3, çekinme, Ek 19 ----------
for rn, cov, w in (("id_60_20_20", ("79", "91", "86.8"), "241.8"), ("wave_ood", ("110", "148", "74.3"), "247.4")):
    cf = g(s4, f"regimes/{rn}/scenarios/archive_available/arms/StaticDOpt_k5/conformal")
    chk(f"S4 {rn} coverage count", cov[0], round(cf["simultaneous_coverage"] * cf["n_eval"]))
    chk(f"S4 {rn} n_eval", cov[1], cf["n_eval"])
    chk(f"S4 {rn} coverage %", cov[2], cf["simultaneous_coverage"], 100)
    chk(f"S4 {rn} width", w, cf["width_sim"])
lo, hi = wilson(79, 91)
chk("Wilson 79/91 lo", "78.4", lo)
chk("Wilson 79/91 hi", "92.3", hi)
T3 = {
    "id_60_20_20": {
        "EqualWidthAUC": ("width_sim", "241.8", 91, "100.0", "18.70", "20.0", "98.7", "0.0", "116.77"),
        "RankConformal": ("k_hat", "6", 85, "93.4", "6.00", "6.0", "35.3", "0.0", "66.03"),
        "RelRankConformal": ("q_rel", "0.429", 85, "93.4", "8.64", "9.0", "45.6", "0.0", "79.39"),
        "Top1GapConformal": ("m_hat", "74.2", 84, "92.3", "5.45", "5.0", "33.6", "9.9", "53.83")},
    "wave_ood": {
        "EqualWidthAUC": ("width_sim", "247.4", 148, "100.0", "21.41", "22.0", "98.7", "0.0", "130.90"),
        "RankConformal": ("k_hat", "6", 137, "92.6", "6.00", "6.0", "29.1", "0.0", "73.51"),
        "Top1GapConformal": ("m_hat", "63.6", 128, "86.5", "4.76", "4.0", "23.1", "16.9", "52.04")}}
for rn, methods in T3.items():
    reg = g(A, f"track_b_shortlist/regimes/{rn}")
    chk(f"T3 {rn} n_cal", "90" if rn == "id_60_20_20" else "70", reg["n_cal"])
    chk(f"T3 {rn} n_test", "91" if rn == "id_60_20_20" else "148", reg["n_test"])
    for meth, (tk, tv, top, topp, size, med, frac, sing, mreg) in methods.items():
        node = reg["arms"]["StaticDOpt_k5"]["methods"][meth]
        t = node["test"]
        chk(f"T3 {rn} {meth} threshold", tv, node["calibration"][tk])
        chk(f"T3 {rn} {meth} top1 count", str(top), round(t["shortlist_contains_true_top1_rate"] * t["n_test"]))
        chk(f"T3 {rn} {meth} top1 %", topp, t["shortlist_contains_true_top1_rate"], 100)
        chk(f"T3 {rn} {meth} mean size", size, t["shortlist_mean_size_obs"])
        chk(f"T3 {rn} {meth} median size", med, t["shortlist_median_size_obs"])
        chk(f"T3 {rn} {meth} mean fraction", frac, t["shortlist_mean_fraction_obs"], 100)
        chk(f"T3 {rn} {meth} singleton rate", sing, t["definite_top1_rate_obs"], 100)
        chk(f"T3 {rn} {meth} mean shortlist Reg1", mreg, t["shortlist_mean_regret1"])
for rn, (x, n, lo_c, hi_c) in (("id_60_20_20", (5, 9, "26.7", "81.1")), ("wave_ood", (19, 25, "56.6", "88.5"))):
    base = f"track_b/{rn}/StaticDOpt_k5/Top1GapConformal/"
    d1, ok = Z5[base + "patient_definite_top1"], Z5[base + "patient_shortlist_contains_top1"]
    s = np.isfinite(d1) & (d1 == 1)
    chk(f"singleton correct {rn}", str(x), np.nansum(ok[s]))
    chk(f"singleton n {rn}", str(n), s.sum())
    w = wilson(x, n)
    chk(f"singleton Wilson lo {rn}", lo_c, w[0])
    chk(f"singleton Wilson hi {rn}", hi_c, w[1])
cmp_id = g(A, "track_b_shortlist/regimes/id_60_20_20/arms/StaticDOpt_k5/comparisons/RankConformal_vs_EqualWidthAUC")
size_key = [k for k in cmp_id if "size" in k][0]
ci("Rank-EqualWidth size ID", ("−12.70", "−13.69", "−11.69"), cmp_id[size_key])
chk("Rank-EqualWidth size ID p", "0.002", cmp_id[size_key]["p_two_sided"])
cmp_w = g(A, "track_b_shortlist/regimes/wave_ood/arms/StaticDOpt_k5/comparisons/RankConformal_vs_EqualWidthAUC")
chk("Rank-EqualWidth size W34", "−15.41", cmp_w[size_key]["diff"])
ab = g(s4, "regimes/wave_ood/scenarios/archive_available/arms/StaticDOpt_k5/abstention")
chk("AURC omics W34", "32.44", ab["omics_pca_dist"]["risk_coverage_regret1"]["aurc"])
chk("AURC random W34", "24.60", ab["omics_pca_dist"]["risk_coverage_regret1"]["random_aurc"])
chk("AURC m3 W34", "20.35", ab["m3_post_scale"]["risk_coverage_regret1"]["aurc"])
ag = g(A, "track_b_shortlist/regimes/wave_ood/augrc_sensitivity/StaticDOpt_k5")
chk("AUGRC omics W34", "14.27", ag["omics_pca_dist"]["augrc"])
chk("AUGRC random W34", "12.30", ag["omics_pca_dist"]["random_augrc"])
chk("AUGRC m3 W34", "10.57", ag["m3_post_scale"]["augrc"])
for rn, (acc_oof, acc_new, n) in (("id_60_20_20", ("91", "84", "91")), ("wave_ood", ("146", "65", "148"))):
    tev = g(s4, f"regimes/{rn}/scenarios/archive_available/arms/StaticDOpt_k5/abstention/m3_post_scale/threshold_eval")
    chk(f"m3 OOF accepted {rn}", acc_oof, tev["n_accepted"])
    d19 = g("sprint04_v2_dev_opus19_results", f"regimes/{rn}/archive_available/arms/StaticDOpt_k5/final_fit_threshold_eval")
    chk(f"Add19 accepted {rn}", acc_new, d19["n_accepted"])
    chk(f"Add19 n {rn}", n, d19["n_available"])
d19w = g("sprint04_v2_dev_opus19_results", "regimes/wave_ood/archive_available/arms/StaticDOpt_k5/final_fit_threshold_eval")
chk("Add19 W34 accepted Reg1", "20.6", d19w["accepted_mean_regret1"])
chk("Add19 W34 abstained Reg1", "27.7", d19w["abstained_mean_regret1"])

# ---------- §3.4 Tablo 4, C1/C2, OPUS-028 ----------
c1 = tc["experiment_c1_frozen_transfer"]
TOP8 = c1["reselected_beataml_static_dopt_top8"]
chk("C1 prefix", "1", float(all(t.split(" (")[0] == m for t, m in zip(TOP8[:5], ["Flavopiridol", "Sunitinib", "XAV-939", "CYT387", "Tivozanib"]))))
ix = c1["cohorts"]["index_109_patients"]
T4C1 = {"C1_M0_k0": ("0.571", "0.310", "0.186", "11.23"), "C1_M1_rank_k0": ("0.651", "0.372", "0.258", "7.41"),
        "C1_M3_StaticDOpt_k2": ("0.683", "0.404", "0.270", "6.66"), "C1_M3_StaticDOpt_k5": ("0.683", "0.408", "0.280", "7.18"),
        "C1_M3_Random20x_k5": ("0.649", "0.362", "0.281", "7.44"), "C1_M3_StaticDOpt_k8": ("0.670", "0.390", "0.279", "7.83")}
for arm, (nd, rc, sp, rg) in T4C1.items():
    a = ix["arms"][arm]
    chk(f"T4 {arm} NDCG", nd, a["ndcg_at_3"])
    chk(f"T4 {arm} Recall", rc, a["recall_at_3"])
    chk(f"T4 {arm} Spearman", sp, a["within_patient_spearman"])
    chk(f"T4 {arm} Reg1", rg, a["regret_at_1_sdss"])
T4C2 = {"M0_k0": ("5.27", "0.713", "0.438", "0.000", "6.89"), "M0_k5": ("4.72", "0.713", None, None, None),
        "M1_k0": ("4.83", "0.765", "0.486", "0.325", "4.92"), "M2a_StaticDOpt_k5": ("4.06", "0.797", "0.510", "0.453", "3.33"),
        "M3_StaticDOpt_k5": ("3.83", "0.843", "0.569", "0.521", "2.58"), "M3_Random20x_k5": ("4.29", "0.802", "0.523", "0.420", "3.65")}
for arm, vals in T4C2.items():
    a = c2["arms"][arm]
    for key, cl in zip(("rmse", "ndcg_at_3", "recall_at_3", "within_patient_residual_spearman", "regret_at_1"), vals):
        if cl is not None:
            chk(f"T4 C2 {arm} {key}", cl, a[key])
cm = ix["comparisons"]
ci("C1 k5-M1 NDCG", ("+0.032", "+0.007", "+0.058"), cm["C1_M3_k5_vs_M1_k0"]["ndcg3_diff"])
chk("C1 k5-M1 Spearman p (makale r2)", "0.048", cm["C1_M3_k5_vs_M1_k0"]["spearman_diff"]["p_two_sided"])
chk("C1 k5-M1 NDCG p", "0.016", cm["C1_M3_k5_vs_M1_k0"]["ndcg3_diff"]["p_two_sided"])
ci("C1 k5-M1 Recall (makale: Opus bootstrap)", ("+0.037", "+0.006", "+0.067"), cm["C1_M3_k5_vs_M1_k0"]["recall3_diff"])
ci("C1 k5-Random NDCG", ("+0.033", "+0.013", "+0.054"), cm["C1_M3_Static_k5_vs_Random_k5"]["ndcg3_diff"])
chk("C1 k5-Random NDCG p", "0.004", cm["C1_M3_Static_k5_vs_Random_k5"]["ndcg3_diff"]["p_two_sided"])
ci("C1 k5-M1 Spearman (birincil arşiv; makale r2)", ("+0.022", "+0.000", "+0.045"), cm["C1_M3_k5_vs_M1_k0"]["spearman_diff"])
ci("C1 k5-M1 Reg1", ("−0.23", "−1.26", "+0.76"), cm["C1_M3_k5_vs_M1_k0"]["regret1_sdss_diff"])
ci("C1 k5-Random Reg1", ("−0.26", "−0.93", "+0.41"), cm["C1_M3_Static_k5_vs_Random_k5"]["regret1_sdss_diff"])
cl132 = c1["cohorts"]["all_132_specimens_clustered"]
chk("C1 132 NDCG", "0.680", cl132["arms"]["C1_M3_StaticDOpt_k5"]["ndcg_at_3"])
chk("C1 132 vs M1 p", "0.008", cl132["comparisons"]["C1_M3_k5_vs_M1_k0"]["ndcg3_diff"]["p_two_sided"])
chk("C1 132 vs Random p", "0.002", cl132["comparisons"]["C1_M3_Static_k5_vs_Random_k5"]["ndcg3_diff"]["p_two_sided"])
cc = c2["comparisons"]
for key, cls in (("M3_Static_k5_vs_M1_k0", (("rmse_diff", ("−1.00", "−1.26", "−0.76")), ("ndcg3_diff", ("+0.077", "+0.052", "+0.103")), ("regret1_diff", ("−2.34", "−3.48", "−1.28")))),
                 ("M3_Static_k5_vs_Random20x_k5", (("rmse_diff", ("−0.46", "−0.61", "−0.32")), ("ndcg3_diff", ("+0.041", "+0.026", "+0.056")), ("regret1_diff", ("−1.07", "−1.60", "−0.52"))))):
    for mk, trip in cls:
        ci(f"C2 {key} {mk}", trip, cc[key][mk])
        chk(f"C2 {key} {mk} p", "0.002", cc[key][mk]["p_two_sided"])
chk("C2 n_eval", "109", c2["n_eval"])
o = J["opus028"]
chk("O28 train", "284", o["n_train"])
chk("O28 eval", "91", o["n_eval"])
chk("O28 drugs", "102", o["n_common_drugs"])
chk("O28 targets", "23", o["n_targets"])
chk("O28 probes", "62", o["n_train_only_probes"])
chk("O28 k2 ±1", "91", o["budgets"]["2"]["n_probe_vectors_with_abs_values_all_one"])
pd5, pd2 = o["budgets"]["5"]["paired_diffs"], o["budgets"]["2"]["paired_diffs"]
ci("O28 k5 kprobe-M1 NDCG", ("+0.034", "+0.014", "+0.055"), pd5["k_probe_minus_M1_k0"]["ndcg_at_3"])
ci("O28 k5 kprobe-M1 Reg1 (Sol)", ("−6.91", "−14.88", "+0.21"), pd5["k_probe_minus_M1_k0"]["regret_at_1_auc"])
ci("O28 k5 kprobe-oracle NDCG", ("−0.005", "−0.020", "+0.007"), pd5["k_probe_minus_full_row_oracle"]["ndcg_at_3"])
ci("O28 k5 kprobe-oracle Reg1", ("+2.2", "−3.5", "+8.9"), pd5["k_probe_minus_full_row_oracle"]["regret_at_1_auc"])
ci("O28 k5 kprobe-trainref Reg1 (r2)", ("+1.61", "−3.62", "+7.73"), pd5["k_probe_minus_train_reference"]["regret_at_1_auc"])
ci("O28 k5 kprobe-trainref NDCG", ("−0.010", "−0.021", "+0.001"), pd5["k_probe_minus_train_reference"]["ndcg_at_3"])
ci("O28 k2 kprobe-oracle NDCG", ("−0.018", "−0.032", "−0.005"), pd2["k_probe_minus_full_row_oracle"]["ndcg_at_3"])
ci("O28 k2 kprobe-trainref NDCG", ("−0.016", "−0.031", "−0.002"), pd2["k_probe_minus_train_reference"]["ndcg_at_3"])

if "--with-fimm" in sys.argv:  # JSON'da tutulmayan FIMM sayımları veriden yeniden hesaplanır
    from src.oncotwin.data.dataset import load_beataml_aml_dataset
    from src.oncotwin.data.fimm_dataset import load_fimm_aml_dataset, select_fimm_index_specimen_positions
    ba = load_beataml_aml_dataset()
    f, _ = load_fimm_aml_dataset(ba.drug_names, common_gene_ensembl_ids=[x.split(".")[0] for x in ba.gene_names])
    counts = f.meta["Patient_ID"].value_counts()
    chk("FIMM single-specimen patients", "92", (counts == 1).sum())
    chk("FIMM multi-specimen patients", "17", (counts > 1).sum())
    chk("FIMM mutation columns", "15", len(f.mut_feature_names))
    RESULTS.append(("OK" if "KIT" not in f.mut_feature_names else "FAIL", "KIT absent", "absent", str("KIT" in f.mut_feature_names)))
    fi = f.subset_by_positions(select_fimm_index_specimen_positions(f.meta))
    di = {d: i for i, d in enumerate(fi.drug_names)}
    prefix = [di[d] for d in TOP8]
    for k, cl in ((2, "109"), (5, "98"), (8, "98")):
        chk(f"FIMM prefix complete k={k}", cl, np.isfinite(fi.Y_aligned[:, prefix[:k]]).all(axis=1).sum())

bad = [r for r in RESULTS if r[0] != "OK"]
for r in bad:
    print("FAIL", r[1], "| makale:", r[2], "| kaynak:", r[3])
print(f"\n{len(RESULTS)} kontrol, {len(RESULTS) - len(bad)} OK, {len(bad)} FAIL")
