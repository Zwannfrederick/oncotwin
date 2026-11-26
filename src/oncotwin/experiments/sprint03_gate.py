"""Sprint 3 Kapı 2 karar kuralları (dondurulmuş PROTOCOL_SPRINT03.md §5)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.oncotwin.experiments.sprint03_report import ABLATION_NAMES


def evaluate_gate2(
    cv_comps: dict[str, Any],
    wave_comps: dict[str, Any],
) -> dict[str, Any]:
    """Dondurulmuş PROTOCOL_SPRINT03.md Bölüm 5 Kapı 2 kriterlerini mekanik olarak değerlendirir."""
    # 1. Ablasyon geçerliliği (k in {5, 8}): protokol §5.1 gereği hem CV hem W3+4'te %95 GA 0'ı dışlamalı
    abl_ok = True
    abl_failures: list[str] = []
    for k in (5, 8):
        for ps in ("StaticDOpt", "Random50"):
            for abl in ABLATION_NAMES[1:]:
                c_cv = cv_comps["ablations"][f"k{k}_{ps}_M3_full_vs_{abl}"]
                c_wv = wave_comps["ablations"][f"k{k}_{ps}_M3_full_vs_{abl}"]
                ok_cv = c_cv["rmse_diff"]["ci_high"] < 0.0 and c_cv["res_spearman_diff"]["ci_low"] > 0.0
                ok_wv = c_wv["rmse_diff"]["ci_high"] < 0.0 and c_wv["res_spearman_diff"]["ci_low"] > 0.0
                if not (ok_cv and ok_wv):
                    abl_ok = False
                    abl_failures.append(f"k{k}_{ps}_{abl}")

    # 2. Panel seçimi vs Random 50x (k in {3, 5, 8})
    passing_policies: list[str] = []
    for k in (3, 5, 8):
        for pol in ("StaticDiversity", "StaticDOpt", "AdaptiveOED"):
            c_cv = cv_comps["panel"][f"k{k}_{pol}_vs_Random50"]
            c_wv = wave_comps["panel"][f"k{k}_{pol}_vs_Random50"]
            rmse_cv_ok = c_cv["rmse_diff"]["ci_high"] < 0.0
            rank_cv_ok = (
                c_cv["ndcg3_diff"]["ci_low"] > 0.0
                or c_cv["recall3_diff"]["ci_low"] > 0.0
                or c_cv["regret1_diff"]["ci_high"] < 0.0
            )
            wv_dir_ok = c_wv["rmse_diff"]["diff"] < 0.0
            if rmse_cv_ok and rank_cv_ok and wv_dir_ok:
                passing_policies.append(f"{pol}(k={k})")

    # 3. Adaptive vs Static-D-Optimal alt-kararı (v1 kuralı, k in {3,5,8}; asimetrik -> OPUS-001,
    #    simetrik Holm yeniden analizi sprint03_reanalysis.py içinde)
    adapt_wins = False
    static_wins = False
    for k in (3, 5, 8):
        c_ad = cv_comps["panel"][f"k{k}_AdaptiveOED_vs_StaticDOpt"]
        if c_ad["rmse_diff"]["ci_high"] < 0.0 or c_ad["ndcg3_diff"]["ci_low"] > 0.0 or c_ad["regret1_diff"]["ci_high"] < 0.0:
            adapt_wins = True
        if c_ad["rmse_diff"]["ci_low"] > 0.0 and c_ad["regret1_diff"]["ci_low"] > 0.0:
            static_wins = True
    if adapt_wins and not static_wins:
        adaptive_vs_static = "ADAPTIVE_WINS"
    elif static_wins and not adapt_wins:
        adaptive_vs_static = "STATIC_WINS"
    else:
        adaptive_vs_static = "STATIC_EQUIVALENT"

    # 4. Doygunluk bütçesi k* (StaticDOpt üzerinde adım başı < 1.0 AUC veya GA 0'a dokunma)
    k_star = 12
    for k_prev, k_next in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 12)):
        step_diff = cv_comps["voi"][f"StaticDOpt_k{k_prev}_to_k{k_next}"]["rmse_diff"]
        per_probe_gain = -step_diff["diff"] / float(k_next - k_prev)
        if step_diff["ci_high"] >= 0.0 or per_probe_gain < 1.0:
            k_star = k_prev
            break

    status = "PASS" if (abl_ok and len(passing_policies) > 0) else "FAIL"
    return {
        "status": status,
        "ablation_sanity_passed": abl_ok,
        "ablation_failures": abl_failures,
        "passing_policies": passing_policies,
        "adaptive_vs_static": adaptive_vs_static,
        "k_star_saturation": k_star,
    }


# Sol §3.2/3 (keşifsel): k={3,5,8} x {RMSE, NDCG@3, Regret@1}, iki yönlü, 9 p-değerine Holm.
# Adaptif lehine yön: RMSE farkı < 0, NDCG@3 farkı > 0, Regret@1 farkı < 0 (fark = Adaptive - StaticDOpt).
HOLM_FAMILY = tuple((k, m) for k in (3, 5, 8) for m in ("rmse_diff", "ndcg3_diff", "regret1_diff"))
ADAPTIVE_SIGN = {"rmse_diff": -1.0, "ndcg3_diff": +1.0, "regret1_diff": -1.0}


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni düzeltilmiş p-değerleri (giriş sırasında)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out


def adaptive_vs_static_symmetric_holm(comps: dict[str, Any], alpha: float = 0.05) -> dict[str, Any]:
    """Simetrik, çokluk-düzeltmeli Adaptive-vs-StaticDOpt etiketi (eski asimetrik kural yerine keşifsel)."""
    tests = []
    for k, m in HOLM_FAMILY:
        c = comps["panel"][f"k{k}_AdaptiveOED_vs_StaticDOpt"][m]
        if c["n_paired"] > 0 and np.isfinite(c["p_two_sided"]):
            tests.append({"k": k, "metric": m, "diff": c["diff"], "ci_low": c["ci_low"],
                          "ci_high": c["ci_high"], "p_raw": c["p_two_sided"]})
    adj = holm_adjust(np.array([t["p_raw"] for t in tests])) if tests else np.array([])
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)
        t["significant"] = bool(pa < alpha)
        t["favors"] = "AdaptiveOED" if t["diff"] * ADAPTIVE_SIGN[t["metric"]] > 0 else "StaticDOpt"
    sig = {t["favors"] for t in tests if t["significant"]}
    label = ("INSUFFICIENT_EVIDENCE" if not sig else "MIXED" if len(sig) == 2
             else "ADAPTIVE_FAVORED" if sig == {"AdaptiveOED"} else "STATIC_FAVORED")
    return {"label": label, "alpha": alpha, "n_tests": len(tests), "tests": tests,
            "note": "Keşifsel (Sprint 3 sonrası tanımlandı); pratik eşdeğerlik iddiası içermez."}
