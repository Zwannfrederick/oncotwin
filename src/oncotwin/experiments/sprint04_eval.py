"""Sprint 4 değerlendirme, çekinme ve kilitli senaryo kıyas yardımcıları."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.metrics import (
    VECTOR_KEYS,
    evaluate_masked_predictions,
    paired_bootstrap_ci,
    paired_metric_diffs,
)
from src.oncotwin.eval.uncertainty import (
    evaluate_conformal_and_shortlist,
    evaluate_locked_threshold_abstention,
    fit_split_conformal_simultaneous,
    selective_risk_coverage_curve,
)


def _safe_eval_masked(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    base_d: np.ndarray,
    valid_mask: np.ndarray,
    target_idx: list[int],
) -> dict[str, Any]:
    """`n_obs=0` iken ValueError yerine `hesaplanamadı` döndürür; hasta vektörlerini tam satır boyuna taşır."""
    n_rows = len(Y_true)
    t_idx = np.asarray(target_idx, dtype=np.int64)
    has_any = int(np.sum(valid_mask)) > 0 and bool(
        np.any(np.isfinite(Y_true[valid_mask][:, t_idx]) & np.isfinite(Y_pred[valid_mask][:, t_idx]))
    )
    if not has_any:
        return {
            "status": "hesaplanamadı",
            "n_obs": 0,
            "n_eval": 0,
            "rmse": float("nan"),
            "obs_rmse": float("nan"),
            "residual_r2": float("nan"),
            "within_patient_residual_spearman": float("nan"),
            "per_drug_spearman": float("nan"),
            "ndcg_at_3": float("nan"),
            "residual_ndcg_at_3": float("nan"),
            "recall_at_3": float("nan"),
            "regret_at_1": float("nan"),
            "regret_at_3": float("nan"),
            "vectors": {k: np.full(n_rows, np.nan, dtype=np.float64) for k in VECTOR_KEYS},
        }
    ev = evaluate_masked_predictions(Y_true, Y_pred, base_d, valid_mask, target_idx, fold_ids=None)
    lifted: dict[str, np.ndarray] = {}
    for vk in VECTOR_KEYS:
        arr = np.full(n_rows, np.nan, dtype=np.float64)
        arr[valid_mask] = ev["vectors"][vk]
        lifted[vk] = arr
    ev["status"] = "ok"
    ev["vectors"] = lifted
    return ev


def _evaluate_arm_outputs(
    ds_cal: BeatAMLDataset,
    ds_te: BeatAMLDataset,
    base_d_cal: np.ndarray,
    base_d_te: np.ndarray,
    Yp_cal: np.ndarray,
    Yp_te: np.ndarray,
    m3_te: np.ndarray,
    omics_te: np.ndarray,
    valid_cal: np.ndarray,
    valid_te: np.ndarray,
    avail_rc_mask_te: np.ndarray,
    target_idx: list[int],
    tau_omics: float,
    tau_m3: float,
    alpha: float,
    coverage_cap: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Tek bir (senaryo, kol) için conformal, kısa liste, kilitli eşik çekinme ve AURC çıktılarını üretir."""
    ev_cal = _safe_eval_masked(ds_cal.Y_auc, Yp_cal, base_d_cal, valid_cal, target_idx)
    ev_te = _safe_eval_masked(ds_te.Y_auc, Yp_te, base_d_te, valid_te, target_idx)

    conf_fit = fit_split_conformal_simultaneous(
        ds_cal.Y_auc, Yp_cal, target_idx, alpha=alpha, valid_mask=valid_cal
    )
    conf_te = evaluate_conformal_and_shortlist(
        ds_te.Y_auc, Yp_te, target_idx, q_sim=conf_fit["q_sim"], q_per_drug=conf_fit["q_per_drug"], valid_mask=valid_te
    )

    # Risk–kapsam yalnızca hedef-uygun hastalar üzerinde hesaplanır (archive için valid_te = tüm eval_te)
    sc_om = omics_te[avail_rc_mask_te]
    sc_m3 = m3_te[avail_rc_mask_te]
    reg_rc = ev_te["vectors"]["patient_regret1"][avail_rc_mask_te]
    rsq_rc = ev_te["vectors"]["patient_rmse_sq"][avail_rc_mask_te]
    cov_rc = conf_te["vectors"]["patient_covered_sim"][avail_rc_mask_te]
    comp_rc = valid_te[avail_rc_mask_te]

    abst: dict[str, Any] = {}
    for s_key, sc_arr, tau in (("omics_pca_dist", sc_om, tau_omics), ("m3_post_scale", sc_m3, tau_m3)):
        abst[s_key] = {
            "tau_locked": float(tau),
            "threshold_eval": evaluate_locked_threshold_abstention(
                sc_arr, reg_rc, rsq_rc, cov_rc, tau_locked=tau, available_mask=comp_rc
            ),
            "risk_coverage_regret1": selective_risk_coverage_curve(
                sc_arr, reg_rc, available_mask=comp_rc, coverage_cap=coverage_cap
            ),
            "risk_coverage_rmse_sq": selective_risk_coverage_curve(
                sc_arr, rsq_rc, available_mask=comp_rc, coverage_cap=coverage_cap
            ),
        }

    node = {
        "cal_n_eval": ev_cal["n_eval"],
        "test_metrics": {k: v for k, v in ev_te.items() if k not in ("vectors", "drug_spearmans")},
        "conformal": {
            "alpha": float(alpha),
            "n_cal": conf_fit["n_cal"],
            "order_stat_rank": conf_fit["order_stat_rank"],
            "q_sim": conf_fit["q_sim"],
            "width_sim": conf_fit["width_sim"],
            "finite_interval_published": conf_fit["finite_interval_published"],
            **{k: v for k, v in conf_te.items() if k != "vectors"},
        },
        "abstention": abst,
    }
    vecs = {
        **ev_te["vectors"],
        "patient_max_abs_err": conf_te["vectors"]["patient_max_abs_err"],
        "patient_covered_sim": conf_te["vectors"]["patient_covered_sim"],
        "patient_shortlist_size_obs": conf_te["vectors"]["patient_shortlist_size_obs"],
        "patient_shortlist_contains_top1": conf_te["vectors"]["patient_shortlist_contains_top1"],
        "m3_post_scale": m3_te,
        "valid_mask": valid_te,
    }
    return node, vecs


def _compare_pair_node(va: dict[str, np.ndarray], vb: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, Any]:
    out = paired_metric_diffs({"vectors": va}, {"vectors": vb}, n_boot=n_boot)
    out["shortlist_size_diff"] = paired_bootstrap_ci(
        va["patient_shortlist_size_obs"], vb["patient_shortlist_size_obs"], n_boot=n_boot, seed=seed
    )
    return out


def _build_locked_comparisons(
    lock_vecs: dict[str, dict[str, np.ndarray]],
    omics_te: np.ndarray,
    k_budgets: tuple[int, ...],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Kilitli senaryoda yalnızca ortak tam-panel hastalar üzerinde eşleştirilmiş kıyas kurar (n<5 -> hesaplanamadı)."""
    pair_out: dict[str, Any] = {}
    three_way: dict[str, Any] = {}
    for k in k_budgets:
        for a, b in (("AdaptiveOED", "EqualWeightVarRed"), ("AdaptiveOED", "StaticDOpt"), ("EqualWeightVarRed", "StaticDOpt")):
            va, vb = lock_vecs[f"{a}_k{k}"], lock_vecs[f"{b}_k{k}"]
            common = va["valid_mask"] & vb["valid_mask"]
            n_p = int(np.sum(common))
            if n_p < 5:
                pair_out[f"k{k}_{a}_vs_{b}"] = {
                    "status": "hesaplanamadı",
                    "reason": "insufficient_common_complete_patients (n < 5)",
                    "n_paired": n_p,
                }
            else:
                va_c = {kk: vv[common] for kk, vv in va.items() if isinstance(vv, np.ndarray) and len(vv) == len(common)}
                vb_c = {kk: vv[common] for kk, vv in vb.items() if isinstance(vv, np.ndarray) and len(vv) == len(common)}
                node = _compare_pair_node(va_c, vb_c, n_boot=n_boot, seed=seed)
                rc_a = selective_risk_coverage_curve(omics_te[common], va["patient_regret1"][common])
                rc_b = selective_risk_coverage_curve(omics_te[common], vb["patient_regret1"][common])
                node["status"] = "ok"
                node["n_paired"] = n_p
                node["aurc_omics_a"] = rc_a["aurc"]
                node["aurc_omics_b"] = rc_b["aurc"]
                node["aurc_omics_diff"] = float(rc_a["aurc"] - rc_b["aurc"])
                pair_out[f"k{k}_{a}_vs_{b}"] = node

        pols = ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED")
        common3 = lock_vecs[f"StaticDOpt_k{k}"]["valid_mask"] & lock_vecs[f"EqualWeightVarRed_k{k}"]["valid_mask"] & lock_vecs[f"AdaptiveOED_k{k}"]["valid_mask"]
        n3 = int(np.sum(common3))
        if n3 < 5:
            three_way[f"k{k}"] = {
                "status": "hesaplanamadı",
                "reason": "insufficient_common_complete_patients (n < 5)",
                "n_paired": n3,
            }
        else:
            arms_c = {}
            for p in pols:
                vp = lock_vecs[f"{p}_k{k}"]
                rc_p = selective_risk_coverage_curve(omics_te[common3], vp["patient_regret1"][common3])
                arms_c[p] = {
                    "rmse": float(np.sqrt(np.mean(vp["patient_rmse_sq"][common3]))),
                    "regret_at_1": float(np.mean(vp["patient_regret1"][common3])),
                    "ndcg_at_3": float(np.mean(vp["patient_ndcg3"][common3])),
                    "aurc_omics_regret1": float(rc_p["aurc"]),
                }
            three_way[f"k{k}"] = {"status": "ok", "n_paired": n3, "arms": arms_c}
    return {"policy_pairs": pair_out, "three_way_common": three_way}
