"""Sprint 5 Kulvar B: Hasta-düzeyi sıra-kalibreli konformal kısa liste ve AUGRC duyarlılığı."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.metrics import paired_bootstrap_ci
from src.oncotwin.eval.rank_conformal import (
    build_shortlist_masks,
    evaluate_shortlist_masks,
    fit_rank_conformal_shortlist,
    fit_top1_gap_conformal_shortlist,
    generalized_risk_coverage_augrc,
)
from src.oncotwin.eval.splits import get_evaluable_mask, make_sprint04_splits
from src.oncotwin.eval.uncertainty import fit_split_conformal_simultaneous
from src.oncotwin.experiments.run_sprint04 import (
    ALPHA,
    K_BUDGETS,
    MIN_EVAL_PROBES_ARCHIVE,
    MIN_EVAL_TARGETS,
    MIN_TRAIN_PROBE_OBS,
    SPRINT04_SEED,
    _fit_sprint04_fold,
    _predict_arm,
    _select_archive_panels,
)
from src.oncotwin.models.panel_selection import compute_contender_weights

SHORTLIST_ARMS: tuple[tuple[str, str, int], ...] = (
    ("M0_k0", "M0", 0),
    ("M1_k0", "M1", 0),
    ("StaticDOpt_k2", "StaticDOpt", 2),
    ("StaticDOpt_k5", "StaticDOpt", 5),
    ("StaticDOpt_k8", "StaticDOpt", 8),
    ("AdaptiveOED_k5", "AdaptiveOED", 5),
)


def _strip_vec(ev: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in ev.items() if k != "vectors"}


def _paired_shortlist_diffs(
    va: dict[str, np.ndarray],
    vb: dict[str, np.ndarray],
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict[str, Any]:
    keys = (
        ("contains_top1_diff", "patient_shortlist_contains_top1"),
        ("size_diff", "patient_shortlist_size_obs"),
        ("fraction_diff", "patient_shortlist_fraction_obs"),
        ("definite_top1_diff", "patient_definite_top1"),
        ("best_regret1_diff", "patient_shortlist_best_regret1"),
        ("mean_regret1_diff", "patient_shortlist_mean_regret1"),
        ("worst_regret1_diff", "patient_shortlist_worst_regret1"),
    )
    return {out: paired_bootstrap_ci(va[vk], vb[vk], n_boot=n_boot, seed=seed) for out, vk in keys}


def run_track_b_shortlist(
    ds_idx: BeatAMLDataset,
    target_idx: list[int],
    v2_results_json: Path = Path("data/processed/sprint04_v2_results.json"),
    v2_vectors_npz: Path = Path("data/processed/sprint04_v2_vectors.npz"),
    alpha: float = ALPHA,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Sprint 4 v2 tahminleri üzerinde 4 konformal kısa liste yöntemi ve AUGRC kıyasını çalıştırır."""
    t0 = time.perf_counter()
    v2_res = json.loads(v2_results_json.read_text(encoding="utf-8"))
    z_v2 = np.load(v2_vectors_npz)
    splits = make_sprint04_splits(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects, seed=SPRINT04_SEED)
    uniform_w = np.full(len(target_idx), 1.0 / len(target_idx), dtype=np.float64)
    target_set = set(target_idx)
    n_drugs = ds_idx.Y_auc.shape[1]

    regimes_out: dict[str, Any] = {}
    all_vecs: dict[str, np.ndarray] = {}

    for rn in ("id_60_20_20", "wave_ood"):
        part = splits[rn]
        full_fit = _fit_sprint04_fold(
            ds_idx, part["train_tune"], target_idx, k_budgets=K_BUDGETS, inner_seed=SPRINT04_SEED, min_train_obs=MIN_TRAIN_PROBE_OBS
        )
        ds_cal = ds_idx.subset_by_positions(part["calibration"])
        ds_te = ds_idx.subset_by_positions(part["test"])
        m1_cal = full_fit["M1"].predict(ds_cal.X_rna, ds_cal.X_mut, n_drugs=n_drugs)
        m1_te = full_fit["M1"].predict(ds_te.X_rna, ds_te.X_mut, n_drugs=n_drugs)
        cov_cc = full_fit["cov_ref"][np.ix_(full_fit["target_local"], full_fit["target_local"])]
        c_seed = full_fit["contender_seed"]
        cont_cal = compute_contender_weights(m1_cal[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=c_seed)
        cont_te = compute_contender_weights(m1_te[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=c_seed)

        arch_cal = get_evaluable_mask(ds_cal.Y_auc, target_idx, full_fit["probe_pool_idx"], MIN_EVAL_TARGETS, MIN_EVAL_PROBES_ARCHIVE)
        arch_te = get_evaluable_mask(ds_te.Y_auc, target_idx, full_fit["probe_pool_idx"], MIN_EVAL_TARGETS, MIN_EVAL_PROBES_ARCHIVE)
        assert np.array_equal(arch_te, z_v2[f"{rn}/test_archive_eval_mask"].astype(bool))

        arms_out: dict[str, Any] = {}
        augrc_out: dict[str, Any] = {}
        om_te = z_v2[f"{rn}/test_omics_pca_dist"][arch_te]

        for arm_name, pol, k in SHORTLIST_ARMS:
            p_cal = _select_archive_panels(ds_cal.Y_auc, full_fit, cont_cal, pol, k, arch_cal, uniform_w, target_set)
            p_te = _select_archive_panels(ds_te.Y_auc, full_fit, cont_te, pol, k, arch_te, uniform_w, target_set)
            Yp_cal, _ = _predict_arm(ds_cal, p_cal, full_fit, m1_cal, target_idx, k=k, complete_mask=arch_cal, policy=pol)
            Yp_te, m3_te = _predict_arm(ds_te, p_te, full_fit, m1_te, target_idx, k=k, complete_mask=arch_te, policy=pol)

            # Sprint 4 v2 ile bit-düzeyinde uyum doğrulaması
            v2_m3_sc = z_v2[f"{rn}/archive_available/{arm_name}/m3_post_scale"]
            assert np.allclose(m3_te[arch_te], v2_m3_sc[arch_te], atol=1e-12)

            eq_fit = fit_split_conformal_simultaneous(ds_cal.Y_auc, Yp_cal, target_idx, alpha=alpha, valid_mask=arch_cal)
            rk_fit = fit_rank_conformal_shortlist(ds_cal.Y_auc, Yp_cal, target_idx, alpha=alpha, valid_mask=arch_cal)
            gp_fit = fit_top1_gap_conformal_shortlist(ds_cal.Y_auc, Yp_cal, target_idx, alpha=alpha, valid_mask=arch_cal)

            method_specs = (
                ("EqualWidthAUC", "equal_width_auc", eq_fit["q_sim"], eq_fit["finite_interval_published"], {"q_sim": eq_fit["q_sim"], "width_sim": eq_fit["width_sim"], "n_cal": eq_fit["n_cal"]}),
                ("RankConformal", "abs_rank", rk_fit["k_hat_float"], rk_fit["finite_published"], {"k_hat": rk_fit["k_hat"], "n_cal": rk_fit["n_cal"]}),
                ("RelRankConformal", "rel_rank", rk_fit["q_rel"], rk_fit["finite_published"], {"q_rel": rk_fit["q_rel"], "n_cal": rk_fit["n_cal"]}),
                ("Top1GapConformal", "top1_gap", gp_fit["m_hat"], gp_fit["finite_published"], {"m_hat": gp_fit["m_hat"], "n_cal": gp_fit["n_cal"]}),
            )

            m_evals: dict[str, dict[str, Any]] = {}
            m_vecs: dict[str, dict[str, np.ndarray]] = {}
            for m_label, m_code, thresh, fin_pub, cal_info in method_specs:
                sl_masks = build_shortlist_masks(ds_te.Y_auc, Yp_te, target_idx, m_code, float(thresh), valid_mask=arch_te)
                ev = evaluate_shortlist_masks(ds_te.Y_auc, sl_masks, target_idx, valid_mask=arch_te, finite_published=fin_pub)
                m_evals[m_label] = {"calibration": cal_info, "test": _strip_vec(ev)}
                m_vecs[m_label] = ev["vectors"]
                for vk, varr in ev["vectors"].items():
                    all_vecs[f"track_b/{rn}/{arm_name}/{m_label}/{vk}"] = varr

            arms_out[arm_name] = {
                "methods": m_evals,
                "comparisons": {
                    "RankConformal_vs_EqualWidthAUC": _paired_shortlist_diffs(m_vecs["RankConformal"], m_vecs["EqualWidthAUC"], n_boot=n_boot, seed=SPRINT04_SEED),
                    "RelRankConformal_vs_EqualWidthAUC": _paired_shortlist_diffs(m_vecs["RelRankConformal"], m_vecs["EqualWidthAUC"], n_boot=n_boot, seed=SPRINT04_SEED),
                    "Top1GapConformal_vs_EqualWidthAUC": _paired_shortlist_diffs(m_vecs["Top1GapConformal"], m_vecs["EqualWidthAUC"], n_boot=n_boot, seed=SPRINT04_SEED),
                    "Top1GapConformal_vs_RankConformal": _paired_shortlist_diffs(m_vecs["Top1GapConformal"], m_vecs["RankConformal"], n_boot=n_boot, seed=SPRINT04_SEED),
                },
            }

            # Ek duyarlılık: Traub et al. (NeurIPS 2024) AUGRC (Sprint 4 v2 çekinme skorları)
            reg_te = z_v2[f"{rn}/archive_available/{arm_name}/patient_regret1"][arch_te]
            v2_abst = v2_res["regimes"][rn]["scenarios"]["archive_available"]["arms"][arm_name]["abstention"]
            augrc_out[arm_name] = {
                "omics_pca_dist": {
                    **generalized_risk_coverage_augrc(om_te, reg_te),
                    "v2_aurc_regret1": v2_abst["omics_pca_dist"]["risk_coverage_regret1"]["aurc"],
                    "v2_random_aurc": v2_abst["omics_pca_dist"]["risk_coverage_regret1"]["random_aurc"],
                },
                "m3_post_scale": {
                    **generalized_risk_coverage_augrc(v2_m3_sc[arch_te], reg_te),
                    "v2_aurc_regret1": v2_abst["m3_post_scale"]["risk_coverage_regret1"]["aurc"],
                    "v2_random_aurc": v2_abst["m3_post_scale"]["risk_coverage_regret1"]["random_aurc"],
                },
            }

        regimes_out[rn] = {
            "n_cal": int(np.sum(arch_cal)),
            "n_test": int(np.sum(arch_te)),
            "arms": arms_out,
            "augrc_sensitivity": augrc_out,
        }

    return {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "alpha": float(alpha),
        "note": "W3+4 hiçbir yöntem veya eşik seçiminde kullanılmamıştır (yalnız keşifsel stres testi).",
        "regimes": regimes_out,
    }, all_vecs
