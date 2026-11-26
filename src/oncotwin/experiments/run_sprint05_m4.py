"""Sprint 5 Kulvar A: M4 yolak/hedef-ailesi yapılı kovaryans, derece-korumalı kontrol ve Kapı 3."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.metrics import (
    VECTOR_KEYS,
    average_metric_nodes,
    evaluate_masked_predictions,
    paired_metric_diffs,
)
from src.oncotwin.eval.splits import (
    get_evaluable_mask,
    make_clean_wave_split,
    make_patient_disjoint_kfold,
    make_sprint04_splits,
    select_train_probe_pool,
)
from src.oncotwin.models.baselines import MeanEffectM0, OmicsRidgeM1
from src.oncotwin.models.panel_selection import (
    require_full_budget,
    select_static_d_optimal_ranking,
    take_first_k_available,
)
from src.oncotwin.models.pathway_cov import (
    build_cosine_similarity_gram,
    combine_psd_with_pathway_prior,
    conditional_gaussian_predict_fast,
    eval_inner_cv_mse_grid,
    load_beataml_drug_family_incidence,
    permute_bipartite_incidence_preserving_degrees,
    precompute_inner_cv_context,
)

M4_ALPHA_GRID = (0.05, 0.15, 0.30, 0.50)
M4_RHO_POS_GRID = (0.15, 0.30, 0.50, 0.70)
M4_LAM_GRID = (50.0, 200.0, 500.0, 1500.0, 5000.0)
M4_K_BUDGETS = (2, 5, 8)
M4_N_PERM_DRAWS = 20
M4_CV_SEEDS = (2026, 2027, 2028)
GATE3_RMSE_THRESHOLD = -0.50


def _eval_rho0_inner_cv(
    ctx: dict[str, Any],
    k_values: tuple[int, ...],
    alpha_grid: tuple[float, ...],
    lam_grid: tuple[float, ...],
) -> tuple[dict[int, tuple[float, float, float, float]], dict[int, np.ndarray]]:
    """rho=0 (M3) için iç-CV en iyi (mse, alpha, 0.0, lam) ve nihai kovaryansları hesaplar."""
    fold_base = ctx["fold_base"]
    n_act = len(ctx["active_idx"])
    eye_g = np.eye(n_act, dtype=np.float64)

    covs_a = {
        a: [combine_psd_with_pathway_prior(cp, dp, eye_g, alpha=a, rho=0.0) for cp, dp, _, _ in fold_base]
        for a in alpha_grid
    }
    best_by_k: dict[int, tuple[float, float, float, float]] = {}
    for k in k_values:
        best = (float("inf"), alpha_grid[0], 0.0, lam_grid[0])
        for a in alpha_grid:
            mses = eval_inner_cv_mse_grid(ctx, covs_a[a], k, lam_grid)
            for lam, mse in zip(lam_grid, mses):
                if mse < best[0]:
                    best = (mse, a, 0.0, lam)
        best_by_k[k] = best

    cov_full = {
        k: combine_psd_with_pathway_prior(ctx["cov_psd_full"], ctx["diag_psd_full"], eye_g, alpha=best_by_k[k][1], rho=0.0)
        for k in k_values
    }
    return best_by_k, cov_full


def _fit_pathway_cov_pair(
    ctx: dict[str, Any],
    G_unit_full: np.ndarray,
    k_values: tuple[int, ...],
    rho0_best: dict[int, tuple[float, float, float, float]],
    alpha_grid: tuple[float, ...] = M4_ALPHA_GRID,
    rho_pos_grid: tuple[float, ...] = M4_RHO_POS_GRID,
    lam_grid: tuple[float, ...] = M4_LAM_GRID,
) -> dict[str, Any]:
    """Tek bir G_unit için hem iç-CV serbest (rho>=0) hem zorunlu pozitif (rho>0) kovaryansları seçer."""
    fold_base = ctx["fold_base"]
    active_idx: np.ndarray = ctx["active_idx"]
    G_act = G_unit_full[np.ix_(active_idx, active_idx)]

    combo_covs = {
        (a, r): [combine_psd_with_pathway_prior(cp, dp, G_act, alpha=a, rho=r) for cp, dp, _, _ in fold_base]
        for a in alpha_grid
        for r in rho_pos_grid
    }

    best_free: dict[int, tuple[float, float, float]] = {}
    best_pos: dict[int, tuple[float, float, float]] = {}
    for k in k_values:
        b_pos = (float("inf"), alpha_grid[0], rho_pos_grid[0], lam_grid[0])
        for a in alpha_grid:
            for r in rho_pos_grid:
                mses = eval_inner_cv_mse_grid(ctx, combo_covs[(a, r)], k, lam_grid)
                for lam, mse in zip(lam_grid, mses):
                    if mse < b_pos[0]:
                        b_pos = (mse, a, r, lam)
        best_pos[k] = (b_pos[1], b_pos[2], b_pos[3])
        r0 = rho0_best[k]
        best_free[k] = (b_pos[1], b_pos[2], b_pos[3]) if b_pos[0] < r0[0] else (r0[1], 0.0, r0[3])

    cov_psd_f, diag_psd_f = ctx["cov_psd_full"], ctx["diag_psd_full"]
    return {
        "free_params": best_free,
        "pos_params": best_pos,
        "free_cov": {
            k: combine_psd_with_pathway_prior(cov_psd_f, diag_psd_f, G_act, alpha=p[0], rho=p[1])
            for k, p in best_free.items()
        },
        "pos_cov": {
            k: combine_psd_with_pathway_prior(cov_psd_f, diag_psd_f, G_act, alpha=p[0], rho=p[1])
            for k, p in best_pos.items()
        },
    }


def _predict_from_cov(
    Y_eval: np.ndarray,
    prior_eval: np.ndarray,
    cov_k: np.ndarray,
    lam_k: float,
    probes_global: list[np.ndarray],
    ctx: dict[str, Any],
    target_idx: list[int],
) -> np.ndarray:
    active_idx: np.ndarray = ctx["active_idx"]
    g2l: dict[int, int] = ctx["global_to_local"]
    t_local: np.ndarray = ctx["t_local"]
    R_eval_act = (Y_eval - prior_eval)[:, active_idx]
    probes_loc = [np.asarray([g2l[int(g)] for g in s], dtype=np.int64) for s in probes_global]
    delta = conditional_gaussian_predict_fast(R_eval_act, cov_k, probes_loc, t_local, lam=lam_k)
    out = prior_eval.copy()
    out[:, np.asarray(target_idx, dtype=np.int64)] += delta
    return out


def _select_static_dopt_panels(
    Y_eval: np.ndarray,
    cov_ref: np.ndarray,
    lam_ref: float,
    ctx: dict[str, Any],
    probe_pool_idx: list[int],
    eval_mask: np.ndarray,
    k: int,
    label: str,
) -> tuple[list[np.ndarray], list[int]]:
    g2l: dict[int, int] = ctx["global_to_local"]
    active_idx: np.ndarray = ctx["active_idx"]
    p_loc = np.asarray([g2l[g] for g in probe_pool_idx], dtype=np.int64)
    ranked_global = active_idx[select_static_d_optimal_ranking(cov_ref, p_loc, ctx["t_local"], lam=lam_ref)]
    probes = [take_first_k_available(ranked_global, Y_eval[i], k)[0] for i in range(len(Y_eval))]
    require_full_budget(probes, eval_mask, k, label)
    return probes, [int(x) for x in ranked_global[:k]]


def _run_split_m4(
    ds: BeatAMLDataset,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_idx: list[int],
    G_true_full: np.ndarray,
    G_perm_list: list[np.ndarray],
    k_budgets: tuple[int, ...],
    seed: int,
    min_train_obs: int,
    min_targets: int,
    min_probes: int,
    top_k_genes: int = 2000,
) -> dict[str, Any]:
    """Bir CV tohumu veya tek bölme için M3, M4, M4_forced_pos, M4_perm_null (A1) ve A2 çalıştırır."""
    n_pat, n_drugs = ds.Y_auc.shape
    target_names = [ds.drug_names[i] for i in target_idx]
    n_perm = len(G_perm_list)
    k_ref = 5 if 5 in k_budgets else k_budgets[0]

    base_d_oof = np.full((n_pat, n_drugs), np.nan, dtype=np.float64)
    eval_mask_all = np.zeros(n_pat, dtype=bool)
    fold_ids = np.full(n_pat, -1, dtype=np.int64)

    preds_a1: dict[str, dict[int, np.ndarray]] = {
        m: {k: np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for k in k_budgets}
        for m in ("M3", "M4", "M4_forced_pos")
    }
    preds_perm_free = {
        k: [np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for _ in range(n_perm)] for k in k_budgets
    }
    preds_perm_pos = {
        k: [np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for _ in range(n_perm)] for k in k_budgets
    }
    preds_a2_m4 = {k: np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for k in k_budgets}

    selected_rhos_m4: dict[int, list[float]] = {k: [] for k in k_budgets}
    panel_jaccards_k5: list[float] = []

    for f_idx, (tr_idx, te_idx) in enumerate(folds):
        fold_ids[te_idx] = f_idx
        Y_tr = ds.Y_auc[tr_idx]
        subj_tr = ds.meta.iloc[tr_idx]["dbgap_subject_id"].to_numpy()
        probe_pool_idx = select_train_probe_pool(Y_tr, ds.drug_names, target_names, min_train_obs=min_train_obs)
        fit_drugs = sorted(set(target_idx) | set(probe_pool_idx))

        m0 = MeanEffectM0().fit(Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=k_budgets, inner_seed=seed + f_idx)
        base_d_oof[te_idx] = (m0.mu_ + m0.b_d_)[None, :]
        m1, Y_oof = OmicsRidgeM1(top_k_genes=min(top_k_genes, ds.X_rna.shape[1])).fit_and_oof(
            ds.X_rna[tr_idx], ds.X_mut[tr_idx], Y_tr, subj_tr, fit_drug_idx=fit_drugs, target_idx=target_idx, inner_seed=seed + f_idx
        )
        m1_te = m1.predict(ds.X_rna[te_idx], ds.X_mut[te_idx], n_drugs=n_drugs)
        Y_te = ds.Y_auc[te_idx]
        te_mask = get_evaluable_mask(Y_te, target_idx, probe_pool_idx, min_targets=min_targets, min_probes=min_probes)
        eval_mask_all[te_idx] = te_mask

        ctx = precompute_inner_cv_context(
            Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=k_budgets, prior_train=Y_oof, inner_seed=seed + f_idx
        )
        rho0_best, m3_covs = _eval_rho0_inner_cv(ctx, k_budgets, M4_ALPHA_GRID, M4_LAM_GRID)
        m4_fit = _fit_pathway_cov_pair(ctx, G_true_full, k_budgets, rho0_best)
        perm_fits = [_fit_pathway_cov_pair(ctx, Gp, k_budgets, rho0_best) for Gp in G_perm_list]

        for k in k_budgets:
            selected_rhos_m4[k].append(float(m4_fit["free_params"][k][1]))
            # Deney A1: Sabit StaticDOpt(M3) paneli
            probes_m3, top_m3 = _select_static_dopt_panels(
                Y_te, m3_covs[k_ref], rho0_best[k_ref][3], ctx, probe_pool_idx, te_mask, k, f"A1_M3_k{k}"
            )
            preds_a1["M3"][k][te_idx] = _predict_from_cov(Y_te, m1_te, m3_covs[k], rho0_best[k][3], probes_m3, ctx, target_idx)
            preds_a1["M4"][k][te_idx] = _predict_from_cov(
                Y_te, m1_te, m4_fit["free_cov"][k], m4_fit["free_params"][k][2], probes_m3, ctx, target_idx
            )
            preds_a1["M4_forced_pos"][k][te_idx] = _predict_from_cov(
                Y_te, m1_te, m4_fit["pos_cov"][k], m4_fit["pos_params"][k][2], probes_m3, ctx, target_idx
            )
            for p_i, pfit in enumerate(perm_fits):
                preds_perm_free[k][p_i][te_idx] = _predict_from_cov(
                    Y_te, m1_te, pfit["free_cov"][k], pfit["free_params"][k][2], probes_m3, ctx, target_idx
                )
                preds_perm_pos[k][p_i][te_idx] = _predict_from_cov(
                    Y_te, m1_te, pfit["pos_cov"][k], pfit["pos_params"][k][2], probes_m3, ctx, target_idx
                )

            # Deney A2: M4 kovaryansıyla yeniden seçilmiş StaticDOpt(M4) paneli
            probes_m4, top_m4 = _select_static_dopt_panels(
                Y_te, m4_fit["free_cov"][k_ref], m4_fit["free_params"][k_ref][2], ctx, probe_pool_idx, te_mask, k, f"A2_M4_k{k}"
            )
            preds_a2_m4[k][te_idx] = _predict_from_cov(
                Y_te, m1_te, m4_fit["free_cov"][k], m4_fit["free_params"][k][2], probes_m4, ctx, target_idx
            )
            if k == k_ref:
                s1, s2 = set(top_m3), set(top_m4)
                panel_jaccards_k5.append(len(s1 & s2) / float(len(s1 | s2)))

    f_arg = fold_ids if len(folds) > 1 else None
    out_a1: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in ("M3", "M4", "M4_forced_pos", "M4_perm_null", "M4_perm_forced")}
    out_a2: dict[int, dict[str, Any]] = {}
    for k in k_budgets:
        for m in ("M3", "M4", "M4_forced_pos"):
            out_a1[m][k] = evaluate_masked_predictions(ds.Y_auc, preds_a1[m][k], base_d_oof, eval_mask_all, target_idx, fold_ids=f_arg)
        out_a1["M4_perm_null"][k] = average_metric_nodes([
            evaluate_masked_predictions(ds.Y_auc, preds_perm_free[k][p_i], base_d_oof, eval_mask_all, target_idx, fold_ids=f_arg)
            for p_i in range(n_perm)
        ])
        out_a1["M4_perm_forced"][k] = average_metric_nodes([
            evaluate_masked_predictions(ds.Y_auc, preds_perm_pos[k][p_i], base_d_oof, eval_mask_all, target_idx, fold_ids=f_arg)
            for p_i in range(n_perm)
        ])
        out_a2[k] = evaluate_masked_predictions(ds.Y_auc, preds_a2_m4[k], base_d_oof, eval_mask_all, target_idx, fold_ids=f_arg)

    return {
        "a1": out_a1,
        "a2_m4_reselected": out_a2,
        "eval_mask": eval_mask_all,
        "selected_rhos_m4": selected_rhos_m4,
        "mean_panel_jaccard_k5": float(np.mean(panel_jaccards_k5)) if panel_jaccards_k5 else float("nan"),
    }


def _strip_vectors(node: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k not in ("vectors", "drug_spearmans")}


def _summarize_regime_m4(
    a1_nodes: dict[str, dict[int, dict[str, Any]]],
    a2_nodes: dict[int, dict[str, Any]],
    k_budgets: tuple[int, ...],
    rho_hist: dict[int, list[float]],
    mean_jaccard: float,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    res_a1: dict[str, Any] = {}
    res_a2: dict[str, Any] = {}
    vecs: dict[str, np.ndarray] = {}
    for k in k_budgets:
        res_a1[f"k{k}"] = {
            "models": {m: _strip_vectors(a1_nodes[m][k]) for m in a1_nodes},
            "selected_rho_m4_values": rho_hist[k],
            "selected_rho_m4_pos_fraction": float(np.mean([r > 0.0 for r in rho_hist[k]])),
            "comparisons": {
                "M4_vs_M3": paired_metric_diffs(a1_nodes["M4"][k], a1_nodes["M3"][k], n_boot=n_boot),
                "M4_vs_M4_perm_null": paired_metric_diffs(a1_nodes["M4"][k], a1_nodes["M4_perm_null"][k], n_boot=n_boot),
                "M4_forced_pos_vs_M3": paired_metric_diffs(a1_nodes["M4_forced_pos"][k], a1_nodes["M3"][k], n_boot=n_boot),
                "M4_forced_pos_vs_M4_perm_forced": paired_metric_diffs(
                    a1_nodes["M4_forced_pos"][k], a1_nodes["M4_perm_forced"][k], n_boot=n_boot
                ),
            },
        }
        res_a2[f"k{k}"] = {
            "M4_reselected_panel": _strip_vectors(a2_nodes[k]),
            "M3_reference_panel": _strip_vectors(a1_nodes["M3"][k]),
            "M4_reselected_vs_M3": paired_metric_diffs(a2_nodes[k], a1_nodes["M3"][k], n_boot=n_boot),
            "M4_reselected_vs_M4_fixed": paired_metric_diffs(a2_nodes[k], a1_nodes["M4"][k], n_boot=n_boot),
        }
        for m in ("M3", "M4", "M4_forced_pos", "M4_perm_null", "M4_perm_forced"):
            for vk in VECTOR_KEYS:
                vecs[f"A1/k{k}/{m}/{vk}"] = a1_nodes[m][k]["vectors"][vk]
        for vk in VECTOR_KEYS:
            vecs[f"A2/k{k}/M4_reselected/{vk}"] = a2_nodes[k]["vectors"][vk]
    return {"experiment_a1_fixed_panel": res_a1, "experiment_a2_reselected_panel": res_a2, "mean_panel_jaccard_k5": mean_jaccard}, vecs


def evaluate_gate3(m4_results: dict[str, Any]) -> dict[str, Any]:
    """PROTOCOL_SPRINT05.md §1.4 ön kayıtlı Kapı 3 (AD-001) karar kuralını uygular."""
    cv_k5 = m4_results["regimes"]["id_cv_5fold_3seed"]["experiment_a1_fixed_panel"]["k5"]["comparisons"]
    w_k5 = m4_results["regimes"]["wave_ood_s3"]["experiment_a1_fixed_panel"]["k5"]["comparisons"]

    c_m3 = cv_k5["M4_vs_M3"]
    c_perm = cv_k5["M4_vs_M4_perm_null"]
    w_m3 = w_k5["M4_vs_M3"]
    w_perm = w_k5["M4_vs_M4_perm_null"]

    cond1_rmse = (
        c_m3["rmse_diff"]["diff"] <= GATE3_RMSE_THRESHOLD
        and c_m3["rmse_diff"]["ci_high"] < 0.0
        and c_m3["rmse_diff"]["p_two_sided"] < 0.05
    )
    cond1_no_deg = not (c_m3["regret1_diff"]["ci_low"] > 0.0 or c_m3["ndcg3_diff"]["ci_high"] < 0.0)
    cond1 = bool(cond1_rmse and cond1_no_deg)

    cond2 = bool(
        c_perm["rmse_diff"]["diff"] <= GATE3_RMSE_THRESHOLD
        and c_perm["rmse_diff"]["ci_high"] < 0.0
        and c_perm["rmse_diff"]["p_two_sided"] < 0.05
    )
    cond3 = bool(w_m3["rmse_diff"]["diff"] < 0.0 and w_perm["rmse_diff"]["diff"] < 0.0)
    passed = bool(cond1 and cond2 and cond3)
    return {
        "decision": "PASS" if passed else "NO-GO (v2 ODE ERTELENİR)",
        "passed": passed,
        "rmse_threshold_auc": GATE3_RMSE_THRESHOLD,
        "conditions": {
            "1_id_cv_beats_m3_ge_050_no_ranking_deg": cond1,
            "2_id_cv_beats_degree_preserving_perm_null_ge_050": cond2,
            "3_exploratory_wave_ood_consistent_direction": cond3,
        },
        "observed_k5_a1": {
            "id_cv_m4_minus_m3_rmse": c_m3["rmse_diff"],
            "id_cv_m4_minus_perm_rmse": c_perm["rmse_diff"],
            "wave_ood_m4_minus_m3_rmse": w_m3["rmse_diff"],
            "wave_ood_m4_minus_perm_rmse": w_perm["rmse_diff"],
        },
    }


def run_track_a_m4(
    ds_idx: BeatAMLDataset,
    target_idx: list[int],
    k_budgets: tuple[int, ...] = M4_K_BUDGETS,
    n_perm: int = M4_N_PERM_DRAWS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Kulvar A'nın tüm Beat AML bölmelerini (ID CV, ID 60/20/20, W3+4 S3 & S4) koşturur."""
    t0 = time.perf_counter()
    B_full, fam_names = load_beataml_drug_family_incidence(ds_idx.drug_names, include_gene_sheet=False)
    G_true_full = build_cosine_similarity_gram(B_full)
    G_perm_list = [
        build_cosine_similarity_gram(permute_bipartite_incidence_preserving_degrees(B_full, seed=5000 + m))
        for m in range(n_perm)
    ]

    regimes_out: dict[str, Any] = {}
    all_vecs: dict[str, np.ndarray] = {}

    # 1. Birincil ID: Sprint 3 5-Fold CV x 3 seed (n_paired=427)
    seed_runs = []
    for s in M4_CV_SEEDS:
        folds = make_patient_disjoint_kfold(ds_idx.meta, n_splits=5, seed=s, strata_col="cohort")
        seed_runs.append(
            _run_split_m4(ds_idx, folds, target_idx, G_true_full, G_perm_list, k_budgets, seed=s, min_train_obs=100, min_targets=5, min_probes=12)
        )
    a1_cv = {
        m: {k: average_metric_nodes([r["a1"][m][k] for r in seed_runs]) for k in k_budgets}
        for m in ("M3", "M4", "M4_forced_pos", "M4_perm_null", "M4_perm_forced")
    }
    a2_cv = {k: average_metric_nodes([r["a2_m4_reselected"][k] for r in seed_runs]) for k in k_budgets}
    rho_cv = {k: [v for r in seed_runs for v in r["selected_rhos_m4"][k]] for k in k_budgets}
    jac_cv = float(np.mean([r["mean_panel_jaccard_k5"] for r in seed_runs]))
    regimes_out["id_cv_5fold_3seed"], v_cv = _summarize_regime_m4(a1_cv, a2_cv, k_budgets, rho_cv, jac_cv)
    for vk, arr in v_cv.items():
        all_vecs[f"track_a/id_cv_5fold_3seed/{vk}"] = arr

    # 2. İkincil ID Holdout: Sprint 4 id_60_20_20 (n_paired=91) & Keşifsel W3+4 Sprint 4 (n_paired=148)
    s4_splits = make_sprint04_splits(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects, seed=2026)
    for rname, s_key, m_prb in (
        ("id_60_20_20", "id_60_20_20", 8),
        ("wave_ood_s4", "wave_ood", 8),
    ):
        part = s4_splits[s_key]
        run_one = _run_split_m4(
            ds_idx, [(part["train_tune"], part["test"])], target_idx, G_true_full, G_perm_list,
            k_budgets, seed=2026, min_train_obs=100, min_targets=5, min_probes=m_prb,
        )
        regimes_out[rname], v_r = _summarize_regime_m4(
            run_one["a1"], run_one["a2_m4_reselected"], k_budgets, run_one["selected_rhos_m4"], run_one["mean_panel_jaccard_k5"]
        )
        for vk, arr in v_r.items():
            all_vecs[f"track_a/{rname}/{vk}"] = arr

    # 3. Keşifsel Zamansal Stres Testi: Sprint 3 temiz Waves1+2 -> Waves3+4 (n_paired=146)
    w_tr, w_te = make_clean_wave_split(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects)
    run_w3 = _run_split_m4(
        ds_idx, [(w_tr, w_te)], target_idx, G_true_full, G_perm_list,
        k_budgets, seed=2026, min_train_obs=100, min_targets=5, min_probes=12,
    )
    regimes_out["wave_ood_s3"], v_w3 = _summarize_regime_m4(
        run_w3["a1"], run_w3["a2_m4_reselected"], k_budgets, run_w3["selected_rhos_m4"], run_w3["mean_panel_jaccard_k5"]
    )
    for vk, arr in v_w3.items():
        all_vecs[f"track_a/wave_ood_s3/{vk}"] = arr

    out = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "drug_family_graph": {
            "n_drugs_total": int(B_full.shape[0]),
            "n_families": int(len(fam_names)),
            "n_annotated_drugs": int(np.sum(B_full.sum(axis=1) > 0)),
            "n_incidence_edges": int(B_full.sum()),
            "n_perm_draws": int(n_perm),
        },
        "regimes": regimes_out,
    }
    out["gate3"] = evaluate_gate3(out)
    return out, all_vecs
