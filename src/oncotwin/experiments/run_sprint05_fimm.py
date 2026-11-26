"""Sprint 5 Kulvar C: FIMM / Malani (Zenodo 7370747) donmuş aktarım (C1) ve FIMM içi yöntem tekrarı (C2)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.data.fimm_dataset import (
    FIMMDataset,
    load_fimm_aml_dataset,
    select_fimm_index_specimen_positions,
    within_sample_rank_normalize,
)
from src.oncotwin.eval.metrics import (
    VECTOR_KEYS,
    average_metric_nodes,
    evaluate_masked_predictions,
    ndcg_at_k,
    paired_bootstrap_ci,
    paired_metric_diffs,
    recall_at_k,
    regret_at_k,
    safe_spearman,
)
from src.oncotwin.eval.splits import (
    get_evaluable_mask,
    make_patient_disjoint_kfold,
    sample_probe_sets,
    select_train_probe_pool,
)
from src.oncotwin.experiments.run_sprint05_m4 import (
    M4_ALPHA_GRID,
    _eval_rho0_inner_cv,
    _fit_pathway_cov_pair,
    _predict_from_cov,
)
from src.oncotwin.models.baselines import CovarianceFewShotM2a, HybridFewShotM3, MeanEffectM0, OmicsRidgeM1
from src.oncotwin.models.panel_selection import (
    require_full_budget,
    select_static_d_optimal_ranking,
    take_first_k_available,
)
from src.oncotwin.models.pathway_cov import (
    build_cosine_similarity_gram,
    load_beataml_drug_family_incidence,
    permute_bipartite_incidence_preserving_degrees,
    precompute_inner_cv_context,
)

FIMM_K_BUDGETS = (2, 5, 8)
FIMM_CV_SEEDS = (2026, 2027, 2028)
FIMM_RANDOM_REPS = 20
# Boyut analizi: [lam] = [varyans]. C1 (z-skor, var~1) ve C2 (-sDSS, var~50-150) için ölçek-uyumlu ızgaralar
C1_Z_LAM_GRID = (0.05, 0.20, 0.50, 1.50, 5.00)
C2_SDSS_LAM_GRID = (1.0, 5.0, 20.0, 75.0, 250.0)


def _within_patient_zscore_rows(Y_mat: np.ndarray) -> np.ndarray:
    """Her hastayı yalnız kendi sonlu ilaç gözlemleri üzerinden boyutsuz z-skora çevirir."""
    out = np.full_like(Y_mat, np.nan, dtype=np.float64)
    for i in range(Y_mat.shape[0]):
        m = np.isfinite(Y_mat[i])
        if int(np.sum(m)) < 2:
            continue
        vals = Y_mat[i, m]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        out[i, m] = (vals - mu) / sd if sd > 1e-6 else 0.0
    return out


def _build_single_patient_probe_z_matrix(Y_aligned: np.ndarray, probe_sets: list[np.ndarray]) -> np.ndarray:
    """FIMM test hastasının yalnız kendi k gözlenen probunu z-skorlar; hedefler ve diğer hastalar NaN kalır."""
    out = np.full_like(Y_aligned, np.nan, dtype=np.float64)
    for i, S in enumerate(probe_sets):
        if len(S) == 0:
            continue
        vals = Y_aligned[i, S]
        fin = np.isfinite(vals)
        if int(np.sum(fin)) < 2:
            if int(np.sum(fin)) == 1:
                out[i, S[fin]] = 0.0
            continue
        v = vals[fin]
        mu = float(np.mean(v))
        sd = float(np.std(v))
        out[i, S[fin]] = (v - mu) / sd if sd > 1e-6 else 0.0
    return out


def _eval_scale_free_ranking(
    Y_true_aligned: np.ndarray,
    Y_pred: np.ndarray,
    valid_mask: np.ndarray,
    target_idx: list[int],
    cluster_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ölçekten bağımsız sıralama (NDCG@3, Recall@3, Within-Patient Spearman) ve sDSS Regret@1/3."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    rows = np.flatnonzero(valid_mask)
    n_rows = len(Y_true_aligned)
    vecs = {
        "patient_ndcg3": np.full(n_rows, np.nan, dtype=np.float64),
        "patient_recall3": np.full(n_rows, np.nan, dtype=np.float64),
        "patient_spearman": np.full(n_rows, np.nan, dtype=np.float64),
        "patient_regret1": np.full(n_rows, np.nan, dtype=np.float64),
        "patient_regret3": np.full(n_rows, np.nan, dtype=np.float64),
    }
    for i in rows:
        yt = Y_true_aligned[i, t_idx]
        yp = Y_pred[i, t_idx]
        m = np.isfinite(yt) & np.isfinite(yp)
        if int(np.sum(m)) < 3:
            continue
        yo, po = yt[m], yp[m]
        vecs["patient_ndcg3"][i] = ndcg_at_k(yo, po, k=3)
        vecs["patient_recall3"][i] = recall_at_k(yo, po, k=3)
        vecs["patient_regret1"][i] = regret_at_k(yo, po, k=1)
        vecs["patient_regret3"][i] = regret_at_k(yo, po, k=3)
        vecs["patient_spearman"][i] = safe_spearman(yo, po)

    # Çoklu örnekli hastalarda hasta-kümeli ortalama (sözde-tekrarı önler)
    if cluster_ids is not None:
        uniq = np.unique(cluster_ids[rows])
        c_vecs: dict[str, np.ndarray] = {}
        for vk, arr in vecs.items():
            out_c = np.full(len(uniq), np.nan, dtype=np.float64)
            for idx_c, cid in enumerate(uniq):
                vals = arr[(cluster_ids == cid) & np.isfinite(arr)]
                if len(vals) > 0:
                    out_c[idx_c] = float(np.mean(vals))
            c_vecs[vk] = out_c
        eval_vecs = c_vecs
        n_units = int(len(uniq))
    else:
        eval_vecs = {vk: arr[rows] for vk, arr in vecs.items()}
        n_units = int(len(rows))

    def _safe_mean(a: np.ndarray) -> float:
        fin = a[np.isfinite(a)]
        return float(np.mean(fin)) if len(fin) > 0 else float("nan")

    return {
        "n_eval_units": n_units,
        "ndcg_at_3": _safe_mean(eval_vecs["patient_ndcg3"]),
        "recall_at_3": _safe_mean(eval_vecs["patient_recall3"]),
        "within_patient_spearman": _safe_mean(eval_vecs["patient_spearman"]),
        "regret_at_1_sdss": _safe_mean(eval_vecs["patient_regret1"]),
        "regret_at_3_sdss": _safe_mean(eval_vecs["patient_regret3"]),
        "vectors": eval_vecs,
    }


def _average_scale_free_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    keys = list(nodes[0]["vectors"].keys())
    avg_vecs: dict[str, np.ndarray] = {}
    for vk in keys:
        stacked = np.stack([nd["vectors"][vk] for nd in nodes], axis=0)
        has_fin = np.any(np.isfinite(stacked), axis=0)
        out = np.full(stacked.shape[1], np.nan, dtype=np.float64)
        if np.any(has_fin):
            out[has_fin] = np.nanmean(stacked[:, has_fin], axis=0)
        avg_vecs[vk] = out

    def _safe_mean(a: np.ndarray) -> float:
        fin = a[np.isfinite(a)]
        return float(np.mean(fin)) if len(fin) > 0 else float("nan")

    return {
        "n_eval_units": nodes[0]["n_eval_units"],
        "ndcg_at_3": _safe_mean(avg_vecs["patient_ndcg3"]),
        "recall_at_3": _safe_mean(avg_vecs["patient_recall3"]),
        "within_patient_spearman": _safe_mean(avg_vecs["patient_spearman"]),
        "regret_at_1_sdss": _safe_mean(avg_vecs["patient_regret1"]),
        "regret_at_3_sdss": _safe_mean(avg_vecs["patient_regret3"]),
        "vectors": avg_vecs,
    }


def _paired_scale_free_diffs(node_a: dict[str, Any], node_b: dict[str, Any], n_boot: int = 1000) -> dict[str, Any]:
    pairs = (
        ("ndcg3_diff", "patient_ndcg3"),
        ("recall3_diff", "patient_recall3"),
        ("spearman_diff", "patient_spearman"),
        ("regret1_sdss_diff", "patient_regret1"),
        ("regret3_sdss_diff", "patient_regret3"),
    )
    return {
        out: paired_bootstrap_ci(node_a["vectors"][vk], node_b["vectors"][vk], n_boot=n_boot, seed=2026)
        for out, vk in pairs
    }


def _align_beataml_to_fimm(
    ds_ba: BeatAMLDataset,
    fimm_ds: FIMMDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beat AML RNA/Mut/AUC matrislerini FIMM ortak gen, mutasyon ve ilaç sırasına hizalar."""
    ba_ens_map = {g.split(".")[0]: idx for idx, g in enumerate(ds_ba.gene_names)}
    gene_cols = [ba_ens_map[g] for g in fimm_ds.gene_ensembl_ids]
    X_rna_ba = ds_ba.X_rna[:, gene_cols].astype(np.float64)

    ba_mut_map = {name: idx for idx, name in enumerate(ds_ba.mut_feature_names)}
    mut_cols: list[int] = []
    for mg in fimm_ds.mut_feature_names:
        if mg == "FLT3":
            mut_cols.append(ba_mut_map["clin_FLT3_ITD"])
        elif mg == "NPM1":
            mut_cols.append(ba_mut_map["clin_NPM1"])
        else:
            mut_cols.append(ba_mut_map[f"wes_{mg}"])
    X_mut_ba = ds_ba.X_mut[:, mut_cols].astype(np.float64)

    ba_drug_map = {d: idx for idx, d in enumerate(ds_ba.drug_names)}
    drug_cols = [ba_drug_map[d] for d in fimm_ds.drug_names]
    Y_ba = ds_ba.Y_auc[:, drug_cols].astype(np.float64)
    return X_rna_ba, X_mut_ba, Y_ba


def _run_experiment_c1_frozen_transfer(
    ds_ba: BeatAMLDataset,
    fimm_all: FIMMDataset,
    fimm_idx: FIMMDataset,
    target_idx: list[int],
    common_probes: list[int],
    k_budgets: tuple[int, ...] = FIMM_K_BUDGETS,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Deney C1: Beat AML'de dondurulmuş modelin FIMM'e sıfır-yeniden-fit aktarımı."""
    X_rna_ba, X_mut_ba, Y_ba = _align_beataml_to_fimm(ds_ba, fimm_all)
    X_rna_ba_rank = within_sample_rank_normalize(X_rna_ba)
    Z_ba = _within_patient_zscore_rows(Y_ba)
    subj_ba = ds_ba.meta["dbgap_subject_id"].to_numpy()
    fit_drugs = sorted(set(target_idx) | set(common_probes))
    n_drugs = Z_ba.shape[1]

    m0_z = MeanEffectM0().fit(Z_ba, subj_ba, common_probes, target_idx, k_values=k_budgets, inner_seed=2026)
    m1_z, Y_oof_z = OmicsRidgeM1(top_k_genes=min(2000, X_rna_ba_rank.shape[1])).fit_and_oof(
        X_rna_ba_rank, X_mut_ba, Z_ba, subj_ba, fit_drug_idx=fit_drugs, target_idx=target_idx, inner_seed=2026
    )
    m2a_z = CovarianceFewShotM2a(lam_grid=C1_Z_LAM_GRID).fit(
        Z_ba, subj_ba, common_probes, target_idx, k_values=k_budgets, inner_seed=2026
    )
    m3_z = HybridFewShotM3(lam_grid=C1_Z_LAM_GRID).fit(
        m1_z, Y_oof_z, Z_ba, subj_ba, common_probes, target_idx, k_values=k_budgets, inner_seed=2026
    )

    # Ortak katalogda StaticDOpt paneli YALNIZCA Beat AML eğitim kovaryansından seçilir
    cm = m3_z._cov_model
    g2l = cm.global_to_local_
    p_loc = np.asarray([g2l[g] for g in common_probes], dtype=np.int64)
    t_loc = np.asarray([g2l[g] for g in target_idx], dtype=np.int64)
    k_ref = 5 if 5 in k_budgets else k_budgets[0]
    _, lam_ref = cm.best_params_by_k_[k_ref]
    ranked_common = cm.active_idx_[select_static_d_optimal_ranking(cm.cov_by_k_[k_ref], p_loc, t_loc, lam=lam_ref)]
    top8_names = [fimm_all.drug_names[int(g)] for g in ranked_common[:8]]

    out_cohorts: dict[str, Any] = {}
    vecs: dict[str, np.ndarray] = {}
    for cname, fds, use_clusters in (
        ("index_109_patients", fimm_idx, False),
        ("all_132_specimens_clustered", fimm_all, True),
    ):
        X_rna_f_rank = within_sample_rank_normalize(fds.X_rna)
        ev_mask = get_evaluable_mask(fds.Y_aligned, target_idx, common_probes, min_targets=5, min_probes=8)
        cl_ids = fds.meta["patient_int_id"].to_numpy() if use_clusters else None

        m0_pred = np.broadcast_to((m0_z.mu_ + m0_z.b_d_)[None, :], fds.Y_aligned.shape)
        m1_pred = m1_z.predict(X_rna_f_rank, fds.X_mut, n_drugs=n_drugs)

        arm_nodes: dict[str, dict[str, Any]] = {
            "C1_M0_k0": _eval_scale_free_ranking(fds.Y_aligned, m0_pred, ev_mask, target_idx, cluster_ids=cl_ids),
            "C1_M1_rank_k0": _eval_scale_free_ranking(fds.Y_aligned, m1_pred, ev_mask, target_idx, cluster_ids=cl_ids),
        }
        for k in k_budgets:
            probes_k = [take_first_k_available(ranked_common, fds.Y_aligned[i], k)[0] for i in range(len(fds.meta))]
            require_full_budget(probes_k, ev_mask, k, f"FIMM_C1_{cname}_k{k}")
            Z_obs_k = _build_single_patient_probe_z_matrix(fds.Y_aligned, probes_k)
            p_m2a = m2a_z.predict(Z_obs_k, probes_k, target_idx, k=k)
            p_m3 = m3_z.predict(X_rna_f_rank, fds.X_mut, Z_obs_k, probes_k, target_idx, k=k, prior_eval=m1_pred)
            arm_nodes[f"C1_M2a_StaticDOpt_k{k}"] = _eval_scale_free_ranking(fds.Y_aligned, p_m2a, ev_mask, target_idx, cluster_ids=cl_ids)
            arm_nodes[f"C1_M3_StaticDOpt_k{k}"] = _eval_scale_free_ranking(fds.Y_aligned, p_m3, ev_mask, target_idx, cluster_ids=cl_ids)

            if k == k_ref:
                rand_reps = sample_probe_sets(fds.Y_aligned, common_probes, k=k, n_repeats=FIMM_RANDOM_REPS, seed=2026 + k)
                r_m3_nodes = []
                for rep_p in rand_reps:
                    Z_r = _build_single_patient_probe_z_matrix(fds.Y_aligned, rep_p)
                    pr = m3_z.predict(X_rna_f_rank, fds.X_mut, Z_r, rep_p, target_idx, k=k, prior_eval=m1_pred)
                    r_m3_nodes.append(_eval_scale_free_ranking(fds.Y_aligned, pr, ev_mask, target_idx, cluster_ids=cl_ids))
                arm_nodes[f"C1_M3_Random20x_k{k}"] = _average_scale_free_nodes(r_m3_nodes)

        comps = {
            "C1_M3_k5_vs_M1_k0": _paired_scale_free_diffs(arm_nodes["C1_M3_StaticDOpt_k5"], arm_nodes["C1_M1_rank_k0"], n_boot=n_boot),
            "C1_M3_k5_vs_M0_k0": _paired_scale_free_diffs(arm_nodes["C1_M3_StaticDOpt_k5"], arm_nodes["C1_M0_k0"], n_boot=n_boot),
            "C1_M3_Static_k5_vs_Random_k5": _paired_scale_free_diffs(arm_nodes["C1_M3_StaticDOpt_k5"], arm_nodes["C1_M3_Random20x_k5"], n_boot=n_boot),
            "C1_M2a_k5_vs_M0_k0": _paired_scale_free_diffs(arm_nodes["C1_M2a_StaticDOpt_k5"], arm_nodes["C1_M0_k0"], n_boot=n_boot),
            "C1_M1_rank_k0_vs_M0_k0": _paired_scale_free_diffs(arm_nodes["C1_M1_rank_k0"], arm_nodes["C1_M0_k0"], n_boot=n_boot),
        }
        out_cohorts[cname] = {
            "n_specimens": int(len(fds.meta)),
            "n_evaluable_specimens": int(np.sum(ev_mask)),
            "arms": {ak: {k: v for k, v in av.items() if k != "vectors"} for ak, av in arm_nodes.items()},
            "comparisons": comps,
        }
        for ak, av in arm_nodes.items():
            for vk, arr in av["vectors"].items():
                vecs[f"track_c/C1/{cname}/{ak}/{vk}"] = arr

    return {
        "label": "Donmuş Beat AML Model Aktarımı (Sıfır FIMM Yeniden Fit, Sıfır FIMM Kohort İstatistiği)",
        "reselected_beataml_static_dopt_top8": top8_names,
        "conformal_intervals_transferred": False,
        "cohorts": out_cohorts,
    }, vecs


def _run_experiment_c2_within_fimm(
    fimm_idx: FIMMDataset,
    target_idx: list[int],
    k_budgets: tuple[int, ...] = FIMM_K_BUDGETS,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Deney C2: FIMM 109 hasta içinde hasta-ayrık 5-Kat CV x 3 tohum yöntem tekrarı (ayrı etiketli)."""
    n_pat, n_drugs = fimm_idx.Y_aligned.shape
    target_names = [fimm_idx.drug_names[i] for i in target_idx]
    k_ref = 5 if 5 in k_budgets else k_budgets[0]

    # Beat AML drug_family grafı ortak 102 FIMM ilacına eşlenir
    B_fimm, _ = load_beataml_drug_family_incidence(fimm_idx.drug_names, include_gene_sheet=False)
    G_true = build_cosine_similarity_gram(B_fimm)
    G_perms = [
        build_cosine_similarity_gram(permute_bipartite_incidence_preserving_degrees(B_fimm, seed=6000 + m))
        for m in range(10)
    ]

    meta_cv = fimm_idx.meta.copy()
    meta_cv["dbgap_subject_id"] = meta_cv["patient_int_id"]

    arm_names = (
        ["M0_k0", "M0_k5", "M1_k0"]
        + [f"M2a_StaticDOpt_k{k}" for k in k_budgets]
        + [f"M3_StaticDOpt_k{k}" for k in k_budgets]
        + ["M3_Random20x_k5", "M4_StaticDOpt_k5", "M4_perm_null_k5"]
    )
    seed_results: dict[str, list[dict[str, Any]]] = {a: [] for a in arm_names}

    for s in FIMM_CV_SEEDS:
        folds = make_patient_disjoint_kfold(meta_cv, n_splits=5, seed=s, strata_col="cohort")
        base_d_oof = np.full((n_pat, n_drugs), np.nan, dtype=np.float64)
        eval_mask = np.zeros(n_pat, dtype=bool)
        fold_ids = np.full(n_pat, -1, dtype=np.int64)
        preds = {a: np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for a in arm_names if a not in ("M3_Random20x_k5", "M4_perm_null_k5")}
        preds_rand = [np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for _ in range(FIMM_RANDOM_REPS)]
        preds_perm = [np.full((n_pat, n_drugs), np.nan, dtype=np.float64) for _ in range(len(G_perms))]

        for f_idx, (tr_idx, te_idx) in enumerate(folds):
            fold_ids[te_idx] = f_idx
            Y_tr, Y_te = fimm_idx.Y_aligned[tr_idx], fimm_idx.Y_aligned[te_idx]
            subj_tr = meta_cv.iloc[tr_idx]["dbgap_subject_id"].to_numpy()
            probes_pool = select_train_probe_pool(Y_tr, fimm_idx.drug_names, target_names, min_train_obs=50)
            fit_drugs = sorted(set(target_idx) | set(probes_pool))

            m0 = MeanEffectM0().fit(Y_tr, subj_tr, probes_pool, target_idx, k_values=k_budgets, inner_seed=s + f_idx)
            base_d_oof[te_idx] = (m0.mu_ + m0.b_d_)[None, :]
            m1, Y_oof = OmicsRidgeM1(top_k_genes=min(2000, fimm_idx.X_rna.shape[1])).fit_and_oof(
                fimm_idx.X_rna[tr_idx], fimm_idx.X_mut[tr_idx], Y_tr, subj_tr, fit_drug_idx=fit_drugs, target_idx=target_idx, inner_seed=s + f_idx
            )
            m1_te = m1.predict(fimm_idx.X_rna[te_idx], fimm_idx.X_mut[te_idx], n_drugs=n_drugs)
            m2a = CovarianceFewShotM2a(lam_grid=C2_SDSS_LAM_GRID).fit(
                Y_tr, subj_tr, probes_pool, target_idx, k_values=k_budgets, inner_seed=s + f_idx
            )

            ctx = precompute_inner_cv_context(
                Y_tr, subj_tr, probes_pool, target_idx, k_values=k_budgets, prior_train=Y_oof, inner_seed=s + f_idx
            )
            rho0_best, m3_covs = _eval_rho0_inner_cv(ctx, k_budgets, M4_ALPHA_GRID, C2_SDSS_LAM_GRID)
            m4_fit = _fit_pathway_cov_pair(ctx, G_true, (k_ref,), rho0_best, lam_grid=C2_SDSS_LAM_GRID)
            perm_fits = [_fit_pathway_cov_pair(ctx, Gp, (k_ref,), rho0_best, lam_grid=C2_SDSS_LAM_GRID) for Gp in G_perms]

            te_mask = get_evaluable_mask(Y_te, target_idx, probes_pool, min_targets=5, min_probes=8)
            eval_mask[te_idx] = te_mask

            preds["M0_k0"][te_idx] = (m0.mu_ + m0.b_d_)[None, :]
            preds["M1_k0"][te_idx] = m1_te

            g2l = ctx["global_to_local"]
            p_loc = np.asarray([g2l[g] for g in probes_pool], dtype=np.int64)
            ranked = ctx["active_idx"][select_static_d_optimal_ranking(m3_covs[k_ref], p_loc, ctx["t_local"], lam=rho0_best[k_ref][3])]

            for k in k_budgets:
                p_k = [take_first_k_available(ranked, Y_te[i], k)[0] for i in range(len(te_idx))]
                require_full_budget(p_k, te_mask, k, f"FIMM_C2_s{s}_f{f_idx}_k{k}")
                preds[f"M2a_StaticDOpt_k{k}"][te_idx] = m2a.predict(Y_te, p_k, target_idx, k=k)
                preds[f"M3_StaticDOpt_k{k}"][te_idx] = _predict_from_cov(Y_te, m1_te, m3_covs[k], rho0_best[k][3], p_k, ctx, target_idx)
                if k == k_ref:
                    preds["M0_k5"][te_idx] = m0.predict(Y_te, p_k, k=k)
                    preds["M4_StaticDOpt_k5"][te_idx] = _predict_from_cov(
                        Y_te, m1_te, m4_fit["free_cov"][k], m4_fit["free_params"][k][2], p_k, ctx, target_idx
                    )
                    for pi, pf in enumerate(perm_fits):
                        preds_perm[pi][te_idx] = _predict_from_cov(
                            Y_te, m1_te, pf["free_cov"][k], pf["free_params"][k][2], p_k, ctx, target_idx
                        )
                    r_reps = sample_probe_sets(Y_te, probes_pool, k=k, n_repeats=FIMM_RANDOM_REPS, seed=s + 100 + f_idx)
                    for ri, rp in enumerate(r_reps):
                        preds_rand[ri][te_idx] = _predict_from_cov(Y_te, m1_te, m3_covs[k], rho0_best[k][3], rp, ctx, target_idx)

        for a, pmat in preds.items():
            seed_results[a].append(evaluate_masked_predictions(fimm_idx.Y_aligned, pmat, base_d_oof, eval_mask, target_idx, fold_ids=fold_ids))
        seed_results["M3_Random20x_k5"].append(average_metric_nodes([
            evaluate_masked_predictions(fimm_idx.Y_aligned, pr, base_d_oof, eval_mask, target_idx, fold_ids=fold_ids)
            for pr in preds_rand
        ]))
        seed_results["M4_perm_null_k5"].append(average_metric_nodes([
            evaluate_masked_predictions(fimm_idx.Y_aligned, pp, base_d_oof, eval_mask, target_idx, fold_ids=fold_ids)
            for pp in preds_perm
        ]))

    avg_nodes = {a: average_metric_nodes(seed_results[a]) for a in arm_names}
    comps = {
        "M3_Static_k5_vs_M1_k0": paired_metric_diffs(avg_nodes["M3_StaticDOpt_k5"], avg_nodes["M1_k0"], n_boot=n_boot),
        "M3_Static_k5_vs_M0_k0": paired_metric_diffs(avg_nodes["M3_StaticDOpt_k5"], avg_nodes["M0_k0"], n_boot=n_boot),
        "M3_Static_k5_vs_M0_k5": paired_metric_diffs(avg_nodes["M3_StaticDOpt_k5"], avg_nodes["M0_k5"], n_boot=n_boot),
        "M3_Static_k5_vs_Random20x_k5": paired_metric_diffs(avg_nodes["M3_StaticDOpt_k5"], avg_nodes["M3_Random20x_k5"], n_boot=n_boot),
        "M4_Static_k5_vs_M3_Static_k5": paired_metric_diffs(avg_nodes["M4_StaticDOpt_k5"], avg_nodes["M3_StaticDOpt_k5"], n_boot=n_boot),
        "M4_Static_k5_vs_M4_perm_null_k5": paired_metric_diffs(avg_nodes["M4_StaticDOpt_k5"], avg_nodes["M4_perm_null_k5"], n_boot=n_boot),
    }
    vecs: dict[str, np.ndarray] = {}
    for a, nd in avg_nodes.items():
        for vk in VECTOR_KEYS:
            vecs[f"track_c/C2/index_109_patients/{a}/{vk}"] = nd["vectors"][vk]

    return {
        "label": "FIMM İçi Yöntem Tekrarı (5-Kat Hasta-Ayrık CV x 3 Tohum; Donmuş Aktarımdan Ayrıdır)",
        "n_patients_total": int(n_pat),
        "n_eval": int(avg_nodes["M3_StaticDOpt_k5"]["n_eval"]),
        "arms": {a: {k: v for k, v in nd.items() if k not in ("vectors", "drug_spearmans")} for a, nd in avg_nodes.items()},
        "comparisons": comps,
    }, vecs


def run_track_c_fimm(
    ds_ba: BeatAMLDataset,
    n_boot: int = 1000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Kulvar C: FIMM/Malani C1 (donmuş aktarım) ve C2 (FIMM içi yöntem tekrarı) koşucusu."""
    t0 = time.perf_counter()
    ba_ens = [g.split(".")[0] for g in ds_ba.gene_names]
    fimm_all, _ = load_fimm_aml_dataset(ds_ba.drug_names, common_gene_ensembl_ids=ba_ens)
    idx_pos = select_fimm_index_specimen_positions(fimm_all.meta)
    fimm_idx = fimm_all.subset_by_positions(idx_pos)

    fimm_drug_to_i = {d: i for i, d in enumerate(fimm_all.drug_names)}
    target_names = [d for d in CLINICAL_CANDIDATE_DRUGS if d in fimm_drug_to_i]
    target_idx = [fimm_drug_to_i[d] for d in target_names]

    # Beat AML all-471 prob havuzu ile FIMM ortak katalog kesişimi (yalnız Beat AML gözlem sayılarıyla)
    ba_probes_all = select_train_probe_pool(ds_ba.Y_auc, ds_ba.drug_names, CLINICAL_CANDIDATE_DRUGS, min_train_obs=100)
    common_probes = [fimm_drug_to_i[ds_ba.drug_names[i]] for i in ba_probes_all if ds_ba.drug_names[i] in fimm_drug_to_i]

    c1_res, v_c1 = _run_experiment_c1_frozen_transfer(ds_ba, fimm_all, fimm_idx, target_idx, common_probes, n_boot=n_boot)
    c2_res, v_c2 = _run_experiment_c2_within_fimm(fimm_idx, target_idx, n_boot=n_boot)

    all_vecs = {**v_c1, **v_c2}
    return {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "common_targets_count": int(len(target_idx)),
        "common_probes_count": int(len(common_probes)),
        "common_genes_count": int(len(fimm_all.gene_ensembl_ids)),
        "experiment_c1_frozen_transfer": c1_res,
        "experiment_c2_within_fimm_replication": c2_res,
    }, all_vecs
