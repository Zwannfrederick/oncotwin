"""Sprint 4 deney koşucusu: Hasta-birimli split-conformal, kısa liste ve OOD çekinme (`ABSTAIN`/`AURC`)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import (
    BeatAMLDataset,
    load_beataml_aml_dataset,
    select_index_specimen_positions,
)
from src.oncotwin.eval.splits import (
    get_evaluable_mask,
    make_patient_disjoint_kfold,
    make_sprint04_splits,
    select_train_probe_pool,
)
from src.oncotwin.eval.uncertainty import (
    OmicsPCADistanceScorer,
    json_sanitize,
    m3_conditional_posterior_scale,
    paired_aurc_bootstrap_ci,
)
from src.oncotwin.experiments.sprint04_eval import (
    _build_locked_comparisons,
    _compare_pair_node,
    _evaluate_arm_outputs,
)
from src.oncotwin.experiments.sprint04_report import render_sprint04_md
from src.oncotwin.models.baselines import HybridFewShotM3, MeanEffectM0, OmicsRidgeM1
from src.oncotwin.models.panel_selection import (
    compute_contender_weights,
    require_full_budget,
    select_adaptive_batch_oed,
    select_static_d_optimal_ranking,
    take_first_k_available,
)

PROTOCOL_S04_PATH = Path("docs/PROTOCOL_SPRINT04.md")
DEFAULT_MANIFEST_JSON = Path("data/processed/sprint04_v2_split_manifest.json")
DEFAULT_OUT_JSON = Path("data/processed/sprint04_v2_results.json")
DEFAULT_OUT_NPZ = Path("data/processed/sprint04_v2_vectors.npz")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-04-v2-conformal-abstain.md")

SPRINT04_SEED = 2026
K_BUDGETS = (2, 5, 8)
MAIN_K = 5
ALPHA = 0.10
TARGET_ACCEPT_RATE = 0.80
MIN_TRAIN_PROBE_OBS = 100
MIN_EVAL_TARGETS = 5
MIN_EVAL_PROBES_ARCHIVE = 8
TOP_K_GENES = 2000


def _arm_specs(k_budgets: tuple[int, ...]) -> list[tuple[str, str, int]]:
    specs: list[tuple[str, str, int]] = [("M0_k0", "M0", 0), ("M1_k0", "M1", 0)]
    for k in k_budgets:
        for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
            specs.append((f"{pol}_k{k}", pol, k))
    return specs


def _fit_sprint04_fold(
    ds: BeatAMLDataset,
    train_idx: np.ndarray,
    target_idx: list[int],
    k_budgets: tuple[int, ...],
    inner_seed: int,
    min_train_obs: int,
    top_k_genes: int = TOP_K_GENES,
) -> dict[str, Any]:
    Y_tr = ds.Y_auc[train_idx]
    X_rna_tr = ds.X_rna[train_idx]
    X_mut_tr = ds.X_mut[train_idx]
    subj_tr = ds.meta.iloc[train_idx]["dbgap_subject_id"].to_numpy()

    target_names = [ds.drug_names[i] for i in target_idx]
    target_set = set(target_idx)
    probe_pool_idx = select_train_probe_pool(Y_tr, ds.drug_names, target_names, min_train_obs=min_train_obs)
    fit_drugs = sorted(target_set | set(probe_pool_idx))
    k_ref = 5 if 5 in k_budgets else k_budgets[0]

    m0 = MeanEffectM0().fit(Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=k_budgets, inner_seed=inner_seed)
    m1, Y_oof = OmicsRidgeM1(top_k_genes=min(top_k_genes, X_rna_tr.shape[1])).fit_and_oof(
        X_rna_tr, X_mut_tr, Y_tr, subj_tr, fit_drug_idx=fit_drugs, target_idx=target_idx, inner_seed=inner_seed
    )
    m3 = HybridFewShotM3().fit(
        m1, Y_oof, Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=k_budgets, inner_seed=inner_seed
    )
    pca_scorer = OmicsPCADistanceScorer(top_k_genes=min(top_k_genes, X_rna_tr.shape[1]), max_components=20).fit(
        X_rna_tr, X_mut_tr
    )

    cov_model = m3._cov_model
    g2l = cov_model.global_to_local_
    probe_local = np.asarray([g2l[g] for g in probe_pool_idx], dtype=np.int64)
    target_local = np.asarray([g2l[g] for g in target_idx], dtype=np.int64)
    active_arr = cov_model.active_idx_
    cov_ref = cov_model.cov_by_k_[k_ref]
    _, lam_ref = cov_model.best_params_by_k_[k_ref]
    static_dopt_global = active_arr[
        select_static_d_optimal_ranking(cov_ref, probe_local, target_local, lam=lam_ref)
    ]

    return {
        "probe_pool_idx": probe_pool_idx,
        "probe_local": probe_local,
        "target_local": target_local,
        "active_arr": active_arr,
        "g2l": g2l,
        "mu_tr": m0.mu_,
        "b_d_tr": m0.b_d_,
        "M0": m0,
        "M1": m1,
        "M3": m3,
        "pca_scorer": pca_scorer,
        "cov_ref": cov_ref,
        "lam_ref": lam_ref,
        "static_dopt_global": static_dopt_global,
        "k_ref": k_ref,
        "contender_seed": int(inner_seed) + 1,
    }


def _choose_locked_panels(
    fit_obj: dict[str, Any],
    contender_w: np.ndarray,
    policy: str,
    k: int,
    uniform_w: np.ndarray,
    target_set: set[int],
) -> list[np.ndarray]:
    """Yalnız eğitim kataloğu + (adaptif için) omik ağırlıklarla kilitli k-prob paneli seçer.

    Y_eval veya değerlendirme maskesi parametre olarak dahi alınmaz (Sol §5 #6(d)).
    """
    n_pat = len(contender_w)
    if k <= 0 or policy in ("M0", "M1"):
        empty = np.array([], dtype=np.int64)
        return [empty for _ in range(n_pat)]

    cov_model = fit_obj["M3"]._cov_model
    cov_k = cov_model.cov_by_k_[k]
    _, lam_k = cov_model.best_params_by_k_[k]
    probe_local = fit_obj["probe_local"]
    target_local = fit_obj["target_local"]
    active_arr = fit_obj["active_arr"]

    if policy == "StaticDOpt":
        locked = fit_obj["static_dopt_global"][:k].copy()
        assert len(locked) == k and set(int(x) for x in locked).isdisjoint(target_set)
        return [locked.copy() for _ in range(n_pat)]
    if policy == "EqualWeightVarRed":
        locked = active_arr[select_adaptive_batch_oed(cov_k, probe_local, target_local, uniform_w, k=k, lam=lam_k)]
        assert len(locked) == k and set(int(x) for x in locked).isdisjoint(target_set)
        return [locked.copy() for _ in range(n_pat)]
    if policy == "AdaptiveOED":
        panels: list[np.ndarray] = []
        for i in range(n_pat):
            chosen = active_arr[
                select_adaptive_batch_oed(cov_k, probe_local, target_local, contender_w[i], k=k, lam=lam_k)
            ]
            assert len(chosen) == k and set(int(x) for x in chosen).isdisjoint(target_set)
            panels.append(chosen)
        return panels
    raise ValueError(f"Bilinmeyen politika: {policy}")


def _open_archive_mask_for_locked(
    locked_panels: list[np.ndarray],
    Y_eval: np.ndarray,
    k: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Kilitli panelleri arşiv maskesiyle eşleştirir; eksik prob varsa geri doldurmaz, teknik çekinme işaretler."""
    n_pat = Y_eval.shape[0]
    probes_out: list[np.ndarray] = []
    complete_mask = np.zeros(n_pat, dtype=bool)
    n_obs = np.zeros(n_pat, dtype=np.int64)
    for i in range(n_pat):
        locked = locked_panels[i]
        cnt = int(np.sum(np.isfinite(Y_eval[i, locked]))) if k > 0 else 0
        n_obs[i] = cnt
        is_full = (cnt == k)
        complete_mask[i] = is_full
        probes_out.append(locked if is_full and k > 0 else np.array([], dtype=np.int64))
    return probes_out, complete_mask, n_obs


def _select_archive_panels(
    Y_eval: np.ndarray,
    fit_obj: dict[str, Any],
    contender_w: np.ndarray,
    policy: str,
    k: int,
    eval_mask: np.ndarray,
    uniform_w: np.ndarray,
    target_set: set[int],
) -> list[np.ndarray]:
    """Mevcut ölçümler içinden (AUC değerlerini görmeden) tam k prob seçer."""
    n_pat = Y_eval.shape[0]
    if k <= 0 or policy in ("M0", "M1"):
        empty = np.array([], dtype=np.int64)
        return [empty for _ in range(n_pat)]

    probes: list[np.ndarray] = []
    if policy == "StaticDOpt":
        ranked = fit_obj["static_dopt_global"]
        for i in range(n_pat):
            chosen, _ = take_first_k_available(ranked, Y_eval[i], k)
            probes.append(chosen)
    elif policy in ("EqualWeightVarRed", "AdaptiveOED"):
        cov_model = fit_obj["M3"]._cov_model
        cov_k = cov_model.cov_by_k_[k]
        _, lam_k = cov_model.best_params_by_k_[k]
        g2l = fit_obj["g2l"]
        pool_arr = np.asarray(fit_obj["probe_pool_idx"], dtype=np.int64)
        active_arr = fit_obj["active_arr"]
        target_local = fit_obj["target_local"]
        for i in range(n_pat):
            avail_g = pool_arr[np.isfinite(Y_eval[i, pool_arr])]
            avail_loc = np.asarray([g2l[int(g)] for g in avail_g], dtype=np.int64)
            w_i = uniform_w if policy == "EqualWeightVarRed" else contender_w[i]
            chosen = active_arr[select_adaptive_batch_oed(cov_k, avail_loc, target_local, w_i, k=k, lam=lam_k)]
            probes.append(chosen)
    else:
        raise ValueError(f"Bilinmeyen politika: {policy}")

    require_full_budget(probes, eval_mask, k, f"archive_available/{policy}_k{k}")
    for i in np.flatnonzero(eval_mask):
        assert set(int(x) for x in probes[i]).isdisjoint(target_set)
    return probes


def _predict_arm(
    ds_sub: BeatAMLDataset,
    probes: list[np.ndarray],
    fit_obj: dict[str, Any],
    m1_pred: np.ndarray,
    target_idx: list[int],
    k: int,
    complete_mask: np.ndarray | None = None,
    policy: str = "M3",
) -> tuple[np.ndarray, np.ndarray]:
    """Kol tahminlerini ve seçilen prob seti başına M3 koşullu posterior ölçeğini üretir."""
    n_pat = len(ds_sub.meta)
    g2l = fit_obj["g2l"]
    if k <= 0:
        Y_pred = fit_obj["M0"].predict(ds_sub.Y_auc, probes, k=0) if policy == "M0" else m1_pred.copy()
        empty_loc = [np.array([], dtype=np.int64) for _ in range(n_pat)]
        m3_sc = m3_conditional_posterior_scale(fit_obj["cov_ref"], empty_loc, fit_obj["target_local"], fit_obj["lam_ref"])
    else:
        Y_pred = fit_obj["M3"].predict(
            ds_sub.X_rna, ds_sub.X_mut, ds_sub.Y_auc, probes, target_idx, k=k, prior_eval=m1_pred
        )
        cov_model = fit_obj["M3"]._cov_model
        cov_k = cov_model.cov_by_k_[k]
        _, lam_k = cov_model.best_params_by_k_[k]
        probes_loc = [
            np.asarray([g2l[int(g)] for g in s], dtype=np.int64) for s in probes
        ]
        m3_sc = m3_conditional_posterior_scale(cov_k, probes_loc, fit_obj["target_local"], lam=lam_k)

    if complete_mask is not None:
        Y_pred = Y_pred.copy()
        Y_pred[~complete_mask, :] = np.nan
        m3_sc = m3_sc.copy()
        m3_sc[~complete_mask] = np.nan
    return Y_pred, m3_sc


def _lock_threshold_quantile(vals: np.ndarray, valid_mask: np.ndarray, target_accept: float) -> float:
    m = valid_mask & np.isfinite(vals)
    if not np.any(m):
        return float("nan")
    return float(np.quantile(vals[m], target_accept, method="higher"))


def _compute_inner_oof_thresholds(
    ds: BeatAMLDataset,
    train_idx: np.ndarray,
    target_idx: list[int],
    k_budgets: tuple[int, ...],
    seed: int,
    target_accept: float,
    min_train_obs: int,
    min_targets: int,
    min_probes: int,
    inner_n_splits: int,
) -> dict[str, Any]:
    """train_tune içi hasta-ayrık OOF doğrulamasıyla sabit %80 kabul eşiklerini kilitler."""
    meta_tr = ds.meta.iloc[train_idx].reset_index(drop=True)
    n_tr = len(train_idx)
    folds = make_patient_disjoint_kfold(meta_tr, n_splits=inner_n_splits, seed=seed, strata_col="cohort")
    target_set = set(target_idx)
    uniform_w = np.full(len(target_idx), 1.0 / len(target_idx), dtype=np.float64)
    specs = _arm_specs(k_budgets)

    oof_omics = np.full(n_tr, np.nan, dtype=np.float64)
    oof_arch_mask = np.zeros(n_tr, dtype=bool)
    oof_tgt_mask = np.zeros(n_tr, dtype=bool)
    oof_m3: dict[tuple[str, str], np.ndarray] = {
        (sc, arm): np.full(n_tr, np.nan, dtype=np.float64)
        for sc in ("archive_available", "locked_train_catalog")
        for arm, _, _ in specs
    }
    oof_lock_complete = {arm: np.zeros(n_tr, dtype=bool) for arm, _, _ in specs}

    for f_idx, (itr_loc, iva_loc) in enumerate(folds):
        g_tr = train_idx[itr_loc]
        g_va = train_idx[iva_loc]
        fit_in = _fit_sprint04_fold(
            ds, g_tr, target_idx, k_budgets=k_budgets, inner_seed=seed + 100 + f_idx, min_train_obs=min_train_obs
        )
        ds_va = ds.subset_by_positions(g_va)
        oof_omics[iva_loc] = fit_in["pca_scorer"].score(ds_va.X_rna, ds_va.X_mut)
        oof_arch_mask[iva_loc] = get_evaluable_mask(
            ds_va.Y_auc, target_idx, fit_in["probe_pool_idx"], min_targets=min_targets, min_probes=min_probes
        )
        oof_tgt_mask[iva_loc] = np.sum(np.isfinite(ds_va.Y_auc[:, target_idx]), axis=1) >= min_targets

        m1_va = fit_in["M1"].predict(ds_va.X_rna, ds_va.X_mut, n_drugs=ds.Y_auc.shape[1])
        cov_cc = fit_in["cov_ref"][np.ix_(fit_in["target_local"], fit_in["target_local"])]
        cont_va = compute_contender_weights(m1_va[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=fit_in["contender_seed"])

        for arm_name, pol, k in specs:
            probes_arch = _select_archive_panels(
                ds_va.Y_auc, fit_in, cont_va, pol, k, oof_arch_mask[iva_loc], uniform_w, target_set
            )
            _, m3_arch = _predict_arm(
                ds_va, probes_arch, fit_in, m1_va, target_idx, k=k, complete_mask=oof_arch_mask[iva_loc], policy=pol
            )
            oof_m3[("archive_available", arm_name)][iva_loc] = m3_arch

            locked_p = _choose_locked_panels(fit_in, cont_va, pol, k, uniform_w, target_set)
            probes_lock, comp_mask, _ = _open_archive_mask_for_locked(locked_p, ds_va.Y_auc, k)
            valid_lock = oof_tgt_mask[iva_loc] & comp_mask
            oof_lock_complete[arm_name][iva_loc] = valid_lock
            _, m3_lock = _predict_arm(
                ds_va, probes_lock, fit_in, m1_va, target_idx, k=k, complete_mask=valid_lock, policy=pol
            )
            oof_m3[("locked_train_catalog", arm_name)][iva_loc] = m3_lock

    taus: dict[str, Any] = {
        "archive_available": {
            "omics_pca_dist": _lock_threshold_quantile(oof_omics, oof_arch_mask, target_accept),
            "m3_post_scale": {
                arm: _lock_threshold_quantile(oof_m3[("archive_available", arm)], oof_arch_mask, target_accept)
                for arm, _, _ in specs
            },
        },
        "locked_train_catalog": {
            "omics_pca_dist_target_elig": _lock_threshold_quantile(oof_omics, oof_tgt_mask, target_accept),
            "omics_pca_dist": {
                arm: _lock_threshold_quantile(oof_omics, oof_lock_complete[arm], target_accept)
                for arm, _, _ in specs
            },
            "m3_post_scale": {
                arm: _lock_threshold_quantile(oof_m3[("locked_train_catalog", arm)], oof_lock_complete[arm], target_accept)
                for arm, _, _ in specs
            },
        },
    }
    return taus


def run_regime_sprint04(
    ds: BeatAMLDataset,
    regime_split: dict[str, np.ndarray],
    target_idx: list[int],
    regime_name: str,
    seed: int = SPRINT04_SEED,
    alpha: float = ALPHA,
    target_accept: float = TARGET_ACCEPT_RATE,
    min_train_obs: int = MIN_TRAIN_PROBE_OBS,
    min_targets: int = MIN_EVAL_TARGETS,
    min_probes: int = MIN_EVAL_PROBES_ARCHIVE,
    k_budgets: tuple[int, ...] = K_BUDGETS,
    inner_n_splits: int = 5,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Tek rejim (ID 60/20/20 veya Clean Waves 1+2 -> Waves 3+4) için tam Sprint 4 akışını koşar."""
    tr_idx = regime_split["train_tune"]
    cal_idx = regime_split["calibration"]
    te_idx = regime_split["test"]
    target_set = set(target_idx)
    uniform_w = np.full(len(target_idx), 1.0 / len(target_idx), dtype=np.float64)
    specs = _arm_specs(k_budgets)

    taus = _compute_inner_oof_thresholds(
        ds, tr_idx, target_idx, k_budgets=k_budgets, seed=seed, target_accept=target_accept,
        min_train_obs=min_train_obs, min_targets=min_targets, min_probes=min_probes, inner_n_splits=inner_n_splits,
    )
    full_fit = _fit_sprint04_fold(
        ds, tr_idx, target_idx, k_budgets=k_budgets, inner_seed=seed, min_train_obs=min_train_obs
    )

    ds_cal = ds.subset_by_positions(cal_idx)
    ds_te = ds.subset_by_positions(te_idx)
    base_d_cal = np.broadcast_to((full_fit["mu_tr"] + full_fit["b_d_tr"])[None, :], ds_cal.Y_auc.shape)
    base_d_te = np.broadcast_to((full_fit["mu_tr"] + full_fit["b_d_tr"])[None, :], ds_te.Y_auc.shape)

    n_drugs = ds.Y_auc.shape[1]
    m1_cal = full_fit["M1"].predict(ds_cal.X_rna, ds_cal.X_mut, n_drugs=n_drugs)
    m1_te = full_fit["M1"].predict(ds_te.X_rna, ds_te.X_mut, n_drugs=n_drugs)
    cov_cc = full_fit["cov_ref"][np.ix_(full_fit["target_local"], full_fit["target_local"])]
    c_seed = full_fit["contender_seed"]
    cont_cal = compute_contender_weights(m1_cal[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=c_seed)
    cont_te = compute_contender_weights(m1_te[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=c_seed)
    omics_te = full_fit["pca_scorer"].score(ds_te.X_rna, ds_te.X_mut)

    arch_cal_mask = get_evaluable_mask(ds_cal.Y_auc, target_idx, full_fit["probe_pool_idx"], min_targets, min_probes)
    arch_te_mask = get_evaluable_mask(ds_te.Y_auc, target_idx, full_fit["probe_pool_idx"], min_targets, min_probes)
    tgt_cal_mask = np.sum(np.isfinite(ds_cal.Y_auc[:, target_idx]), axis=1) >= min_targets
    tgt_te_mask = np.sum(np.isfinite(ds_te.Y_auc[:, target_idx]), axis=1) >= min_targets

    vectors_out: dict[str, np.ndarray] = {
        "cal_subject_ids": ds_cal.meta["dbgap_subject_id"].to_numpy(dtype=np.int64),
        "test_subject_ids": ds_te.meta["dbgap_subject_id"].to_numpy(dtype=np.int64),
        "test_omics_pca_dist": omics_te,
        "test_archive_eval_mask": arch_te_mask,
        "test_target_eligible_mask": tgt_te_mask,
    }

    # 1. archive_available
    arch_arms: dict[str, Any] = {}
    arch_vecs: dict[str, dict[str, np.ndarray]] = {}
    for arm_name, pol, k in specs:
        p_cal = _select_archive_panels(ds_cal.Y_auc, full_fit, cont_cal, pol, k, arch_cal_mask, uniform_w, target_set)
        p_te = _select_archive_panels(ds_te.Y_auc, full_fit, cont_te, pol, k, arch_te_mask, uniform_w, target_set)
        Yp_cal, _ = _predict_arm(ds_cal, p_cal, full_fit, m1_cal, target_idx, k=k, complete_mask=arch_cal_mask, policy=pol)
        Yp_te, m3_te = _predict_arm(ds_te, p_te, full_fit, m1_te, target_idx, k=k, complete_mask=arch_te_mask, policy=pol)
        node, vdict = _evaluate_arm_outputs(
            ds_cal, ds_te, base_d_cal, base_d_te, Yp_cal, Yp_te, m3_te, omics_te,
            arch_cal_mask, arch_te_mask, arch_te_mask, target_idx,
            tau_omics=taus["archive_available"]["omics_pca_dist"],
            tau_m3=taus["archive_available"]["m3_post_scale"][arm_name],
            alpha=alpha, coverage_cap=None,
        )
        arch_arms[arm_name] = node
        arch_vecs[arm_name] = vdict
        for vn, varr in vdict.items():
            vectors_out[f"archive_available/{arm_name}/{vn}"] = varr

    # 2. locked_train_catalog (her kol kendi tam-panelinde betimlenir; kıyaslar yalnızca ortak maskede yapılır)
    lock_arms: dict[str, Any] = {}
    lock_vecs: dict[str, dict[str, np.ndarray]] = {}
    for arm_name, pol, k in specs:
        l_cal = _choose_locked_panels(full_fit, cont_cal, pol, k, uniform_w, target_set)
        l_te = _choose_locked_panels(full_fit, cont_te, pol, k, uniform_w, target_set)
        p_cal, comp_cal, _ = _open_archive_mask_for_locked(l_cal, ds_cal.Y_auc, k)
        p_te, comp_te, nobs_te = _open_archive_mask_for_locked(l_te, ds_te.Y_auc, k)
        v_cal = tgt_cal_mask & comp_cal
        v_te = tgt_te_mask & comp_te
        Yp_cal, _ = _predict_arm(ds_cal, p_cal, full_fit, m1_cal, target_idx, k=k, complete_mask=v_cal, policy=pol)
        Yp_te, m3_te = _predict_arm(ds_te, p_te, full_fit, m1_te, target_idx, k=k, complete_mask=v_te, policy=pol)

        n_tgt_te = int(np.sum(tgt_te_mask))
        n_comp_te = int(np.sum(v_te))
        rate_te = (n_comp_te / float(n_tgt_te)) if n_tgt_te > 0 else 0.0
        node, vdict = _evaluate_arm_outputs(
            ds_cal, ds_te, base_d_cal, base_d_te, Yp_cal, Yp_te, m3_te, omics_te,
            v_cal, v_te, tgt_te_mask, target_idx,
            tau_omics=taus["locked_train_catalog"]["omics_pca_dist"][arm_name],
            tau_m3=taus["locked_train_catalog"]["m3_post_scale"][arm_name],
            alpha=alpha, coverage_cap=None,
        )
        node["availability"] = {
            "n_target_eligible_cal": int(np.sum(tgt_cal_mask)),
            "n_complete_cal": int(np.sum(v_cal)),
            "panel_availability_rate_cal": (int(np.sum(v_cal)) / float(np.sum(tgt_cal_mask))) if np.any(tgt_cal_mask) else 0.0,
            "n_target_eligible_test": n_tgt_te,
            "n_complete_test": n_comp_te,
            "n_technical_abstain": n_tgt_te - n_comp_te,
            "panel_availability_rate_test": rate_te,
            "technical_abstain_rate_test": 1.0 - rate_te if n_tgt_te > 0 else float("nan"),
            "mean_observed_probes_of_k": float(np.mean(nobs_te[tgt_te_mask])) if n_tgt_te > 0 else float("nan"),
            "locked_panel_example": l_te[0].tolist() if k > 0 and pol in ("StaticDOpt", "EqualWeightVarRed") else None,
        }
        lock_arms[arm_name] = node
        lock_vecs[arm_name] = vdict
        for vn, varr in vdict.items():
            vectors_out[f"locked_train_catalog/{arm_name}/{vn}"] = varr

    # 3. Eşleştirilmiş karşılaştırmalar
    policy_pairs_arch: dict[str, Any] = {}
    for k in k_budgets:
        for a, b in (("AdaptiveOED", "EqualWeightVarRed"), ("AdaptiveOED", "StaticDOpt"), ("EqualWeightVarRed", "StaticDOpt")):
            policy_pairs_arch[f"k{k}_{a}_vs_{b}"] = _compare_pair_node(
                arch_vecs[f"{a}_k{k}"], arch_vecs[f"{b}_k{k}"], n_boot=n_boot, seed=seed
            )
    k_main = 5 if 5 in k_budgets else k_budgets[0]
    policy_pairs_arch[f"StaticDOpt_k{k_main}_vs_M1_k0"] = _compare_pair_node(
        arch_vecs[f"StaticDOpt_k{k_main}"], arch_vecs["M1_k0"], n_boot=n_boot, seed=seed
    )
    policy_pairs_arch[f"StaticDOpt_k{k_main}_vs_M0_k0"] = _compare_pair_node(
        arch_vecs[f"StaticDOpt_k{k_main}"], arch_vecs["M0_k0"], n_boot=n_boot, seed=seed
    )
    policy_pairs_arch["M1_k0_vs_M0_k0"] = _compare_pair_node(
        arch_vecs["M1_k0"], arch_vecs["M0_k0"], n_boot=n_boot, seed=seed
    )

    aurc_pairs_arch: dict[str, Any] = {}
    for arm_name in [f"StaticDOpt_k{k}" for k in k_budgets] + [f"AdaptiveOED_k{k_main}"]:
        reg_vec = arch_vecs[arm_name]["patient_regret1"][arch_te_mask]
        om_vec = omics_te[arch_te_mask]
        m3_vec = arch_vecs[arm_name]["m3_post_scale"][arch_te_mask]
        aurc_pairs_arch[f"{arm_name}_omics_vs_m3_post_scale"] = paired_aurc_bootstrap_ci(
            om_vec, m3_vec, reg_vec, n_boot=n_boot, seed=seed
        )
        aurc_pairs_arch[f"{arm_name}_omics_vs_random"] = paired_aurc_bootstrap_ci(
            om_vec, None, reg_vec, n_boot=n_boot, seed=seed
        )

    lock_comps = _build_locked_comparisons(lock_vecs, omics_te, k_budgets, n_boot=n_boot, seed=seed)

    return {
        "regime": regime_name,
        "documented_plate_catalog_status": "hesaplanamadı (Beat AML 2.0 içinde bağımsız plaka manifesti yok)",
        "inner_oof_taus": taus,
        "counts": {
            "train_tune": int(len(tr_idx)),
            "calibration": int(len(cal_idx)),
            "test": int(len(te_idx)),
            "archive_eval_cal": int(np.sum(arch_cal_mask)),
            "archive_eval_test": int(np.sum(arch_te_mask)),
            "target_eligible_cal": int(np.sum(tgt_cal_mask)),
            "target_eligible_test": int(np.sum(tgt_te_mask)),
            "train_probe_pool_size": int(len(full_fit["probe_pool_idx"])),
        },
        "scenarios": {
            "archive_available": {"arms": arch_arms},
            "locked_train_catalog": {"arms": lock_arms},
        },
        "comparisons": {
            "archive_available": {"policy_pairs": policy_pairs_arch, "aurc_pairs": aurc_pairs_arch},
            "locked_train_catalog": lock_comps,
        },
        "vectors": vectors_out,
    }


def run_sprint04(
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_npz: Path = DEFAULT_OUT_NPZ,
    manifest_json: Path = DEFAULT_MANIFEST_JSON,
) -> dict[str, Any]:
    if not PROTOCOL_S04_PATH.exists():
        raise FileNotFoundError(f"Önce protokol dondurulmalı: {PROTOCOL_S04_PATH}")

    t0 = time.perf_counter()
    ds_all = load_beataml_aml_dataset()
    ds_idx = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    target_idx = [ds_idx.drug_names.index(d) for d in CLINICAL_CANDIDATE_DRUGS]

    splits = make_sprint04_splits(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects, seed=SPRINT04_SEED)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(splits["manifest"], f, indent=2, ensure_ascii=False)

    regimes_out: dict[str, Any] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for rname in ("id_60_20_20", "wave_ood"):
        r_res = run_regime_sprint04(ds_idx, splits[rname], target_idx, regime_name=rname)
        vecs = r_res.pop("vectors")
        regimes_out[rname] = r_res
        for vk, varr in vecs.items():
            npz_payload[f"{rname}/{vk}"] = varr

    res = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "seed": SPRINT04_SEED,
        "alpha": ALPHA,
        "target_accept_rate": TARGET_ACCEPT_RATE,
        "protocol_notes": {
            "version": "v2 (DEV-OPUS-13..16 / SOL-09..12)",
            "v1_preserved": True,
            "sol_09_fix": "Fold başına tek kilitli contender_seed (kalibrasyon ve test aynı sabit seçici fonksiyon).",
            "sol_10_fix": "Kilitli senaryoda ortak tam-panel hasta maskesinde eşleştirilmiş kıyas (n<5 -> hesaplanamadı) ve bağ-kesen cap etiketi.",
            "sol_11_fix": "Sonlu aralık yayımlanmadığında (q=inf) kapsam ve kısa liste alanları NaN/hesaplanamadı.",
            "sol_12_fix": "PROTOCOL_SPRINT04.md §1.2 gereği M0(0) ek taban kolu v2'ye eklendi (v1 sapması kapatıldı).",
            "pending_addendum_19": "DEV-OPUS-19 (PROTOCOL_SPRINT04_ADDENDUM_19.md) Sol A1 netleştirmesi bekliyor; v2'de uygulanmadı.",
        },
        "regimes": regimes_out,
    }
    clean_res = json_sanitize(res)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_res, f, indent=2, ensure_ascii=False, allow_nan=False)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **npz_payload)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_sprint04_md(clean_res), encoding="utf-8")
    return clean_res


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoTwin Sprint 4 split-conformal & OOD abstention.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    args = parser.parse_args()
    res = run_sprint04(args.out_json, args.out_md, args.out_npz, args.manifest_json)
    print(f"Sprint 4 tamamlandı ({res['elapsed_seconds']:.1f} s) -> {args.out_md}")


if __name__ == "__main__":
    main()
