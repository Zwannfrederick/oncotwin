"""Sprint 2 deney koşucusu: Protokol uyumlu baseline'lar (M0, M1, M2a, M2b, M3) ve sızıntı testleri."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS, DEFAULT_RAW_DIR
from src.oncotwin.data.dataset import (
    BeatAMLDataset,
    load_beataml_aml_dataset,
    select_index_specimen_positions,
)
from src.oncotwin.eval.metrics import (
    evaluate_masked_predictions,
    evaluate_predictions,
    paired_bootstrap_ci,
)
from src.oncotwin.eval.splits import (
    get_evaluable_mask,
    make_clean_wave_split,
    make_patient_disjoint_kfold,
    sample_probe_sets,
    select_train_probe_pool,
)
from src.oncotwin.models.baselines import (
    CovarianceFewShotM2a,
    HybridFewShotM3,
    LowRankSVDM2b,
    MeanEffectM0,
    OmicsRidgeM1,
)

PROTOCOL_PATH = Path("docs/PROTOCOL.md")
DEFAULT_OUT_JSON = Path("data/processed/sprint02_results.json")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-02-baselines.md")

CV_SEEDS = (42, 137, 2026)
K_BUDGETS = (0, 3, 5, 8)
N_PROBE_REPEATS = 10
MIN_TRAIN_PROBE_OBS = 100
MODELS = ("M0", "M1", "M2a", "M2b", "M3")


def _fit_split_models(
    ds: BeatAMLDataset,
    train_idx: np.ndarray,
    target_idx: list[int],
    inner_seed: int,
) -> dict[str, Any]:
    """Yalnızca train_idx üzerinde tüm modelleri ve prob havuzunu fit eder."""
    Y_tr = ds.Y_auc[train_idx]
    X_rna_tr = ds.X_rna[train_idx]
    X_mut_tr = ds.X_mut[train_idx]
    subj_tr = ds.meta.iloc[train_idx]["dbgap_subject_id"].to_numpy()

    probe_pool_idx = select_train_probe_pool(
        Y_tr, ds.drug_names, CLINICAL_CANDIDATE_DRUGS, min_train_obs=MIN_TRAIN_PROBE_OBS
    )
    fit_drugs = sorted(set(target_idx) | set(probe_pool_idx))

    m0 = MeanEffectM0().fit(Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=(3, 5, 8), inner_seed=inner_seed)
    m1, Y_oof = OmicsRidgeM1().fit_and_oof(
        X_rna_tr, X_mut_tr, Y_tr, subj_tr, fit_drug_idx=fit_drugs, target_idx=target_idx, inner_seed=inner_seed
    )
    m2a = CovarianceFewShotM2a().fit(
        Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=(3, 5, 8), inner_seed=inner_seed
    )
    m2b = LowRankSVDM2b().fit(
        Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=(3, 5, 8), inner_seed=inner_seed
    )
    m3 = HybridFewShotM3().fit(
        m1, Y_oof, Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=(3, 5, 8), inner_seed=inner_seed
    )
    return {
        "probe_pool_idx": probe_pool_idx,
        "mu_tr": m0.mu_,
        "b_d_tr": m0.b_d_,
        "M0": m0,
        "M1": m1,
        "M2a": m2a,
        "M2b": m2b,
        "M3": m3,
    }


def run_permutation_leakage_check(ds_idx: BeatAMLDataset, target_idx: list[int]) -> dict[str, Any]:
    """Eğitim hasta etiketlerini karıştırarak M1 sızıntı kontrolü yapar."""
    results: dict[str, Any] = {}
    rng = np.random.default_rng(999)

    # 1) Clean Wave Split üzerinde permütasyon testi
    tr_w, te_w = make_clean_wave_split(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects)
    perm_order = rng.permutation(len(tr_w))
    Y_tr_perm = ds_idx.Y_auc[tr_w][perm_order]
    subj_tr_perm = ds_idx.meta.iloc[tr_w]["dbgap_subject_id"].to_numpy()[perm_order]

    probe_pool = select_train_probe_pool(Y_tr_perm, ds_idx.drug_names, CLINICAL_CANDIDATE_DRUGS, MIN_TRAIN_PROBE_OBS)
    fit_drugs = sorted(set(target_idx) | set(probe_pool))
    m1 = OmicsRidgeM1()
    m1.fit_and_oof(
        ds_idx.X_rna[tr_w], ds_idx.X_mut[tr_w], Y_tr_perm, subj_tr_perm, fit_drugs, target_idx, inner_seed=17
    )
    assert m1.fold_model_ is not None
    pred_te = m1.predict(ds_idx.X_rna[te_w], ds_idx.X_mut[te_w], n_drugs=ds_idx.Y_auc.shape[1])
    eval_mask = get_evaluable_mask(ds_idx.Y_auc[te_w], target_idx, probe_pool, min_targets=5, min_probes=8)
    ev = evaluate_predictions(
        ds_idx.Y_auc[te_w][eval_mask],
        pred_te[eval_mask],
        target_idx,
        m1.fold_model_.mu,
        m1.fold_model_.b_d,
    )
    ci_drug = paired_bootstrap_ci(ev["drug_spearmans"], np.zeros_like(ev["drug_spearmans"]), n_boot=1000, seed=2026)
    passed_wave = bool(ci_drug["ci_low"] <= 0.0 <= ci_drug["ci_high"] and ev["residual_r2"] <= 0.01)
    results["clean_wave_perm"] = {
        "per_drug_spearman": float(ev["per_drug_spearman"]),
        "per_drug_spearman_ci": [float(ci_drug["ci_low"]), float(ci_drug["ci_high"])],
        "residual_r2": float(ev["residual_r2"]),
        "within_patient_residual_spearman": float(ev["within_patient_residual_spearman"]),
        "passed": passed_wave,
    }

    # 2) 5-Fold CV (seed=42) üzerinde permütasyon testi (tüm Y bir kez permüte edilip kat-içi Spearman alınır)
    folds = make_patient_disjoint_kfold(ds_idx.meta, n_splits=5, seed=42)
    n_all, n_drugs = ds_idx.Y_auc.shape
    global_perm = rng.permutation(n_all)
    Y_perm_cv = ds_idx.Y_auc[global_perm]
    subj_perm_cv = ds_idx.meta["dbgap_subject_id"].to_numpy()[global_perm]
    oof_pred = np.full((n_all, n_drugs), np.nan, dtype=np.float64)
    oof_base = np.full((n_all, n_drugs), np.nan, dtype=np.float64)
    oof_eval = np.zeros(n_all, dtype=bool)
    fold_ids = np.full(n_all, -1, dtype=np.int64)
    for f_idx, (tr_pos, te_pos) in enumerate(folds):
        p_pool = select_train_probe_pool(Y_perm_cv[tr_pos], ds_idx.drug_names, CLINICAL_CANDIDATE_DRUGS, MIN_TRAIN_PROBE_OBS)
        f_drugs = sorted(set(target_idx) | set(p_pool))
        m1_f = OmicsRidgeM1()
        m1_f.fit_and_oof(
            ds_idx.X_rna[tr_pos], ds_idx.X_mut[tr_pos], Y_perm_cv[tr_pos], subj_perm_cv[tr_pos],
            f_drugs, target_idx, inner_seed=17 + f_idx
        )
        assert m1_f.fold_model_ is not None
        oof_pred[te_pos] = m1_f.predict(ds_idx.X_rna[te_pos], ds_idx.X_mut[te_pos], n_drugs=n_drugs)
        oof_base[te_pos] = (m1_f.fold_model_.mu + m1_f.fold_model_.b_d)[None, :]
        oof_eval[te_pos] = get_evaluable_mask(Y_perm_cv[te_pos], target_idx, p_pool, min_targets=5, min_probes=8)
        fold_ids[te_pos] = f_idx

    ev_cv = evaluate_masked_predictions(Y_perm_cv, oof_pred, oof_base, oof_eval, target_idx, fold_ids=fold_ids)
    ci_cv = paired_bootstrap_ci(
        ev_cv["drug_spearmans"], np.zeros_like(ev_cv["drug_spearmans"]), n_boot=1000, seed=2026
    )
    passed_cv = bool(ci_cv["ci_low"] <= 0.0 <= ci_cv["ci_high"] and ev_cv["residual_r2"] <= 0.01)
    results["cv_seed42_perm"] = {
        "per_drug_spearman": float(ev_cv["per_drug_spearman"]),
        "per_drug_spearman_ci": [float(ci_cv["ci_low"]), float(ci_cv["ci_high"])],
        "residual_r2": float(ev_cv["residual_r2"]),
        "within_patient_residual_spearman": float(ev_cv["within_patient_residual_spearman"]),
        "passed": passed_cv,
    }
    assert passed_wave and passed_cv, f"Permütasyon sızıntı kontrolü başarısız: {results}"
    return results


def _evaluate_regime_splits(
    ds: BeatAMLDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_idx: list[int],
    seed_offset: int,
    cluster_by_subject: bool = False,
) -> dict[str, Any]:
    """Bir tek seed için tüm splitlerde modelleri eğitir ve k in {0,3,5,8} x R=10 için değerlendirir."""
    n_rows, n_drugs = ds.Y_auc.shape
    fitted_splits = []
    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        fit_obj = _fit_split_models(ds, tr_idx, target_idx, inner_seed=seed_offset + f_idx)
        eval_mask_te = get_evaluable_mask(
            ds.Y_auc[te_idx], target_idx, fit_obj["probe_pool_idx"], min_targets=5, min_probes=8
        )
        m1_pred_te = fit_obj["M1"].predict(ds.X_rna[te_idx], ds.X_mut[te_idx], n_drugs=n_drugs)
        fitted_splits.append((te_idx, fit_obj, eval_mask_te, m1_pred_te))

    global_eval_mask = np.zeros(n_rows, dtype=bool)
    base_d_all = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
    fold_ids = np.full(n_rows, -1, dtype=np.int64)
    for f_idx, (te_idx, fit_obj, eval_mask_te, _) in enumerate(fitted_splits):
        global_eval_mask[te_idx] = eval_mask_te
        base_d_all[te_idx] = (fit_obj["mu_tr"] + fit_obj["b_d_tr"])[None, :]
        fold_ids[te_idx] = f_idx

    eval_fold_ids = fold_ids if len(fitted_splits) > 1 else None
    subjects_all = ds.meta["dbgap_subject_id"].to_numpy() if cluster_by_subject else None

    out_by_k: dict[int, dict[str, Any]] = {}
    for k in K_BUDGETS:
        n_rep = 1 if k == 0 else N_PROBE_REPEATS
        # Her tekrar için tüm foldlardan OOF tahminleri topla
        rep_metrics: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}

        for r in range(n_rep):
            preds_by_model = {m: np.full((n_rows, n_drugs), np.nan, dtype=np.float64) for m in MODELS}
            for f_idx, (te_idx, fit_obj, _, m1_pred_te) in enumerate(fitted_splits):
                Y_te = ds.Y_auc[te_idx]
                probes = sample_probe_sets(
                    Y_te, fit_obj["probe_pool_idx"], k=k, n_repeats=1, seed=seed_offset * 100 + k * 31 + f_idx * 7 + r
                )[0]
                preds_by_model["M0"][te_idx] = fit_obj["M0"].predict(Y_te, probes, k=k)
                preds_by_model["M1"][te_idx] = m1_pred_te
                preds_by_model["M2a"][te_idx] = fit_obj["M2a"].predict(Y_te, probes, target_idx, k=k)
                preds_by_model["M2b"][te_idx] = fit_obj["M2b"].predict(Y_te, probes, target_idx, k=k)
                preds_by_model["M3"][te_idx] = fit_obj["M3"].predict(
                    ds.X_rna[te_idx], ds.X_mut[te_idx], Y_te, probes, target_idx, k=k, prior_eval=m1_pred_te
                )

            for m in MODELS:
                ev = evaluate_masked_predictions(
                    ds.Y_auc,
                    preds_by_model[m],
                    base_d_all,
                    global_eval_mask,
                    target_idx,
                    subjects=subjects_all,
                    fold_ids=eval_fold_ids,
                )
                rep_metrics[m].append(ev)

        # R tekrar üzerinden hasta vektörlerini ve özet istatistikleri ortala
        agg_k: dict[str, Any] = {}
        for m in MODELS:
            evs = rep_metrics[m]
            vec_keys = ("patient_rmse_sq", "patient_res_spearman", "patient_ndcg3", "patient_res_ndcg3")
            mean_vecs = {
                vk: np.nanmean(np.stack([e["vectors"][vk] for e in evs], axis=0), axis=0)
                for vk in vec_keys
            }
            agg_k[m] = {
                "n_eval": evs[0]["n_eval"],
                "n_obs": evs[0]["n_obs"],
                "rmse": float(np.sqrt(np.nanmean(mean_vecs["patient_rmse_sq"]))),
                "obs_rmse": float(np.mean([e["obs_rmse"] for e in evs])),
                "residual_r2": float(np.mean([e["residual_r2"] for e in evs])),
                "within_patient_residual_spearman": float(np.nanmean(mean_vecs["patient_res_spearman"])),
                "per_drug_spearman": float(np.mean([e["per_drug_spearman"] for e in evs])),
                "ndcg_at_3": float(np.nanmean(mean_vecs["patient_ndcg3"])),
                "residual_ndcg_at_3": float(np.nanmean(mean_vecs["patient_res_ndcg3"])),
                "vectors": mean_vecs,
            }
        out_by_k[k] = agg_k

    # Seçilen hiperparametre özeti
    chosen_hparams = {
        "M0_lam_f0": fitted_splits[0][1]["M0"].best_lambda_by_k_,
        "M1_lam_f0": fitted_splits[0][1]["M1"].best_lambda_,
        "M2a_params_f0": fitted_splits[0][1]["M2a"].best_params_by_k_,
        "M2b_params_f0": fitted_splits[0][1]["M2b"].best_params_by_k_,
        "M3_params_f0": fitted_splits[0][1]["M3"]._cov_model.best_params_by_k_,
        "n_probes_f0": len(fitted_splits[0][1]["probe_pool_idx"]),
    }
    return {"by_k": out_by_k, "eval_mask": global_eval_mask, "hparams": chosen_hparams}


def _compute_paired_comparisons(by_k_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Protokoldeki tüm eşleştirilmiş hasta bootstrap %95 GA karşılaştırmalarını üretir."""
    comps: dict[str, Any] = {}
    rmse_fn = lambda v: float(np.sqrt(np.mean(v)))

    # M1 vs M0(0)
    m1_v = by_k_results[0]["M1"]["vectors"]
    m0_v = by_k_results[0]["M0"]["vectors"]
    comps["k0_M1_vs_M0"] = {
        "rmse_diff": paired_bootstrap_ci(m1_v["patient_rmse_sq"], m0_v["patient_rmse_sq"], agg_fn=rmse_fn),
        "res_spearman_diff": paired_bootstrap_ci(m1_v["patient_res_spearman"], m0_v["patient_res_spearman"]),
        "ndcg3_diff": paired_bootstrap_ci(m1_v["patient_ndcg3"], m0_v["patient_ndcg3"]),
        "res_ndcg3_diff": paired_bootstrap_ci(m1_v["patient_res_ndcg3"], m0_v["patient_res_ndcg3"]),
    }

    for k in (3, 5, 8):
        m0k = by_k_results[k]["M0"]["vectors"]
        m1k = by_k_results[0]["M1"]["vectors"]
        m2ak = by_k_results[k]["M2a"]["vectors"]
        for model_name in ("M2a", "M2b", "M3"):
            mv = by_k_results[k][model_name]["vectors"]
            comps[f"k{k}_{model_name}_vs_M0"] = {
                "rmse_diff": paired_bootstrap_ci(mv["patient_rmse_sq"], m0k["patient_rmse_sq"], agg_fn=rmse_fn),
                "res_spearman_diff": paired_bootstrap_ci(mv["patient_res_spearman"], m0k["patient_res_spearman"]),
                "ndcg3_diff": paired_bootstrap_ci(mv["patient_ndcg3"], m0k["patient_ndcg3"]),
                "res_ndcg3_diff": paired_bootstrap_ci(mv["patient_res_ndcg3"], m0k["patient_res_ndcg3"]),
            }
            comps[f"k{k}_{model_name}_vs_M1"] = {
                "rmse_diff": paired_bootstrap_ci(mv["patient_rmse_sq"], m1k["patient_rmse_sq"], agg_fn=rmse_fn),
                "res_spearman_diff": paired_bootstrap_ci(mv["patient_res_spearman"], m1k["patient_res_spearman"]),
                "ndcg3_diff": paired_bootstrap_ci(mv["patient_ndcg3"], m1k["patient_ndcg3"]),
                "res_ndcg3_diff": paired_bootstrap_ci(mv["patient_res_ndcg3"], m1k["patient_res_ndcg3"]),
            }
        m3v = by_k_results[k]["M3"]["vectors"]
        comps[f"k{k}_M3_vs_M2a"] = {
            "rmse_diff": paired_bootstrap_ci(m3v["patient_rmse_sq"], m2ak["patient_rmse_sq"], agg_fn=rmse_fn),
            "res_spearman_diff": paired_bootstrap_ci(m3v["patient_res_spearman"], m2ak["patient_res_spearman"]),
            "ndcg3_diff": paired_bootstrap_ci(m3v["patient_ndcg3"], m2ak["patient_ndcg3"]),
            "res_ndcg3_diff": paired_bootstrap_ci(m3v["patient_res_ndcg3"], m2ak["patient_res_ndcg3"]),
        }
    return comps


def _aggregate_cv_seeds(seed_runs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """3 CV seed'ini aynı benzersiz hasta indeksleri üzerinde ortalar."""
    common_mask = seed_runs[0]["eval_mask"].copy()
    for sr in seed_runs[1:]:
        common_mask &= sr["eval_mask"]

    # Her seed'in eval_mask'ı fold'daki probe_pool'a göre minör farklılık gösterebilir;
    # gerçek vektör boyutu eval_mask uzunluğundadır -> her satırı tam uzunlukta hizalayalım
    n_total = len(common_mask)
    out: dict[int, dict[str, Any]] = {}
    vec_keys = ("patient_rmse_sq", "patient_res_spearman", "patient_ndcg3", "patient_res_ndcg3")

    for k in K_BUDGETS:
        out[k] = {}
        for m in MODELS:
            aligned_vecs = {vk: [] for vk in vec_keys}
            for sr in seed_runs:
                mask_s = sr["eval_mask"]
                for vk in vec_keys:
                    full_v = np.full(n_total, np.nan, dtype=np.float64)
                    full_v[mask_s] = sr["by_k"][k][m]["vectors"][vk]
                    aligned_vecs[vk].append(full_v[common_mask])

            mean_vecs = {
                vk: np.nanmean(np.stack(aligned_vecs[vk], axis=0), axis=0)
                for vk in vec_keys
            }
            out[k][m] = {
                "n_eval": int(common_mask.sum()),
                "n_obs": int(np.mean([sr["by_k"][k][m]["n_obs"] for sr in seed_runs])),
                "rmse": float(np.sqrt(np.nanmean(mean_vecs["patient_rmse_sq"]))),
                "obs_rmse": float(np.mean([sr["by_k"][k][m]["obs_rmse"] for sr in seed_runs])),
                "residual_r2": float(np.mean([sr["by_k"][k][m]["residual_r2"] for sr in seed_runs])),
                "within_patient_residual_spearman": float(np.nanmean(mean_vecs["patient_res_spearman"])),
                "per_drug_spearman": float(np.mean([sr["by_k"][k][m]["per_drug_spearman"] for sr in seed_runs])),
                "ndcg_at_3": float(np.nanmean(mean_vecs["patient_ndcg3"])),
                "residual_ndcg_at_3": float(np.nanmean(mean_vecs["patient_res_ndcg3"])),
                "vectors": mean_vecs,
            }
    return out


def _strip_vectors(by_k: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        str(k): {
            m: {key: val for key, val in md.items() if key != "vectors"}
            for m, md in mdict.items()
        }
        for k, mdict in by_k.items()
    }


def evaluate_gate1(
    perm_res: dict[str, Any],
    cv_comps: dict[str, Any],
    wave_comps: dict[str, Any],
) -> dict[str, Any]:
    """Dondurulmuş PROTOCOL.md Bölüm 8 Kapı 1 kriterlerini mekanik olarak değerlendirir."""
    perm_ok = bool(perm_res["clean_wave_perm"]["passed"] and perm_res["cv_seed42_perm"]["passed"])
    passing_models: list[str] = []
    full_ndcg_models: list[str] = []

    for k in (5, 8):
        for m in ("M2a", "M2b", "M3"):
            cv_m0 = cv_comps[f"k{k}_{m}_vs_M0"]
            cv_m1 = cv_comps[f"k{k}_{m}_vs_M1"]
            wv_m0 = wave_comps[f"k{k}_{m}_vs_M0"]

            cv_ok = (
                cv_m0["rmse_diff"]["ci_high"] < 0.0
                and cv_m0["res_spearman_diff"]["ci_low"] > 0.0
                and cv_m1["rmse_diff"]["ci_high"] < 0.0
                and cv_m1["res_spearman_diff"]["ci_low"] > 0.0
            )
            wv_ok = (
                wv_m0["rmse_diff"]["ci_high"] < 0.0
                and wv_m0["res_spearman_diff"]["ci_low"] > 0.0
            )
            if perm_ok and cv_ok and wv_ok:
                tag = f"{m}(k={k})"
                passing_models.append(tag)
                if cv_m0["ndcg3_diff"]["ci_low"] > 0.0:
                    full_ndcg_models.append(tag)

    if full_ndcg_models:
        status = "PASS"
    elif passing_models:
        status = "CONDITIONAL_PASS"
    else:
        status = "FAIL"

    return {
        "status": status,
        "perm_ok": perm_ok,
        "passing_models": passing_models,
        "full_ndcg_models": full_ndcg_models,
    }


def run_sprint02(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Protokol bulunamadı: {PROTOCOL_PATH}. Deneyler protokolsüz koşulamaz.")

    t0 = time.perf_counter()
    ds_all = load_beataml_aml_dataset(raw_dir)
    idx_pos = select_index_specimen_positions(ds_all.meta)
    ds_idx = ds_all.subset_by_positions(idx_pos)

    drug_to_idx = {d: i for i, d in enumerate(ds_all.drug_names)}
    target_idx = [drug_to_idx[d] for d in CLINICAL_CANDIDATE_DRUGS]

    # 1. Zorunlu Permütasyon Sızıntı Kontrolü
    perm_res = run_permutation_leakage_check(ds_idx, target_idx)

    # 2. Birincil ID Rejimi: Patient-Disjoint 5-Fold CV x 3 Seed
    seed_runs = []
    for s in CV_SEEDS:
        folds = make_patient_disjoint_kfold(ds_idx.meta, n_splits=5, seed=s)
        seed_runs.append(_evaluate_regime_splits(ds_idx, folds, target_idx, seed_offset=s))
    cv_agg = _aggregate_cv_seeds(seed_runs)
    cv_comps = _compute_paired_comparisons(cv_agg)

    # 3. Birincil Zamansal OOD Rejimi: Clean Waves1+2 -> Waves3+4 (indeks örnek)
    tr_w, te_w = make_clean_wave_split(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects)
    wave_run = _evaluate_regime_splits(ds_idx, [(tr_w, te_w)], target_idx, seed_offset=777)
    wave_comps = _compute_paired_comparisons(wave_run["by_k"])

    # 4. İkincil Duyarlılık Analizi: Waves1+2 -> Waves3+4 tüm eşleşmiş örnekler (küme ortalamalı)
    tr_all, te_all = make_clean_wave_split(ds_all.meta, ds_all.w12_all_subjects, ds_all.both_all_subjects)
    wave_all_spec_run = _evaluate_regime_splits(
        ds_all, [(tr_all, te_all)], target_idx, seed_offset=888, cluster_by_subject=True
    )
    wave_all_spec_comps = _compute_paired_comparisons(wave_all_spec_run["by_k"])

    gate1 = evaluate_gate1(perm_res, cv_comps, wave_comps)
    elapsed = time.perf_counter() - t0

    results: dict[str, Any] = {
        "elapsed_seconds": round(elapsed, 2),
        "cohort_counts": {
            "all_matched_specimens": len(ds_all.meta),
            "unique_patients_index": len(ds_idx.meta),
            "wave_train_patients": int(len(tr_w)),
            "wave_test_clean_patients": int(len(te_w)),
            "wave_test_clean_all_specimens": int(len(te_all)),
            "n_target_drugs_C": len(target_idx),
        },
        "permutation_leakage_check": perm_res,
        "cv_5fold_3seeds": {
            "metrics": _strip_vectors(cv_agg),
            "paired_bootstrap": cv_comps,
            "sample_hparams_seed42_fold0": seed_runs[0]["hparams"],
        },
        "wave_ood_index": {
            "metrics": _strip_vectors(wave_run["by_k"]),
            "paired_bootstrap": wave_comps,
            "hparams": wave_run["hparams"],
        },
        "wave_ood_all_specimens_sensitivity": {
            "metrics": _strip_vectors(wave_all_spec_run["by_k"]),
            "paired_bootstrap": wave_all_spec_comps,
        },
        "gate1_decision": gate1,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_sprint02_md(results), encoding="utf-8")
    return results


def _fmt_ci(ci: dict[str, Any], digits: int = 3) -> str:
    return f"{ci['diff']:+.{digits}f} [{ci['ci_low']:+.{digits}f}, {ci['ci_high']:+.{digits}f}]"


def _render_metric_table(metrics_by_k: dict[str, Any]) -> list[str]:
    rows = [
        "| $k$ | Model | `n_eval` | `RMSE` ↓ | `Residual R²` ↑ | Hasta-İçi $\\rho^{\\text{res}}$ ↑ | İlaç-İçi $\\rho_{\\text{drug}}$ ↑ | `NDCG@3` ↑ | Kalıntı `NDCG@3` ↑ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k_str in ("0", "3", "5", "8"):
        mdict = metrics_by_k[k_str]
        for m in MODELS:
            if k_str != "0" and m == "M1":
                continue
            r = mdict[m]
            rows.append(
                f"| {k_str} | **{m}** | {r['n_eval']} | {r['rmse']:.2f} | {r['residual_r2']:+.3f} | "
                f"{r['within_patient_residual_spearman']:+.3f} | {r['per_drug_spearman']:+.3f} | "
                f"{r['ndcg_at_3']:.3f} | {r['residual_ndcg_at_3']:.3f} |"
            )
    return rows


def _render_sprint02_md(res: dict[str, Any]) -> str:
    cc = res["cohort_counts"]
    perm = res["permutation_leakage_check"]
    cv_m = res["cv_5fold_3seeds"]["metrics"]
    cv_c = res["cv_5fold_3seeds"]["paired_bootstrap"]
    wv_m = res["wave_ood_index"]["metrics"]
    wv_c = res["wave_ood_index"]["paired_bootstrap"]
    wva_m = res["wave_ood_all_specimens_sensitivity"]["metrics"]
    g1 = res["gate1_decision"]

    lines = [
        "# Sprint 2 — Sızıntısız Baseline'lar (`M0`–`M3`) ve Kapı 1 Değerlendirmesi",
        "",
        f"Tüm deneyler dondurulmuş [PROTOCOL.md](../PROTOCOL.md) uyarınca **{res['elapsed_seconds']} sn** içinde CPU üzerinde çalıştırılmıştır. Eğitim katı dışında hiçbir istatistik hesaplanmamıştır.",
        "",
        f"- **Saf AML İndeks Kohortu:** {cc['unique_patients_index']} hasta (tüm eşleşmiş örnekler: {cc['all_matched_specimens']})",
        f"- **Temiz Zamansal Bölme (`Waves1+2 → Waves3+4`):** Eğitim = {cc['wave_train_patients']} hasta, Test = {cc['wave_test_clean_patients']} temiz hasta (duyarlılık tüm örnekler: {cc['wave_test_clean_all_specimens']})",
        f"- **Hedef Küme ($\\mathcal{{C}}$):** {cc['n_target_drugs_C']} AML klinik adayı; düşük `auc` = yüksek duyarlılık.",
        "",
        "---",
        "",
        "## 1. Zorunlu Permütasyon Sızıntı Kontrolü",
        "",
        "| Kontrol Rejimi | `per_drug_spearman` [%95 GA] | `residual_r2` | Hasta-İçi $\\rho^{\\text{res}}$ | Durum |",
        "|---|---:|---:|---:|---|",
        f"| Clean `Waves1+2 → Waves3+4` | {perm['clean_wave_perm']['per_drug_spearman']:+.3f} [{perm['clean_wave_perm']['per_drug_spearman_ci'][0]:+.3f}, {perm['clean_wave_perm']['per_drug_spearman_ci'][1]:+.3f}] | {perm['clean_wave_perm']['residual_r2']:+.4f} | {perm['clean_wave_perm']['within_patient_residual_spearman']:+.3f} | {'GEÇTİ' if perm['clean_wave_perm']['passed'] else 'KALDI'} |",
        f"| 5-Fold CV (Seed 42) | {perm['cv_seed42_perm']['per_drug_spearman']:+.3f} [{perm['cv_seed42_perm']['per_drug_spearman_ci'][0]:+.3f}, {perm['cv_seed42_perm']['per_drug_spearman_ci'][1]:+.3f}] | {perm['cv_seed42_perm']['residual_r2']:+.4f} | {perm['cv_seed42_perm']['within_patient_residual_spearman']:+.3f} | {'GEÇTİ' if perm['cv_seed42_perm']['passed'] else 'KALDI'} |",
        "",
        "---",
        "",
        "## 2. Birincil ID Rejimi — `Patient-Disjoint 5-Fold CV` (3 Seed × 10 Prob Tekrarı)",
        "",
        *_render_metric_table(cv_m),
        "",
        "### Eşleştirilmiş Hasta Bootstrap Farkları (%95 GA, 1.000 tekrar)",
        "",
        "| Karşılaştırma | `ΔRMSE` (negatif = daha iyi) | `Δ(Hasta-İçi ρ^res)` (pozitif = daha iyi) | `ΔNDCG@3` (pozitif = daha iyi) | `ΔKalıntı NDCG@3` |",
        "|---|---:|---:|---:|---:|",
        f"| `M1 vs M0(0)` | {_fmt_ci(cv_c['k0_M1_vs_M0']['rmse_diff'], 2)} | {_fmt_ci(cv_c['k0_M1_vs_M0']['res_spearman_diff'])} | {_fmt_ci(cv_c['k0_M1_vs_M0']['ndcg3_diff'])} | {_fmt_ci(cv_c['k0_M1_vs_M0']['res_ndcg3_diff'])} |",
    ]
    for k in (3, 5, 8):
        for comp_name in (f"k{k}_M2a_vs_M0", f"k{k}_M2b_vs_M0", f"k{k}_M3_vs_M0", f"k{k}_M3_vs_M1", f"k{k}_M3_vs_M2a"):
            c = cv_c[comp_name]
            lines.append(
                f"| `{comp_name}` | {_fmt_ci(c['rmse_diff'], 2)} | {_fmt_ci(c['res_spearman_diff'])} | "
                f"{_fmt_ci(c['ndcg3_diff'])} | {_fmt_ci(c['res_ndcg3_diff'])} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Birincil Zamansal OOD Rejimi — `Clean Waves1+2 → Waves3+4` (10 Prob Tekrarı)",
            "",
            *_render_metric_table(wv_m),
            "",
            "### Zamansal OOD Eşleştirilmiş Hasta Bootstrap Farkları (%95 GA)",
            "",
            "| Karşılaştırma | `ΔRMSE` (negatif = daha iyi) | `Δ(Hasta-İçi ρ^res)` (pozitif = daha iyi) | `ΔNDCG@3` (pozitif = daha iyi) | `ΔKalıntı NDCG@3` |",
            "|---|---:|---:|---:|---:|",
            f"| `M1 vs M0(0)` | {_fmt_ci(wv_c['k0_M1_vs_M0']['rmse_diff'], 2)} | {_fmt_ci(wv_c['k0_M1_vs_M0']['res_spearman_diff'])} | {_fmt_ci(wv_c['k0_M1_vs_M0']['ndcg3_diff'])} | {_fmt_ci(wv_c['k0_M1_vs_M0']['res_ndcg3_diff'])} |",
        ]
    )
    for k in (3, 5, 8):
        for comp_name in (f"k{k}_M2a_vs_M0", f"k{k}_M2b_vs_M0", f"k{k}_M3_vs_M0", f"k{k}_M3_vs_M1", f"k{k}_M3_vs_M2a"):
            c = wv_c[comp_name]
            lines.append(
                f"| `{comp_name}` | {_fmt_ci(c['rmse_diff'], 2)} | {_fmt_ci(c['res_spearman_diff'])} | "
                f"{_fmt_ci(c['ndcg3_diff'])} | {_fmt_ci(c['res_ndcg3_diff'])} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. İkincil Duyarlılık Analizi — `Clean Waves1+2 → Waves3+4` Tüm Örnekler (Hasta-Küme Ortalamalı)",
            "",
            *_render_metric_table(wva_m),
            "",
            "---",
            "",
            "## 5. Önceden Kayıtlı Kapı 1 Kararı",
            "",
            f"- **Kapı 1 Sonucu:** **`{g1['status']}`**",
            f"- **Permütasyon Sızıntı Kontrolü:** {'GEÇTİ' if g1['perm_ok'] else 'KALDI'}",
            f"- **`RMSE` + Hasta-İçi $\\rho^{{\\text{{res}}}}$ Kriterini Geçen Modeller ($k \\in {{5,8}}$):** `{g1['passing_models']}`",
            f"- **Ek `ΔNDCG@3` Kriterini de Geçen Modeller:** `{g1['full_ndcg_models']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 2 baseline deneylerini çalıştır.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    res = run_sprint02(args.raw_dir, args.out_json, args.out_md)
    print(json.dumps({"elapsed_seconds": res["elapsed_seconds"], "gate1": res["gate1_decision"]}, indent=2))


if __name__ == "__main__":
    main()
