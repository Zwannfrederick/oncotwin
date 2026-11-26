"""Sprint 3 deney koşucusu: Toplu panel seçimi (k=0..12), M3 ablasyonları ve Kapı 2.

v2 (DEV-OPUS-01..11): tek değerlendirici, bütçe guard'ı, hasta vektörlerinin `.npz` kaydı, v2 duyarlılık
ablasyonları, W3+4 tam-panel raporu ve keşifsel ortak-havuz duyarlılığı. Varsayılan çıktılar v2 yollarıdır;
v1 sonuçları (`sprint03_results.json`) ezilmez.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS, DEFAULT_RAW_DIR
from src.oncotwin.data.dataset import (
    BeatAMLDataset,
    load_beataml_aml_dataset,
    select_index_specimen_positions,
)
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
    sample_probe_sets,
    select_train_probe_pool,
)
from src.oncotwin.experiments.sprint03_ablations import N_OFFSET_FIXED_DRAWS, fit_ablation_models, run_ablations_for_k
from src.oncotwin.experiments.sprint03_gate import evaluate_gate2
from src.oncotwin.experiments.sprint03_report import (
    ABLATION_K_BUDGETS,
    ABLATION_NAMES,
    POLICIES,
    render_sprint03_md,
    render_sprint03_v2_md,
)
from src.oncotwin.models.baselines import HybridFewShotM3, MeanEffectM0, OmicsRidgeM1
from src.oncotwin.models.panel_selection import (
    compute_contender_weights,
    require_full_budget,
    select_adaptive_batch_oed,
    select_static_d_optimal_ranking,
    select_static_diversity_ranking,
    take_first_k_available,
)

PROTOCOL_PATH = Path("docs/PROTOCOL.md")
PROTOCOL_S03_PATH = Path("docs/PROTOCOL_SPRINT03.md")
DEFAULT_OUT_JSON = Path("data/processed/sprint03_v2_results.json")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-03-v2-panel-selection.md")
DEFAULT_OUT_NPZ = Path("data/processed/sprint03_v2_vectors.npz")

CV_SEEDS = (42, 137, 2026)
PANEL_K_BUDGETS = (1, 2, 3, 5, 8, 12)
N_RANDOM_REPEATS = 50
MIN_TRAIN_PROBE_OBS = 100
MIN_EVAL_PROBES = 12
# Keşifsel W3+4 ortak havuz: test hastalarının >= %50'sinde ölçülen prob (arşivde 25 ile 101 gözlem arasında boşluk)
COMMON_POOL_MIN_FRAC = 0.5
V2_ABLATION_NAMES = ("StaticDOpt/M3_offset_only", "Random50/M3_offset_only", "StaticDOpt/M3_perm_offset_fixed")
JSON_DROP_KEYS = frozenset({"vectors", "full_panel_mask_eval", "drug_spearmans", "draws"})


def _fit_sprint03_split(
    ds: BeatAMLDataset,
    train_idx: np.ndarray,
    target_idx: list[int],
    inner_seed: int,
    fit_ablations: bool = True,
    probe_filter: set[int] | None = None,
) -> dict[str, Any]:
    """Tek bir eğitim katında M0, M1, M3, statik panel sıralamaları ve ablasyon modellerini fit eder.

    `probe_filter` (keşifsel): modeller tam eğitim havuzuyla fit edilir; yalnız seçim havuzu daraltılır.
    """
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
    m3 = HybridFewShotM3().fit(
        m1, Y_oof, Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=PANEL_K_BUDGETS, inner_seed=inner_seed
    )

    select_pool = probe_pool_idx if probe_filter is None else [g for g in probe_pool_idx if g in probe_filter]
    cov_model = m3._cov_model
    g2l = cov_model.global_to_local_
    probe_local = np.asarray([g2l[g] for g in select_pool], dtype=np.int64)
    target_local = np.asarray([g2l[g] for g in target_idx], dtype=np.int64)

    # Statik sıralamalar k-bağımsız/tutarlı kovaryans üzerinde: k=5 seçili iç-CV parametreleriyle
    cov_for_static = cov_model.cov_by_k_[5]
    _, lam_static = cov_model.best_params_by_k_[5]
    active_arr = cov_model.active_idx_
    out: dict[str, Any] = {
        "probe_pool_idx": select_pool,
        "mu_tr": m0.mu_,
        "b_d_tr": m0.b_d_,
        "M0": m0,
        "M1": m1,
        "M3": m3,
        "static_div_global": active_arr[select_static_diversity_ranking(cov_for_static, probe_local)],
        "static_dopt_global": active_arr[
            select_static_d_optimal_ranking(cov_for_static, probe_local, target_local, lam=lam_static)
        ],
        "target_local": target_local,
        "active_arr": active_arr,
    }
    if fit_ablations:
        out.update(fit_ablation_models(
            m1, Y_oof, Y_tr, subj_tr, probe_pool_idx, target_idx, fit_drugs, ABLATION_K_BUDGETS, inner_seed
        ))
    return out


def _evaluate_policies_for_k(
    ds: BeatAMLDataset,
    fitted_splits: list[tuple],
    k: int,
    target_idx: list[int],
    evaluate: Any,
    seed_offset: int,
) -> tuple[dict[str, Any], list[list[np.ndarray]], list[list[list[np.ndarray]]]]:
    """Bir k bütçesinde Random 50x, Static-Diversity, Static-D-Optimal ve Adaptive OED politikalarını değerlendirir."""
    n_rows, n_drugs = ds.Y_auc.shape

    # 1. Random 50x (50 tekrar; ablasyonlarda aynı paneller yeniden kullanılır)
    rand_evs: list[dict[str, Any]] = []
    rand_probes_by_rep_split: list[list[list[np.ndarray]]] = []
    for r in range(N_RANDOM_REPEATS):
        pred_rand = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
        rep_splits_probes: list[list[np.ndarray]] = []
        for f_idx, (te_idx, fit_obj, eval_mask_te, m1_pred_te, _) in enumerate(fitted_splits):
            Y_te = ds.Y_auc[te_idx]
            probes = sample_probe_sets(
                Y_te, fit_obj["probe_pool_idx"], k=k, n_repeats=1, seed=seed_offset * 1000 + k * 97 + f_idx * 13 + r
            )[0]
            require_full_budget(probes, eval_mask_te, k, "Random50")
            rep_splits_probes.append(probes)
            pred_rand[te_idx] = fit_obj["M3"].predict(
                ds.X_rna[te_idx], ds.X_mut[te_idx], Y_te, probes, target_idx, k=k, prior_eval=m1_pred_te
            )
        rand_probes_by_rep_split.append(rep_splits_probes)
        rand_evs.append(evaluate(pred_rand))

    rand_node = average_metric_nodes(rand_evs)
    rand_node["rep_distribution"] = {
        "rmse_per_rep": [float(e["rmse"]) for e in rand_evs],
        "ndcg3_per_rep": [float(e["ndcg_at_3"]) for e in rand_evs],
        "recall3_per_rep": [float(e["recall_at_3"]) for e in rand_evs],
        "regret1_per_rep": [float(e["regret_at_1"]) for e in rand_evs],
    }
    policy_results: dict[str, Any] = {"Random50": rand_node}

    # 2. Static-Diversity, Static-D-Optimal ve Patient-Adaptive Batch OED
    static_dopt_probes_by_split: list[list[np.ndarray]] = []
    adaptive_unique_panels: set[tuple[int, ...]] = set()
    adaptive_panel_list: list[set[int]] = []
    for pol_name in ("StaticDiversity", "StaticDOpt", "AdaptiveOED"):
        pred_pol = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
        hit_rates: list[float] = []
        full_hits: list[int] = []
        for te_idx, fit_obj, eval_mask_te, m1_pred_te, contender_w in fitted_splits:
            Y_te = ds.Y_auc[te_idx]
            probes_pol: list[np.ndarray] = []
            if pol_name in ("StaticDiversity", "StaticDOpt"):
                ranked_g = fit_obj["static_div_global" if pol_name == "StaticDiversity" else "static_dopt_global"]
                for i in range(len(te_idx)):
                    chosen, d_hits = take_first_k_available(ranked_g, Y_te[i], k)
                    probes_pol.append(chosen)
                    if eval_mask_te[i]:
                        hit_rates.append(d_hits / float(k))
                        full_hits.append(int(d_hits == k))
                if pol_name == "StaticDOpt":
                    static_dopt_probes_by_split.append(probes_pol)
            else:
                cov_model = fit_obj["M3"]._cov_model
                cov_k = cov_model.cov_by_k_[k]
                _, lam_k = cov_model.best_params_by_k_[k]
                pool_arr = np.asarray(fit_obj["probe_pool_idx"], dtype=np.int64)
                for i in range(len(te_idx)):
                    avail_g = pool_arr[np.isfinite(Y_te[i, pool_arr])]
                    avail_loc = np.asarray([cov_model.global_to_local_[int(g)] for g in avail_g], dtype=np.int64)
                    chosen_g = fit_obj["active_arr"][select_adaptive_batch_oed(
                        cov_k, avail_loc, fit_obj["target_local"], contender_w[i], k=k, lam=lam_k
                    )]
                    probes_pol.append(chosen_g)
                    if eval_mask_te[i]:
                        adaptive_unique_panels.add(tuple(sorted(int(x) for x in chosen_g)))
                        adaptive_panel_list.append(set(int(x) for x in chosen_g))
            require_full_budget(probes_pol, eval_mask_te, k, pol_name)
            pred_pol[te_idx] = fit_obj["M3"].predict(
                ds.X_rna[te_idx], ds.X_mut[te_idx], Y_te, probes_pol, target_idx, k=k, prior_eval=m1_pred_te
            )

        ev_pol = evaluate(pred_pol)
        if pol_name != "AdaptiveOED":
            ev_pol["top_k_direct_hit_rate"] = float(np.mean(hit_rates)) if hit_rates else float("nan")
            ev_pol["full_panel_patient_rate"] = float(np.mean(full_hits)) if full_hits else float("nan")
            ev_pol["full_panel_mask_eval"] = np.asarray(full_hits, dtype=bool)
        else:
            ev_pol["n_unique_panels"] = len(adaptive_unique_panels)
            ev_pol["mean_pairwise_jaccard"] = _mean_pairwise_jaccard(adaptive_panel_list, seed=123 + k)
        policy_results[pol_name] = ev_pol
    return policy_results, static_dopt_probes_by_split, rand_probes_by_rep_split


def _mean_pairwise_jaccard(panels: list[set[int]], seed: int) -> float:
    """Adaptif panel çeşitliliği: en fazla 300 rastgele hasta çiftinde ortalama Jaccard."""
    if len(panels) <= 1:
        return 1.0
    rng_j = np.random.default_rng(seed)
    vals = []
    for _ in range(min(300, len(panels))):
        a_i, b_i = rng_j.integers(0, len(panels), size=2)
        if a_i != b_i:
            vals.append(len(panels[a_i] & panels[b_i]) / float(len(panels[a_i] | panels[b_i])))
    return float(np.mean(vals)) if vals else 1.0


def _evaluate_sprint03_splits(
    ds: BeatAMLDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_idx: list[int],
    seed_offset: int,
    run_ablations: bool = True,
    probe_filter: set[int] | None = None,
) -> dict[str, Any]:
    """Verilen bölme listesinde (5-Fold veya tek Wave split) tüm panel politikalarını ve ablasyonları çalıştırır."""
    n_rows, n_drugs = ds.Y_auc.shape
    fitted_splits = []
    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        fit_obj = _fit_sprint03_split(
            ds, tr_idx, target_idx, inner_seed=seed_offset + f_idx, fit_ablations=run_ablations, probe_filter=probe_filter
        )
        eval_mask_te = get_evaluable_mask(
            ds.Y_auc[te_idx], target_idx, fit_obj["probe_pool_idx"], min_targets=5, min_probes=MIN_EVAL_PROBES
        )
        m1_pred_te = fit_obj["M1"].predict(ds.X_rna[te_idx], ds.X_mut[te_idx], n_drugs=n_drugs)
        # Seçim öncesi Monte Carlo yarışmacı ağırlıkları (ASLA test AUC kullanılmaz!)
        t_loc = fit_obj["target_local"]
        cov_CC = fit_obj["M3"]._cov_model.cov_by_k_[5][np.ix_(t_loc, t_loc)]
        contender_w = compute_contender_weights(m1_pred_te[:, target_idx], cov_CC, top_k=3, n_mc=500, seed=seed_offset + f_idx)
        fitted_splits.append((te_idx, fit_obj, eval_mask_te, m1_pred_te, contender_w))

    global_eval_mask = np.zeros(n_rows, dtype=bool)
    base_d_all = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
    fold_ids = np.full(n_rows, -1, dtype=np.int64)
    pred_m0_k0 = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
    pred_m1_k0 = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
    for f_idx, (te_idx, fit_obj, eval_mask_te, m1_pred_te, _) in enumerate(fitted_splits):
        global_eval_mask[te_idx] = eval_mask_te
        base_d_all[te_idx] = (fit_obj["mu_tr"] + fit_obj["b_d_tr"])[None, :]
        fold_ids[te_idx] = f_idx
        empty_p = [np.array([], dtype=np.int64) for _ in range(len(te_idx))]
        pred_m0_k0[te_idx] = fit_obj["M0"].predict(ds.Y_auc[te_idx], empty_p, k=0)
        pred_m1_k0[te_idx] = m1_pred_te
    eval_fold_ids = fold_ids if len(fitted_splits) > 1 else None

    def evaluate(pred: np.ndarray) -> dict[str, Any]:
        return evaluate_masked_predictions(ds.Y_auc, pred, base_d_all, global_eval_mask, target_idx, fold_ids=eval_fold_ids)

    by_k: dict[int, dict[str, Any]] = {}
    ablations_by_k: dict[int, dict[str, Any]] = {}
    ablations_v2_by_k: dict[int, dict[str, Any]] = {}
    for k in PANEL_K_BUDGETS:
        by_k[k], static_probes, rand_probes = _evaluate_policies_for_k(ds, fitted_splits, k, target_idx, evaluate, seed_offset)
        if run_ablations and k in ABLATION_K_BUDGETS:
            ablations_by_k[k], ablations_v2_by_k[k] = run_ablations_for_k(
                ds, fitted_splits, k, target_idx, static_probes, rand_probes, by_k[k]["Random50"], evaluate, seed_offset
            )

    return {
        "eval_mask": global_eval_mask,
        "k0_refs": {"M0": evaluate(pred_m0_k0), "M1": evaluate(pred_m1_k0)},
        "by_k": by_k,
        "ablations_by_k": ablations_by_k,
        "ablations_v2_by_k": ablations_v2_by_k,
        # Tek split için öğrenilen ilk-12 statik ilaç isimleri
        "top12_static_dopt": [ds.drug_names[int(g)] for g in fitted_splits[0][1]["static_dopt_global"][:12]],
        "top12_static_div": [ds.drug_names[int(g)] for g in fitted_splits[0][1]["static_div_global"][:12]],
    }


def _aggregate_cv_seed_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """3 CV seed'ini ortak değerlendirilebilir benzersiz hastalar üzerinde birleştirir (seed'ler hasta başına ortalanır)."""
    common_mask = seed_runs[0]["eval_mask"].copy()
    for sr in seed_runs[1:]:
        common_mask &= sr["eval_mask"]
    n_total = len(common_mask)

    def _agg_node(getter) -> dict[str, Any]:
        aligned = []
        for sr in seed_runs:
            node = getter(sr)
            vecs = {}
            for vk in VECTOR_KEYS:
                full_v = np.full(n_total, np.nan, dtype=np.float64)
                full_v[sr["eval_mask"]] = node["vectors"][vk]
                vecs[vk] = full_v[common_mask]
            aligned.append({**node, "vectors": vecs})
        res = average_metric_nodes(aligned)
        if "top_k_direct_hit_rate" in aligned[0]:
            res["top_k_direct_hit_rate"] = float(np.mean([a["top_k_direct_hit_rate"] for a in aligned]))
            res["full_panel_patient_rate"] = float(np.mean([a["full_panel_patient_rate"] for a in aligned]))
        if "n_unique_panels" in aligned[0]:
            res["n_unique_panels"] = float(np.mean([a["n_unique_panels"] for a in aligned]))
            res["mean_pairwise_jaccard"] = float(np.mean([a["mean_pairwise_jaccard"] for a in aligned]))
        return res

    ablations_v2 = {}
    for k in ABLATION_K_BUDGETS:
        node_k = {name: _agg_node(lambda sr, k=k, n=name: sr["ablations_v2_by_k"][k][n]) for name in V2_ABLATION_NAMES}
        node_k["draws"] = [
            _agg_node(lambda sr, k=k, d=d: sr["ablations_v2_by_k"][k]["draws"][d]) for d in range(N_OFFSET_FIXED_DRAWS)
        ]
        ablations_v2[k] = node_k

    return {
        "eval_mask": common_mask,
        "k0_refs": {m: _agg_node(lambda sr, m=m: sr["k0_refs"][m]) for m in ("M0", "M1")},
        "by_k": {k: {pol: _agg_node(lambda sr, k=k, p=pol: sr["by_k"][k][p]) for pol in POLICIES} for k in PANEL_K_BUDGETS},
        "ablations_by_k": {
            k: {
                ps: {abl: _agg_node(lambda sr, k=k, ps=ps, ab=abl: sr["ablations_by_k"][k][ps][ab]) for abl in ABLATION_NAMES}
                for ps in ("StaticDOpt", "Random50")
            }
            for k in ABLATION_K_BUDGETS
        },
        "ablations_v2_by_k": ablations_v2,
    }


def _compute_sprint03_comparisons(agg: dict[str, Any]) -> dict[str, Any]:
    by_k = agg["by_k"]
    m1_ref = agg["k0_refs"]["M1"]
    panel_comps: dict[str, Any] = {}
    for k in PANEL_K_BUDGETS:
        rand_node = by_k[k]["Random50"]
        for pol in ("StaticDiversity", "StaticDOpt", "AdaptiveOED"):
            panel_comps[f"k{k}_{pol}_vs_Random50"] = paired_metric_diffs(by_k[k][pol], rand_node)
            panel_comps[f"k{k}_{pol}_vs_M1"] = paired_metric_diffs(by_k[k][pol], m1_ref)
        panel_comps[f"k{k}_Random50_vs_M1"] = paired_metric_diffs(rand_node, m1_ref)
        panel_comps[f"k{k}_AdaptiveOED_vs_StaticDOpt"] = paired_metric_diffs(by_k[k]["AdaptiveOED"], by_k[k]["StaticDOpt"])

    # Ardışık bütçe marjinal kazançları (VoI)
    voi_comps = {
        f"{pol}_k{a}_to_k{b}": paired_metric_diffs(by_k[b][pol], by_k[a][pol])
        for pol in ("Random50", "StaticDOpt", "AdaptiveOED")
        for a, b in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 12))
    }
    abl_comps = {
        f"k{k}_{ps}_M3_full_vs_{abl}": paired_metric_diffs(
            agg["ablations_by_k"][k][ps]["M3_full"], agg["ablations_by_k"][k][ps][abl]
        )
        for k in agg["ablations_by_k"]  # ablasyonsuz koşularda (keşifsel ortak havuz) boş
        for ps in ("StaticDOpt", "Random50")
        for abl in ABLATION_NAMES[1:]
    }
    return {"panel": panel_comps, "voi": voi_comps, "ablations": abl_comps}


def _ablation_v2_summary(agg: dict[str, Any]) -> dict[str, Any]:
    """v2 duyarlılık ablasyonları: eşleştirilmiş farklar, offset payı ve 20 çekiliş dağılımı (keşifsel)."""
    out: dict[str, Any] = {}
    m1 = agg["k0_refs"]["M1"]
    for k in ABLATION_K_BUDGETS:
        v2 = agg["ablations_v2_by_k"][k]
        full = {ps: agg["ablations_by_k"][k][ps]["M3_full"] for ps in ("StaticDOpt", "Random50")}
        share = {}
        for ps in ("StaticDOpt", "Random50"):
            gain = m1["rmse"] - full[ps]["rmse"]
            share[ps] = (m1["rmse"] - v2[f"{ps}/M3_offset_only"]["rmse"]) / gain if gain > 0 else float("nan")
        draws_rmse = [d["rmse"] for d in v2["draws"]]
        draws_rho = [d["within_patient_residual_spearman"] for d in v2["draws"]]
        out[str(k)] = {
            "metrics": {name: _strip_for_json(v2[name]) for name in V2_ABLATION_NAMES},
            "comparisons": {
                "StaticDOpt_M3_full_vs_M3_offset_only": paired_metric_diffs(full["StaticDOpt"], v2["StaticDOpt/M3_offset_only"]),
                "Random50_M3_full_vs_M3_offset_only": paired_metric_diffs(full["Random50"], v2["Random50/M3_offset_only"]),
                "StaticDOpt_M3_offset_only_vs_M1": paired_metric_diffs(v2["StaticDOpt/M3_offset_only"], m1),
                "StaticDOpt_M3_full_vs_M3_perm_offset_fixed": paired_metric_diffs(
                    full["StaticDOpt"], v2["StaticDOpt/M3_perm_offset_fixed"]
                ),
            },
            "offset_share_of_M3_rmse_gain": share,
            "perm_offset_fixed_draws": {
                "n_draws": len(draws_rmse),
                "rmse": {"mean": float(np.mean(draws_rmse)), "sd": float(np.std(draws_rmse, ddof=1)),
                         "min": float(np.min(draws_rmse)), "max": float(np.max(draws_rmse))},
                "res_spearman": {"mean": float(np.mean(draws_rho)), "sd": float(np.std(draws_rho, ddof=1)),
                                 "min": float(np.min(draws_rho)), "max": float(np.max(draws_rho))},
            },
        }
    return out


def _wave_full_panel_report(wave_run: dict[str, Any]) -> dict[str, Any]:
    """W3+4 StaticDOpt/StaticDiversity: tam panel, doğrudan isabet ve backfill oranı; tam panel < 5 ise hesaplanamadı."""
    report: dict[str, Any] = {}
    for k in PANEL_K_BUDGETS:
        rand_v = wave_run["by_k"][k]["Random50"]["vectors"]
        for pol in ("StaticDOpt", "StaticDiversity"):
            node = wave_run["by_k"][k][pol]
            fmask = node["full_panel_mask_eval"]
            entry: dict[str, Any] = {
                "n_full_panel_patients": int(fmask.sum()),
                "n_total_eval": int(len(fmask)),
                "top_k_direct_hit_rate": node["top_k_direct_hit_rate"],
                "backfilled_slot_rate": 1.0 - node["top_k_direct_hit_rate"],
            }
            if fmask.sum() >= 5:
                v = node["vectors"]
                entry.update({
                    "status": "hesaplandı",
                    "policy_rmse": float(np.sqrt(np.nanmean(v["patient_rmse_sq"][fmask]))),
                    "random_rmse": float(np.sqrt(np.nanmean(rand_v["patient_rmse_sq"][fmask]))),
                    "policy_ndcg3": float(np.nanmean(v["patient_ndcg3"][fmask])),
                    "random_ndcg3": float(np.nanmean(rand_v["patient_ndcg3"][fmask])),
                })
            else:
                entry["status"] = "hesaplanamadı (tam panelli hasta < 5)"
            report[f"k{k}_{pol}"] = entry
    return report


def _strip_for_json(obj: Any) -> Any:
    """Vektörleri ve dizi alanlarını JSON çıktısından ayıklar (vektörler .npz'ye yazılır); int anahtarlar str olur."""
    if isinstance(obj, dict):
        return {str(k): _strip_for_json(v) for k, v in obj.items() if k not in JSON_DROP_KEYS}
    if isinstance(obj, list):
        return [_strip_for_json(v) for v in obj]
    return obj


def _metrics_json(run: dict[str, Any]) -> dict[str, Any]:
    """v1 ile aynı metrik alanları (k0_refs, by_k, ablations_by_k), vektörsüz."""
    return {key: _strip_for_json(run[key]) for key in ("k0_refs", "by_k", "ablations_by_k")}


def _collect_vectors(obj: Any, prefix: str, out: dict[str, np.ndarray]) -> None:
    """İç içe düğümlerdeki hasta vektörlerini `prefix/.../anahtar` adlarıyla düz sözlüğe toplar."""
    if isinstance(obj, dict):
        if "vectors" in obj:
            for vk, v in obj["vectors"].items():
                out[f"{prefix}/{vk}"] = np.asarray(v, dtype=np.float64)
        for key, val in obj.items():
            if key != "vectors":
                _collect_vectors(val, f"{prefix}/{key}", out)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            _collect_vectors(val, f"{prefix}/{i}", out)


def run_sprint03(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_npz: Path = DEFAULT_OUT_NPZ,
) -> dict[str, Any]:
    if not PROTOCOL_PATH.exists() or not PROTOCOL_S03_PATH.exists():
        raise FileNotFoundError("PROTOCOL.md ve PROTOCOL_SPRINT03.md bulunmadan deney koşulamaz.")

    t0 = time.perf_counter()
    ds_all = load_beataml_aml_dataset(raw_dir)
    ds_idx = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    dmap = {d: i for i, d in enumerate(ds_all.drug_names)}
    target_idx = [dmap[d] for d in CLINICAL_CANDIDATE_DRUGS]
    subjects = ds_idx.meta["dbgap_subject_id"].to_numpy()

    # 1. Patient-Disjoint 5-Fold CV x 3 Seed
    seed_runs = [
        _evaluate_sprint03_splits(ds_idx, make_patient_disjoint_kfold(ds_idx.meta, n_splits=5, seed=s), target_idx, seed_offset=s)
        for s in CV_SEEDS
    ]
    cv_agg = _aggregate_cv_seed_runs(seed_runs)
    cv_comps = _compute_sprint03_comparisons(cv_agg)

    # 2. Clean Waves1+2 -> Waves3+4
    tr_w, te_w = make_clean_wave_split(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects)
    wave_run = _evaluate_sprint03_splits(ds_idx, [(tr_w, te_w)], target_idx, seed_offset=777)
    wave_comps = _compute_sprint03_comparisons(wave_run)

    # 3. Keşifsel: W3+4 arşiv mevcudiyetiyle tanımlı ortak prob havuzu (yanıt değil, ölçüm varlığı kullanılır)
    w34_frac = np.mean(np.isfinite(ds_idx.Y_auc[te_w]), axis=0)
    common_pool = {int(g) for g in np.flatnonzero(w34_frac >= COMMON_POOL_MIN_FRAC)}
    train_pool_w = select_train_probe_pool(ds_idx.Y_auc[tr_w], ds_idx.drug_names, CLINICAL_CANDIDATE_DRUGS, MIN_TRAIN_PROBE_OBS)
    excluded_w = sorted(ds_idx.drug_names[g] for g in train_pool_w if g not in common_pool)
    wave_common = _evaluate_sprint03_splits(
        ds_idx, [(tr_w, te_w)], target_idx, seed_offset=777, run_ablations=False, probe_filter=common_pool
    )

    # Random 50x dağılım yüzdelik özeti (Waves3+4'te yapısal panellerin konumu)
    wave_rand_percentiles: dict[str, Any] = {}
    for k in PANEL_K_BUDGETS:
        dist = wave_run["by_k"][k]["Random50"]["rep_distribution"]
        rmse_arr = np.asarray(dist["rmse_per_rep"])
        ndcg_arr = np.asarray(dist["ndcg3_per_rep"])
        wave_rand_percentiles[str(k)] = {
            "random_rmse_median": float(np.median(rmse_arr)),
            "random_rmse_p10_best": float(np.percentile(rmse_arr, 10)),
            "random_rmse_p90_worst": float(np.percentile(rmse_arr, 90)),
            "dopt_beats_random_rmse_pct": float(np.mean(wave_run["by_k"][k]["StaticDOpt"]["rmse"] < rmse_arr) * 100.0),
            "adaptive_beats_random_rmse_pct": float(np.mean(wave_run["by_k"][k]["AdaptiveOED"]["rmse"] < rmse_arr) * 100.0),
            "dopt_beats_random_ndcg3_pct": float(np.mean(wave_run["by_k"][k]["StaticDOpt"]["ndcg_at_3"] > ndcg_arr) * 100.0),
        }

    # v1 uyumlu tam-panel alt-grup özeti (k=3, 5, 8; yalnız n_full >= 5)
    full_report = _wave_full_panel_report(wave_run)
    wave_full_panel_subgroups = {
        str(k): {
            "n_full_panel_patients": e["n_full_panel_patients"], "n_total_eval": e["n_total_eval"],
            "dopt_rmse": e["policy_rmse"], "random_rmse": e["random_rmse"],
            "dopt_ndcg3": e["policy_ndcg3"], "random_ndcg3": e["random_ndcg3"],
        }
        for k in (3, 5, 8)
        if (e := full_report[f"k{k}_StaticDOpt"])["status"] == "hesaplandı"
    }

    # İlaç ailelerini ekle (Waves1+2 ilk 12 seçilen panel için)
    dfam = pd.read_excel(raw_dir / "beataml_drug_families.xlsx", sheet_name="drug_family")
    fam_map = dfam.groupby("inhibitor")["family"].apply(lambda s: "; ".join(sorted(set(s.astype(str))))).to_dict()

    def _family_rows(names: list[str]) -> list[dict[str, Any]]:
        return [{"rank": i + 1, "inhibitor": n, "family": fam_map.get(n, "Unlisted / Non-kinase")} for i, n in enumerate(names)]

    gate2 = evaluate_gate2(cv_comps, wave_comps)
    results: dict[str, Any] = {
        "version": "sprint03-v2",
        "cv_5fold_3seeds": {"metrics": _metrics_json(cv_agg), "comparisons": cv_comps},
        "wave_ood": {
            "metrics": _metrics_json(wave_run),
            "comparisons": wave_comps,
            "random50_percentiles": wave_rand_percentiles,
            "full_panel_subgroups_dopt": wave_full_panel_subgroups,
            "full_panel_report": full_report,
        },
        "wave_ood_common_pool_exploratory": {
            "definition": f"Prob havuzu: eğitim havuzu ∩ W3+4 test hastalarının >= %{COMMON_POOL_MIN_FRAC * 100:.0f}'inde ölçülen ilaçlar (arşiv mevcudiyeti; keşifsel).",
            "n_selection_pool_probes": len(train_pool_w) - len(excluded_w),
            "excluded_train_pool_drugs": excluded_w,
            "metrics": _metrics_json(wave_common),
            "comparisons": {"panel": _compute_sprint03_comparisons(wave_common)["panel"]},
            "full_panel_report": _wave_full_panel_report(wave_common),
        },
        "ablations_v2_exploratory": {
            "cv_5fold_3seeds": _ablation_v2_summary(cv_agg),
            "wave_ood": _ablation_v2_summary(wave_run),
        },
        "learned_panels_waves12": {
            "static_d_optimal_top12": _family_rows(wave_run["top12_static_dopt"]),
            "static_diversity_top12": _family_rows(wave_run["top12_static_div"]),
        },
        "gate2_decision": gate2,
        "metric_notes": {
            "rmse": "Hasta-ağırlıklı: sqrt(hasta MSE ortalaması). Protokol §7'nin havuzlanmış tanımı `obs_rmse` alanındadır.",
            "confidence_intervals": "Hasta düzeyi eşleştirilmiş bootstrap (1000, seed 2026); sabit (fit edilmiş) model tahminlerinin belirsizliği, kat yeniden-eğitim varyansı hariç.",
            "p_two_sided": "İki yönlü yüzdelik-bootstrap p-değeri, (sayım+1)/(B+1); yalnız keşifsel Holm yeniden analizinde kullanılır.",
            "sprint02_leakage_ci": "Sprint 2 permütasyon sızıntı testinin GA'sı 24 ilaç üzerinden bootstrap edilmiştir (protokol §7 hasta düzeyi ister).",
        },
    }
    results["elapsed_seconds"] = round(time.perf_counter() - t0, 2)

    vectors: dict[str, np.ndarray] = {
        "cv/subject_ids": subjects[cv_agg["eval_mask"]].astype(np.int64),
        "wave/subject_ids": subjects[wave_run["eval_mask"]].astype(np.int64),
        "wave_common/subject_ids": subjects[wave_common["eval_mask"]].astype(np.int64),
    }
    for prefix, node in (("cv", cv_agg), ("wave", wave_run), ("wave_common", wave_common)):
        _collect_vectors({k: v for k, v in node.items() if k != "eval_mask"}, prefix, vectors)

    for path in (out_json, out_md, out_npz):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **vectors)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_sprint03_md(results) + render_sprint03_v2_md(results), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3 toplu panel seçimi ve M3 ablasyon deneylerini çalıştır (v2 çıktıları).")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    args = parser.parse_args()
    res = run_sprint03(args.raw_dir, args.out_json, args.out_md, args.out_npz)
    print(json.dumps({"elapsed_seconds": res["elapsed_seconds"], "gate2": res["gate2_decision"]}, indent=2))


if __name__ == "__main__":
    main()
