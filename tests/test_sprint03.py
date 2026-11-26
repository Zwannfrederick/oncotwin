"""Sprint 3 birim, sızıntı, brute-force MI doğrulaması ve ablasyon testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oncotwin.eval.metrics import (
    _rankdata_average,
    evaluate_predictions,
    recall_at_k,
    regret_at_k,
)
from src.oncotwin.eval.splits import select_train_probe_pool
from src.oncotwin.experiments.sprint03_ablations import (
    _permute_across_patient_probes,
    _permute_within_patient_probes,
)
from src.oncotwin.models.baselines import HybridFewShotM3, OmicsRidgeM1
from src.oncotwin.models.panel_selection import (
    compute_contender_weights,
    select_adaptive_batch_oed,
    select_static_d_optimal_ranking,
    select_static_diversity_ranking,
    take_first_k_available,
)
from tests.test_sprint02 import _synthetic_cohort


def test_fast_rank_matches_pandas_and_recall_regret_direction() -> None:
    arr = np.array([10.0, 20.0, 20.0, 5.0, 30.0, 20.0, 40.0])
    r_np = _rankdata_average(arr)
    r_pd = pd.Series(arr).rank(method="average").to_numpy(dtype=np.float64)
    assert np.allclose(r_np, r_pd)

    y_true = np.array([40.0, 70.0, 90.0, 160.0, 220.0, 250.0])
    perf_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    worst_pred = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    const_pred = np.array([100.0] * 6)

    assert np.isclose(recall_at_k(y_true, perf_pred, k=3), 1.0)
    assert np.isclose(recall_at_k(y_true, worst_pred, k=3), 0.0)
    assert np.isclose(recall_at_k(y_true, const_pred, k=3), 0.5)

    assert np.isclose(regret_at_k(y_true, perf_pred, k=1), 0.0)
    assert np.isclose(regret_at_k(y_true, perf_pred, k=3), 0.0)
    assert np.isclose(regret_at_k(y_true, worst_pred, k=1), 210.0)
    assert regret_at_k(y_true, worst_pred, k=3) <= regret_at_k(y_true, worst_pred, k=1)
    assert regret_at_k(y_true, const_pred, k=1) > 0.0


def test_static_d_optimal_matches_exact_logdet_mi() -> None:
    rng = np.random.default_rng(101)
    A = rng.normal(0.0, 1.0, size=(12, 8))
    cov = A @ A.T + 0.5 * np.eye(12)
    target_idx = np.array([0, 1, 2], dtype=np.int64)
    probe_idx = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)
    lam = 2.5

    ranked = select_static_d_optimal_ranking(cov, probe_idx, target_idx, lam=lam)

    # İlk seçilen prob brute-force 0.5 * (logdet(C_CC) - logdet(C_CC | j)) ile birebir aynı olmalı
    C_TT = cov[np.ix_(target_idx, target_idx)]
    _, ld_prior = np.linalg.slogdet(C_TT)
    exact_mi = {}
    for j in probe_idx:
        c_tj = cov[target_idx, j : j + 1]
        C_post = C_TT - (c_tj @ c_tj.T) / (cov[j, j] + lam)
        _, ld_post = np.linalg.slogdet(C_post)
        exact_mi[int(j)] = 0.5 * (ld_prior - ld_post)
    best_exact = max(exact_mi, key=exact_mi.get)  # type: ignore[arg-type]
    assert int(ranked[0]) == best_exact


def test_panel_selectors_equal_budget_exclusion_and_zero_auc_leakage() -> None:
    # Blok 1: hedefler [0, 1] yalnızca prob [4, 5] ile korele
    # Blok 2: hedefler [2, 3] yalnızca prob [6, 7] ile korele
    cov = np.diag([100.0] * 8)
    for t, p in [(0, 4), (0, 5), (1, 4), (1, 5), (2, 6), (2, 7), (3, 6), (3, 7)]:
        cov[t, p] = cov[p, t] = 85.0

    target_local = np.array([0, 1, 2, 3], dtype=np.int64)
    probe_local = np.array([4, 5, 6, 7], dtype=np.int64)

    # İlk test: M1 öncülü Blok 1 hedeflerini öne çıkardığında AdaptiveOED prob [4,5]'i seçmeli
    prior_b1 = np.array([[50.0, 55.0, 220.0, 230.0], [220.0, 230.0, 50.0, 55.0]])
    w = compute_contender_weights(prior_b1, cov[:4, :4], top_k=2, n_mc=400, seed=42)
    sel_p0 = select_adaptive_batch_oed(cov, probe_local, target_local, w[0], k=2, lam=10.0)
    sel_p1 = select_adaptive_batch_oed(cov, probe_local, target_local, w[1], k=2, lam=10.0)
    assert set(sel_p0.tolist()) == {4, 5}
    assert set(sel_p1.tolist()) == {6, 7}
    assert set(sel_p0.tolist()).isdisjoint(set(target_local.tolist()))

    # İkinci test: Test AUC değerleri +/-500 bozulsa bile statik ve adaptif seçimler bit düzeyinde aynı kalmalı
    y_row_a = np.array([10.0, 20.0, 30.0, 40.0, np.nan, 110.0, 120.0, 130.0])
    y_row_b = np.array([900.0, -500.0, 30.0, 40.0, np.nan, -999.0, 999.0, 500.0])

    div_rank = select_static_diversity_ranking(cov, probe_local)
    dopt_rank = select_static_d_optimal_ranking(cov, probe_local, target_local, lam=10.0)

    c_div_a, hits_a = take_first_k_available(div_rank, y_row_a, k=2)
    c_div_b, hits_b = take_first_k_available(div_rank, y_row_b, k=2)
    assert np.array_equal(c_div_a, c_div_b) and hits_a == hits_b and len(c_div_a) == 2

    c_dopt_a, _ = take_first_k_available(dopt_rank, y_row_a, k=2)
    c_dopt_b, _ = take_first_k_available(dopt_rank, y_row_b, k=2)
    assert np.array_equal(c_dopt_a, c_dopt_b) and len(c_dopt_a) == 2
    assert set(c_dopt_a.tolist()).isdisjoint(set(target_local.tolist()))


def test_m3_ablations_on_synthetic_correlated_data() -> None:
    d = _synthetic_cohort(n_patients=60, n_genes=20, n_drugs=14, seed=31)
    tr = np.arange(0, 40)
    te = np.arange(40, 60)
    target_idx = [0, 1, 2, 3, 4, 5]
    probe_pool = list(range(6, 14))
    subj_tr = d["meta"].iloc[tr]["dbgap_subject_id"].to_numpy()

    m1, oof = OmicsRidgeM1(top_k_genes=15).fit_and_oof(
        d["X_rna"][tr], d["X_mut"][tr], d["Y"][tr], subj_tr, list(range(14)), target_idx
    )
    m3 = HybridFewShotM3().fit(m1, oof, d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(5,))

    # Eğitim kovaryansını permüte eden M3_perm_cov
    rng = np.random.default_rng(77)
    E_oof = d["Y"][tr] - oof
    E_perm = E_oof.copy()
    for c in range(14):
        E_perm[:, c] = E_perm[rng.permutation(len(tr)), c]
    m3_perm_cov = HybridFewShotM3().fit(m1, oof, oof + E_perm, subj_tr, probe_pool, target_idx, k_values=(5,))

    Y_te = d["Y"][te]
    prior_te = m1.predict(d["X_rna"][te], d["X_mut"][te], n_drugs=14)
    probes = [np.array(probe_pool[:5], dtype=np.int64) for _ in range(len(te))]
    eval_mask = np.ones(len(te), dtype=bool)

    p_full = m3.predict(d["X_rna"][te], d["X_mut"][te], Y_te, probes, target_idx, k=5, prior_eval=prior_te)
    Y_within = _permute_within_patient_probes(Y_te, prior_te, probes, seed=11)
    p_within = m3.predict(d["X_rna"][te], d["X_mut"][te], Y_within, probes, target_idx, k=5, prior_eval=prior_te)
    Y_across = _permute_across_patient_probes(Y_te, prior_te, probes, eval_mask, seed=22)
    p_across = m3.predict(d["X_rna"][te], d["X_mut"][te], Y_across, probes, target_idx, k=5, prior_eval=prior_te)
    p_tr_cov = m3_perm_cov.predict(d["X_rna"][te], d["X_mut"][te], Y_te, probes, target_idx, k=5, prior_eval=prior_te)

    assert m1.fold_model_ is not None
    ev_full = evaluate_predictions(Y_te, p_full, target_idx, m1.fold_model_.mu, m1.fold_model_.b_d)
    ev_within = evaluate_predictions(Y_te, p_within, target_idx, m1.fold_model_.mu, m1.fold_model_.b_d)
    ev_across = evaluate_predictions(Y_te, p_across, target_idx, m1.fold_model_.mu, m1.fold_model_.b_d)
    ev_tr_cov = evaluate_predictions(Y_te, p_tr_cov, target_idx, m1.fold_model_.mu, m1.fold_model_.b_d)

    assert ev_full["rmse"] < ev_within["rmse"]
    assert ev_full["rmse"] < ev_across["rmse"]
    assert ev_full["rmse"] < ev_tr_cov["rmse"]


def test_sprint03_results_consistency() -> None:
    res_path = Path("data/processed/sprint03_results.json")
    if not res_path.exists():
        pytest.skip("sprint03_results.json henüz üretilmedi.")

    res = json.loads(res_path.read_text(encoding="utf-8"))
    for regime in ("cv_5fold_3seeds", "wave_ood"):
        m = res[regime]["metrics"]
        n_ref = m["k0_refs"]["M1"]["n_eval"]
        for k_str in ("1", "2", "3", "5", "8", "12"):
            for pol in ("Random50", "StaticDiversity", "StaticDOpt", "AdaptiveOED"):
                node = m["by_k"][k_str][pol]
                assert node["n_eval"] == n_ref
                assert np.isfinite(node["rmse"])
                assert 0.0 <= node["recall_at_3"] <= 1.0
                assert node["regret_at_3"] <= node["regret_at_1"] + 1e-9
