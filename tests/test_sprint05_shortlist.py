"""Sprint 5 Kulvar B (sıra-kalibreli konformal kısa liste ve AUGRC) testleri."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.oncotwin.eval.rank_conformal import (
    build_shortlist_masks,
    evaluate_shortlist_masks,
    fit_rank_conformal_shortlist,
    fit_top1_gap_conformal_shortlist,
    generalized_risk_coverage_augrc,
)


def test_rank_and_top1_gap_conformal_monte_carlo_coverage_guarantee() -> None:
    """Değiştirilebilir sentetik hastalarda RankConformal ve Top1GapConformal [1-α, 1-α+1/(n+1)] bandındadır."""
    rng = np.random.default_rng(2026)
    alpha = 0.10
    n_cal, n_te, n_drugs = 49, 25, 12
    target_idx = list(range(n_drugs))
    cov_rank, cov_gap, sz_rank, sz_gap = [], [], [], []

    for _ in range(300):
        Y_pred = rng.normal(140.0, 30.0, size=(n_cal + n_te, n_drugs))
        Y_true = Y_pred + rng.normal(0.0, 18.0, size=(n_cal + n_te, n_drugs))
        # Rastgele eksik hedefler (en az 6 gözlenen bırak)
        miss = rng.uniform(size=Y_true.shape) < 0.15
        miss[:, :6] = False
        Y_true[miss] = np.nan

        rc = fit_rank_conformal_shortlist(Y_true[:n_cal], Y_pred[:n_cal], target_idx, alpha=alpha)
        gc = fit_top1_gap_conformal_shortlist(Y_true[:n_cal], Y_pred[:n_cal], target_idx, alpha=alpha)
        assert rc["finite_published"] and gc["finite_published"]

        m_r = build_shortlist_masks(Y_true[n_cal:], Y_pred[n_cal:], target_idx, "abs_rank", float(rc["k_hat"]))
        m_g = build_shortlist_masks(Y_true[n_cal:], Y_pred[n_cal:], target_idx, "top1_gap", gc["m_hat"])
        ev_r = evaluate_shortlist_masks(Y_true[n_cal:], m_r, target_idx, finite_published=True)
        ev_g = evaluate_shortlist_masks(Y_true[n_cal:], m_g, target_idx, finite_published=True)
        cov_rank.append(ev_r["shortlist_contains_true_top1_rate"])
        cov_gap.append(ev_g["shortlist_contains_true_top1_rate"])
        sz_rank.append(ev_r["shortlist_mean_size_obs"])
        sz_gap.append(ev_g["shortlist_mean_size_obs"])

    # Sonlu-örneklem konformal kapsam >= 1 - alpha (Monte Carlo SE ~ 0.005)
    assert float(np.mean(cov_rank)) >= 1.0 - alpha - 0.015
    assert float(np.mean(cov_gap)) >= 1.0 - alpha - 0.015
    assert float(np.mean(sz_rank)) < n_drugs
    assert float(np.mean(sz_gap)) < n_drugs


def test_shortlist_ties_infinite_guard_and_auc_loss_identities() -> None:
    """SOL-11 sonlu-eşik koruması, bağ dahil etme ve best/mean/worst regret kimlikleri."""
    target_idx = [0, 1, 2, 3]
    # n_cal=5, alpha=0.10 => ceil(6*0.9)=6 > 5 -> sonlu eşik yayımlanmamalı
    Y_small = np.arange(20, dtype=np.float64).reshape(5, 4)
    rc_inf = fit_rank_conformal_shortlist(Y_small, Y_small, target_idx, alpha=0.10)
    gc_inf = fit_top1_gap_conformal_shortlist(Y_small, Y_small, target_idx, alpha=0.10)
    assert rc_inf["finite_published"] is False and rc_inf["k_hat"] is None
    assert gc_inf["finite_published"] is False and not math.isfinite(gc_inf["m_hat"])

    m_inf = build_shortlist_masks(Y_small, Y_small, target_idx, "abs_rank", rc_inf["k_hat_float"])
    ev_inf = evaluate_shortlist_masks(Y_small, m_inf, target_idx, finite_published=False)
    assert ev_inf["status"].startswith("hesaplanamadı")
    assert not math.isfinite(ev_inf["shortlist_contains_true_top1_rate"])
    assert not math.isfinite(ev_inf["shortlist_mean_size_obs"])

    # Bağ durumu: yp = [10, 20, 20, 30], k_hat=2 -> index 0,1,2 dahil (boyut 3), gerçek en iyi index 2 (y=5)
    Yt = np.array([[50.0, 40.0, 5.0, 80.0]], dtype=np.float64)
    Yp = np.array([[10.0, 20.0, 20.0, 30.0]], dtype=np.float64)
    m_tie = build_shortlist_masks(Yt, Yp, target_idx, "abs_rank", 2.0)
    ev_tie = evaluate_shortlist_masks(Yt, m_tie, target_idx, finite_published=True)
    assert ev_tie["shortlist_mean_size_obs"] == 3.0
    assert ev_tie["shortlist_contains_true_top1_rate"] == 1.0
    assert ev_tie["shortlist_best_regret1"] == 0.0
    assert abs(ev_tie["shortlist_mean_regret1"] - ((50.0 + 40.0 + 5.0) / 3.0 - 5.0)) <= 1e-12
    assert ev_tie["shortlist_worst_regret1"] == 45.0


def test_augrc_constant_score_equals_exact_random_and_oracle_bounds() -> None:
    """Traub et al. (NeurIPS 2024) AUGRC: sabit skor == 0.5*mean(loss) == Random AUGRC; oracle <= actual."""
    losses = np.array([0.0, 10.0, 30.0, 60.0, 100.0], dtype=np.float64)
    const_scores = np.full(5, 7.0, dtype=np.float64)
    good_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    bad_scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)

    r_const = generalized_risk_coverage_augrc(const_scores, losses)
    r_good = generalized_risk_coverage_augrc(good_scores, losses)
    r_bad = generalized_risk_coverage_augrc(bad_scores, losses)

    expected_rand = 0.5 * float(np.mean(losses))
    assert abs(r_const["augrc"] - expected_rand) <= 1e-12
    assert abs(r_const["random_augrc"] - expected_rand) <= 1e-12
    assert abs(r_good["augrc"] - r_good["oracle_augrc"]) <= 1e-12
    assert r_good["e_augrc"] <= 1e-12
    assert r_good["augrc"] < r_const["augrc"] < r_bad["augrc"]

    r_empty = generalized_risk_coverage_augrc(np.array([np.nan]), np.array([1.0]))
    assert r_empty["status"] == "hesaplanamadı" and not math.isfinite(r_empty["augrc"])


def test_sprint05_shortlist_results_and_sprint04_v2_equality() -> None:
    """Kulvar B EqualWidthAUC Sprint 4 v2 ile <=1e-12 eşit olmalı; RankConformal/Top1Gap ve AUGRC tutarlı olmalı."""
    s5 = json.loads(Path("data/processed/sprint05_results.json").read_text(encoding="utf-8"))
    s4 = json.loads(Path("data/processed/sprint04_v2_results.json").read_text(encoding="utf-8"))
    tb = s5["track_b_shortlist"]

    for rn in ("id_60_20_20", "wave_ood"):
        for arm in ("M0_k0", "M1_k0", "StaticDOpt_k2", "StaticDOpt_k5", "StaticDOpt_k8", "AdaptiveOED_k5"):
            methods = tb["regimes"][rn]["arms"][arm]["methods"]
            assert set(methods.keys()) == {"EqualWidthAUC", "RankConformal", "RelRankConformal", "Top1GapConformal"}
            e5 = methods["EqualWidthAUC"]
            e4 = s4["regimes"][rn]["scenarios"]["archive_available"]["arms"][arm]["conformal"]
            assert abs(e5["calibration"]["width_sim"] - e4["width_sim"]) <= 1e-12
            assert abs(e5["test"]["shortlist_mean_size_obs"] - e4["mean_shortlist_size_obs"]) <= 1e-12
            assert abs(e5["test"]["shortlist_contains_true_top1_rate"] - e4["shortlist_contains_true_top1_rate"]) <= 1e-12

            au = tb["regimes"][rn]["augrc_sensitivity"][arm]
            assert math.isfinite(au["omics_pca_dist"]["augrc"])
            assert math.isfinite(au["m3_post_scale"]["augrc"])
            if arm in ("M0_k0", "M1_k0"):
                # Sabit skorlu M0/M1 kollarında m3_post_scale AUGRC == exact Random AUGRC
                assert abs(au["m3_post_scale"]["augrc"] - au["m3_post_scale"]["random_augrc"]) <= 1e-12

    # Birincil ID 60/20/20 StaticDOpt_k5: RankConformal ve Top1GapConformal EqualWidthAUC'den belirgin kısa
    m_main = tb["regimes"]["id_60_20_20"]["arms"]["StaticDOpt_k5"]["methods"]
    assert m_main["RankConformal"]["test"]["shortlist_mean_size_obs"] < m_main["EqualWidthAUC"]["test"]["shortlist_mean_size_obs"] - 10.0
    assert m_main["Top1GapConformal"]["test"]["shortlist_mean_size_obs"] < m_main["EqualWidthAUC"]["test"]["shortlist_mean_size_obs"] - 10.0
    assert m_main["Top1GapConformal"]["test"]["definite_top1_rate_obs"] > 0.0

    z = np.load(Path("data/processed/sprint05_vectors.npz"))
    arr = z["track_b/id_60_20_20/StaticDOpt_k5/RankConformal/patient_shortlist_size_obs"]
    assert len(arr) == 93 and int(np.isfinite(arr).sum()) == 91
