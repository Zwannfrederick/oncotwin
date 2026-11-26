"""Opus Sprint 4 adversaryal kontrolleri (üretim test paketinden ayrı; üretim çıktısına yazmaz).

Çalıştırma: PYTHONPATH=. python3 -m pytest -q -W error::RuntimeWarning audits/opus-2026-11-22-sprint04/opus_s04_adversarial_checks.py
Durum (2026-11-25): SOL-09 ve SOL-11 Sprint 4 v2'de kapandı; xfail işareti kalmadı.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.oncotwin.eval.uncertainty import (
    OmicsPCADistanceScorer,
    evaluate_conformal_and_shortlist,
    evaluate_locked_threshold_abstention,
    fit_split_conformal_simultaneous,
    m3_conditional_posterior_scale,
    paired_aurc_bootstrap_ci,
    selective_risk_coverage_curve,
)
from src.oncotwin.models.panel_selection import compute_contender_weights, select_adaptive_batch_oed


def _psd(n: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n + 3))
    return a @ a.T / n


# 1. Split-conformal uygulaması değiştirilebilir veride nominal eşzamanlı kapsamı sağlar
#    (alt sınır 1-α; üst sınır 1-α+1/(n+1), sürekli skorlar için).
def test_conformal_marginal_coverage_monte_carlo() -> None:
    rng = np.random.default_rng(1)
    n_cal, n_te, n_c, alpha, reps = 40, 200, 6, 0.10, 400
    covs = []
    for _ in range(reps):
        def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
            pred = rng.normal(size=(n, n_c)) * 10
            y = pred + rng.standard_t(df=3, size=(n, n_c)) * 5
            y[rng.random((n, n_c)) < 0.3] = np.nan  # rastgele eksik hedef
            y[:, 0] = np.where(np.isnan(y[:, 0]), pred[:, 0] + 1.0, y[:, 0])  # her satırda >=1 hedef
            return y, pred
        yc, pc = draw(n_cal)
        yt, pt = draw(n_te)
        fit = fit_split_conformal_simultaneous(yc, pc, list(range(n_c)), alpha=alpha)
        ev = evaluate_conformal_and_shortlist(yt, pt, list(range(n_c)), q_sim=fit["q_sim"])
        covs.append(ev["simultaneous_coverage"])
    mean_cov = float(np.mean(covs))
    se = float(np.std(covs) / math.sqrt(reps))
    assert mean_cov >= 1 - alpha - 3 * se
    assert mean_cov <= 1 - alpha + 1 / (n_cal + 1) + 3 * se


# 2. Konformal sıra istatistiği kayan nokta yuvarlamasıyla bir fazla alınmaz (n<=5000, α=0.10).
def test_conformal_rank_exact_integer_arithmetic() -> None:
    for n in range(1, 5001):
        fit = fit_split_conformal_simultaneous(
            np.arange(n, dtype=float)[:, None], np.zeros((n, 1)), [0], alpha=0.10
        )
        assert fit["order_stat_rank"] == -(-(n + 1) * 9 // 10)


# 3. SOL-11: q=∞ iken kapsam/birinciyi içerme 1.0 olarak yayımlanıyor (beklenen: hesaplanamadı/NaN).
# SOL-11 / OPUS-021: Sprint 4 v2 (2026-11-22) ile kapandı.
def test_infinite_q_does_not_publish_coverage() -> None:
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    ev = evaluate_conformal_and_shortlist(y, y + 5.0, [0, 1], q_sim=float("inf"))
    assert math.isnan(ev["simultaneous_coverage"])


# 4. SOL-09: aynı hasta profili, yalnız MC tohumu farklı -> adaptif panel değişebiliyor
#    (seçici kalibrasyon ve test için aynı sabit fonksiyon değil).
# Fonksiyon düzeyi özellik (hata değil): farklı MC tohumu paneli değiştirir. Bu yüzden koşucu kalibrasyon ve
# test için tek `contender_seed` kilitlemelidir (SOL-09 / OPUS-017, v2'de kapandı; koşucu düzeyi kabul testi
# tests/test_sprint04_contracts.py::test_selector_is_same_fixed_function_for_calibration_and_test).
def test_adaptive_selector_depends_on_mc_seed() -> None:
    n_c, n_p = 8, 30
    cov = _psd(n_c + n_p, 3)
    t_loc, p_loc = np.arange(n_c), np.arange(n_c, n_c + n_p)
    prior = np.random.default_rng(5).normal(size=(200, n_c)) * 3  # yakın yarışan adaylar
    w1 = compute_contender_weights(prior, cov[np.ix_(t_loc, t_loc)], top_k=3, n_mc=500, seed=2027)
    w2 = compute_contender_weights(prior, cov[np.ix_(t_loc, t_loc)], top_k=3, n_mc=500, seed=2028)
    changed = sum(
        set(select_adaptive_batch_oed(cov, p_loc, t_loc, w1[i], k=5, lam=1.0).tolist())
        != set(select_adaptive_batch_oed(cov, p_loc, t_loc, w2[i], k=5, lam=1.0).tolist())
        for i in range(len(prior))
    )
    assert changed > 0


# 5. Aynı tohumla ağırlıklar hasta sırasından bağımsızdır (gürültü bankası çağrı başına tek) ->
#    SOL-09 düzeltmesi için tek kilitli tohum yeterli.
def test_contender_weights_row_order_invariant_with_same_seed() -> None:
    n_c = 6
    cov = _psd(n_c, 9)
    prior = np.random.default_rng(2).normal(size=(25, n_c)) * 4
    perm = np.random.default_rng(3).permutation(25)
    w = compute_contender_weights(prior, cov, seed=11)
    w_perm = compute_contender_weights(prior[perm], cov, seed=11)
    assert np.array_equal(w[perm], w_perm)


# 6. m3_conditional_posterior_scale yoğun ters matris formülüyle eşleşir.
def test_posterior_scale_matches_dense_formula() -> None:
    cov = _psd(15, 4)
    t = np.arange(5)
    S = np.array([7, 9, 12])
    lam = 0.7
    sc = m3_conditional_posterior_scale(cov, [S, np.array([], dtype=np.int64)], t, lam)
    C_ss = cov[np.ix_(S, S)] + lam * np.eye(3)
    post = np.diag(cov[np.ix_(t, t)] - cov[np.ix_(t, S)] @ np.linalg.inv(C_ss) @ cov[np.ix_(S, t)])
    assert np.isclose(sc[0], math.sqrt(np.mean(post)))
    assert np.isclose(sc[1], math.sqrt(np.mean(np.diag(cov)[t])))


# 7. Sabit skor (bilgisiz sıralama) AURC'si rastgele referansa eşit; eşleştirilmiş fark tam 0.
def test_constant_score_aurc_equals_random_reference() -> None:
    loss = np.random.default_rng(6).exponential(10, size=50)
    rc = selective_risk_coverage_curve(np.full(50, 3.0), loss)
    assert np.isclose(rc["aurc"], rc["random_aurc"])
    boot = paired_aurc_bootstrap_ci(np.full(50, 3.0), None, loss, n_boot=50)
    assert boot["diff"] == 0.0 and boot["ci_low"] == 0.0 and boot["ci_high"] == 0.0


# 8. OPUS-019: ortak tavan bir bağ grubunun içinden geçtiğinde kesirli kabul varsayılır —
#    gerçekleşebilir eşik noktasındaki değerden farklı sonuç (SOL-10 ikincil madde).
def test_capped_aurc_fractional_tie_group_is_not_a_realizable_threshold() -> None:
    sc = np.array([0.0, 1.0, 1.0, 1.0])
    loss = np.array([0.0, 30.0, 0.0, 0.0])
    rc = selective_risk_coverage_curve(sc, loss, coverage_cap=0.5)
    # gerçekleşebilir noktalar: kapsam 0.25 (risk 0) ve 1.0 (risk 7.5); 0.5'te kabul edilebilir eşik yok
    assert rc["cap_status"] == "ok"
    assert np.isclose(rc["aurc_capped"], (0.25 * 0.0 + 0.25 * 7.5) / 0.5)


# 9. Kilitli eşik bağlı skorları topluca kabul eder: %80 hedef kabul, ayrık skorda %100'e sıçrayabilir.
def test_locked_threshold_with_discrete_scores_overshoots_target() -> None:
    oof = np.array([1.0] * 9 + [2.0])
    tau = float(np.quantile(oof, 0.80, method="higher"))
    te = np.array([1.0] * 10)
    ab = evaluate_locked_threshold_abstention(te, np.ones(10), np.ones(10), np.ones(10), tau)
    assert ab["acceptance_rate_total"] == 1.0


# 10. PCA uzaklığı: eğitim satır sırasından bağımsız ve eğitim merkezinde ~0.
def test_pca_scorer_row_order_invariant() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 60))
    M = (rng.random((40, 5)) < 0.3).astype(float)
    perm = rng.permutation(40)
    a = OmicsPCADistanceScorer(top_k_genes=50, max_components=10).fit(X, M)
    b = OmicsPCADistanceScorer(top_k_genes=50, max_components=10).fit(X[perm], M[perm])
    Xt, Mt = rng.normal(size=(7, 60)), (rng.random((7, 5)) < 0.3).astype(float)
    assert np.allclose(a.score(Xt, Mt), b.score(Xt, Mt), atol=1e-8)
