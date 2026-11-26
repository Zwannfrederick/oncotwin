"""OPUS bağımsız adversarial denetim testleri (üretim koduna dokunmaz; scratchpad'de koşar)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.oncotwin.eval.metrics import (
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
    regret_at_k,
    safe_spearman,
)
from src.oncotwin.eval.splits import make_patient_disjoint_kfold, sample_probe_sets
from src.oncotwin.experiments.run_sprint03 import _permute_within_patient_probes
from src.oncotwin.models.baselines import (
    HybridFewShotM3,
    OmicsRidgeM1,
    conditional_gaussian_predict_matrix,
    estimate_psd_shrunk_cov,
)
from src.oncotwin.models.panel_selection import (
    compute_contender_weights,
    select_adaptive_batch_oed,
    select_static_d_optimal_ranking,
    take_first_k_available,
)
from tests.test_sprint02 import _synthetic_cohort


# ---------- brute-force referanslar ----------

def _mi_bruteforce(cov, T, S, lam):
    """I(Y_T ; Y_S + eps) = 0.5[logdet(C_TT) - logdet(C_TT|S)], eps ~ N(0, lam I)."""
    C_TT = cov[np.ix_(T, T)]
    if len(S) == 0:
        return 0.0
    C_SS = cov[np.ix_(S, S)] + lam * np.eye(len(S))
    C_TS = cov[np.ix_(T, S)]
    post = C_TT - C_TS @ np.linalg.solve(C_SS, C_TS.T)
    return 0.5 * (np.linalg.slogdet(C_TT)[1] - np.linalg.slogdet(post)[1])


def _greedy_mi_bruteforce(cov, T, P, lam):
    S: list[int] = []
    rem = list(P)
    while rem:
        base = _mi_bruteforce(cov, T, S, lam)
        gains = [(_mi_bruteforce(cov, T, S + [j], lam) - base) for j in rem]
        j = rem[int(np.argmax(gains))]
        S.append(j)
        rem.remove(j)
    return np.asarray(S)


def _rand_cov(n, rank, seed, scale=1500.0, ridge=1e-3):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, rank))
    c = A @ A.T
    c = scale * c / np.mean(np.diag(c))
    return c + ridge * np.eye(n)


def _expected_over_ties(fn, yt, yp, k):
    """Tahmin bağlarını tüm permütasyonlarla kırarak beklenen metrik (kesin referans)."""
    n = len(yp)
    vals = []
    for perm in itertools.permutations(range(n)):
        # Bağ grubu içi sırayı perm ile kır: anahtar (yp, perm-rank)
        order = sorted(range(n), key=lambda i: (yp[i], perm[i]))
        vals.append(fn(yt, order, k))
    return float(np.mean(vals))


def _recall_order(yt, order, k):
    m = min(k, len(yt))
    true_w = _true_topk_weights(yt, m)
    return float(sum(true_w[i] for i in order[:m]) / m)


def _true_topk_weights(yt, m):
    # Gerçek tarafta da bağlar beklenen üyelikle (metrics._top_k_membership_weights eşdeğeri, brute-force)
    n = len(yt)
    acc = np.zeros(n)
    perms = list(itertools.permutations(range(n)))
    for perm in perms:
        order = sorted(range(n), key=lambda i: (yt[i], perm[i]))
        for i in order[:m]:
            acc[i] += 1
    return acc / len(perms)


def _regret_order(yt, order, k):
    m = min(k, len(yt))
    return float(min(yt[i] for i in order[:m]) - min(yt))


def _ndcg_order(yt, order, k):
    rel = np.max(yt) - np.asarray(yt)
    m = min(k, len(yt))
    disc = 1.0 / np.log2(np.arange(2, m + 2))
    idcg = float(np.sum(np.sort(rel)[::-1][:m] * disc))
    return float(np.sum(rel[order[:m]] * disc) / idcg)


# ---------- C. Sayısal doğruluk ----------

@pytest.mark.parametrize("seed,rank", [(1, 12), (2, 3), (3, 2)])  # 2 ve 3: rank-eksik (near-singular)
def test_static_dopt_full_ranking_matches_bruteforce_greedy(seed, rank):
    cov = _rand_cov(14, rank, seed)
    T, P = np.arange(0, 5), np.arange(5, 14)
    for lam in (50.0, 500.0, 5000.0):
        got = select_static_d_optimal_ranking(cov, P, T, lam=lam)
        ref = _greedy_mi_bruteforce(cov, T, P, lam)
        # Kazanç bağlarında sıra farklı olabilir; ilk 5 seçim aynı ve MI eğrisi eşit olmalı
        for kk in range(1, 6):
            assert np.isclose(_mi_bruteforce(cov, T, list(got[:kk]), lam),
                              _mi_bruteforce(cov, T, list(ref[:kk]), lam), rtol=1e-6, atol=1e-9)


def _adaptive_bruteforce(cov, avail, T, w, k, lam):
    S: list[int] = []
    rem = list(avail)
    for _ in range(min(k, len(rem))):
        def post_var(Sx):
            if not Sx:
                return np.diag(cov)[T]
            C_SS = cov[np.ix_(Sx, Sx)] + lam * np.eye(len(Sx))
            C_TS = cov[np.ix_(T, Sx)]
            return np.diag(cov[np.ix_(T, T)] - C_TS @ np.linalg.solve(C_SS, C_TS.T))
        base = w @ post_var(S)
        gains = [base - w @ post_var(S + [j]) for j in rem]
        j = rem[int(np.argmax(gains))]
        S.append(j)
        rem.remove(j)
    return S


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_adaptive_oed_matches_bruteforce_posterior_recompute(seed):
    cov = _rand_cov(16, 4, seed)
    T = np.arange(0, 6)
    avail = np.array([6, 8, 9, 11, 12, 13, 15])
    w = np.random.default_rng(seed).dirichlet(np.ones(6))
    got = select_adaptive_batch_oed(cov, avail, T, w, k=5, lam=200.0)
    ref = _adaptive_bruteforce(cov, avail, T, w, 5, 200.0)
    assert list(got) == ref


def test_rank1_downdate_drift_after_many_steps():
    """60 ardışık rank-1 downdate sonrası Schur matrisi doğrudan hesapla eşleşmeli."""
    cov = _rand_cov(90, 20, 7)
    T, P = np.arange(0, 24), np.arange(24, 90)
    lam = 50.0
    ranked = select_static_d_optimal_ranking(cov, P, T, lam=lam)
    S = list(ranked[:60])
    # Fonksiyon içindeki Sigma_free'yi yeniden üret
    Sig = 0.5 * (cov + cov.T)
    for j in S:
        v = Sig[:, j].copy()
        Sig = Sig - np.outer(v, v) / (max(Sig[j, j], 0.0) + lam)
    C_SS = cov[np.ix_(S, S)] + lam * np.eye(len(S))
    direct = cov - cov[:, S] @ np.linalg.solve(C_SS, cov[S, :])
    rel = np.max(np.abs(Sig - direct)) / np.max(np.abs(cov))
    assert rel < 1e-9, rel


def test_conditional_gaussian_matches_explicit_formula_and_skips_nan():
    cov = _rand_cov(10, 3, 8)
    rng = np.random.default_rng(8)
    R = rng.normal(scale=30, size=(3, 10))
    R[1, 7] = np.nan  # eksik prob
    T = np.array([0, 1, 2])
    S = [np.array([5, 6, 7]), np.array([5, 6, 7]), np.array([], dtype=np.int64)]
    out = conditional_gaussian_predict_matrix(R, cov, S, T, lam=100.0)
    for i in range(2):
        Sv = [s for s in S[i] if np.isfinite(R[i, s])]
        ref = cov[np.ix_(T, Sv)] @ np.linalg.solve(cov[np.ix_(Sv, Sv)] + 100 * np.eye(len(Sv)), R[i, Sv])
        assert np.allclose(out[i], ref)
    assert np.all(out[2] == 0.0)
    # Not: satır 1'de bütçe 3 iken etkin bütçe sessizce 2'ye düşer (uyarı yok)


def test_psd_shrunk_cov_edge_cases():
    rng = np.random.default_rng(9)
    R = rng.normal(scale=40, size=(50, 8))
    R[:, 3] = np.nan          # tamamen eksik sütun
    R[:, 4] = 7.0             # sabit sütun
    R[:40, 5] = np.nan        # min_pair altı
    R[:, 6] = R[:, 7] * 1.0   # tam kolineer
    for a in (0.0, 0.05, 0.5):
        c = estimate_psd_shrunk_cov(R, alpha=a)
        assert np.all(np.isfinite(c))
        assert np.allclose(c, c.T)
        assert np.min(np.linalg.eigvalsh(c)) > 0  # kesin pozitif (min_eig=1e-4)
        np.linalg.cholesky(c)


def test_contender_weights_cholesky_on_exactly_singular_large_scale_cov():
    """Ham (shrinkage'sız) rank-1 kovaryansta 1e-6 jitter yeterli mi?"""
    v = np.linspace(20, 60, 24)
    cov = np.outer(v, v)  # rank-1, ölçek ~3600
    prior = np.random.default_rng(1).normal(150, 30, size=(4, 24))
    try:
        w = compute_contender_weights(prior, cov, n_mc=200)
        ok = np.allclose(w.sum(axis=1), 1.0)
    except np.linalg.LinAlgError:
        ok = False
    # Kayıt amaçlı: üretimde cov_CC her zaman estimate_psd_shrunk_cov'dan (min_eig 1e-4) gelir
    print("singular_cov_cholesky_ok=", ok)


# ---------- Metrikler: tie-enumerasyonlu kesin referans ----------

CASES = [
    (np.array([40., 70, 90, 160, 220]), np.array([1., 1, 1, 2, 3])),
    (np.array([50., 50, 50, 50, 50]), np.array([3., 1, 2, 5, 4])),       # tüm gerçek değerler eşit
    (np.array([10., 20, 30, 40, 50]), np.array([7., 7, 7, 7, 7])),       # sabit tahmin
    (np.array([90., 30, 30, 60, 10, 75]), np.array([5., 2, 2, 2, 9, 1])),
]


@pytest.mark.parametrize("yt,yp", CASES)
def test_ranking_metrics_equal_tie_enumeration_reference(yt, yp):
    for k in (1, 3):
        ref_reg = _expected_over_ties(_regret_order, yt, yp, k)
        assert np.isclose(regret_at_k(yt, yp, k=k), ref_reg), (k, regret_at_k(yt, yp, k=k), ref_reg)
    ref_rec = _expected_over_ties(_recall_order, yt, yp, 3)
    assert np.isclose(recall_at_k(yt, yp, k=3), ref_rec)
    if np.ptp(yt) > 0:
        assert np.isclose(ndcg_at_k(yt, yp, k=3), _expected_over_ties(_ndcg_order, yt, yp, 3))
    else:
        assert np.isnan(ndcg_at_k(yt, yp, k=3))  # tüm yanıtlar eşit -> NaN (nanmean ile düşer)


def test_metric_degenerate_inputs():
    one = np.array([100.0])
    assert regret_at_k(one, one, 3) == 0.0 and recall_at_k(one, one, 3) == 1.0
    nan = np.array([np.nan, np.nan])
    assert np.isnan(recall_at_k(nan, nan)) and np.isnan(regret_at_k(nan, nan)) and np.isnan(ndcg_at_k(nan, nan))
    assert safe_spearman(np.array([1., 1, 1, 1]), np.array([1., 2, 3, 4])) == 0.0
    assert np.isnan(safe_spearman(np.array([1., 2]), np.array([1., 2])))


# ---------- E. Bootstrap ----------

def test_paired_bootstrap_matches_manual_patient_resampling():
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 2, 80), rng.normal(9, 2, 80)
    a[5] = np.nan
    fn = lambda v: float(np.sqrt(np.mean(v)))
    res = paired_bootstrap_ci(a ** 2, b ** 2, n_boot=500, seed=11, agg_fn=fn)
    m = np.isfinite(a)
    av, bv = (a ** 2)[m], (b ** 2)[m]
    r2 = np.random.default_rng(11)
    d = [fn(av[i]) - fn(bv[i]) for i in (r2.integers(0, len(av), len(av)) for _ in range(500))]
    assert np.isclose(res["ci_low"], np.percentile(d, 2.5)) and res["n_paired"] == 79
    one = paired_bootstrap_ci(np.array([1.0]), np.array([0.0]))
    assert one["ci_low"] == one["ci_high"] == 1.0  # n=1: sıfır genişlikli GA (uyarı yok)


# ---------- B / A. Panel bütçesi, backfill ve sızıntı invaryansı ----------

def test_budget_exceeding_available_is_silently_truncated():
    ranked = np.array([0, 1, 2, 3, 4])
    y = np.array([np.nan, 1.0, np.nan, 2.0, np.nan])
    chosen, hits = take_first_k_available(ranked, y, k=5)
    assert len(chosen) == 2  # k=5 istendi, 2 verildi; hata/uyarı yok
    cov = _rand_cov(8, 3, 2)
    sel = select_adaptive_batch_oed(cov, np.array([5, 6]), np.arange(3), np.ones(3) / 3, k=5, lam=50.0)
    assert len(sel) == 2
    sp = sample_probe_sets(y[None, :], [0, 1, 2, 3, 4], k=5, n_repeats=1)[0][0]
    assert len(sp) == 2


def test_ood_backfill_preserves_budget_when_enough_available():
    ranked = np.arange(20)
    y = np.full(20, np.nan)
    y[[3, 7, 8, 11, 15, 16, 17, 18, 19]] = 1.0  # ilk 3 sırada hiçbiri yok (ABT-737/XAV-939 benzeri)
    for k in (1, 2, 3, 5, 8):
        chosen, hits = take_first_k_available(ranked, y, k=k)
        assert len(chosen) == k and np.all(np.isfinite(y[chosen]))
    assert take_first_k_available(ranked, y, k=3)[1] == 0  # direct-hit 0 -> "tam=0%"


def test_selection_and_m3_target_predictions_invariant_to_hidden_test_auc():
    d = _synthetic_cohort(n_patients=70, n_genes=20, n_drugs=16, seed=5)
    tr, te = np.arange(0, 50), np.arange(50, 70)
    T, P = list(range(6)), list(range(6, 16))
    subj = d["meta"].iloc[tr]["dbgap_subject_id"].to_numpy()
    m1, oof = OmicsRidgeM1(top_k_genes=15).fit_and_oof(d["X_rna"][tr], d["X_mut"][tr], d["Y"][tr], subj, list(range(16)), T)
    m3 = HybridFewShotM3().fit(m1, oof, d["Y"][tr], subj, P, T, k_values=(5,))
    cm = m3._cov_model
    prior = m1.predict(d["X_rna"][te], d["X_mut"][te], n_drugs=16)
    t_loc = np.array([cm.global_to_local_[g] for g in T])
    w = compute_contender_weights(prior[:, T], cm.cov_by_k_[5][np.ix_(t_loc, t_loc)])

    Y = d["Y"][te].copy()
    rng = np.random.default_rng(0)
    Y_pert = Y.copy()
    fin = np.isfinite(Y_pert)
    Y_pert[fin] = rng.permutation(Y_pert[fin]) + rng.normal(0, 500, fin.sum())  # tüm değerler bozuldu, maske aynı

    def select(Yx):
        out = []
        for i in range(len(te)):
            avail = np.array([cm.global_to_local_[g] for g in P if np.isfinite(Yx[i, g])])
            out.append(tuple(select_adaptive_batch_oed(cm.cov_by_k_[5], avail, t_loc, w[i], 5, 200.0)))
        return out, sample_probe_sets(Yx, P, k=5, n_repeats=1, seed=1)[0]

    (ad_a, rd_a), (ad_b, rd_b) = select(Y), select(Y_pert)
    assert ad_a == ad_b
    assert all(np.array_equal(x, y) for x, y in zip(rd_a, rd_b))

    # M3 hedef tahminleri yalnızca prob sütunlarına bağlı olmalı: hedef AUC'leri bozulunca değişmemeli
    probes = rd_a
    Y_tgt_pert = Y.copy()
    Y_tgt_pert[:, T] = Y_tgt_pert[:, T] + 1000.0
    p1 = m3.predict(None, None, Y, probes, T, k=5, prior_eval=prior)
    p2 = m3.predict(None, None, Y_tgt_pert, probes, T, k=5, prior_eval=prior)
    assert np.array_equal(p1[:, T], p2[:, T])


def test_determinism_same_seed_identical_different_seed_differs():
    prior = np.random.default_rng(2).normal(150, 20, size=(30, 24))
    cov = _rand_cov(24, 6, 3)
    a = compute_contender_weights(prior, cov, seed=42)
    b = compute_contender_weights(prior, cov, seed=42)
    c = compute_contender_weights(prior, cov, seed=43)
    assert np.array_equal(a, b)
    print("contender_weight_seed_sensitivity_max_abs=", float(np.max(np.abs(a - c))))
    s1 = sample_probe_sets(np.ones((5, 30)), list(range(30)), 5, 2, seed=9)
    s2 = sample_probe_sets(np.ones((5, 30)), list(range(30)), 5, 2, seed=9)
    assert all(np.array_equal(x, y) for r1, r2 in zip(s1, s2) for x, y in zip(r1, r2))


# ---------- Tekrarlanan specimen ----------

def test_kfold_with_repeated_specimens_keeps_patient_disjoint():
    meta = pd.DataFrame({"dbgap_subject_id": [1, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10],
                         "cohort": ["Waves1+2"] * 7 + ["Waves3+4"] * 7})
    for tr, te in make_patient_disjoint_kfold(meta, n_splits=3, seed=1):
        assert set(meta.dbgap_subject_id[tr]).isdisjoint(set(meta.dbgap_subject_id[te]))


# ---------- D. Ablasyon mekanizması ----------

def test_perm_within_patient_does_not_preserve_offset_under_unequal_weights():
    """Saf-ofset dünyada (ilaç-kimliğine özgü bilgi YOK) perm_within kayıp yaratıyor mu?"""
    rng = np.random.default_rng(12)
    n_t, n_p, n_pat = 24, 40, 3000
    sig_b = 25.0
    noise_sd = rng.uniform(10, 45, n_t + n_p)            # heteroskedastik ilaç gürültüsü
    cov = sig_b ** 2 * np.ones((n_t + n_p,) * 2) + np.diag(noise_sd ** 2)
    b = rng.normal(0, sig_b, n_pat)
    R = b[:, None] + rng.normal(size=(n_pat, n_t + n_p)) * noise_sd[None, :]
    T = np.arange(n_t)
    k = 5
    probes = [np.sort(rng.choice(np.arange(n_t, n_t + n_p), k, replace=False)) for _ in range(n_pat)]
    zero = np.zeros_like(R)
    full = conditional_gaussian_predict_matrix(R, cov, probes, T, lam=1e-6)
    R_perm = _permute_within_patient_probes(R, zero, probes, seed=3)
    perm = conditional_gaussian_predict_matrix(R_perm, cov, probes, T, lam=1e-6)
    rmse = lambda P: float(np.sqrt(np.mean((R[:, T] - P) ** 2)))
    base = rmse(np.zeros((n_pat, n_t)))
    gain_full, gain_perm = base - rmse(full), base - rmse(perm)
    print(f"pure-offset world: base={base:.2f} full={rmse(full):.2f} perm_within={rmse(perm):.2f} "
          f"retained_gain={gain_perm / gain_full:.2%}")
    # Homoskedastik kontrol: ağırlıklar eşit -> perm_within tam olarak aynı tahmini vermeli
    cov_h = sig_b ** 2 * np.ones((n_t + n_p,) * 2) + 30.0 ** 2 * np.eye(n_t + n_p)
    f_h = conditional_gaussian_predict_matrix(R, cov_h, probes, T, lam=1e-6)
    p_h = conditional_gaussian_predict_matrix(R_perm, cov_h, probes, T, lam=1e-6)
    assert np.allclose(f_h, p_h)


def test_perm_within_identity_rate():
    zero = np.zeros((20000, 10))
    R = np.random.default_rng(1).normal(size=(20000, 10))
    for k in (3, 5, 8):
        probes = [np.arange(k)] * 20000
        Rp = _permute_within_patient_probes(R, zero, probes, seed=4)
        ident = float(np.mean(np.all(Rp[:, :k] == R[:, :k], axis=1)))
        print(f"k={k} identity_perm_rate={ident:.4f}")
