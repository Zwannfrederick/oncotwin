"""Opus regresyon testleri (DEV-OPUS-12): audits/opus-2026-11-21 denetim testlerinden kalıcı olanlar
+ DEV-OPUS-01/02 bütçe ve Kapı 2 guard testleri."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.oncotwin.eval.metrics import (
    _rankdata_average,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
    regret_at_k,
    safe_spearman,
)
from src.oncotwin.eval.splits import make_patient_disjoint_kfold, sample_probe_sets
from src.oncotwin.experiments.sprint03_ablations import _permute_within_patient_probes
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
    assert np.array_equal(a, b) and not np.array_equal(a, c)
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
    assert gain_perm / gain_full < 0.9  # OPUS-002: heteroskedastik ağırlıkta ofset korunmaz
    # Homoskedastik kontrol: ağırlıklar eşit -> perm_within tam olarak aynı tahmini vermeli
    cov_h = sig_b ** 2 * np.ones((n_t + n_p,) * 2) + 30.0 ** 2 * np.eye(n_t + n_p)
    f_h = conditional_gaussian_predict_matrix(R, cov_h, probes, T, lam=1e-6)
    p_h = conditional_gaussian_predict_matrix(R_perm, cov_h, probes, T, lam=1e-6)
    assert np.allclose(f_h, p_h)



# ---------- DEV-OPUS-01: koşucu bütçe guard'ı ----------

def test_runner_rejects_short_panel_for_evaluable_patient():
    # DEV-OPUS-20: yardımcı 4 koşucuda kullanıldığı için ortak panel modülüne taşındı.
    from src.oncotwin.models.panel_selection import require_full_budget

    probes = [np.array([1, 2, 3]), np.array([4, 5]), np.array([7])]
    eval_mask = np.array([True, False, False])
    require_full_budget(probes, eval_mask, k=3, policy="StaticDOpt")  # değerlendirilmeyen kısa satırlar serbest
    with pytest.raises(ValueError, match="StaticDOpt"):
        require_full_budget(probes, np.array([True, True, False]), k=3, policy="StaticDOpt")


# ---------- DEV-OPUS-02: Kapı 2 W3+4 ablasyonu %95 GA ile ----------

def _ci(diff, lo, hi):
    return {"diff": diff, "ci_low": lo, "ci_high": hi, "n_paired": 100}


def _bundle(rmse, rho, ndcg=(0.0, -0.1, 0.1), rec=(0.0, -0.1, 0.1), reg=(0.0, -1.0, 1.0)):
    return {"rmse_diff": _ci(*rmse), "res_spearman_diff": _ci(*rho), "ndcg3_diff": _ci(*ndcg),
            "recall3_diff": _ci(*rec), "regret1_diff": _ci(*reg), "res_ndcg3_diff": _ci(0, -1, 1),
            "regret3_diff": _ci(0, -1, 1)}


def _passing_comps():
    from src.oncotwin.experiments.sprint03_report import ABLATION_NAMES

    good = _bundle((-3.0, -4.0, -2.0), (0.1, 0.05, 0.15), ndcg=(0.02, 0.01, 0.03))
    abl = {f"k{k}_{ps}_M3_full_vs_{a}": good for k in (3, 5, 8) for ps in ("StaticDOpt", "Random50")
           for a in ABLATION_NAMES[1:]}
    panel = {f"k{k}_{p}_vs_Random50": good for k in (1, 2, 3, 5, 8, 12)
             for p in ("StaticDiversity", "StaticDOpt", "AdaptiveOED")}
    panel.update({f"k{k}_AdaptiveOED_vs_StaticDOpt": _bundle((0, -1, 1), (0, -1, 1)) for k in (1, 2, 3, 5, 8, 12)})
    voi = {f"StaticDOpt_k{a}_to_k{b}": _bundle((-3.0 * (b - a), -9, -1), (0, -1, 1))
           for a, b in ((1, 2), (2, 3), (3, 5), (5, 8), (8, 12))}
    return {"ablations": abl, "panel": panel, "voi": voi}


def test_gate2_wave_ablation_requires_ci_not_point_estimate():
    from src.oncotwin.experiments.sprint03_gate import evaluate_gate2

    cv, wave = _passing_comps(), _passing_comps()
    assert evaluate_gate2(cv, wave)["ablation_sanity_passed"]
    # W3+4: nokta tahmini doğru yönde, fakat %95 GA 0'ı içeriyor -> protokol §5.1'e göre başarısız
    wave["ablations"]["k5_StaticDOpt_M3_full_vs_M3_no_omics"] = _bundle((-1.0, -2.5, 0.4), (0.05, -0.01, 0.1))
    res = evaluate_gate2(cv, wave)
    assert not res["ablation_sanity_passed"]
    assert "k5_StaticDOpt_M3_no_omics" in res["ablation_failures"]


# ---------- DEV-OPUS-04/05: tek, vektörleştirilmiş değerlendirici ----------

def _scalar_reference_vectors(Yt, Yp, Bd, min_targets=5):
    """Tanım gereği referans: hasta başına skaler metrik fonksiyonları."""
    n = Yt.shape[0]
    out = {k: np.full(n, np.nan) for k in ("patient_rmse_sq", "patient_res_spearman", "patient_ndcg3",
                                            "patient_res_ndcg3", "patient_recall3", "patient_regret1", "patient_regret3")}
    for i in range(n):
        m = np.isfinite(Yt[i]) & np.isfinite(Yp[i])
        if m.sum() > 0:
            out["patient_rmse_sq"][i] = np.mean((Yt[i, m] - Yp[i, m]) ** 2)
        if m.sum() >= min_targets:
            yt, yp, rt, rp = Yt[i, m], Yp[i, m], (Yt - Bd)[i, m], (Yp - Bd)[i, m]
            out["patient_res_spearman"][i] = safe_spearman(rt, rp)
            out["patient_ndcg3"][i] = ndcg_at_k(yt, yp, 3)
            out["patient_res_ndcg3"][i] = ndcg_at_k(rt, rp, 3)
            out["patient_recall3"][i] = recall_at_k(yt, yp, 3)
            out["patient_regret1"][i] = regret_at_k(yt, yp, 1)
            out["patient_regret3"][i] = regret_at_k(yt, yp, 3)
    return out


def _random_eval_matrices(seed, n=300, d=24):
    rng = np.random.default_rng(seed)
    Yt = rng.normal(150, 50, (n, d))
    Yp = Yt + rng.normal(0, 30, (n, d))
    Yt[rng.random((n, d)) < 0.25] = np.nan
    Yp[:20] = np.round(Yp[:20] / 25.0) * 25.0            # yoğun tahmin bağları
    Yt[20:40] = np.round(Yt[20:40] / 40.0) * 40.0        # yoğun gerçek bağları
    Yp[40:45] = 100.0                                    # sabit tahmin
    Yt[45:50] = np.where(np.isfinite(Yt[45:50]), 90.0, np.nan)  # tüm gerçekler eşit
    Yt[50:55, 4:] = np.nan                               # < 5 hedef
    Yt[55] = np.nan                                      # tamamen boş satır
    Bd = rng.normal(150, 20, d)[None, :].repeat(n, axis=0)
    return Yt, Yp, Bd


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_evaluator_matches_scalar_definitions(seed):
    from src.oncotwin.eval.metrics import evaluate_masked_predictions

    Yt, Yp, Bd = _random_eval_matrices(seed)
    ev = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(len(Yt), bool), list(range(24)))
    ref = _scalar_reference_vectors(Yt, Yp, Bd)
    for key, r in ref.items():
        np.testing.assert_allclose(ev["vectors"][key], r, rtol=1e-12, atol=1e-12, err_msg=key)
    m = np.isfinite(Yt) & np.isfinite(Yp)
    assert np.isclose(ev["obs_rmse"], np.sqrt(np.mean((Yt[m] - Yp[m]) ** 2)))
    assert np.isclose(ev["rmse"], np.sqrt(np.nanmean(ref["patient_rmse_sq"])))
    assert ev["n_eval"] == len(Yt) and ev["n_obs"] == int(m.sum())


def test_evaluator_fold_wise_drug_spearman_and_subject_clustering():
    from src.oncotwin.eval.metrics import evaluate_masked_predictions

    Yt, Yp, Bd = _random_eval_matrices(5, n=120)
    folds = np.arange(120) % 3
    ev = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(120, bool), [0, 1], fold_ids=folds)
    expected = []
    for j in (0, 1):
        vals = []
        for f in range(3):
            mm = (folds == f) & np.isfinite(Yt[:, j]) & np.isfinite(Yp[:, j])
            if mm.sum() >= 10:
                vals.append(safe_spearman(Yt[mm, j], Yp[mm, j]))
        expected.append(np.mean(vals))
    assert np.isclose(ev["per_drug_spearman"], np.mean(expected))

    subj = np.repeat(np.arange(60), 2)
    evc = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(120, bool), list(range(24)), subjects=subj)
    flat = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(120, bool), list(range(24)))
    assert evc["n_eval"] == 60
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        exp = np.nanmean(flat["vectors"]["patient_ndcg3"].reshape(60, 2), axis=1)
    np.testing.assert_allclose(evc["vectors"]["patient_ndcg3"], exp)


def test_average_metric_nodes_matches_manual_rep_averaging():
    from src.oncotwin.eval.metrics import average_metric_nodes, evaluate_masked_predictions

    nodes = []
    for s in range(3):
        Yt, Yp, Bd = _random_eval_matrices(10 + s, n=80)
        nodes.append(evaluate_masked_predictions(Yt, Yp, Bd, np.ones(80, bool), list(range(24))))
    avg = average_metric_nodes(nodes)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        v = np.nanmean(np.stack([n["vectors"]["patient_rmse_sq"] for n in nodes]), axis=0)
        assert np.isclose(avg["rmse"], np.sqrt(np.nanmean(v)))
    assert np.isclose(avg["residual_r2"], np.mean([n["residual_r2"] for n in nodes]))
    assert np.isclose(avg["obs_rmse"], np.mean([n["obs_rmse"] for n in nodes]))
    assert avg["n_eval"] == 80


def test_fast_rankdata_matches_pandas_with_and_without_ties():
    rng = np.random.default_rng(3)
    for arr in (rng.normal(size=200), np.round(rng.normal(size=200), 1)):
        np.testing.assert_allclose(_rankdata_average(arr), pd.Series(arr).rank(method="average").to_numpy())


# ---------- DEV-OPUS-08: simetrik Holm Adaptive-vs-Static yeniden analizi ----------

def test_bootstrap_two_sided_p_value_is_consistent_with_ci():
    rng = np.random.default_rng(1)
    strong = paired_bootstrap_ci(rng.normal(1.0, 1, 200), rng.normal(0.0, 1, 200), n_boot=2000)
    null = paired_bootstrap_ci(rng.normal(0.0, 1, 200), rng.normal(0.0, 1, 200), n_boot=2000)
    assert strong["p_two_sided"] < 0.01 and strong["ci_low"] > 0
    assert null["p_two_sided"] > 0.05
    assert 0.0 < strong["p_two_sided"] <= 1.0  # (sayım+1)/(B+1) ile asla 0 değil


def test_holm_adjust_matches_manual():
    from src.oncotwin.experiments.sprint03_gate import holm_adjust

    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sıralı: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monoton: 0.02,0.03,0.06,0.06
    np.testing.assert_allclose(holm_adjust(p), [0.03, 0.06, 0.06, 0.02])


def _ad_comps(cells):
    """cells: {(k, metric): (diff, p)} -> panel karşılaştırma sözlüğü."""
    key = {"rmse": "rmse_diff", "ndcg3": "ndcg3_diff", "regret1": "regret1_diff"}
    out = {}
    for k in (3, 5, 8):
        b = {}
        for m, jk in key.items():
            diff, p = cells.get((k, m), (0.0, 0.9))
            b[jk] = {"diff": diff, "ci_low": diff - 1, "ci_high": diff + 1, "n_paired": 100, "p_two_sided": p}
        out[f"k{k}_AdaptiveOED_vs_StaticDOpt"] = b
    return {"panel": out}


def test_symmetric_holm_labels():
    from src.oncotwin.experiments.sprint03_gate import adaptive_vs_static_symmetric_holm as rule

    assert rule(_ad_comps({}))["label"] == "INSUFFICIENT_EVIDENCE"
    # tek düzeltilmemiş p=0.02 -> Holm 9 test altında anlamsız (v1 bunu ADAPTIVE_WINS sayardı)
    assert rule(_ad_comps({(5, "ndcg3"): (0.006, 0.02)}))["label"] == "INSUFFICIENT_EVIDENCE"
    assert rule(_ad_comps({(5, "ndcg3"): (0.006, 0.001)}))["label"] == "ADAPTIVE_FAVORED"
    # simetri: statik lehine aynı kanıt aynı güçte
    assert rule(_ad_comps({(5, "rmse"): (+0.8, 0.001)}))["label"] == "STATIC_FAVORED"
    assert rule(_ad_comps({(5, "regret1"): (+2.0, 0.001)}))["label"] == "STATIC_FAVORED"
    mixed = rule(_ad_comps({(5, "ndcg3"): (0.006, 0.001), (3, "rmse"): (+0.9, 0.001)}))
    assert mixed["label"] == "MIXED" and mixed["n_tests"] == 9


# ---------- DEV-OPUS-09: offset-only ve ofset-sabit permütasyon ----------

def _pure_offset_world(seed, n=1500, n_t=12, n_p=30, sig_b=25.0, noise=30.0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, sig_b, n)
    E = b[:, None] + rng.normal(0, noise, (n, n_t + n_p))
    return E, list(range(n_t)), list(range(n_t, n_t + n_p))


def test_offset_only_matches_oracle_gaussian_in_pure_offset_world():
    from src.oncotwin.experiments.sprint03_ablations import OffsetOnlyM3

    E, T, P = _pure_offset_world(0)
    tr, te = slice(0, 1000), slice(1000, 1500)
    model = OffsetOnlyM3().fit(E[tr], E[tr], P, T, k_values=(5,), seed=3)
    probes = sample_probe_sets(E[te], P, k=5, n_repeats=1, seed=4)[0]
    prior = np.zeros_like(E[te])
    pred = model.predict(E[te], prior, probes, T, k=5)
    cov = 25.0 ** 2 * np.ones((42, 42)) + 30.0 ** 2 * np.eye(42)
    oracle = conditional_gaussian_predict_matrix(E[te], cov, probes, np.array(T), lam=1e-9)
    rmse = lambda X: float(np.sqrt(np.mean((E[te][:, T] - X) ** 2)))
    assert abs(rmse(pred[:, T]) - rmse(oracle)) / rmse(oracle) < 0.01
    assert model.lam_by_k_[5] in OffsetOnlyM3.LAMBDA_GRID


def test_offset_fixed_permutation_keeps_offset_and_changes_assignment():
    from src.oncotwin.experiments.sprint03_ablations import permute_offset_fixed_standardized

    rng = np.random.default_rng(2)
    Y = rng.normal(150, 40, (50, 20))
    prior = rng.normal(150, 10, (50, 20))
    scale = rng.uniform(10, 50, 20)
    probes = [np.sort(rng.choice(20, 5, replace=False)) for _ in range(50)]
    Ym = permute_offset_fixed_standardized(Y, prior, probes, scale, seed=7)
    for i, s in enumerate(probes):
        r, rm = Y[i, s] - prior[i, s], Ym[i, s] - prior[i, s]
        z, zm = (r - r.mean()) / scale[s], (rm - r.mean()) / scale[s]
        np.testing.assert_allclose(np.sort(z), np.sort(zm))   # aynı standardize artıklar
        assert not np.allclose(z, zm)                          # özdeş permütasyon yok
    others = np.ones(20, bool)
    others[probes[0]] = False
    assert np.array_equal(Ym[0, others], Y[0, others])        # prob dışı sütunlar değişmez


def test_reanalysis_reads_vector_archive_and_labels_symmetrically():
    from src.oncotwin.experiments.sprint03_reanalysis import reanalyze_adaptive_vs_static

    rng = np.random.default_rng(0)
    n = 300
    vec = {}
    for regime in ("cv", "wave"):
        for k in (3, 5, 8):
            for pol in ("AdaptiveOED", "StaticDOpt"):
                for vk in ("patient_rmse_sq", "patient_ndcg3", "patient_regret1"):
                    base = {"patient_rmse_sq": 2000.0, "patient_ndcg3": 0.85, "patient_regret1": 20.0}[vk]
                    v = base + rng.normal(0, base * 0.05, n)
                    if regime == "cv" and k == 5 and vk == "patient_ndcg3" and pol == "AdaptiveOED":
                        v = v + 0.05  # güçlü adaptif NDCG etkisi yalnız CV k=5
                    vec[f"{regime}/by_k/{k}/{pol}/{vk}"] = v
    res = reanalyze_adaptive_vs_static(vec, n_boot=2000)
    assert res["cv_5fold_3seeds"]["label"] == "ADAPTIVE_FAVORED"
    assert res["wave_ood"]["label"] == "INSUFFICIENT_EVIDENCE"
    assert res["cv_5fold_3seeds"]["n_tests"] == 9


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_evaluator_empty_ranking_slices_return_nan_without_warning():
    """OPUS-022 / DEV-OPUS-18: hiçbir hasta min_targets'ı sağlamıyorsa sıralama metrikleri sessiz NaN, RMSE sonlu."""
    from src.oncotwin.eval.metrics import evaluate_masked_predictions

    Yt, Yp, Bd = _random_eval_matrices(3, n=60)
    ev = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(60, bool), [0, 1])  # 2 hedef < min 5
    assert np.isfinite(ev["rmse"]) and ev["n_obs"] > 0
    for key in ("ndcg_at_3", "recall_at_3", "regret_at_1", "regret_at_3", "within_patient_residual_spearman"):
        assert np.isnan(ev[key]), key
    evc = evaluate_masked_predictions(Yt, Yp, Bd, np.ones(60, bool), [0, 1], subjects=np.repeat(np.arange(30), 2))
    assert evc["n_eval"] == 30 and np.isnan(evc["regret_at_1"])


def test_sprint04_report_marks_k0_m3_post_scale_as_constant_score():
    """DEV-OPUS-20 / OPUS-023: k=0 kollarında m3_post_scale sabit (bilgisiz) skor olarak etiketlenir."""
    from src.oncotwin.experiments.sprint04_report import _render_abstention_table

    ab = {"tau_locked": 1.0, "threshold_eval": {"status": "ok"}, "risk_coverage_regret1": {}}
    arms = {n: {"abstention": {"omics_pca_dist": ab, "m3_post_scale": ab}} for n in ("M1_k0", "StaticDOpt_k5")}
    rows = {r.split("|")[1].strip() + r.split("|")[2].strip(): r for r in _render_abstention_table(arms)[2:]}
    assert "sabit skor" in rows["`M1_k0``m3_post_scale` (sabit skor)"]
    assert "sabit skor" not in rows["`StaticDOpt_k5``m3_post_scale`"]
    assert "sabit skor" not in rows["`M1_k0``omics_pca_dist`"]
