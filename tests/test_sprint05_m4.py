"""Sprint 5 Kulvar A (M4 yolak-yapılı kovaryans & derece-korumalı permütasyon) testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.oncotwin.models.baselines import (
    CovarianceFewShotM2a,
    conditional_gaussian_predict_matrix,
    estimate_psd_shrunk_cov,
)
from src.oncotwin.models.pathway_cov import (
    PathwayCovarianceFewShotM4,
    build_cosine_similarity_gram,
    combine_psd_with_pathway_prior,
    conditional_gaussian_predict_fast,
    estimate_pathway_structured_cov,
    eval_inner_cv_mse_grid,
    permute_bipartite_incidence_preserving_degrees,
    precompute_inner_cv_context,
)


def test_m4_rho_zero_matches_m3_bit_for_bit() -> None:
    rng = np.random.default_rng(42)
    R = rng.normal(0.0, 25.0, size=(60, 14))
    R[rng.uniform(size=R.shape) < 0.15] = np.nan
    B = (rng.uniform(size=(14, 5)) < 0.35).astype(np.int8)
    G = build_cosine_similarity_gram(B)
    for alpha in (0.05, 0.25, 0.50):
        c_m3 = estimate_psd_shrunk_cov(R, alpha=alpha)
        c_m4 = estimate_pathway_structured_cov(R, G, alpha=alpha, rho=0.0)
        assert np.allclose(c_m3, c_m4, atol=1e-13, rtol=1e-13)


def test_gram_and_m4_cov_are_symmetric_pd_and_batched_solve_matches_scalar() -> None:
    rng = np.random.default_rng(101)
    B = np.zeros((16, 6), dtype=np.int8)
    B[:12] = (rng.uniform(size=(12, 6)) < 0.4).astype(np.int8)  # son 4 ilaç izole (anotasyonsuz)
    G = build_cosine_similarity_gram(B)
    assert np.allclose(G, G.T, atol=1e-15)
    assert np.allclose(np.diag(G), 1.0, atol=1e-15)
    assert float(np.min(np.linalg.eigvalsh(G))) >= -1e-12
    # İzole ilaçların dış-diyagonal benzerliği tam 0
    assert np.allclose(G[12:, :12], 0.0)

    R = rng.normal(0.0, 20.0, size=(50, 16))
    cov_m4 = estimate_pathway_structured_cov(R, G, alpha=0.30, rho=0.60)
    assert np.allclose(cov_m4, cov_m4.T, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(cov_m4))) > 0.0

    # Vektörel hızlı solve == referans skaler solve
    probes = [np.asarray([6, 7, 8, 9, 10], dtype=np.int64) for _ in range(12)]
    t_loc = np.arange(6, dtype=np.int64)
    d_ref = conditional_gaussian_predict_matrix(R[:12], cov_m4, probes, t_loc, lam=200.0)
    d_fast = conditional_gaussian_predict_fast(R[:12], cov_m4, probes, t_loc, lam=200.0)
    assert np.allclose(d_ref, d_fast, atol=1e-12)


def test_bipartite_permutation_preserves_exact_drug_degrees_and_family_sizes() -> None:
    rng = np.random.default_rng(7)
    B = (rng.uniform(size=(40, 18)) < 0.22).astype(np.int8)
    B[35:] = 0  # izole düğümler
    B_p1 = permute_bipartite_incidence_preserving_degrees(B, seed=5001, n_swap_mult=30)
    B_p2 = permute_bipartite_incidence_preserving_degrees(B, seed=5002, n_swap_mult=30)
    assert np.array_equal(B_p1.sum(axis=0), B.sum(axis=0))
    assert np.array_equal(B_p1.sum(axis=1), B.sum(axis=1))
    assert np.all(B_p1[35:] == 0)
    assert not np.array_equal(B_p1, B)
    assert not np.array_equal(B_p1, B_p2)


def test_m4_beats_m3_and_permuted_null_in_small_sample_pathway_world() -> None:
    """Az örnekli sentetik yolak-blok dünyasında M4, M3'ü ve derece-korumalı null'ı yenmelidir."""
    rng = np.random.default_rng(2026)
    n_fam = 4
    drugs_per_fam = 4  # 16 ilaç: her 4'lü blok aynı yolak (2 hedef + 2 prob)
    n_drugs = n_fam * drugs_per_fam
    B = np.zeros((n_drugs, n_fam), dtype=np.int8)
    for f in range(n_fam):
        B[f * drugs_per_fam : (f + 1) * drugs_per_fam, f] = 1
    G_true = build_cosine_similarity_gram(B)
    G_perm = build_cosine_similarity_gram(permute_bipartite_incidence_preserving_degrees(B, seed=99))

    # Gerçek kovaryans: güçlü yolak içi korelasyon (0.85), küçük örneklemde gürültülü ampirik tahmin
    true_cov = 400.0 * (0.15 * np.eye(n_drugs) + 0.85 * G_true)
    L = np.linalg.cholesky(true_cov)
    n_tr, n_te = 24, 180
    Y_all = 150.0 + rng.normal(size=(n_tr + n_te, n_drugs)) @ L.T
    Y_tr, Y_te = Y_all[:n_tr], Y_all[n_tr:]
    subj_tr = np.arange(n_tr, dtype=np.int64)
    target_idx = [f * drugs_per_fam + j for f in range(n_fam) for j in (0, 1)]
    probe_pool = [f * drugs_per_fam + j for f in range(n_fam) for j in (2, 3)]
    # Sabit 4-prob paneli (her yolaktan 1 prob)
    fixed_probes = [np.asarray([f * drugs_per_fam + 2 for f in range(n_fam)], dtype=np.int64) for _ in range(n_te)]

    m3 = CovarianceFewShotM2a().fit(Y_tr, subj_tr, probe_pool, target_idx, k_values=(4,), inner_seed=17)
    m4 = PathwayCovarianceFewShotM4(rho_grid=(0.0, 0.3, 0.6, 0.85)).fit(
        Y_tr, subj_tr, probe_pool, target_idx, G_true, k_values=(4,), inner_seed=17
    )
    m4_null = PathwayCovarianceFewShotM4(rho_grid=(0.3, 0.6, 0.85)).fit(
        Y_tr, subj_tr, probe_pool, target_idx, G_perm, k_values=(4,), inner_seed=17
    )

    p_m3 = m3.predict(Y_te, fixed_probes, target_idx, k=4)[:, target_idx]
    p_m4 = m4.predict(Y_te, fixed_probes, target_idx, k=4)[:, target_idx]
    p_null = m4_null.predict(Y_te, fixed_probes, target_idx, k=4)[:, target_idx]
    Yt = Y_te[:, target_idx]
    rmse_m3 = float(np.sqrt(np.mean((Yt - p_m3) ** 2)))
    rmse_m4 = float(np.sqrt(np.mean((Yt - p_m4) ** 2)))
    rmse_null = float(np.sqrt(np.mean((Yt - p_null) ** 2)))

    assert m4.best_params_by_k_[4][1] > 0.0
    assert rmse_m4 < rmse_m3 - 0.5
    assert rmse_m4 < rmse_null - 0.3

    # Gizli hedef AUC değerlerini değiştirince M4 tahmini değişmemeli
    Y_te_mut = Y_te.copy()
    Y_te_mut[:, target_idx] += 1000.0
    p_m4_mut = m4.predict(Y_te_mut, fixed_probes, target_idx, k=4)[:, target_idx]
    assert np.allclose(p_m4, p_m4_mut, atol=1e-13)


def test_fast_predict_and_inner_cv_grid_match_scalar_with_mixed_lengths_and_nans() -> None:
    """Değişken prob sayıları ve NaN'lar altında toplu solve ve iç-CV ızgarası referansla birebir eşleşir."""
    rng = np.random.default_rng(123)
    Y = rng.normal(140.0, 28.0, size=(36, 14))
    Y[rng.uniform(size=Y.shape) < 0.20] = np.nan
    Y[0, 6:] = np.nan  # 0 sonlu prob satırı
    Y[1, :6] = np.nan  # 0 sonlu hedef satırı
    subj = np.arange(36, dtype=np.int64)
    target_idx = list(range(6))
    probe_pool = list(range(6, 14))

    ctx = precompute_inner_cv_context(Y, subj, probe_pool, target_idx, k_values=(2, 5), inner_seed=31)
    G_act = np.eye(len(ctx["active_idx"]), dtype=np.float64)
    covs_f = [combine_psd_with_pathway_prior(cp, dp, G_act, alpha=0.15, rho=0.0) for cp, dp, _, _ in ctx["fold_base"]]  # type: ignore[misc]
    lam_grid = (50.0, 500.0, 2000.0)

    # Karışık prob boyutlu conditional_gaussian_predict_fast == skaler referans
    mixed_probes = [
        np.asarray([], dtype=np.int64),
        np.asarray([6, 7], dtype=np.int64),
        np.asarray([6, 8, 9, 10], dtype=np.int64),
        np.asarray([7, 8, 9, 10, 11], dtype=np.int64),
    ]
    R_sub = ctx["fold_base"][0][2][:4]  # type: ignore[index]
    d_fast = conditional_gaussian_predict_fast(R_sub, covs_f[0], mixed_probes, ctx["t_local"], lam=250.0)  # type: ignore[arg-type]
    d_ref = conditional_gaussian_predict_matrix(R_sub, covs_f[0], mixed_probes, ctx["t_local"], lam=250.0)  # type: ignore[arg-type]
    assert np.allclose(d_fast, d_ref, atol=1e-12)

    # eval_inner_cv_mse_grid == açık skaler iç-CV döngüsü
    for k in (2, 5):
        fast_mses = eval_inner_cv_mse_grid(ctx, covs_f, k, lam_grid)
        for lam, f_mse in zip(lam_grid, fast_mses):
            sq_errs = []
            for f_idx, (_, va_pos) in enumerate(ctx["folds"]):  # type: ignore[arg-type]
                _, _, R_va_act, prior_va_act = ctx["fold_base"][f_idx]  # type: ignore[index]
                Yt = Y[va_pos][:, ctx["t_global"]]  # type: ignore[index]
                tm = np.isfinite(Yt)
                if not np.any(tm):
                    continue
                for rep_p in ctx["fold_probes_by_k"][k][f_idx]:  # type: ignore[index]
                    delta = conditional_gaussian_predict_matrix(R_va_act, covs_f[f_idx], rep_p, ctx["t_local"], lam=lam)  # type: ignore[arg-type]
                    sq_errs.append((Yt[tm] - (prior_va_act[:, ctx["t_local"]] + delta)[tm]) ** 2)  # type: ignore[index]
            ref_mse = float(np.mean(np.concatenate(sq_errs)))
            assert abs(f_mse - ref_mse) <= 1e-10


def test_sprint05_m4_results_consistency() -> None:
    """Sprint 5 Kulvar A JSON/NPZ/MD çıktıları, Kapı 3 kuralı ve Sprint 4 v2 eşitliğini doğrular."""
    res_path = Path("data/processed/sprint05_results.json")
    npz_path = Path("data/processed/sprint05_vectors.npz")
    md_path = Path("docs/sprints/sprint-05-m4-shortlist-fimm.md")
    s4_path = Path("data/processed/sprint04_v2_results.json")
    assert res_path.exists() and npz_path.exists() and md_path.exists()

    s5 = json.loads(res_path.read_text(encoding="utf-8"))
    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    ta = s5["track_a_m4"]
    assert ta["drug_family_graph"]["n_perm_draws"] == 20
    assert set(ta["regimes"].keys()) == {"id_cv_5fold_3seed", "id_60_20_20", "wave_ood_s3", "wave_ood_s4"}

    # M3 (rho=0) sonuçları Sprint 4 v2 ile aynı bölmede birebir eşit
    m3_id_s5 = ta["regimes"]["id_60_20_20"]["experiment_a1_fixed_panel"]["k5"]["models"]["M3"]["rmse"]
    m3_id_s4 = s4["regimes"]["id_60_20_20"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["test_metrics"]["rmse"]
    assert abs(m3_id_s5 - m3_id_s4) <= 1e-12

    m3_w_s5 = ta["regimes"]["wave_ood_s4"]["experiment_a1_fixed_panel"]["k5"]["models"]["M3"]["rmse"]
    m3_w_s4 = s4["regimes"]["wave_ood"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["test_metrics"]["rmse"]
    assert abs(m3_w_s5 - m3_w_s4) <= 1e-12

    # Kapı 3 karar tutarlılığı
    g3 = ta["gate3"]
    all_pass = all(bool(v) for v in g3["conditions"].values())
    assert g3["decision"] == ("PASS" if all_pass else "NO-GO (v2 ODE ERTELENİR)")
    assert g3["decision"] in md_path.read_text(encoding="utf-8")

    # Vektör NPZ doğrulaması
    z = np.load(npz_path)
    assert "track_a/id_cv_5fold_3seed/A1/k5/M4/patient_rmse_sq" in z.files
    assert "track_a/id_cv_5fold_3seed/A1/k5/M4_perm_null/patient_rmse_sq" in z.files
    assert len(z["track_a/id_cv_5fold_3seed/A1/k5/M4/patient_rmse_sq"]) == 427
    assert int(np.isfinite(z["track_a/id_cv_5fold_3seed/A1/k5/M4/patient_rmse_sq"]).sum()) == 427
