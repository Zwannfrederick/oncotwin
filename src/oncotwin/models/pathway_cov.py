"""M4: Yolak / hedef-ailesi yapılı kovaryans büzüştürme ve derece-korumalı permütasyon kontrolü."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.oncotwin.eval.splits import sample_probe_sets
from src.oncotwin.models.baselines import (
    OmicsRidgeM1,
    _build_inner_cv,
    fit_drug_means,
)


def load_beataml_drug_family_incidence(
    drug_names: list[str],
    raw_dir: Path = Path("data/raw/beataml2"),
    include_gene_sheet: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Beat AML drug_families tablosundan (ilaç x aile) ikili üyelik matrisi üretir."""
    xlsx_path = raw_dir / "beataml_drug_families.xlsx"
    dfam = pd.read_excel(xlsx_path, sheet_name="drug_family")
    dfam = dfam[dfam["inhibitor"].isin(drug_names)].dropna(subset=["family"])
    pairs: set[tuple[str, str]] = {
        (str(r["inhibitor"]), f"FAM::{str(r['family'])}") for _, r in dfam.iterrows()
    }
    if include_gene_sheet:
        dgene = pd.read_excel(xlsx_path, sheet_name="drug_gene")
        dgene = dgene[dgene["inhibitor"].isin(drug_names)].dropna(subset=["Symbol"])
        for _, r in dgene.iterrows():
            pairs.add((str(r["inhibitor"]), f"GENE::{str(r['Symbol'])}"))

    features = sorted({feat for _, feat in pairs})
    drug_to_row = {name: i for i, name in enumerate(drug_names)}
    feat_to_col = {feat: j for j, feat in enumerate(features)}
    B = np.zeros((len(drug_names), len(features)), dtype=np.int8)
    for drug, feat in pairs:
        B[drug_to_row[drug], feat_to_col[feat]] = 1
    return B, features


def build_cosine_similarity_gram(B: np.ndarray) -> np.ndarray:
    """İkili üyelik matrisinden birim diyagonalli, PSD kosinüs benzerlik matrisi G_unit üretir."""
    M = np.asarray(B, dtype=np.float64)
    n = M.shape[0]
    if M.ndim != 2:
        raise ValueError("Üyelik matrisi 2-boyutlu olmalıdır.")
    if M.shape[1] == 0:
        return np.eye(n, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M_norm = np.divide(M, norms, out=np.zeros_like(M), where=norms > 0.0)
    G = M_norm @ M_norm.T
    G = np.clip(0.5 * (G + G.T), 0.0, 1.0)
    # G PSD + diag(1 - diag(G)) PSD -> toplam her zaman PSD ve diyagonali 1.0
    diag_fix = np.maximum(1.0 - np.diag(G), 0.0)
    G_unit = G + np.diag(diag_fix)
    np.fill_diagonal(G_unit, 1.0)
    return G_unit


def permute_bipartite_incidence_preserving_degrees(
    B: np.ndarray,
    seed: int,
    n_swap_mult: int = 30,
) -> np.ndarray:
    """Her ilacın aile derecesini ve her ailenin boyutunu birebir koruyarak ikili grafı karıştırır."""
    B_orig = (np.asarray(B) > 0).astype(np.int8)
    B_sw = B_orig.copy()
    edges = list(zip(*np.nonzero(B_sw)))
    n_e = len(edges)
    if n_e < 2:
        return B_sw

    rng = np.random.default_rng(seed)
    n_attempts = max(1, int(n_swap_mult) * n_e)
    idx_pairs = rng.integers(0, n_e, size=(n_attempts, 2))
    for i1, i2 in idx_pairs:
        if i1 == i2:
            continue
        u1, f1 = edges[i1]
        u2, f2 = edges[i2]
        if u1 == u2 or f1 == f2:
            continue
        if B_sw[u1, f2] == 0 and B_sw[u2, f1] == 0:
            B_sw[u1, f1] = 0
            B_sw[u2, f2] = 0
            B_sw[u1, f2] = 1
            B_sw[u2, f1] = 1
            edges[i1] = (u1, f2)
            edges[i2] = (u2, f1)

    assert np.array_equal(B_sw.sum(axis=0), B_orig.sum(axis=0)), "Aile boyutları korunamadı."
    assert np.array_equal(B_sw.sum(axis=1), B_orig.sum(axis=1)), "İlaç dereceleri korunamadı."
    return B_sw


def _base_empirical_psd_cov(
    R_train: np.ndarray,
    min_pair: int = 15,
    min_eig: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Maskeli kalıntılardan PSD-projekte ampirik kovaryans ve marjinal varyans diyagonalini döndürür."""
    mask = np.isfinite(R_train).astype(np.float64)
    R_zero = np.where(mask > 0.5, R_train, 0.0)
    pair_counts = mask.T @ mask
    cross_prod = R_zero.T @ R_zero

    cov = np.divide(
        cross_prod,
        pair_counts,
        out=np.zeros_like(cross_prod),
        where=pair_counts >= min_pair,
    )
    diag = np.diag(cov).copy()
    valid_diag = diag[diag > 0]
    fallback_var = float(np.median(valid_diag)) if len(valid_diag) > 0 else 1.0
    diag = np.where(diag > min_eig, diag, fallback_var)
    np.fill_diagonal(cov, diag)

    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_eig)
    cov_psd = (eigvecs * eigvals[None, :]) @ eigvecs.T
    cov_psd = 0.5 * (cov_psd + cov_psd.T)
    diag_psd = np.maximum(np.diag(cov_psd), min_eig)
    return cov_psd, diag_psd


def combine_psd_with_pathway_prior(
    cov_psd: np.ndarray,
    diag_psd: np.ndarray,
    G_unit: np.ndarray,
    alpha: float,
    rho: float,
) -> np.ndarray:
    """Ampirik PSD kovaryansı T_bio(rho) = D^(1/2)[(1-rho)I + rho G_unit]D^(1/2) hedefine büzüştürür."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha [0,1] aralığında olmalı: {alpha}")
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho [0,1) aralığında olmalı: {rho}")
    if rho == 0.0:
        return (1.0 - alpha) * cov_psd + alpha * np.diag(diag_psd)
    sd = np.sqrt(diag_psd)
    n = len(diag_psd)
    C_bio = (1.0 - rho) * np.eye(n, dtype=np.float64) + rho * G_unit
    T_bio = (sd[:, None] * C_bio) * sd[None, :]
    out = (1.0 - alpha) * cov_psd + alpha * T_bio
    return 0.5 * (out + out.T)


def estimate_pathway_structured_cov(
    R_train: np.ndarray,
    G_unit: np.ndarray,
    alpha: float,
    rho: float,
    min_pair: int = 15,
    min_eig: float = 1e-4,
) -> np.ndarray:
    """rho=0 iken estimate_psd_shrunk_cov ile birebir aynı olan yolak-yapılı kovaryans tahmincisi."""
    cov_psd, diag_psd = _base_empirical_psd_cov(R_train, min_pair=min_pair, min_eig=min_eig)
    return combine_psd_with_pathway_prior(cov_psd, diag_psd, G_unit, alpha=alpha, rho=rho)


def conditional_gaussian_predict_fast(
    R_obs_eval: np.ndarray,
    cov_psd: np.ndarray,
    probe_sets: list[np.ndarray],
    target_local_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Sonlu prob sayısına göre gruplanmış toplu LAPACK solve (tüm prob boyutlarında vektörel)."""
    n_eval = R_obs_eval.shape[0]
    n_t = len(target_local_idx)
    out = np.zeros((n_eval, n_t), dtype=np.float64)
    if n_eval == 0:
        return out
    groups: dict[int, tuple[list[int], list[np.ndarray], list[np.ndarray]]] = {}
    for i, s in enumerate(probe_sets):
        if len(s) == 0:
            continue
        r_s = R_obs_eval[i, s]
        fin = np.isfinite(r_s)
        m = int(np.sum(fin))
        if m == 0:
            continue
        s_v = s[fin] if m < len(s) else s
        r_v = r_s[fin] if m < len(s) else r_s
        if m not in groups:
            groups[m] = ([], [], [])
        rows_m, p_m, r_m = groups[m]
        rows_m.append(i)
        p_m.append(s_v)
        r_m.append(r_v)
    for m, (rows_m, p_m, r_m) in groups.items():
        P = np.asarray(p_m, dtype=np.int64)
        R_s = np.asarray(r_m, dtype=np.float64)
        C_ss = cov_psd[P[:, :, None], P[:, None, :]].copy()
        diag_idx = np.arange(m)
        C_ss[:, diag_idx, diag_idx] += lam
        C_ts = cov_psd[target_local_idx[None, :, None], P[:, None, :]]
        sol = np.linalg.solve(C_ss, R_s[:, :, None])
        out[np.asarray(rows_m, dtype=np.int64)] = (C_ts @ sol)[:, :, 0]
    return out


def eval_inner_cv_mse_grid(
    ctx: dict[str, object],
    covs_by_fold: list[np.ndarray],
    k: int,
    lam_grid: tuple[float, ...],
) -> list[float]:
    """Ön-paketlenmiş iç-CV grupları üzerinden tüm lam değerlerinin MSE'sini tek toplu solve ile hesaplar."""
    packed_k = ctx["packed_by_k"][k]  # type: ignore[index]
    t_local: np.ndarray = ctx["t_local"]  # type: ignore[assignment]
    total_obs: int = packed_k["total_obs"]
    n_lam = len(lam_grid)
    if total_obs == 0:
        return [float("inf")] * n_lam
    lam_arr = np.asarray(lam_grid, dtype=np.float64)[:, None, None, None]
    sse = np.full(n_lam, float(packed_k["const_sq_err"]), dtype=np.float64)
    for f_idx, batches in enumerate(packed_k["fold_batches"]):
        cov_f = covs_by_fold[f_idx]
        for m, P_mat, R_s_mat, R_true_zeroed, t_mask_mat in batches:
            C_ss_base = cov_f[P_mat[:, :, None], P_mat[:, None, :]]
            C_ts = cov_f[t_local[None, :, None], P_mat[:, None, :]]
            eye_m = np.eye(m, dtype=np.float64)[None, None, :, :]
            C_ss_all = C_ss_base[None, :, :, :] + lam_arr * eye_m
            rhs = np.broadcast_to(R_s_mat[None, :, :, None], (n_lam, P_mat.shape[0], m, 1))
            sol = np.linalg.solve(C_ss_all, rhs)
            deltas = (C_ts[None, :, :, :] @ sol)[:, :, :, 0]
            diff = R_true_zeroed[None, :, :] - deltas
            sse += np.sum(np.where(t_mask_mat[None, :, :], diff * diff, 0.0), axis=(1, 2))
    return [float(v / total_obs) for v in sse]


def precompute_inner_cv_context(
    Y_train: np.ndarray,
    train_subjects: np.ndarray,
    probe_pool_idx: list[int],
    target_idx: list[int],
    k_values: tuple[int, ...] = (2, 5, 8),
    prior_train: np.ndarray | None = None,
    inner_seed: int = 17,
) -> dict[str, object]:
    """M3/M4 ve permütasyon kontrollerinin paylaşacağı iç-CV ampirik PSD kovaryanslarını ön-hesaplar."""
    mu, b_d = fit_drug_means(Y_train)
    active = sorted(set(target_idx) | set(probe_pool_idx))
    active_idx = np.asarray(active, dtype=np.int64)
    g2l = {g: l for l, g in enumerate(active)}
    t_local = np.asarray([g2l[g] for g in target_idx], dtype=np.int64)
    t_global = np.asarray(target_idx, dtype=np.int64)

    folds = _build_inner_cv(train_subjects, n_splits=4, seed=inner_seed)
    fold_base: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    fold_probes_by_k: dict[int, list[list[list[np.ndarray]]]] = {k: [] for k in k_values}
    packed_by_k: dict[int, dict[str, object]] = {
        k: {"fold_batches": [], "const_sq_err": 0.0, "total_obs": 0} for k in k_values
    }

    for f_idx, (tr_pos, va_pos) in enumerate(folds):
        if prior_train is None:
            mu_f, b_f = fit_drug_means(Y_train[tr_pos])
            base_f = mu_f + b_f
            R_tr_f = Y_train[tr_pos][:, active_idx] - base_f[active_idx][None, :]
            prior_va_act = np.broadcast_to(base_f[active_idx][None, :], (len(va_pos), len(active)))
        else:
            R_tr_f = (Y_train[tr_pos] - prior_train[tr_pos])[:, active_idx]
            prior_va_act = prior_train[va_pos][:, active_idx]
        R_va_act = Y_train[va_pos][:, active_idx] - prior_va_act
        cov_psd_f, diag_psd_f = _base_empirical_psd_cov(R_tr_f)
        fold_base.append((cov_psd_f, diag_psd_f, R_va_act, prior_va_act))

        Yt_va = Y_train[va_pos][:, t_global]
        t_mask_va = np.isfinite(Yt_va)
        R_true_t_va = np.where(t_mask_va, Yt_va - prior_va_act[:, t_local], 0.0)
        row_has_target = np.any(t_mask_va, axis=1)

        for k in k_values:
            probes_raw = sample_probe_sets(
                Y_train[va_pos], probe_pool_idx, k=k, n_repeats=2, seed=inner_seed + k + f_idx
            )
            probes_loc_reps = [
                [np.asarray([g2l[g] for g in s], dtype=np.int64) for s in rep] for rep in probes_raw
            ]
            fold_probes_by_k[k].append(probes_loc_reps)

            groups: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[int]]] = {}
            pk = packed_by_k[k]
            for rep_probes in probes_loc_reps:
                for i, s in enumerate(rep_probes):
                    if not row_has_target[i]:
                        continue
                    pk["total_obs"] = int(pk["total_obs"]) + int(np.sum(t_mask_va[i]))  # type: ignore[arg-type]
                    if len(s) == 0:
                        pk["const_sq_err"] = float(pk["const_sq_err"]) + float(np.sum(R_true_t_va[i] ** 2))  # type: ignore[arg-type]
                        continue
                    r_s = R_va_act[i, s]
                    fin = np.isfinite(r_s)
                    m = int(np.sum(fin))
                    if m == 0:
                        pk["const_sq_err"] = float(pk["const_sq_err"]) + float(np.sum(R_true_t_va[i] ** 2))  # type: ignore[arg-type]
                        continue
                    if m not in groups:
                        groups[m] = ([], [], [])
                    p_l, r_l, row_l = groups[m]
                    p_l.append(s[fin] if m < len(s) else s)
                    r_l.append(r_s[fin] if m < len(s) else r_s)
                    row_l.append(i)
            batches_f = []
            for m, (p_l, r_l, row_l) in sorted(groups.items()):
                r_idx = np.asarray(row_l, dtype=np.int64)
                batches_f.append(
                    (
                        m,
                        np.asarray(p_l, dtype=np.int64),
                        np.asarray(r_l, dtype=np.float64),
                        R_true_t_va[r_idx],
                        t_mask_va[r_idx],
                    )
                )
            pk["fold_batches"].append(batches_f)  # type: ignore[union-attr]

    R_full = (
        (Y_train - (mu + b_d)[None, :])[:, active_idx]
        if prior_train is None
        else (Y_train - prior_train)[:, active_idx]
    )
    cov_psd_full, diag_psd_full = _base_empirical_psd_cov(R_full)
    return {
        "mu": mu,
        "b_d": b_d,
        "active_idx": active_idx,
        "global_to_local": g2l,
        "t_local": t_local,
        "t_global": t_global,
        "folds": folds,
        "fold_base": fold_base,
        "fold_probes_by_k": fold_probes_by_k,
        "packed_by_k": packed_by_k,
        "cov_psd_full": cov_psd_full,
        "diag_psd_full": diag_psd_full,
        "Y_train": Y_train,
    }


class PathwayCovarianceFewShotM4:
    """Yolak / hedef-ailesi büzüştürme hedefli koşullu Gaussian few-shot kovaryans modeli."""

    def __init__(
        self,
        alpha_grid: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
        rho_grid: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50, 0.70),
        lam_grid: tuple[float, ...] = (50.0, 200.0, 500.0, 1500.0, 5000.0),
    ) -> None:
        self.alpha_grid = alpha_grid
        self.rho_grid = rho_grid
        self.lam_grid = lam_grid
        self.mu_: float = 0.0
        self.b_d_: np.ndarray = np.array([])
        self.active_idx_: np.ndarray = np.array([], dtype=np.int64)
        self.global_to_local_: dict[int, int] = {}
        self.best_params_by_k_: dict[int, tuple[float, float, float]] = {}
        self.cov_by_k_: dict[int, np.ndarray] = {}

    def fit(
        self,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        G_unit_full: np.ndarray,
        k_values: tuple[int, ...] = (2, 5, 8),
        prior_train: np.ndarray | None = None,
        inner_seed: int = 17,
        precomputed_ctx: dict[str, object] | None = None,
    ) -> PathwayCovarianceFewShotM4:
        ctx = (
            precomputed_ctx
            if precomputed_ctx is not None
            else precompute_inner_cv_context(
                Y_train=Y_train,
                train_subjects=train_subjects,
                probe_pool_idx=probe_pool_idx,
                target_idx=target_idx,
                k_values=k_values,
                prior_train=prior_train,
                inner_seed=inner_seed,
            )
        )
        self.mu_ = float(ctx["mu"])  # type: ignore[arg-type]
        self.b_d_ = ctx["b_d"]  # type: ignore[assignment]
        self.active_idx_ = ctx["active_idx"]  # type: ignore[assignment]
        self.global_to_local_ = ctx["global_to_local"]  # type: ignore[assignment]
        fold_base = ctx["fold_base"]  # type: ignore[assignment]
        G_act = np.asarray(G_unit_full, dtype=np.float64)[np.ix_(self.active_idx_, self.active_idx_)]

        combo_covs: dict[tuple[float, float], list[np.ndarray]] = {
            (alpha, rho): [
                combine_psd_with_pathway_prior(c_psd, d_psd, G_act, alpha=alpha, rho=rho)
                for c_psd, d_psd, _, _ in fold_base  # type: ignore[misc]
            ]
            for alpha in self.alpha_grid
            for rho in self.rho_grid
        }

        for k in k_values:
            best_triple = (self.alpha_grid[0], self.rho_grid[0], self.lam_grid[0])
            best_mse = float("inf")
            for alpha in self.alpha_grid:
                for rho in self.rho_grid:
                    mses = eval_inner_cv_mse_grid(ctx, combo_covs[(alpha, rho)], k, self.lam_grid)
                    for lam, mse in zip(self.lam_grid, mses):
                        if mse < best_mse:
                            best_mse = mse
                            best_triple = (alpha, rho, lam)
            self.best_params_by_k_[k] = best_triple

        cov_psd_full: np.ndarray = ctx["cov_psd_full"]  # type: ignore[assignment]
        diag_psd_full: np.ndarray = ctx["diag_psd_full"]  # type: ignore[assignment]
        self.cov_by_k_ = {
            k: combine_psd_with_pathway_prior(cov_psd_full, diag_psd_full, G_act, alpha=a_k, rho=r_k)
            for k, (a_k, r_k, _) in self.best_params_by_k_.items()
        }
        return self

    def predict(
        self,
        Y_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        target_idx: list[int],
        k: int,
        prior_eval: np.ndarray | None = None,
    ) -> np.ndarray:
        base_full = (
            np.broadcast_to((self.mu_ + self.b_d_)[None, :], Y_eval.shape).copy()
            if prior_eval is None
            else prior_eval.copy()
        )
        if k <= 0:
            return base_full
        _, _, lam_k = self.best_params_by_k_[k]
        cov_k = self.cov_by_k_[k]
        R_eval_act = (Y_eval - base_full)[:, self.active_idx_]
        t_local = np.asarray([self.global_to_local_[g] for g in target_idx], dtype=np.int64)
        probes_local = [
            np.asarray([self.global_to_local_[g] for g in s if g in self.global_to_local_], dtype=np.int64)
            for s in probe_sets
        ]
        delta = conditional_gaussian_predict_fast(R_eval_act, cov_k, probes_local, t_local, lam=lam_k)
        out = base_full.copy()
        out[:, np.asarray(target_idx, dtype=np.int64)] += delta
        return out


class PathwayHybridFewShotM4:
    """M1 omik öncülü + yolak/hedef-ailesi yapılı M4 koşullu Gaussian few-shot modeli."""

    def __init__(
        self,
        alpha_grid: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
        rho_grid: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50, 0.70),
        lam_grid: tuple[float, ...] = (50.0, 200.0, 500.0, 1500.0, 5000.0),
    ) -> None:
        self._cov_model = PathwayCovarianceFewShotM4(alpha_grid=alpha_grid, rho_grid=rho_grid, lam_grid=lam_grid)
        self.omics_model: OmicsRidgeM1 | None = None

    def fit(
        self,
        omics_model: OmicsRidgeM1,
        Y_oof_train: np.ndarray,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        G_unit_full: np.ndarray,
        k_values: tuple[int, ...] = (2, 5, 8),
        inner_seed: int = 17,
        precomputed_ctx: dict[str, object] | None = None,
    ) -> PathwayHybridFewShotM4:
        self.omics_model = omics_model
        self._cov_model.fit(
            Y_train=Y_train,
            train_subjects=train_subjects,
            probe_pool_idx=probe_pool_idx,
            target_idx=target_idx,
            G_unit_full=G_unit_full,
            k_values=k_values,
            prior_train=Y_oof_train,
            inner_seed=inner_seed,
            precomputed_ctx=precomputed_ctx,
        )
        return self

    def predict(
        self,
        X_rna_te: np.ndarray,
        X_mut_te: np.ndarray,
        Y_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        target_idx: list[int],
        k: int,
        prior_eval: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.omics_model is None:
            raise RuntimeError("PathwayHybridFewShotM4 fit edilmeden predict çağrılamaz.")
        if prior_eval is None:
            prior_eval = self.omics_model.predict(X_rna_te, X_mut_te, n_drugs=Y_eval.shape[1])
        if k <= 0:
            return prior_eval
        return self._cov_model.predict(
            Y_eval=Y_eval, probe_sets=probe_sets, target_idx=target_idx, k=k, prior_eval=prior_eval
        )
