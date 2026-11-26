"""Protokol uyumlu, yalnızca eğitim katından öğrenen baseline ve few-shot modelleri."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.oncotwin.eval.splits import sample_probe_sets


def fit_drug_means(Y_train: np.ndarray) -> tuple[float, np.ndarray]:
    """Yalnızca Y_train içinden global ortalama ve ilaç bazlı sapmaları hesaplar."""
    mask = np.isfinite(Y_train)
    if not np.any(mask):
        raise ValueError("Eğitim matrisinde sonlu AUC gözlemi yok.")
    mu = float(np.mean(Y_train[mask]))
    counts = np.sum(mask, axis=0)
    sums = np.sum(np.where(mask, Y_train, 0.0), axis=0)
    col_means = np.divide(sums, counts, out=np.full(Y_train.shape[1], mu), where=counts > 0)
    b_d = col_means - mu
    return mu, b_d


def _build_inner_cv(
    train_subjects: np.ndarray,
    n_splits: int = 4,
    seed: int = 17,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Eğitim katı içinde hasta-ayrık iç CV bölmeleri üretir."""
    unique_subj = np.unique(train_subjects)
    rng = np.random.default_rng(seed)
    shuffled = unique_subj.copy()
    rng.shuffle(shuffled)
    fold_map = {int(s): idx % n_splits for idx, s in enumerate(shuffled)}
    row_folds = np.array([fold_map[int(s)] for s in train_subjects], dtype=np.int64)
    return [
        (np.flatnonzero(row_folds != f), np.flatnonzero(row_folds == f))
        for f in range(n_splits)
    ]


def estimate_psd_shrunk_cov(
    R_train: np.ndarray,
    alpha: float,
    min_pair: int = 15,
    min_eig: float = 1e-4,
) -> np.ndarray:
    """Maskeli kalıntı matrisinden ikili PSD-projekte ve büzüştürülmüş kovaryans üretir."""
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
    return (1.0 - alpha) * cov_psd + alpha * np.diag(diag_psd)


def conditional_gaussian_predict_matrix(
    R_obs_eval: np.ndarray,
    cov_psd: np.ndarray,
    probe_sets: list[np.ndarray],
    target_local_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Her hasta için S_p gözlemlerinden koşullu Gaussian/Ridge hedef kalıntılarını tahmin eder."""
    n_eval = R_obs_eval.shape[0]
    out = np.zeros((n_eval, len(target_local_idx)), dtype=np.float64)
    for i in range(n_eval):
        S = probe_sets[i]
        if len(S) == 0:
            continue
        r_s = R_obs_eval[i, S]
        valid = np.isfinite(r_s)
        if not np.any(valid):
            continue
        S_v = S[valid]
        r_v = r_s[valid]
        C_ss = cov_psd[np.ix_(S_v, S_v)].copy()
        C_ss[np.diag_indices_from(C_ss)] += lam
        C_ts = cov_psd[np.ix_(target_local_idx, S_v)]
        out[i] = C_ts @ np.linalg.solve(C_ss, r_v)
    return out


class MeanEffectM0:
    """Adil few-shot ortalama-etki modeli: y = mu + b_d + b_p^(S_p)."""

    def __init__(self, lambda_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)) -> None:
        self.lambda_grid = lambda_grid
        self.mu_: float = 0.0
        self.b_d_: np.ndarray = np.array([])
        self.best_lambda_by_k_: dict[int, float] = {}

    def fit(
        self,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        k_values: tuple[int, ...] = (3, 5, 8),
        inner_seed: int = 17,
    ) -> MeanEffectM0:
        self.mu_, self.b_d_ = fit_drug_means(Y_train)
        folds = _build_inner_cv(train_subjects, n_splits=4, seed=inner_seed)
        t_idx = np.asarray(target_idx, dtype=np.int64)

        for k in k_values:
            losses = {lam: [] for lam in self.lambda_grid}
            for f_idx, (tr_pos, va_pos) in enumerate(folds):
                mu_f, b_f = fit_drug_means(Y_train[tr_pos])
                base_f = mu_f + b_f
                Y_va = Y_train[va_pos]
                R_va = Y_va - base_f[None, :]
                probes_f = sample_probe_sets(Y_va, probe_pool_idx, k=k, n_repeats=3, seed=inner_seed + k + f_idx)
                Yt = Y_va[:, t_idx]
                t_mask = np.isfinite(Yt)
                if not np.any(t_mask):
                    continue
                for rep_probes in probes_f:
                    probe_sums = np.array(
                        [
                            float(np.nansum(R_va[i, s])) if len(s) > 0 else 0.0
                            for i, s in enumerate(rep_probes)
                        ],
                        dtype=np.float64,
                    )
                    probe_cnts = np.array([len(s) for s in rep_probes], dtype=np.float64)
                    for lam in self.lambda_grid:
                        b_p = np.divide(probe_sums, probe_cnts + lam, out=np.zeros_like(probe_sums), where=probe_cnts > 0)
                        pred = base_f[t_idx][None, :] + b_p[:, None]
                        losses[lam].append(float(np.mean((Yt[t_mask] - pred[t_mask]) ** 2)))
            mean_losses = {lam: float(np.mean(v)) if v else float("inf") for lam, v in losses.items()}
            self.best_lambda_by_k_[k] = min(mean_losses, key=mean_losses.get)  # type: ignore[arg-type]
        return self

    def predict(
        self,
        Y_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        k: int,
    ) -> np.ndarray:
        base = self.mu_ + self.b_d_
        pred = np.broadcast_to(base[None, :], Y_eval.shape).copy()
        if k <= 0:
            return pred
        lam = self.best_lambda_by_k_.get(k, 5.0)
        R_eval = Y_eval - base[None, :]
        for i, S in enumerate(probe_sets):
            if len(S) == 0:
                continue
            vals = R_eval[i, S]
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            cnt = int(np.sum(valid))
            b_p = float(np.sum(vals[valid])) / (cnt + lam)
            pred[i] += b_p
        return pred


@dataclass
class _OmicsFoldParams:
    gene_idx: np.ndarray
    rna_mean: np.ndarray
    rna_std: np.ndarray
    mut_mean: np.ndarray
    mut_std: np.ndarray
    X_train_feat: np.ndarray
    mu: float
    b_d: np.ndarray
    weights: dict[int, np.ndarray]  # drug_idx -> dual alphas (N_obs_d,)
    obs_masks: dict[int, np.ndarray]  # drug_idx -> boolean mask in train


def _build_omics_features(
    X_rna_tr: np.ndarray,
    X_mut_tr: np.ndarray,
    X_rna_te: np.ndarray | None = None,
    X_mut_te: np.ndarray | None = None,
    top_k_genes: int = 2000,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Yalnızca X_rna_tr / X_mut_tr içinden gen seçer ve z-skorlar."""
    rna_var = np.var(X_rna_tr.astype(np.float64), axis=0)
    k = min(top_k_genes, X_rna_tr.shape[1])
    gene_idx = np.argpartition(rna_var, -k)[-k:]
    gene_idx = gene_idx[np.argsort(rna_var[gene_idx])[::-1]]

    Xr_tr = X_rna_tr[:, gene_idx].astype(np.float64)
    rna_mean = np.mean(Xr_tr, axis=0)
    rna_std = np.std(Xr_tr, axis=0)
    rna_std = np.where(rna_std < 1e-6, 1.0, rna_std)
    Xr_tr_z = (Xr_tr - rna_mean) / rna_std

    Xm_tr = X_mut_tr.astype(np.float64)
    mut_mean = np.mean(Xm_tr, axis=0)
    mut_std = np.std(Xm_tr, axis=0)
    mut_std = np.where(mut_std < 1e-6, 1.0, mut_std)
    Xm_tr_z = (Xm_tr - mut_mean) / mut_std

    # Mutasyon bloğuna RNA ile orantılı küçük ağırlık koruması yerine saf standartlaştırılmış birleşim
    X_tr = np.hstack([Xr_tr_z, Xm_tr_z])

    X_te: np.ndarray | None = None
    if X_rna_te is not None and X_mut_te is not None:
        Xr_te_z = (X_rna_te[:, gene_idx].astype(np.float64) - rna_mean) / rna_std
        Xm_te_z = (X_mut_te.astype(np.float64) - mut_mean) / mut_std
        X_te = np.hstack([Xr_te_z, Xm_te_z])

    return X_tr, X_te, gene_idx, rna_mean, rna_std, mut_mean, mut_std


class OmicsRidgeM1:
    """Eğitim-içi gen seçimi & z-skorlamalı, dual Gram çekirdek Ridge (M1) + OOF üreticisi."""

    def __init__(
        self,
        lambda_grid: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0),
        top_k_genes: int = 2000,
    ) -> None:
        self.lambda_grid = lambda_grid
        self.top_k_genes = top_k_genes
        self.best_lambda_: float = 1000.0
        self.fold_model_: _OmicsFoldParams | None = None
        self.Y_oof_: np.ndarray | None = None

    def _fit_dual_for_drugs(
        self,
        X_tr: np.ndarray,
        Y_tr: np.ndarray,
        drug_indices: list[int],
        lam: float,
        mu: float,
        b_d: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        K_full = X_tr @ X_tr.T
        weights: dict[int, np.ndarray] = {}
        obs_masks: dict[int, np.ndarray] = {}
        for d in drug_indices:
            m = np.isfinite(Y_tr[:, d])
            obs_masks[d] = m
            n_obs = int(np.sum(m))
            if n_obs < 5:
                weights[d] = np.zeros(n_obs, dtype=np.float64)
                continue
            r = Y_tr[m, d] - (mu + b_d[d])
            K_sub = K_full[np.ix_(m, m)].copy()
            K_sub[np.diag_indices_from(K_sub)] += lam
            weights[d] = np.linalg.solve(K_sub, r)
        return weights, obs_masks

    def fit_and_oof(
        self,
        X_rna_tr: np.ndarray,
        X_mut_tr: np.ndarray,
        Y_tr: np.ndarray,
        train_subjects: np.ndarray,
        fit_drug_idx: list[int],
        target_idx: list[int],
        inner_seed: int = 17,
    ) -> tuple[OmicsRidgeM1, np.ndarray]:
        folds = _build_inner_cv(train_subjects, n_splits=4, seed=inner_seed)
        n_tr, n_drugs = Y_tr.shape

        # Hızlandırma: lambda taraması yalnızca hedef küme (target_idx) üzerinde yapılır;
        # best_lambda_ seçildikten sonra kalan prob sütunları için tek lambda ile solve edilir.
        target_set = set(target_idx)
        target_list = sorted(target_set)
        probe_only_list = sorted(set(fit_drug_idx) - target_set)
        oof_target_by_lam = {lam: np.full((n_tr, len(target_list)), np.nan, dtype=np.float64) for lam in self.lambda_grid}
        fold_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

        for tr_pos, va_pos in folds:
            X_f_tr, X_f_va, *_ = _build_omics_features(
                X_rna_tr[tr_pos],
                X_mut_tr[tr_pos],
                X_rna_tr[va_pos],
                X_mut_tr[va_pos],
                top_k_genes=self.top_k_genes,
            )
            assert X_f_va is not None
            mu_f, b_f = fit_drug_means(Y_tr[tr_pos])
            K_tr = X_f_tr @ X_f_tr.T
            K_va_tr = X_f_va @ X_f_tr.T
            fold_cache.append((tr_pos, va_pos, K_tr, K_va_tr, mu_f, b_f))

            for t_pos_local, d in enumerate(target_list):
                m = np.isfinite(Y_tr[tr_pos, d])
                base = mu_f + b_f[d]
                if int(np.sum(m)) < 5:
                    for lam in self.lambda_grid:
                        oof_target_by_lam[lam][va_pos, t_pos_local] = base
                    continue
                r = Y_tr[tr_pos, d][m] - base
                K_sub = K_tr[np.ix_(m, m)]
                evals, evecs = np.linalg.eigh(K_sub)
                evals = np.maximum(evals, 0.0)
                proj = evecs.T @ r
                K_cross = K_va_tr[:, m]
                for lam in self.lambda_grid:
                    alpha = evecs @ (proj / (evals + lam))
                    oof_target_by_lam[lam][va_pos, t_pos_local] = base + K_cross @ alpha

        # En iyi lambda'yı yalnızca C hedef ilaçları üzerindeki iç-CV MSE ile seç
        t_arr = np.asarray(target_list, dtype=np.int64)
        Yt = Y_tr[:, t_arr]
        t_mask = np.isfinite(Yt)
        best_lam = self.lambda_grid[0]
        best_mse = float("inf")
        for lam in self.lambda_grid:
            Yp_t = oof_target_by_lam[lam]
            m = t_mask & np.isfinite(Yp_t)
            mse = float(np.mean((Yt[m] - Yp_t[m]) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        self.best_lambda_ = best_lam
        Y_oof = np.full((n_tr, n_drugs), np.nan, dtype=np.float64)
        Y_oof[:, t_arr] = oof_target_by_lam[best_lam]

        # Kalan prob ilaçları için yalnızca seçilen best_lam ile tek geçişte solve
        for tr_pos, va_pos, K_tr, K_va_tr, mu_f, b_f in fold_cache:
            for d in probe_only_list:
                m = np.isfinite(Y_tr[tr_pos, d])
                base = mu_f + b_f[d]
                if int(np.sum(m)) < 5:
                    Y_oof[va_pos, d] = base
                    continue
                r = Y_tr[tr_pos, d][m] - base
                K_sub = K_tr[np.ix_(m, m)].copy()
                K_sub[np.diag_indices_from(K_sub)] += best_lam
                alpha = np.linalg.solve(K_sub, r)
                Y_oof[va_pos, d] = base + K_va_tr[:, m] @ alpha

        self.Y_oof_ = Y_oof
        fit_set = sorted(target_set | set(fit_drug_idx))

        # Son model tüm eğitim katı üzerinde fit edilir
        X_full, _, gene_idx, r_mean, r_std, m_mean, m_std = _build_omics_features(
            X_rna_tr, X_mut_tr, top_k_genes=self.top_k_genes
        )
        mu_full, b_full = fit_drug_means(Y_tr)
        weights, obs_masks = self._fit_dual_for_drugs(
            X_full, Y_tr, fit_set, lam=best_lam, mu=mu_full, b_d=b_full
        )
        self.fold_model_ = _OmicsFoldParams(
            gene_idx=gene_idx,
            rna_mean=r_mean,
            rna_std=r_std,
            mut_mean=m_mean,
            mut_std=m_std,
            X_train_feat=X_full,
            mu=mu_full,
            b_d=b_full,
            weights=weights,
            obs_masks=obs_masks,
        )
        return self, self.Y_oof_

    def predict(self, X_rna_te: np.ndarray, X_mut_te: np.ndarray, n_drugs: int) -> np.ndarray:
        if self.fold_model_ is None:
            raise RuntimeError("Model fit edilmeden predict çağrılamaz.")
        fm = self.fold_model_
        Xr_z = (X_rna_te[:, fm.gene_idx].astype(np.float64) - fm.rna_mean) / fm.rna_std
        Xm_z = (X_mut_te.astype(np.float64) - fm.mut_mean) / fm.mut_std
        X_te = np.hstack([Xr_z, Xm_z])
        K_cross = X_te @ fm.X_train_feat.T

        pred = np.broadcast_to((fm.mu + fm.b_d)[None, :], (X_te.shape[0], n_drugs)).copy()
        for d, alpha in fm.weights.items():
            m = fm.obs_masks[d]
            if int(np.sum(m)) >= 5:
                pred[:, d] = (fm.mu + fm.b_d[d]) + K_cross[:, m] @ alpha
        return pred


class CovarianceFewShotM2a:
    """Omiksiz Few-Shot Kovaryans Ridge / Koşullu Gaussian (M2a)."""

    def __init__(
        self,
        alpha_grid: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
        lam_grid: tuple[float, ...] = (50.0, 200.0, 500.0, 1500.0, 5000.0),
    ) -> None:
        self.alpha_grid = alpha_grid
        self.lam_grid = lam_grid
        self.mu_: float = 0.0
        self.b_d_: np.ndarray = np.array([])
        self.active_idx_: np.ndarray = np.array([], dtype=np.int64)
        self.global_to_local_: dict[int, int] = {}
        self.cov_psd_: np.ndarray = np.array([])
        self.best_params_by_k_: dict[int, tuple[float, float]] = {}

    def fit(
        self,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        k_values: tuple[int, ...] = (3, 5, 8),
        prior_train: np.ndarray | None = None,
        inner_seed: int = 17,
    ) -> CovarianceFewShotM2a:
        """prior_train None ise M2a (omiksiz), verilirse M3 (hibrit OOF artık) olarak fit eder."""
        self.mu_, self.b_d_ = fit_drug_means(Y_train)
        active = sorted(set(target_idx) | set(probe_pool_idx))
        self.active_idx_ = np.asarray(active, dtype=np.int64)
        self.global_to_local_ = {g: l for l, g in enumerate(active)}
        t_local = np.asarray([self.global_to_local_[g] for g in target_idx], dtype=np.int64)
        t_global = np.asarray(target_idx, dtype=np.int64)

        folds = _build_inner_cv(train_subjects, n_splits=4, seed=inner_seed)
        # Her iç fold'un (alpha) kovaryansını ön-hesapla
        fold_covs: list[dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        fold_probes_by_k: dict[int, list[list[list[np.ndarray]]]] = {k: [] for k in k_values}

        for f_idx, (tr_pos, va_pos) in enumerate(folds):
            if prior_train is None:
                mu_f, b_f = fit_drug_means(Y_train[tr_pos])
                base_f = mu_f + b_f
                R_tr_f = Y_train[tr_pos][:, self.active_idx_] - base_f[self.active_idx_][None, :]
                prior_va_act = np.broadcast_to(base_f[self.active_idx_][None, :], (len(va_pos), len(active)))
            else:
                R_tr_f = (Y_train[tr_pos] - prior_train[tr_pos])[:, self.active_idx_]
                prior_va_act = prior_train[va_pos][:, self.active_idx_]

            R_va_act = Y_train[va_pos][:, self.active_idx_] - prior_va_act
            covs_alpha: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for alpha in self.alpha_grid:
                covs_alpha[alpha] = (
                    estimate_psd_shrunk_cov(R_tr_f, alpha=alpha),
                    R_va_act,
                    prior_va_act,
                )
            fold_covs.append(covs_alpha)

            for k in k_values:
                probes_raw = sample_probe_sets(
                    Y_train[va_pos], probe_pool_idx, k=k, n_repeats=2, seed=inner_seed + k + f_idx
                )
                probes_local = [
                    [np.asarray([self.global_to_local_[g] for g in s], dtype=np.int64) for s in rep]
                    for rep in probes_raw
                ]
                fold_probes_by_k[k].append(probes_local)

        for k in k_values:
            best_pair = (self.alpha_grid[0], self.lam_grid[0])
            best_mse = float("inf")
            for alpha in self.alpha_grid:
                for lam in self.lam_grid:
                    sq_errs: list[np.ndarray] = []
                    for f_idx, (_, va_pos) in enumerate(folds):
                        cov_f, R_va_act, prior_va_act = fold_covs[f_idx][alpha]
                        Yt = Y_train[va_pos][:, t_global]
                        t_mask = np.isfinite(Yt)
                        if not np.any(t_mask):
                            continue
                        for rep_probes in fold_probes_by_k[k][f_idx]:
                            delta = conditional_gaussian_predict_matrix(
                                R_va_act, cov_f, rep_probes, t_local, lam=lam
                            )
                            pred_t = prior_va_act[:, t_local] + delta
                            sq_errs.append((Yt[t_mask] - pred_t[t_mask]) ** 2)
                    mse = float(np.mean(np.concatenate(sq_errs))) if sq_errs else float("inf")
                    if mse < best_mse:
                        best_mse = mse
                        best_pair = (alpha, lam)
            self.best_params_by_k_[k] = best_pair

        # Nihai kovaryans yalnızca tüm eğitim katı üzerinde (en iyi orta alpha veya k-bazlı dict ile)
        if prior_train is None:
            R_full = (Y_train - (self.mu_ + self.b_d_)[None, :])[:, self.active_idx_]
        else:
            R_full = (Y_train - prior_train)[:, self.active_idx_]

        self.cov_by_k_: dict[int, np.ndarray] = {}
        for k in k_values:
            alpha_k, _ = self.best_params_by_k_[k]
            self.cov_by_k_[k] = estimate_psd_shrunk_cov(R_full, alpha=alpha_k)
        return self

    def predict(
        self,
        Y_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        target_idx: list[int],
        k: int,
        prior_eval: np.ndarray | None = None,
    ) -> np.ndarray:
        if prior_eval is None:
            base_full = np.broadcast_to((self.mu_ + self.b_d_)[None, :], Y_eval.shape).copy()
        else:
            base_full = prior_eval.copy()

        if k <= 0:
            return base_full

        _, lam_k = self.best_params_by_k_[k]
        cov_k = self.cov_by_k_[k]
        R_eval_act = (Y_eval - base_full)[:, self.active_idx_]
        t_local = np.asarray([self.global_to_local_[g] for g in target_idx], dtype=np.int64)
        probes_local = [
            np.asarray([self.global_to_local_[g] for g in s if g in self.global_to_local_], dtype=np.int64)
            for s in probe_sets
        ]
        delta = conditional_gaussian_predict_matrix(
            R_eval_act, cov_k, probes_local, t_local, lam=lam_k
        )
        out = base_full.copy()
        out[:, np.asarray(target_idx, dtype=np.int64)] += delta
        return out


def _fit_masked_low_rank_factors(
    R_train: np.ndarray,
    rank: int,
    max_iter: int = 35,
    tol: float = 1e-3,
) -> np.ndarray:
    """Maskeli matriste iteratif rank-r SVD faktör yüklerini (D x r) öğrenir."""
    mask = np.isfinite(R_train)
    X = np.where(mask, R_train, 0.0)
    r_eff = min(rank, min(X.shape) - 1)
    if r_eff < 1:
        return np.zeros((X.shape[1], 1), dtype=np.float64)

    prev_norm = float(np.linalg.norm(X)) + 1e-12
    V_r = np.zeros((X.shape[1], r_eff), dtype=np.float64)
    use_gram = X.shape[0] >= X.shape[1]

    for _ in range(max_iter):
        if use_gram:
            # D x D simetrik Gram üzerinden eşdeğer sağ tekil vektör projeksiyonu
            _, evecs = np.linalg.eigh(X.T @ X)
            V_r = evecs[:, -r_eff:][:, ::-1]
        else:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            V_r = Vt[:r_eff, :].T

        X_rec = (X @ V_r) @ V_r.T
        diff = float(np.linalg.norm(np.where(~mask, X_rec - X, 0.0)))
        X = np.where(mask, R_train, X_rec)
        if diff / prev_norm < tol:
            break
        prev_norm = float(np.linalg.norm(X)) + 1e-12
    return V_r  # shape: (D_active, r_eff)


def _svd_predict_matrix(
    R_obs_eval: np.ndarray,
    V: np.ndarray,
    probe_sets: list[np.ndarray],
    target_local_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    n_eval = R_obs_eval.shape[0]
    r = V.shape[1]
    V_t = V[target_local_idx]
    eye_r = lam * np.eye(r, dtype=np.float64)
    out = np.zeros((n_eval, len(target_local_idx)), dtype=np.float64)
    for i in range(n_eval):
        S = probe_sets[i]
        if len(S) == 0:
            continue
        r_s = R_obs_eval[i, S]
        valid = np.isfinite(r_s)
        if not np.any(valid):
            continue
        V_s = V[S[valid]]
        u_hat = np.linalg.solve(V_s.T @ V_s + eye_r, V_s.T @ r_s[valid])
        out[i] = V_t @ u_hat
    return out


class LowRankSVDM2b:
    """Omiksiz Düşük-Rank Maskeli SVD (M2b)."""

    def __init__(
        self,
        rank_grid: tuple[int, ...] = (3, 5, 8, 12),
        lam_grid: tuple[float, ...] = (0.5, 2.0, 10.0, 50.0),
    ) -> None:
        self.rank_grid = rank_grid
        self.lam_grid = lam_grid
        self.mu_: float = 0.0
        self.b_d_: np.ndarray = np.array([])
        self.active_idx_: np.ndarray = np.array([], dtype=np.int64)
        self.global_to_local_: dict[int, int] = {}
        self.best_params_by_k_: dict[int, tuple[int, float]] = {}
        self.factors_by_rank_: dict[int, np.ndarray] = {}

    def fit(
        self,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        k_values: tuple[int, ...] = (3, 5, 8),
        inner_seed: int = 17,
    ) -> LowRankSVDM2b:
        self.mu_, self.b_d_ = fit_drug_means(Y_train)
        active = sorted(set(target_idx) | set(probe_pool_idx))
        self.active_idx_ = np.asarray(active, dtype=np.int64)
        self.global_to_local_ = {g: l for l, g in enumerate(active)}
        t_local = np.asarray([self.global_to_local_[g] for g in target_idx], dtype=np.int64)
        t_global = np.asarray(target_idx, dtype=np.int64)

        folds = _build_inner_cv(train_subjects, n_splits=4, seed=inner_seed)
        fold_factors: list[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        fold_probes_by_k: dict[int, list[list[list[np.ndarray]]]] = {k: [] for k in k_values}

        for f_idx, (tr_pos, va_pos) in enumerate(folds):
            mu_f, b_f = fit_drug_means(Y_train[tr_pos])
            base_f = mu_f + b_f
            R_tr_f = Y_train[tr_pos][:, self.active_idx_] - base_f[self.active_idx_][None, :]
            R_va_act = Y_train[va_pos][:, self.active_idx_] - base_f[self.active_idx_][None, :]
            base_va_t = base_f[t_global]

            f_dict: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for r in self.rank_grid:
                V_f = _fit_masked_low_rank_factors(R_tr_f, rank=r)
                f_dict[r] = (V_f, R_va_act, base_va_t)
            fold_factors.append(f_dict)

            for k in k_values:
                probes_raw = sample_probe_sets(
                    Y_train[va_pos], probe_pool_idx, k=k, n_repeats=2, seed=inner_seed + k + f_idx
                )
                fold_probes_by_k[k].append(
                    [[np.asarray([self.global_to_local_[g] for g in s], dtype=np.int64) for s in rep] for rep in probes_raw]
                )

        for k in k_values:
            best_pair = (self.rank_grid[0], self.lam_grid[0])
            best_mse = float("inf")
            for r in self.rank_grid:
                for lam in self.lam_grid:
                    sq_errs: list[np.ndarray] = []
                    for f_idx, (_, va_pos) in enumerate(folds):
                        V_f, R_va_act, base_va_t = fold_factors[f_idx][r]
                        Yt = Y_train[va_pos][:, t_global]
                        t_mask = np.isfinite(Yt)
                        if not np.any(t_mask):
                            continue
                        for rep_probes in fold_probes_by_k[k][f_idx]:
                            delta = _svd_predict_matrix(R_va_act, V_f, rep_probes, t_local, lam=lam)
                            pred_t = base_va_t[None, :] + delta
                            sq_errs.append((Yt[t_mask] - pred_t[t_mask]) ** 2)
                    mse = float(np.mean(np.concatenate(sq_errs))) if sq_errs else float("inf")
                    if mse < best_mse:
                        best_mse = mse
                        best_pair = (r, lam)
            self.best_params_by_k_[k] = best_pair

        R_full = (Y_train - (self.mu_ + self.b_d_)[None, :])[:, self.active_idx_]
        for r in sorted({r for r, _ in self.best_params_by_k_.values()}):
            self.factors_by_rank_[r] = _fit_masked_low_rank_factors(R_full, rank=r)
        return self

    def predict(
        self,
        Y_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        target_idx: list[int],
        k: int,
    ) -> np.ndarray:
        base_full = np.broadcast_to((self.mu_ + self.b_d_)[None, :], Y_eval.shape).copy()
        if k <= 0:
            return base_full
        r_k, lam_k = self.best_params_by_k_[k]
        V = self.factors_by_rank_[r_k]
        R_eval_act = (Y_eval - base_full)[:, self.active_idx_]
        t_local = np.asarray([self.global_to_local_[g] for g in target_idx], dtype=np.int64)
        probes_local = [
            np.asarray([self.global_to_local_[g] for g in s if g in self.global_to_local_], dtype=np.int64)
            for s in probe_sets
        ]
        delta = _svd_predict_matrix(R_eval_act, V, probes_local, t_local, lam=lam_k)
        out = base_full.copy()
        out[:, np.asarray(target_idx, dtype=np.int64)] += delta
        return out


class HybridFewShotM3:
    """M1 omik öncülü + M1 OOF kalıntı kovaryansı üzerinden few-shot güncelleme (M3)."""

    def __init__(
        self,
        alpha_grid: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50),
        lam_grid: tuple[float, ...] = (50.0, 200.0, 500.0, 1500.0, 5000.0),
    ) -> None:
        self._cov_model = CovarianceFewShotM2a(alpha_grid=alpha_grid, lam_grid=lam_grid)
        self.omics_model: OmicsRidgeM1 | None = None

    def fit(
        self,
        omics_model: OmicsRidgeM1,
        Y_oof_train: np.ndarray,
        Y_train: np.ndarray,
        train_subjects: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        k_values: tuple[int, ...] = (3, 5, 8),
        inner_seed: int = 17,
    ) -> HybridFewShotM3:
        self.omics_model = omics_model
        self._cov_model.fit(
            Y_train=Y_train,
            train_subjects=train_subjects,
            probe_pool_idx=probe_pool_idx,
            target_idx=target_idx,
            k_values=k_values,
            prior_train=Y_oof_train,
            inner_seed=inner_seed,
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
            raise RuntimeError("HybridFewShotM3 fit edilmeden predict çağrılamaz.")
        if prior_eval is None:
            prior_eval = self.omics_model.predict(X_rna_te, X_mut_te, n_drugs=Y_eval.shape[1])
        if k <= 0:
            return prior_eval
        return self._cov_model.predict(
            Y_eval=Y_eval,
            probe_sets=probe_sets,
            target_idx=target_idx,
            k=k,
            prior_eval=prior_eval,
        )
