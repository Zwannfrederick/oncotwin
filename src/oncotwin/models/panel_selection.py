"""Sprint 3 toplu panel seçim politikaları (Static-Diversity, Static-D-Optimal, Patient-Adaptive Batch OED)."""

from __future__ import annotations

import numpy as np


def select_static_diversity_ranking(
    cov_res: np.ndarray,
    probe_local_idx: np.ndarray,
) -> np.ndarray:
    """Kalıntı korelasyon mesafesi D = 1 - |R| üzerinde açgözlü Farthest-First sıralaması döndürür."""
    probes = np.asarray(probe_local_idx, dtype=np.int64)
    n_p = len(probes)
    if n_p <= 1:
        return probes.copy()

    cov_p = cov_res[np.ix_(probes, probes)]
    sd = np.sqrt(np.maximum(np.diag(cov_p), 1e-12))
    corr = np.clip(cov_p / np.outer(sd, sd), -1.0, 1.0)
    dist = np.maximum(0.0, 1.0 - np.abs(corr))
    np.fill_diagonal(dist, 0.0)

    selected_pos: list[int] = [int(np.argmin(np.mean(dist, axis=1)))]
    in_sel = np.zeros(n_p, dtype=bool)
    in_sel[selected_pos[0]] = True
    min_dist = dist[selected_pos[0]].copy()

    for _ in range(1, n_p):
        cand = np.where(in_sel, -1.0, min_dist)
        nxt = int(np.argmax(cand))
        selected_pos.append(nxt)
        in_sel[nxt] = True
        min_dist = np.minimum(min_dist, dist[nxt])

    return probes[np.asarray(selected_pos, dtype=np.int64)]


def select_static_d_optimal_ranking(
    cov_res: np.ndarray,
    probe_local_idx: np.ndarray,
    target_local_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Gaussian Karşılıklı Bilgi I(Y_C; Y_S | X) maksimizasyonu ile açgözlü statik sıralama döndürür."""
    probes = np.asarray(probe_local_idx, dtype=np.int64)
    targets = np.asarray(target_local_idx, dtype=np.int64)
    n_p = len(probes)
    if n_p <= 1:
        return probes.copy()

    Sigma_free = 0.5 * (cov_res + cov_res.T)
    C_TT = Sigma_free[np.ix_(targets, targets)].copy()
    C_TT[np.diag_indices_from(C_TT)] += 1e-6
    C_T_all = Sigma_free[targets, :]
    Sigma_cond = Sigma_free - C_T_all.T @ np.linalg.solve(C_TT, C_T_all)
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond.T)

    rem = probes.copy()
    selected: list[int] = []

    for _ in range(n_p):
        d_free = np.maximum(np.diag(Sigma_free)[rem], 0.0) + lam
        d_cond = np.maximum(np.diag(Sigma_cond)[rem], 0.0) + lam
        gains = 0.5 * np.log(np.maximum(d_free / d_cond, 1.0))
        best_pos = int(np.argmax(gains))
        j_star = int(rem[best_pos])
        selected.append(j_star)
        rem = np.delete(rem, best_pos)
        if len(rem) == 0:
            break

        v_f = Sigma_free[:, j_star]
        Sigma_free = Sigma_free - np.outer(v_f, v_f) / (max(float(Sigma_free[j_star, j_star]), 0.0) + lam)
        v_c = Sigma_cond[:, j_star]
        Sigma_cond = Sigma_cond - np.outer(v_c, v_c) / (max(float(Sigma_cond[j_star, j_star]), 0.0) + lam)

    return np.asarray(selected, dtype=np.int64)


def compute_contender_weights(
    prior_pred_C: np.ndarray,
    cov_res_CC: np.ndarray,
    top_k: int = 3,
    n_mc: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Seçim öncesi M1 öncülünden her hastada aday ilaçların Top-k olasılık ağırlığını hesaplar."""
    n_pat, n_c = prior_pred_C.shape
    m_k = min(top_k, n_c)
    C_sym = 0.5 * (cov_res_CC + cov_res_CC.T) + 1e-6 * np.eye(n_c, dtype=np.float64)
    L = np.linalg.cholesky(C_sym)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_mc, n_c)) @ L.T  # (n_mc, n_c)

    weights = np.empty((n_pat, n_c), dtype=np.float64)
    for i in range(n_pat):
        samples = prior_pred_C[i][None, :] + noise
        top_idx = np.argpartition(samples, m_k - 1, axis=1)[:, :m_k]
        counts = np.bincount(top_idx.ravel(), minlength=n_c).astype(np.float64)
        weights[i] = counts / float(np.sum(counts))
    return weights


def select_adaptive_batch_oed(
    cov_res: np.ndarray,
    avail_probe_local_idx: np.ndarray,
    target_local_idx: np.ndarray,
    target_weights: np.ndarray,
    k: int,
    lam: float,
) -> np.ndarray:
    """Hasta-özel karar ağırlıklı beklenen varyans azalışıyla k probu toplu seçer."""
    avail = np.asarray(avail_probe_local_idx, dtype=np.int64)
    targets = np.asarray(target_local_idx, dtype=np.int64)
    if k <= 0 or len(avail) == 0:
        return np.array([], dtype=np.int64)
    k_eff = min(k, len(avail))

    Sigma = 0.5 * (cov_res + cov_res.T)
    rem = avail.copy()
    selected: list[int] = []

    for _ in range(k_eff):
        cross = Sigma[np.ix_(targets, rem)]  # (n_C, n_rem)
        var_j = np.maximum(np.diag(Sigma)[rem], 0.0) + lam
        scores = (target_weights @ (cross * cross)) / var_j
        best_pos = int(np.argmax(scores))
        j_star = int(rem[best_pos])
        selected.append(j_star)
        rem = np.delete(rem, best_pos)
        if len(rem) == 0:
            break
        v = Sigma[:, j_star]
        Sigma = Sigma - np.outer(v, v) / (max(float(Sigma[j_star, j_star]), 0.0) + lam)

    return np.asarray(selected, dtype=np.int64)


def take_first_k_available(
    ranked_global_idx: np.ndarray,
    y_row: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    """Sıralı statik öncelik listesinden hastada ölçülmüş ilk k probu (AUC görmeden) seçer."""
    if k <= 0:
        return np.array([], dtype=np.int64), 0

    chosen: list[int] = []
    direct_hits = 0
    for pos, g_idx in enumerate(ranked_global_idx):
        if np.isfinite(y_row[int(g_idx)]):
            chosen.append(int(g_idx))
            if pos < k:
                direct_hits += 1
            if len(chosen) == k:
                break
    return np.asarray(chosen, dtype=np.int64), direct_hits


def require_full_budget(probes: list[np.ndarray], eval_mask: np.ndarray, k: int, policy: str) -> None:
    """Değerlendirilebilir her hastada panelin tam k prob içerdiğini doğrular (sessiz bütçe kesilmesine karşı)."""
    short = [i for i in np.flatnonzero(eval_mask) if len(probes[i]) != k]
    if short:
        raise ValueError(f"{policy}: {len(short)} değerlendirilebilir hastada bütçe k={k} karşılanamadı (ilk satır {short[0]}).")
