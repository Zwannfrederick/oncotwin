"""Hasta-düzeyi sıra-kalibreli konformal kısa aday listesi ve AUGRC (Traub et al., NeurIPS 2024)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.oncotwin.eval.uncertainty import finite_sample_conformal_quantile


def _patient_true_best_mask(y_obs: np.ndarray, tie_tol: float = 1e-12) -> np.ndarray:
    y_min = float(np.min(y_obs))
    return y_obs <= (y_min + tie_tol)


def fit_rank_conformal_shortlist(
    Y_true_cal: np.ndarray,
    Y_pred_cal: np.ndarray,
    target_idx: list[int],
    alpha: float = 0.10,
    valid_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Kalibrasyon hastalarında gerçek en iyi hedefin tahmin sırasını (mutlak & göreli) kalibre eder."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = np.asarray(Y_true_cal[:, t_idx], dtype=np.float64)
    Yp = np.asarray(Y_pred_cal[:, t_idx], dtype=np.float64)
    finite = np.isfinite(Yt) & np.isfinite(Yp)
    row_ok = (
        np.any(finite, axis=1)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.any(finite, axis=1))
    )

    abs_ranks: list[float] = []
    rel_ranks: list[float] = []
    for i in np.flatnonzero(row_ok):
        fin = finite[i]
        y_o = Yt[i, fin]
        yp_o = Yp[i, fin]
        best_pos = np.flatnonzero(_patient_true_best_mask(y_o, tie_tol=tie_tol))
        # Bağ-koruyucu (conservative): tüm gerçek-en-iyi bağlar arasında en kötü 1-tabanlı sıra
        # strict_rank(d*) = 1 + #{d: yp[d] < yp[d*] - tie_tol} <=> yp[d*] <= sorted_yp[k_hat - 1]
        r_i = max(int(1 + np.sum(yp_o < yp_o[b] - tie_tol)) for b in best_pos)
        abs_ranks.append(float(r_i))
        rel_ranks.append(float(r_i) / float(len(y_o)))

    abs_arr = np.asarray(abs_ranks, dtype=np.float64)
    rel_arr = np.asarray(rel_ranks, dtype=np.float64)
    k_hat, r_stat, fin_pub = finite_sample_conformal_quantile(abs_arr, alpha)
    q_rel, _, _ = finite_sample_conformal_quantile(rel_arr, alpha)
    return {
        "alpha": float(alpha),
        "n_cal": int(len(abs_arr)),
        "order_stat_rank": int(r_stat),
        "k_hat": int(round(k_hat)) if fin_pub else None,
        "k_hat_float": float(k_hat),
        "q_rel": float(q_rel),
        "finite_published": bool(fin_pub),
        "cal_abs_ranks": abs_arr,
        "cal_rel_ranks": rel_arr,
    }


def fit_top1_gap_conformal_shortlist(
    Y_true_cal: np.ndarray,
    Y_pred_cal: np.ndarray,
    target_idx: list[int],
    alpha: float = 0.10,
    valid_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Kalibrasyon hastalarında gerçek en iyi hedefin tahmin edilen #1'e göre AUC farkını kalibre eder."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = np.asarray(Y_true_cal[:, t_idx], dtype=np.float64)
    Yp = np.asarray(Y_pred_cal[:, t_idx], dtype=np.float64)
    finite = np.isfinite(Yt) & np.isfinite(Yp)
    row_ok = (
        np.any(finite, axis=1)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.any(finite, axis=1))
    )

    gaps: list[float] = []
    for i in np.flatnonzero(row_ok):
        fin = finite[i]
        y_o = Yt[i, fin]
        yp_o = Yp[i, fin]
        best_pos = np.flatnonzero(_patient_true_best_mask(y_o, tie_tol=tie_tol))
        yp_min = float(np.min(yp_o))
        gap_i = max(float(yp_o[b] - yp_min) for b in best_pos)
        gaps.append(max(0.0, gap_i))

    gap_arr = np.asarray(gaps, dtype=np.float64)
    m_hat, r_stat, fin_pub = finite_sample_conformal_quantile(gap_arr, alpha)
    return {
        "alpha": float(alpha),
        "n_cal": int(len(gap_arr)),
        "order_stat_rank": int(r_stat),
        "m_hat": float(m_hat),
        "finite_published": bool(fin_pub),
        "cal_gaps": gap_arr,
    }


def build_shortlist_masks(
    Y_true_te: np.ndarray,
    Y_pred_te: np.ndarray,
    target_idx: list[int],
    method: str,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
) -> np.ndarray:
    """Verilen yöntem ve kalibreli eşiğe göre (n_te, |C|) ikili kısa liste maskesi üretir."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = np.asarray(Y_true_te[:, t_idx], dtype=np.float64)
    Yp = np.asarray(Y_pred_te[:, t_idx], dtype=np.float64)
    n_te, n_t = Yt.shape
    finite = np.isfinite(Yt) & np.isfinite(Yp)
    row_ok = (
        np.any(finite, axis=1)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.any(finite, axis=1))
    )
    masks = np.zeros((n_te, n_t), dtype=bool)
    if not math.isfinite(float(threshold)):
        return masks

    for i in np.flatnonzero(row_ok):
        fin = finite[i]
        yp_o = Yp[i, fin]
        n_obs = len(yp_o)
        if method == "equal_width_auc":
            # [yp - q, yp + q] alt sınırı <= min üst sınır <=> yp <= min(yp) + 2*q
            cutoff = float(np.min(yp_o)) + 2.0 * float(threshold) + tie_tol
            sel = yp_o <= cutoff
        elif method == "top1_gap":
            cutoff = float(np.min(yp_o)) + float(threshold) + tie_tol
            sel = yp_o <= cutoff
        elif method == "abs_rank":
            k_eff = int(round(float(threshold)))
            if k_eff <= 0:
                sel = np.zeros(n_obs, dtype=bool)
            elif k_eff >= n_obs:
                sel = np.ones(n_obs, dtype=bool)
            else:
                cutoff = float(np.sort(yp_o)[k_eff - 1]) + tie_tol
                sel = yp_o <= cutoff
        elif method == "rel_rank":
            k_eff = int(math.ceil(float(threshold) * n_obs - 1e-12))
            if k_eff <= 0:
                sel = np.zeros(n_obs, dtype=bool)
            elif k_eff >= n_obs:
                sel = np.ones(n_obs, dtype=bool)
            else:
                cutoff = float(np.sort(yp_o)[k_eff - 1]) + tie_tol
                sel = yp_o <= cutoff
        else:
            raise ValueError(f"Bilinmeyen kısa liste yöntemi: {method}")
        masks[i, fin] = sel
    return masks


def evaluate_shortlist_masks(
    Y_true_te: np.ndarray,
    shortlist_masks: np.ndarray,
    target_idx: list[int],
    valid_mask: np.ndarray | None = None,
    finite_published: bool = True,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Kısa liste için kapsama, boyut, tekil oran ve AUC kaybı (best/mean/worst regret) hesaplar."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = np.asarray(Y_true_te[:, t_idx], dtype=np.float64)
    n_te = len(Yt)
    obs_mask = np.isfinite(Yt)
    row_ok = (
        np.any(obs_mask, axis=1)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & np.any(obs_mask, axis=1))
    )

    vec_keys = (
        "patient_shortlist_size_obs",
        "patient_shortlist_fraction_obs",
        "patient_shortlist_contains_top1",
        "patient_definite_top1",
        "patient_shortlist_best_regret1",
        "patient_shortlist_mean_regret1",
        "patient_shortlist_worst_regret1",
    )
    vecs = {k: np.full(n_te, np.nan, dtype=np.float64) for k in vec_keys}

    if not finite_published or not np.any(row_ok):
        status = (
            "hesaplanamadı (sonlu eşik yayımlanmadı)"
            if not finite_published
            else "hesaplanamadı (değerlendirilebilir test hastası yok)"
        )
        return {
            "status": status,
            "n_test": int(np.sum(row_ok)),
            "shortlist_contains_true_top1_rate": float("nan"),
            "shortlist_mean_size_obs": float("nan"),
            "shortlist_median_size_obs": float("nan"),
            "shortlist_mean_fraction_obs": float("nan"),
            "definite_top1_rate_obs": float("nan"),
            "shortlist_best_regret1": float("nan"),
            "shortlist_mean_regret1": float("nan"),
            "shortlist_worst_regret1": float("nan"),
            "vectors": vecs,
        }

    sl = np.asarray(shortlist_masks, dtype=bool)
    for i in np.flatnonzero(row_ok):
        obs = obs_mask[i]
        y_o = Yt[i, obs]
        in_s = sl[i, obs]
        y_min = float(np.min(y_o))
        true_best = y_o <= (y_min + tie_tol)
        n_obs = len(y_o)
        sz = int(np.sum(in_s))

        vecs["patient_shortlist_size_obs"][i] = float(sz)
        vecs["patient_shortlist_fraction_obs"][i] = float(sz) / float(n_obs)
        vecs["patient_shortlist_contains_top1"][i] = float(np.any(in_s & true_best))
        vecs["patient_definite_top1"][i] = float(sz == 1)
        if sz > 0:
            y_sel = y_o[in_s]
            vecs["patient_shortlist_best_regret1"][i] = float(np.min(y_sel) - y_min)
            vecs["patient_shortlist_mean_regret1"][i] = float(np.mean(y_sel) - y_min)
            vecs["patient_shortlist_worst_regret1"][i] = float(np.max(y_sel) - y_min)

    ok_idx = np.flatnonzero(row_ok)
    best_r = vecs["patient_shortlist_best_regret1"][ok_idx]
    mean_r = vecs["patient_shortlist_mean_regret1"][ok_idx]
    worst_r = vecs["patient_shortlist_worst_regret1"][ok_idx]
    return {
        "status": "ok",
        "n_test": int(len(ok_idx)),
        "shortlist_contains_true_top1_rate": float(np.mean(vecs["patient_shortlist_contains_top1"][ok_idx])),
        "shortlist_mean_size_obs": float(np.mean(vecs["patient_shortlist_size_obs"][ok_idx])),
        "shortlist_median_size_obs": float(np.median(vecs["patient_shortlist_size_obs"][ok_idx])),
        "shortlist_mean_fraction_obs": float(np.mean(vecs["patient_shortlist_fraction_obs"][ok_idx])),
        "definite_top1_rate_obs": float(np.mean(vecs["patient_definite_top1"][ok_idx])),
        "shortlist_best_regret1": float(np.mean(best_r[np.isfinite(best_r)])) if np.any(np.isfinite(best_r)) else float("nan"),
        "shortlist_mean_regret1": float(np.mean(mean_r[np.isfinite(mean_r)])) if np.any(np.isfinite(mean_r)) else float("nan"),
        "shortlist_worst_regret1": float(np.mean(worst_r[np.isfinite(worst_r)])) if np.any(np.isfinite(worst_r)) else float("nan"),
        "vectors": vecs,
    }


def _trapezoid_augrc_from_arrays(scores: np.ndarray, losses: np.ndarray, tie_tol: float = 1e-12) -> float:
    """Bağ-duyarlı Genelleştirilmiş Risk-Kapsam Eğrisi Altındaki Alanı (AUGRC) hesaplar.

    G(c) = (1 / N) * sum_{i in A_c} loss_i = c * R(c); (0, 0)'dan başlayarak yamuk integrali alır.
    Rastgele sıralama ve sabit skor için tam olarak 0.5 * mean(losses) değerini verir.
    """
    n = len(losses)
    order = np.argsort(scores, kind="mergesort")
    s_ord = scores[order]
    l_ord = losses[order]

    area = 0.0
    cum_loss = 0.0
    prev_cov = 0.0
    prev_gr = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(s_ord[end] - s_ord[start]) <= tie_tol:
            end += 1
        cum_loss += float(np.sum(l_ord[start:end]))
        cov = end / float(n)
        gr = cum_loss / float(n)
        area += 0.5 * (prev_gr + gr) * (cov - prev_cov)
        prev_cov = cov
        prev_gr = gr
        start = end
    return float(area)


def generalized_risk_coverage_augrc(
    scores: np.ndarray,
    losses: np.ndarray,
    available_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Traub et al. (NeurIPS 2024, arXiv:2407.01032) AUGRC, Oracle-AUGRC, E-AUGRC ve Random-AUGRC."""
    sc = np.asarray(scores, dtype=np.float64)
    lo = np.asarray(losses, dtype=np.float64)
    avail = np.isfinite(sc) & np.isfinite(lo)
    if available_mask is not None:
        avail = avail & np.asarray(available_mask, dtype=bool)
    n_av = int(np.sum(avail))
    if n_av == 0:
        return {
            "status": "hesaplanamadı",
            "n_available": 0,
            "augrc": float("nan"),
            "oracle_augrc": float("nan"),
            "e_augrc": float("nan"),
            "random_augrc": float("nan"),
        }
    s_av = sc[avail]
    l_av = lo[avail]
    augrc = _trapezoid_augrc_from_arrays(s_av, l_av, tie_tol=tie_tol)
    oracle = _trapezoid_augrc_from_arrays(l_av, l_av, tie_tol=tie_tol)
    rand = 0.5 * float(np.mean(l_av))
    return {
        "status": "ok",
        "n_available": n_av,
        "augrc": augrc,
        "oracle_augrc": oracle,
        "e_augrc": float(augrc - oracle),
        "random_augrc": rand,
    }
