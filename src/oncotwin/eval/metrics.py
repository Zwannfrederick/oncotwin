"""Protokol uyumlu değerlendirme metrikleri, AUC yönlü NDCG@3 ve eşleştirilmiş bootstrap."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np


def _rankdata_average(arr: np.ndarray) -> np.ndarray:
    """Saf NumPy tie-duyarlı ortalama sıralama (pd.Series.rank(method='average') eşdeğeri)."""
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    s = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    if np.all(np.diff(s) > 1e-12):  # bağ yok: döngüsüz hızlı yol (aynı sonuç)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        return ranks
    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(s[end] - s[start]) <= 1e-12:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Sonlu çiftler üzerinde tie-duyarlı Spearman korelasyonu; sabit vektörde 0.0."""
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = x[mask].astype(np.float64, copy=False)
    yv = y[mask].astype(np.float64, copy=False)
    if np.std(xv) < 1e-12 or np.std(yv) < 1e-12:
        return 0.0
    rx = _rankdata_average(xv)
    ry = _rankdata_average(yv)
    rx_c = rx - np.mean(rx)
    ry_c = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(rx_c * rx_c) * np.sum(ry_c * ry_c)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(rx_c * ry_c) / denom)


def _top_k_membership_weights(scores: np.ndarray, m: int) -> np.ndarray:
    """En düşük m öğeye tie-duyarlı beklenen üyelik olasılığı [0, 1] atar."""
    n = len(scores)
    w = np.zeros(n, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    pos = 0
    while pos < m and pos < n:
        end = pos + 1
        while end < n and abs(s[end] - s[pos]) <= 1e-12:
            end += 1
        if end <= m:
            w[order[pos:end]] = 1.0
        else:
            rem = m - pos
            w[order[pos:end]] = rem / float(end - pos)
        pos = end
    return w


def recall_at_k(y_true_auc: np.ndarray, y_pred_auc: np.ndarray, k: int = 3) -> float:
    """Düşük AUC = en duyarlı yönünde tie-duyarlı beklenen Recall@k."""
    mask = np.isfinite(y_true_auc) & np.isfinite(y_pred_auc)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    yt = y_true_auc[mask].astype(np.float64)
    yp = y_pred_auc[mask].astype(np.float64)
    m = min(k, n)
    w_true = _top_k_membership_weights(yt, m)
    w_pred = _top_k_membership_weights(yp, m)
    return float(np.sum(w_true * w_pred) / m)


def regret_at_k(y_true_auc: np.ndarray, y_pred_auc: np.ndarray, k: int = 1) -> float:
    """Tahmin edilen ilk-k en duyarlı ilacın gerçek en iyi ilaca göre AUC kaybı (>= 0)."""
    mask = np.isfinite(y_true_auc) & np.isfinite(y_pred_auc)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    yt = y_true_auc[mask].astype(np.float64)
    yp = y_pred_auc[mask].astype(np.float64)
    best_true = float(np.min(yt))
    m = min(k, n)

    order = np.argsort(yp, kind="mergesort")
    yp_s = yp[order]
    yt_s = yt[order]

    # Sınır indeks m-1'i kapsayan tie bloğunu bul
    b_start = m - 1
    while b_start > 0 and abs(yp_s[b_start - 1] - yp_s[m - 1]) <= 1e-12:
        b_start -= 1
    b_end = m
    while b_end < n and abs(yp_s[b_end] - yp_s[m - 1]) <= 1e-12:
        b_end += 1

    if b_end == m:
        return float(np.min(yt_s[:m]) - best_true)

    pre_min = float(np.min(yt_s[:b_start])) if b_start > 0 else float("inf")
    block_yt = np.sort(yt_s[b_start:b_end])
    b_len = len(block_yt)
    r = m - b_start
    total_comb = math.comb(b_len, r)
    exp_min = 0.0
    for j in range(b_len - r + 1):
        prob_j = math.comb(b_len - 1 - j, r - 1) / total_comb
        exp_min += prob_j * min(pre_min, float(block_yt[j]))
    return float(exp_min - best_true)


def ndcg_at_k(y_true_auc: np.ndarray, y_pred_auc: np.ndarray, k: int = 3) -> float:
    """Düşük AUC = yüksek duyarlılık yönünde tie-duyarlı beklenen NDCG@k."""
    mask = np.isfinite(y_true_auc) & np.isfinite(y_pred_auc)
    n = int(mask.sum())
    if n == 0:
        return float("nan")

    yt = y_true_auc[mask].astype(np.float64)
    yp = y_pred_auc[mask].astype(np.float64)
    rel = np.max(yt) - yt

    m = min(k, n)
    discounts = 1.0 / np.log2(np.arange(2, m + 2, dtype=np.float64))

    ideal_rel = np.sort(rel)[::-1][:m]
    idcg = float(np.sum(ideal_rel * discounts))
    if idcg <= 1e-12:
        return float("nan")

    # Eşit tahmin puanları (ör. sabit tahmin) için beklenen grup ortalaması ile adil DCG
    order = np.argsort(yp, kind="mergesort")
    yp_sorted = yp[order]
    rel_sorted = rel[order].copy()

    start = 0
    while start < n:
        end = start + 1
        while end < n and abs(yp_sorted[end] - yp_sorted[start]) <= 1e-12:
            end += 1
        if end - start > 1:
            rel_sorted[start:end] = float(np.mean(rel_sorted[start:end]))
        start = end

    dcg = float(np.sum(rel_sorted[:m] * discounts))
    return dcg / idcg


VECTOR_KEYS = (
    "patient_rmse_sq",
    "patient_res_spearman",
    "patient_ndcg3",
    "patient_res_ndcg3",
    "patient_recall3",
    "patient_regret1",
    "patient_regret3",
)


def _rows_with_close_ties(V: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Geçerli değerleri arasında |fark| <= 1e-12 olan komşu çift içeren satırları işaretler."""
    S = np.sort(np.where(valid, V, np.inf), axis=1)
    both = np.isfinite(S[:, 1:]) & np.isfinite(S[:, :-1])
    with np.errstate(invalid="ignore"):
        return np.any(both & (np.diff(S, axis=1) <= 1e-12), axis=1)


def _fast_rank_metrics(Yt, Yp, Rt, Rp, valid) -> dict[str, np.ndarray]:
    """Bağsız satırlarda NDCG@3, Recall@3, Regret@1/3 ve kalıntı Spearman'ı satır-vektörel hesaplar."""
    n, d = Yt.shape
    rows = np.arange(n)[:, None]
    disc = 1.0 / np.log2(np.arange(2, 5, dtype=np.float64))

    def order(V):
        return np.argsort(np.where(valid, V, np.inf), axis=1, kind="mergesort")

    def ndcg(T, P):
        tv = np.where(valid, T, np.nan)
        rel = np.nanmax(tv, axis=1)[:, None] - tv
        idcg = np.sum(np.take_along_axis(rel, order(T)[:, :3], 1) * disc, axis=1)
        dcg = np.sum(np.take_along_axis(rel, order(P)[:, :3], 1) * disc, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(idcg <= 1e-12, np.nan, dcg / idcg)

    def ranks(V):
        r = np.zeros((n, d), dtype=np.float64)
        r[rows, order(V)] = np.arange(1, d + 1, dtype=np.float64)
        return np.where(valid, r, 0.0)

    ot, op = order(Yt), order(Yp)
    best = np.take_along_axis(Yt, ot[:, :1], 1)[:, 0]
    yt_top3p = np.take_along_axis(Yt, op[:, :3], 1)
    in_t = np.zeros((n, d), dtype=bool)
    in_p = np.zeros((n, d), dtype=bool)
    in_t[rows, ot[:, :3]] = True
    in_p[rows, op[:, :3]] = True

    cnt = valid.sum(axis=1)
    rx, ry = ranks(Rt), ranks(Rp)
    mid = ((cnt + 1) / 2.0)[:, None]
    rx_c, ry_c = np.where(valid, rx - mid, 0.0), np.where(valid, ry - mid, 0.0)
    denom = np.sqrt(np.sum(rx_c * rx_c, axis=1) * np.sum(ry_c * ry_c, axis=1))
    return {
        "patient_res_spearman": np.sum(rx_c * ry_c, axis=1) / denom,
        "patient_ndcg3": ndcg(Yt, Yp),
        "patient_res_ndcg3": ndcg(Rt, Rp),
        "patient_recall3": np.sum(in_t & in_p, axis=1) / 3.0,
        "patient_regret1": yt_top3p[:, 0] - best,
        "patient_regret3": np.min(yt_top3p, axis=1) - best,
    }


def _patient_metric_vectors(Yt, Yp, Bd, min_targets: int) -> dict[str, np.ndarray]:
    """Hasta başı metrik vektörleri; bağlı satırlar tanımsal skaler fonksiyonlara düşer."""
    mask = np.isfinite(Yt) & np.isfinite(Yp)
    cnt = mask.sum(axis=1)
    n = Yt.shape[0]
    vecs = {k: np.full(n, np.nan, dtype=np.float64) for k in VECTOR_KEYS}
    has = cnt > 0
    vecs["patient_rmse_sq"][has] = np.sum(np.where(mask, (Yt - Yp) ** 2, 0.0), axis=1)[has] / cnt[has]

    Rt, Rp = Yt - Bd, Yp - Bd
    elig = cnt >= min_targets
    tied = (_rows_with_close_ties(Yt, mask) | _rows_with_close_ties(Yp, mask)
            | _rows_with_close_ties(Rt, mask) | _rows_with_close_ties(Rp, mask))
    fast = elig & ~tied & (min_targets >= 3)  # hızlı yol Top-3 için en az 3 hedef varsayar
    if np.any(fast):
        for key, v in _fast_rank_metrics(Yt[fast], Yp[fast], Rt[fast], Rp[fast], mask[fast]).items():
            vecs[key][fast] = v
    for i in np.flatnonzero(elig & ~fast):
        m = mask[i]
        yt, yp, rt, rp = Yt[i, m], Yp[i, m], Rt[i, m], Rp[i, m]
        vecs["patient_res_spearman"][i] = safe_spearman(rt, rp)
        vecs["patient_ndcg3"][i] = ndcg_at_k(yt, yp, k=3)
        vecs["patient_res_ndcg3"][i] = ndcg_at_k(rt, rp, k=3)
        vecs["patient_recall3"][i] = recall_at_k(yt, yp, k=3)
        vecs["patient_regret1"][i] = regret_at_k(yt, yp, k=1)
        vecs["patient_regret3"][i] = regret_at_k(yt, yp, k=3)
    return vecs


def _per_drug_spearmans(Yt, Yp, mask, fold_ids, min_patients: int) -> list[float]:
    """İlaç bazlı Spearman; fold_ids verilirse kat-içi hesaplanıp ortalanır (kat-arası b_d yanlılığını önler)."""
    groups = [np.ones(len(Yt), dtype=bool)] if fold_ids is None else [fold_ids == f for f in np.unique(fold_ids)]
    out: list[float] = []
    for j in range(Yt.shape[1]):
        vals = []
        for g in groups:
            m = g & mask[:, j]
            if int(m.sum()) >= min_patients:
                sp = safe_spearman(Yt[m, j], Yp[m, j])
                if np.isfinite(sp):
                    vals.append(sp)
        if vals:
            out.append(float(np.mean(vals)))
    return out


def evaluate_masked_predictions(
    Y_true_all: np.ndarray,
    Y_pred_all: np.ndarray,
    base_d_all: np.ndarray,
    eval_mask: np.ndarray,
    target_idx: list[int],
    fold_ids: np.ndarray | None = None,
    subjects: np.ndarray | None = None,
    min_targets_per_patient: int = 5,
    min_patients_per_drug: int = 10,
) -> dict[str, Any]:
    """Protokol §7 + Sprint 3 §3 metrikleri (Sprint 2/3 koşucularının ortak değerlendiricisi).

    `rmse` hasta-ağırlıklı (hasta MSE ortalamasının kökü), `obs_rmse` gözlenen (p, d) çiftleri üzerinde havuzlanmış.
    `fold_ids`/`subjects` satır düzeyindedir (eval_mask uygulanmadan önceki uzunlukta).
    """
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = Y_true_all[eval_mask][:, t_idx]
    Yp = Y_pred_all[eval_mask][:, t_idx]
    Bd = np.broadcast_to(base_d_all, Y_true_all.shape)[eval_mask][:, t_idx]
    mask = np.isfinite(Yt) & np.isfinite(Yp)
    n_obs = int(mask.sum())
    if n_obs == 0:
        raise ValueError("Değerlendirme için sonlu gözlem yok.")

    ss_res = float(np.sum((Yt[mask] - Yp[mask]) ** 2))
    ss_tot = float(np.sum((Yt - Bd)[mask] ** 2))
    vecs = _patient_metric_vectors(Yt, Yp, Bd, min_targets_per_patient)
    drug_sps = _per_drug_spearmans(Yt, Yp, mask, None if fold_ids is None else fold_ids[eval_mask], min_patients_per_drug)

    with warnings.catch_warnings():
        # Uygun hasta/hedef yoksa sıralama metrikleri bilinçli olarak NaN (OPUS-022).
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        if subjects is not None:  # çok-örnekli duyarlılık: hasta-küme ortalaması
            sub = subjects[eval_mask]
            uniq = np.unique(sub)
            vecs = {k: np.array([np.nanmean(v[sub == s]) for s in uniq], dtype=np.float64) for k, v in vecs.items()}

        return {
            "n_obs": n_obs,
            "n_eval": int(len(vecs["patient_rmse_sq"])),
            "rmse": float(np.sqrt(np.nanmean(vecs["patient_rmse_sq"]))),
            "obs_rmse": float(np.sqrt(ss_res / n_obs)),
            "ss_res": ss_res,
            "ss_tot_res": ss_tot,
            "residual_r2": 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot,
            "within_patient_residual_spearman": float(np.nanmean(vecs["patient_res_spearman"])),
            "per_drug_spearman": float(np.mean(drug_sps)) if drug_sps else float("nan"),
            "n_drugs_spearman": len(drug_sps),
            "drug_spearmans": np.asarray(drug_sps, dtype=np.float64),
            "ndcg_at_3": float(np.nanmean(vecs["patient_ndcg3"])),
            "residual_ndcg_at_3": float(np.nanmean(vecs["patient_res_ndcg3"])),
            "recall_at_3": float(np.nanmean(vecs["patient_recall3"])),
            "regret_at_1": float(np.nanmean(vecs["patient_regret1"])),
            "regret_at_3": float(np.nanmean(vecs["patient_regret3"])),
            "vectors": vecs,
        }


def evaluate_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_idx: list[int],
    mu_tr: float,
    b_d_tr: np.ndarray,
    min_targets_per_patient: int = 5,
    min_patients_per_drug: int = 10,
) -> dict[str, Any]:
    """Tek bölme API'si: `rmse` burada havuzlanmış gözlem RMSE'sidir (geriye dönük uyumluluk)."""
    ev = evaluate_masked_predictions(
        Y_true, Y_pred, (mu_tr + b_d_tr)[None, :], np.ones(len(Y_true), dtype=bool), target_idx,
        min_targets_per_patient=min_targets_per_patient, min_patients_per_drug=min_patients_per_drug,
    )
    return {**ev, "rmse": ev["obs_rmse"], "n_patients": ev["n_eval"], "patient_vectors": ev["vectors"]}


def average_metric_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aynı hasta sırasına hizalı değerlendirmeleri (tekrar/çekiliş/seed) hasta başına ortalayıp özetler."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)  # tümü-NaN hasta sütunu -> NaN (bilinçli)
        v = {k: np.nanmean(np.stack([nd["vectors"][k] for nd in nodes], axis=0), axis=0) for k in VECTOR_KEYS}
    return {
        "n_eval": int(len(v["patient_rmse_sq"])),
        "rmse": float(np.sqrt(np.nanmean(v["patient_rmse_sq"]))),
        "obs_rmse": float(np.mean([nd["obs_rmse"] for nd in nodes])),
        "residual_r2": float(np.mean([nd["residual_r2"] for nd in nodes])),
        "within_patient_residual_spearman": float(np.nanmean(v["patient_res_spearman"])),
        "per_drug_spearman": float(np.mean([nd["per_drug_spearman"] for nd in nodes])),
        "ndcg_at_3": float(np.nanmean(v["patient_ndcg3"])),
        "residual_ndcg_at_3": float(np.nanmean(v["patient_res_ndcg3"])),
        "recall_at_3": float(np.nanmean(v["patient_recall3"])),
        "regret_at_1": float(np.nanmean(v["patient_regret1"])),
        "regret_at_3": float(np.nanmean(v["patient_regret3"])),
        "vectors": v,
    }


def paired_bootstrap_ci(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 2026,
    agg_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, float | int]:
    """Ortak sonlu hastalarda A - B farkı için eşleştirilmiş bootstrap %95 GA hesaplar."""
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_paired": 0,
                "p_two_sided": float("nan")}

    av = a[mask]
    bv = b[mask]
    fn = agg_fn if agg_fn is not None else (lambda v: float(np.mean(v)))
    point = fn(av) - fn(bv)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b_idx] = fn(av[idx]) - fn(bv[idx])

    # İki yönlü yüzdelik-bootstrap p-değeri (H0: fark = 0); (sayım + 1) / (B + 1) ile sıfırdan kaçınılır
    p_two = 2.0 * min(np.sum(diffs <= 0.0) + 1, np.sum(diffs >= 0.0) + 1) / (n_boot + 1)
    return {
        "diff": float(point),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_paired": n,
        "p_two_sided": float(min(1.0, p_two)),
    }


def paired_metric_diffs(node_a: dict[str, Any], node_b: dict[str, Any], n_boot: int = 1000) -> dict[str, Any]:
    """A - B için 7 hasta-düzeyi metrikte eşleştirilmiş bootstrap farkları (RMSE kök-ortalama ölçeğinde)."""
    va, vb = node_a["vectors"], node_b["vectors"]
    rmse_fn = lambda v: float(np.sqrt(np.mean(v)))
    pairs = (
        ("rmse_diff", "patient_rmse_sq"),
        ("res_spearman_diff", "patient_res_spearman"),
        ("ndcg3_diff", "patient_ndcg3"),
        ("res_ndcg3_diff", "patient_res_ndcg3"),
        ("recall3_diff", "patient_recall3"),
        ("regret1_diff", "patient_regret1"),
        ("regret3_diff", "patient_regret3"),
    )
    return {
        out: paired_bootstrap_ci(va[vk], vb[vk], n_boot=n_boot, agg_fn=rmse_fn if vk == "patient_rmse_sq" else None)
        for out, vk in pairs
    }
