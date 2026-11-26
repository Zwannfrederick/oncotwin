"""Sprint 4 belirsizlik, hasta-birimli split-conformal kalibrasyon, kısa aday listesi ve çekinme modülü."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.oncotwin.eval.metrics import paired_bootstrap_ci
from src.oncotwin.models.baselines import _build_omics_features


class OmicsPCADistanceScorer:
    """Yalnızca eğitim omik özellik uzayında fit edilen <=20 bileşenli standardize PCA merkez uzaklığı."""

    def __init__(self, top_k_genes: int = 2000, max_components: int = 20) -> None:
        self.top_k_genes = top_k_genes
        self.max_components = max_components
        self.gene_idx_: np.ndarray = np.array([], dtype=np.int64)
        self.rna_mean_: np.ndarray = np.array([])
        self.rna_std_: np.ndarray = np.array([])
        self.mut_mean_: np.ndarray = np.array([])
        self.mut_std_: np.ndarray = np.array([])
        self.feat_mean_: np.ndarray = np.array([])
        self.components_: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.pc_std_: np.ndarray = np.array([], dtype=np.float64)

    def fit(self, X_rna_tr: np.ndarray, X_mut_tr: np.ndarray) -> OmicsPCADistanceScorer:
        X_tr, _, gene_idx, rna_mean, rna_std, mut_mean, mut_std = _build_omics_features(
            X_rna_tr, X_mut_tr, top_k_genes=self.top_k_genes
        )
        self.gene_idx_ = gene_idx
        self.rna_mean_ = rna_mean
        self.rna_std_ = rna_std
        self.mut_mean_ = mut_mean
        self.mut_std_ = mut_std

        self.feat_mean_ = np.mean(X_tr, axis=0)
        Xc = X_tr - self.feat_mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        n_pos = int(np.sum(s > 1e-8))
        m = min(self.max_components, n_pos)
        if m == 0:
            self.components_ = np.empty((0, X_tr.shape[1]), dtype=np.float64)
            self.pc_std_ = np.array([], dtype=np.float64)
        else:
            denom = math.sqrt(max(X_tr.shape[0] - 1, 1))
            self.components_ = vt[:m].copy()
            self.pc_std_ = np.maximum(s[:m] / denom, 1e-6)
        return self

    def score(self, X_rna_te: np.ndarray, X_mut_te: np.ndarray) -> np.ndarray:
        if len(self.gene_idx_) == 0:
            raise RuntimeError("OmicsPCADistanceScorer fit edilmeden score çağrılamaz.")
        if self.components_.shape[0] == 0:
            return np.zeros(X_rna_te.shape[0], dtype=np.float64)
        Xr = (X_rna_te[:, self.gene_idx_].astype(np.float64) - self.rna_mean_) / self.rna_std_
        Xm = (X_mut_te.astype(np.float64) - self.mut_mean_) / self.mut_std_
        X_te = np.hstack([Xr, Xm])
        proj = (X_te - self.feat_mean_) @ self.components_.T
        z = proj / self.pc_std_[None, :]
        return np.sqrt(np.mean(z * z, axis=1))


def m3_conditional_posterior_scale(
    cov_psd: np.ndarray,
    probe_sets_local: list[np.ndarray],
    target_local_idx: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Seçilen prob seti ve eğitim kovaryansından 24 hedef üzerinde RMS koşullu posterior SD döndürür.

    Hiçbir test veya kalibrasyon AUC değeri kullanmaz.
    """
    t_idx = np.asarray(target_local_idx, dtype=np.int64)
    base_diag = np.maximum(np.diag(cov_psd)[t_idx], 0.0)
    out = np.empty(len(probe_sets_local), dtype=np.float64)
    for i, S in enumerate(probe_sets_local):
        if len(S) == 0:
            out[i] = float(np.sqrt(np.mean(base_diag)))
            continue
        S_arr = np.asarray(S, dtype=np.int64)
        C_ss = cov_psd[np.ix_(S_arr, S_arr)].copy()
        C_ss[np.diag_indices_from(C_ss)] += lam
        C_ts = cov_psd[np.ix_(t_idx, S_arr)]
        sol = np.linalg.solve(C_ss, C_ts.T)
        diag_cond = base_diag - np.sum(C_ts * sol.T, axis=1)
        out[i] = float(np.sqrt(np.mean(np.maximum(diag_cond, 0.0))))
    return out


def finite_sample_conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int, bool]:
    """Sonlu-örneklem split-conformal sıra istatistiği r = ceil((n+1)(1-alpha)) ve eşik q_hat."""
    n = int(len(scores))
    r = int(math.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or r > n or r <= 0:
        return float("inf"), r, False
    q = float(np.sort(scores)[r - 1])
    return q, r, True


def fit_split_conformal_simultaneous(
    Y_true_cal: np.ndarray,
    Y_pred_cal: np.ndarray,
    target_idx: list[int],
    alpha: float = 0.10,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Hasta-birimli eşzamanlı split-conformal (max |AUC residual|) ve yardımcı ilaç-bazlı kuantil hesaplar."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = Y_true_cal[:, t_idx]
    Yp = Y_pred_cal[:, t_idx]
    finite = np.isfinite(Yt) & np.isfinite(Yp)
    row_ok = np.any(finite, axis=1) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & np.any(finite, axis=1))

    abs_err = np.where(finite, np.abs(Yt - Yp), -1.0)
    cal_scores = np.max(abs_err[row_ok], axis=1).astype(np.float64) if np.any(row_ok) else np.array([], dtype=np.float64)
    q_sim, r_sim, fin_pub = finite_sample_conformal_quantile(cal_scores, alpha)

    q_per_drug = np.full(len(t_idx), float("inf"), dtype=np.float64)
    n_cal_per_drug = np.zeros(len(t_idx), dtype=np.int64)
    row_mask = np.ones(len(Yt), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    for j in range(len(t_idx)):
        m_j = row_mask & finite[:, j]
        n_d = int(np.sum(m_j))
        n_cal_per_drug[j] = n_d
        if n_d > 0:
            q_d, _, _ = finite_sample_conformal_quantile(np.abs(Yt[m_j, j] - Yp[m_j, j]), alpha)
            q_per_drug[j] = q_d

    return {
        "alpha": float(alpha),
        "n_cal": int(len(cal_scores)),
        "order_stat_rank": int(r_sim),
        "q_sim": float(q_sim),
        "width_sim": float(2.0 * q_sim) if fin_pub else float("inf"),
        "finite_interval_published": fin_pub,
        "cal_scores": cal_scores,
        "q_per_drug": q_per_drug,
        "n_cal_per_drug": n_cal_per_drug,
    }


def evaluate_conformal_and_shortlist(
    Y_true_te: np.ndarray,
    Y_pred_te: np.ndarray,
    target_idx: list[int],
    q_sim: float,
    q_per_drug: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Eşzamanlı kapsam, ilaç ampirik kapsamı ve konformal aralık elemesiyle kısa aday listesi hesaplar."""
    t_idx = np.asarray(target_idx, dtype=np.int64)
    Yt = Y_true_te[:, t_idx]
    Yp = Y_pred_te[:, t_idx]
    n_rows, n_c = Yt.shape
    finite = np.isfinite(Yt) & np.isfinite(Yp)
    row_ok = np.any(finite, axis=1) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & np.any(finite, axis=1))
    n_eval = int(np.sum(row_ok))
    fin_pub = math.isfinite(q_sim)

    v_max_err = np.full(n_rows, np.nan, dtype=np.float64)
    v_cov = np.full(n_rows, np.nan, dtype=np.float64)
    v_size_obs = np.full(n_rows, np.nan, dtype=np.float64)
    v_frac_obs = np.full(n_rows, np.nan, dtype=np.float64)
    v_top1_ok = np.full(n_rows, np.nan, dtype=np.float64)
    v_def1_obs = np.full(n_rows, np.nan, dtype=np.float64)
    v_size_24 = np.full(n_rows, np.nan, dtype=np.float64)
    v_def1_24 = np.full(n_rows, np.nan, dtype=np.float64)

    empty_vecs = {
        "patient_max_abs_err": v_max_err,
        "patient_covered_sim": v_cov,
        "patient_shortlist_size_obs": v_size_obs,
        "patient_shortlist_frac_obs": v_frac_obs,
        "patient_shortlist_contains_top1": v_top1_ok,
        "patient_definite_top1_obs": v_def1_obs,
        "patient_shortlist_size_all24": v_size_24,
        "patient_definite_top1_all24": v_def1_24,
    }
    if n_eval == 0:
        return {
            "status": "hesaplanamadı",
            "n_eval": 0,
            "finite_interval_published": fin_pub,
            "shortlist_fallback_all_targets": not fin_pub,
            "q_sim": float(q_sim),
            "width_sim": float(2.0 * q_sim) if fin_pub else float("inf"),
            "simultaneous_coverage": float("nan"),
            "pooled_drug_coverage_sim": float("nan"),
            "per_drug_empirical_coverage_sim": float("nan"),
            "n_drugs_empirical_coverage": 0,
            "mean_shortlist_size_obs": float("nan"),
            "mean_shortlist_frac_obs": float("nan"),
            "shortlist_contains_true_top1_rate": float("nan"),
            "definite_top1_rate_obs": float("nan"),
            "mean_shortlist_size_all24": float("nan"),
            "definite_top1_rate_all24": float("nan"),
            "per_drug_marginal_conformal_coverage": float("nan"),
            "n_drugs_finite_marginal": 0,
            "mean_marginal_width": float("nan"),
            "vectors": empty_vecs,
        }

    pooled_hits: list[np.ndarray] = []
    for i in np.flatnonzero(row_ok):
        obs = finite[i]
        yt_i = Yt[i, obs]
        yp_i = Yp[i, obs]
        err_i = np.abs(yt_i - yp_i)
        max_e = float(np.max(err_i))
        v_max_err[i] = max_e
        if not fin_pub:
            continue

        v_cov[i] = 1.0 if max_e <= q_sim else 0.0
        pooled_hits.append(err_i <= q_sim)

        n_obs_i = int(np.sum(obs))
        L_obs = yp_i - q_sim
        min_U_obs = float(np.min(yp_i + q_sim))
        short_obs = L_obs <= min_U_obs + 1e-12
        sz_obs = int(np.sum(short_obs))
        v_size_obs[i] = float(sz_obs)
        v_frac_obs[i] = sz_obs / float(n_obs_i)
        v_def1_obs[i] = 1.0 if sz_obs == 1 else 0.0
        best_true = float(np.min(yt_i))
        v_top1_ok[i] = 1.0 if np.any(yt_i[short_obs] <= best_true + 1e-12) else 0.0

        yp_fin = Yp[i][np.isfinite(Yp[i])]
        if len(yp_fin) > 0:
            sz_24 = int(np.sum((yp_fin - q_sim) <= float(np.min(yp_fin + q_sim)) + 1e-12))
            v_size_24[i] = float(sz_24)
            v_def1_24[i] = 1.0 if sz_24 == 1 else 0.0

    per_drug_cov_sim: list[float] = []
    per_drug_cov_marg: list[float] = []
    marg_widths: list[float] = []
    for j in range(n_c):
        m_j = row_ok & finite[:, j]
        if np.any(m_j):
            err_j = np.abs(Yt[m_j, j] - Yp[m_j, j])
            if fin_pub:
                per_drug_cov_sim.append(float(np.mean(err_j <= q_sim)))
            if q_per_drug is not None and math.isfinite(float(q_per_drug[j])):
                q_j = float(q_per_drug[j])
                per_drug_cov_marg.append(float(np.mean(err_j <= q_j)))
                marg_widths.append(2.0 * q_j)

    if not fin_pub:
        return {
            "status": "hesaplanamadı (sonlu aralık yayımlanmadı)",
            "n_eval": n_eval,
            "finite_interval_published": False,
            "shortlist_fallback_all_targets": True,
            "q_sim": float(q_sim),
            "width_sim": float("inf"),
            "simultaneous_coverage": float("nan"),
            "pooled_drug_coverage_sim": float("nan"),
            "per_drug_empirical_coverage_sim": float("nan"),
            "n_drugs_empirical_coverage": 0,
            "mean_shortlist_size_obs": float("nan"),
            "mean_shortlist_frac_obs": float("nan"),
            "shortlist_contains_true_top1_rate": float("nan"),
            "definite_top1_rate_obs": float("nan"),
            "mean_shortlist_size_all24": float("nan"),
            "definite_top1_rate_all24": float("nan"),
            "per_drug_marginal_conformal_coverage": float(np.mean(per_drug_cov_marg)) if per_drug_cov_marg else float("nan"),
            "n_drugs_finite_marginal": len(per_drug_cov_marg),
            "mean_marginal_width": float(np.mean(marg_widths)) if marg_widths else float("nan"),
            "vectors": empty_vecs,
        }

    pooled_all = np.concatenate(pooled_hits) if pooled_hits else np.array([], dtype=bool)
    return {
        "status": "ok",
        "n_eval": n_eval,
        "finite_interval_published": True,
        "shortlist_fallback_all_targets": False,
        "q_sim": float(q_sim),
        "width_sim": float(2.0 * q_sim),
        "simultaneous_coverage": float(np.mean(v_cov[row_ok])),
        "pooled_drug_coverage_sim": float(np.mean(pooled_all)) if len(pooled_all) > 0 else float("nan"),
        "per_drug_empirical_coverage_sim": float(np.mean(per_drug_cov_sim)) if per_drug_cov_sim else float("nan"),
        "n_drugs_empirical_coverage": len(per_drug_cov_sim),
        "mean_shortlist_size_obs": float(np.mean(v_size_obs[row_ok])),
        "mean_shortlist_frac_obs": float(np.mean(v_frac_obs[row_ok])),
        "shortlist_contains_true_top1_rate": float(np.mean(v_top1_ok[row_ok])),
        "definite_top1_rate_obs": float(np.mean(v_def1_obs[row_ok])),
        "mean_shortlist_size_all24": float(np.mean(v_size_24[row_ok & np.isfinite(v_size_24)])),
        "definite_top1_rate_all24": float(np.mean(v_def1_24[row_ok & np.isfinite(v_def1_24)])),
        "per_drug_marginal_conformal_coverage": float(np.mean(per_drug_cov_marg)) if per_drug_cov_marg else float("nan"),
        "n_drugs_finite_marginal": len(per_drug_cov_marg),
        "mean_marginal_width": float(np.mean(marg_widths)) if marg_widths else float("nan"),
        "vectors": empty_vecs,
    }


def _step_aurc_from_arrays(
    s_av: np.ndarray,
    l_av: np.ndarray,
    n_total: int,
    tie_tol: float = 1e-12,
    coverage_cap: float | None = None,
) -> dict[str, Any]:
    n_av = len(s_av)
    order = np.argsort(s_av, kind="mergesort")
    s_sorted = s_av[order]
    l_sorted = l_av[order]

    cov_pts: list[float] = []
    risk_pts: list[float] = []
    cum_loss = 0.0
    aurc_raw = 0.0
    aurc_cap_raw = 0.0
    prev_cov = 0.0
    cap_cuts_tie = False
    realizable_cap_below = 0.0
    aurc_realizable_raw = 0.0
    start = 0
    while start < n_av:
        end = start + 1
        while end < n_av and abs(float(s_sorted[end]) - float(s_sorted[start])) <= tie_tol:
            end += 1
        cum_loss += float(np.sum(l_sorted[start:end]))
        cov_m = end / float(n_total)
        risk_m = cum_loss / float(end)
        cov_pts.append(cov_m)
        risk_pts.append(risk_m)
        aurc_raw += (cov_m - prev_cov) * risk_m
        if coverage_cap is not None:
            if prev_cov < coverage_cap:
                aurc_cap_raw += (min(cov_m, coverage_cap) - prev_cov) * risk_m
            if prev_cov + 1e-12 < coverage_cap < cov_m - 1e-12:
                cap_cuts_tie = True
            if cov_m <= coverage_cap + 1e-12:
                realizable_cap_below = cov_m
                aurc_realizable_raw = aurc_raw
        prev_cov = cov_m
        start = end

    max_cov = n_av / float(n_total)
    aurc_norm = aurc_raw / max_cov
    out: dict[str, Any] = {
        "aurc": float(aurc_norm),
        "aurc_raw": float(aurc_raw),
        "max_coverage": float(max_cov),
        "coverages": cov_pts,
        "risks": risk_pts,
    }
    if coverage_cap is not None:
        out["cap_cuts_tie_group"] = cap_cuts_tie
        out["cap_interpretation"] = "kesirli_kabul_varsayımı" if cap_cuts_tie else "gerçekleşebilir_eşik_noktası"
        out["realizable_cap_below"] = float(realizable_cap_below)
        out["aurc_realizable_at_or_below_cap"] = (
            float(aurc_realizable_raw / realizable_cap_below) if realizable_cap_below > 0.0 else float("nan")
        )
        if max_cov + 1e-12 >= coverage_cap and coverage_cap > 0.0:
            out["aurc_capped"] = float(aurc_cap_raw / coverage_cap)
            out["cap_status"] = "ok"
        else:
            out["aurc_capped"] = float("nan")
            out["cap_status"] = "below_cap"
    return out


def selective_risk_coverage_curve(
    scores: np.ndarray,
    losses: np.ndarray,
    available_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
    coverage_cap: float | None = None,
) -> dict[str, Any]:
    """Tie-duyarlı basamaklı Risk–Kapsam eğrisi, AURC, Oracle-AURC ve E-AURC hesaplar."""
    sc = np.asarray(scores, dtype=np.float64)
    lo = np.asarray(losses, dtype=np.float64)
    avail = np.isfinite(sc) & np.isfinite(lo)
    if available_mask is not None:
        avail = avail & np.asarray(available_mask, dtype=bool)
    n_total = int(len(lo))
    n_av = int(np.sum(avail))

    if n_total == 0 or n_av == 0:
        return {
            "status": "hesaplanamadı",
            "n_total": n_total,
            "n_available": 0,
            "max_coverage": 0.0,
            "aurc": float("nan"),
            "aurc_raw": float("nan"),
            "oracle_aurc": float("nan"),
            "e_aurc": float("nan"),
            "random_aurc": float("nan"),
            "aurc_capped": float("nan"),
            "oracle_aurc_capped": float("nan"),
            "e_aurc_capped": float("nan"),
            "coverages": [],
            "risks": [],
        }

    s_av = sc[avail]
    l_av = lo[avail]
    main = _step_aurc_from_arrays(s_av, l_av, n_total, tie_tol=tie_tol, coverage_cap=coverage_cap)
    oracle = _step_aurc_from_arrays(l_av, l_av, n_total, tie_tol=tie_tol, coverage_cap=coverage_cap)
    rand_aurc = float(np.mean(l_av))

    res: dict[str, Any] = {
        "status": "ok",
        "n_total": n_total,
        "n_available": n_av,
        "max_coverage": main["max_coverage"],
        "aurc": main["aurc"],
        "aurc_raw": main["aurc_raw"],
        "oracle_aurc": oracle["aurc"],
        "e_aurc": float(main["aurc"] - oracle["aurc"]),
        "random_aurc": rand_aurc,
        "coverages": main["coverages"],
        "risks": main["risks"],
    }
    if coverage_cap is not None:
        res["coverage_cap"] = float(coverage_cap)
        res["cap_status"] = main["cap_status"]
        res["cap_cuts_tie_group"] = main["cap_cuts_tie_group"]
        res["cap_interpretation"] = main["cap_interpretation"]
        res["realizable_cap_below"] = main["realizable_cap_below"]
        res["aurc_realizable_at_or_below_cap"] = main["aurc_realizable_at_or_below_cap"]
        res["aurc_capped"] = main["aurc_capped"]
        res["oracle_aurc_capped"] = oracle["aurc_capped"]
        res["e_aurc_capped"] = (
            float(main["aurc_capped"] - oracle["aurc_capped"])
            if math.isfinite(main["aurc_capped"]) and math.isfinite(oracle["aurc_capped"])
            else float("nan")
        )
    return res


def evaluate_locked_threshold_abstention(
    scores_te: np.ndarray,
    losses_regret1: np.ndarray,
    losses_rmse_sq: np.ndarray,
    covered_sim: np.ndarray,
    tau_locked: float,
    available_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Doğrulamada kilitlenmiş eşik altında test kabul oranı, kabul/çekinme Regret@1 ve RMSE hesaplar."""
    sc = np.asarray(scores_te, dtype=np.float64)
    reg = np.asarray(losses_regret1, dtype=np.float64)
    rsq = np.asarray(losses_rmse_sq, dtype=np.float64)
    cov = np.asarray(covered_sim, dtype=np.float64)
    avail = np.isfinite(sc) & np.isfinite(reg) & np.isfinite(rsq)
    if available_mask is not None:
        avail = avail & np.asarray(available_mask, dtype=bool)

    n_total = int(len(sc))
    n_av = int(np.sum(avail))
    if n_total == 0 or n_av == 0 or not math.isfinite(float(tau_locked)):
        return {
            "status": "hesaplanamadı",
            "tau_locked": float(tau_locked),
            "n_total": n_total,
            "n_available": n_av,
            "n_technical_abstain": n_total - n_av,
            "n_accepted": 0,
            "n_abstained_score": 0,
            "acceptance_rate_total": float("nan"),
            "acceptance_rate_among_available": float("nan"),
            "accepted_mean_regret1": float("nan"),
            "abstained_mean_regret1": float("nan"),
            "accepted_rmse": float("nan"),
            "abstained_rmse": float("nan"),
            "accepted_simultaneous_coverage_descriptive": float("nan"),
        }

    acc = avail & (sc <= float(tau_locked))
    abst = avail & ~acc
    n_acc = int(np.sum(acc))
    n_abst = int(np.sum(abst))

    cov_acc_mask = acc & np.isfinite(cov)
    return {
        "status": "ok",
        "tau_locked": float(tau_locked),
        "n_total": n_total,
        "n_available": n_av,
        "n_technical_abstain": n_total - n_av,
        "n_accepted": n_acc,
        "n_abstained_score": n_abst,
        "acceptance_rate_total": n_acc / float(n_total),
        "acceptance_rate_among_available": n_acc / float(n_av),
        "accepted_mean_regret1": float(np.mean(reg[acc])) if n_acc > 0 else float("nan"),
        "abstained_mean_regret1": float(np.mean(reg[abst])) if n_abst > 0 else float("nan"),
        "accepted_rmse": float(np.sqrt(np.mean(rsq[acc]))) if n_acc > 0 else float("nan"),
        "abstained_rmse": float(np.sqrt(np.mean(rsq[abst]))) if n_abst > 0 else float("nan"),
        "accepted_simultaneous_coverage_descriptive": (
            float(np.mean(cov[cov_acc_mask])) if np.any(cov_acc_mask) else float("nan")
        ),
    }


def paired_aurc_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray | None,
    losses: np.ndarray,
    n_boot: int = 1000,
    seed: int = 2026,
    available_mask: np.ndarray | None = None,
    tie_tol: float = 1e-12,
    coverage_cap: float | None = None,
) -> dict[str, float | int]:
    """Ortak değerlendirilebilir hastalarda hasta-düzeyi eşleştirilmiş AURC farkı (A - B veya A - random)."""
    sa = np.asarray(scores_a, dtype=np.float64)
    lo = np.asarray(losses, dtype=np.float64)
    mask = np.isfinite(sa) & np.isfinite(lo)
    if scores_b is not None:
        sb = np.asarray(scores_b, dtype=np.float64)
        mask = mask & np.isfinite(sb)
    else:
        sb = None
    if available_mask is not None:
        mask = mask & np.asarray(available_mask, dtype=bool)

    n = int(np.sum(mask))
    if n == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_paired": 0,
                "p_two_sided": float("nan")}

    sav = sa[mask]
    lav = lo[mask]
    sbv = sb[mask] if sb is not None else np.zeros(n, dtype=np.float64)

    def _aurc(s_arr: np.ndarray, l_arr: np.ndarray) -> float:
        res = _step_aurc_from_arrays(s_arr, l_arr, len(l_arr), tie_tol=tie_tol, coverage_cap=coverage_cap)
        return float(res["aurc_capped"]) if coverage_cap is not None else float(res["aurc"])

    point = _aurc(sav, lav) - _aurc(sbv, lav)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b_idx in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b_idx] = _aurc(sav[idx], lav[idx]) - _aurc(sbv[idx], lav[idx])

    p_two = 2.0 * min(np.sum(diffs <= 0.0) + 1, np.sum(diffs >= 0.0) + 1) / (n_boot + 1)
    return {
        "diff": float(point),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_paired": n,
        "p_two_sided": float(min(1.0, p_two)),
    }


def json_sanitize(obj: Any) -> Any:
    """NaN/Infinity değerlerini RFC 8259 JSON uyumlu `None` değerine dönüştürür."""
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    return obj
