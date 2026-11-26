"""Sprint 3 M3 mekanizma ablasyonları: dondurulmuş 5 koşul (PROTOCOL_SPRINT03 §4) + v2 duyarlılık (OPUS-002, Sol §3.2/4)."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from src.oncotwin.eval.metrics import average_metric_nodes
from src.oncotwin.eval.splits import sample_probe_sets
from src.oncotwin.experiments.sprint03_report import ABLATION_NAMES
from src.oncotwin.models.baselines import CovarianceFewShotM2a, HybridFewShotM3, OmicsRidgeM1

N_OFFSET_FIXED_DRAWS = 20


def _permute_within_patient_probes(
    Y_eval: np.ndarray,
    prior_eval: np.ndarray,
    probes: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Test hastasının kendi seçili k prob kalıntısını ilaç kimlikleri arasında karıştırır."""
    rng = np.random.default_rng(seed)
    Y_mod = Y_eval.copy()
    for i, s in enumerate(probes):
        if len(s) <= 1:
            continue
        r_s = Y_eval[i, s] - prior_eval[i, s]
        perm = rng.permutation(len(s))
        Y_mod[i, s] = prior_eval[i, s] + r_s[perm]
    return Y_mod


def _permute_across_patient_probes(
    Y_eval: np.ndarray,
    prior_eval: np.ndarray,
    probes: list[np.ndarray],
    eval_mask_te: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Değerlendirilebilir test hastalarının prob kalıntı vektörlerini hastalar arasında değiştirir."""
    rng = np.random.default_rng(seed)
    Y_mod = Y_eval.copy()
    idx_eval = np.flatnonzero(eval_mask_te)
    if len(idx_eval) <= 1:
        return Y_mod
    # Kaydırma + rastgele permütasyon ile hiç kimsenin kendi satırını almamasını garantiye yakın tut
    perm = idx_eval[rng.permutation(len(idx_eval))]
    for pos, p_i in enumerate(idx_eval):
        q_i = int(perm[pos])
        if q_i == p_i:
            q_i = int(perm[(pos + 1) % len(idx_eval)])
        s_p = probes[p_i]
        s_q = probes[q_i]
        if len(s_p) == len(s_q) and len(s_p) > 0:
            r_q = Y_eval[q_i, s_q] - prior_eval[q_i, s_q]
            Y_mod[p_i, s_p] = prior_eval[p_i, s_p] + r_q
    return Y_mod


def permute_offset_fixed_standardized(
    Y_eval: np.ndarray,
    prior_eval: np.ndarray,
    probes: list[np.ndarray],
    scale: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Hasta ofsetini (prob artık ortalaması) sabit tutup eğitim ölçeğiyle standardize edilmiş
    merkezlenmiş artıkları özdeş olmayan bir permütasyonla karıştırır (Sol §3.2/4)."""
    rng = np.random.default_rng(seed)
    Y_mod = Y_eval.copy()
    for i, s in enumerate(probes):
        if len(s) <= 1:
            continue
        r = Y_eval[i, s] - prior_eval[i, s]
        offset = float(np.mean(r))
        z = (r - offset) / scale[s]
        perm = rng.permutation(len(s))
        while np.array_equal(perm, np.arange(len(s))):
            perm = rng.permutation(len(s))
        Y_mod[i, s] = prior_eval[i, s] + offset + z[perm] * scale[s]
    return Y_mod


class OffsetOnlyM3:
    """M1 öncülü + gözlenen prob artıklarından daraltılmış tek hasta ofseti; tüm hedeflere eşit eklenir.

    Daraltma lambda'sı her k için yalnız eğitim katının M1 OOF artıklarıyla seçilir. İlaç ölçeği
    kullanılmaz (eşit ağırlık); ilaca özgü kovaryans bilgisi bilinçli olarak yoktur.
    """

    LAMBDA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

    def __init__(self) -> None:
        self.lam_by_k_: dict[int, float] = {}

    @staticmethod
    def _offset_parts(R: np.ndarray, probe_sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        sums = np.array([float(np.sum(R[i, s])) for i, s in enumerate(probe_sets)], dtype=np.float64)
        if not np.all(np.isfinite(sums)):
            raise ValueError("OffsetOnlyM3: seçili problarda sonlu olmayan artık var.")
        return sums, np.array([len(s) for s in probe_sets], dtype=np.float64)

    def fit(
        self,
        E_train: np.ndarray,
        Y_train: np.ndarray,
        probe_pool_idx: list[int],
        target_idx: list[int],
        k_values: tuple[int, ...],
        seed: int,
    ) -> OffsetOnlyM3:
        E_t = E_train[:, np.asarray(target_idx, dtype=np.int64)]
        obs = np.isfinite(E_t)
        for k in k_values:
            losses = {lam: [] for lam in self.LAMBDA_GRID}
            for rep in sample_probe_sets(Y_train, probe_pool_idx, k=k, n_repeats=3, seed=seed + 31 * k):
                sums, cnts = self._offset_parts(E_train, rep)
                for lam in self.LAMBDA_GRID:
                    losses[lam].append((E_t - (sums / (cnts + lam))[:, None])[obs] ** 2)
            self.lam_by_k_[k] = min(self.LAMBDA_GRID, key=lambda lam: float(np.mean(np.concatenate(losses[lam]))))
        return self

    def predict(
        self,
        Y_eval: np.ndarray,
        prior_eval: np.ndarray,
        probe_sets: list[np.ndarray],
        target_idx: list[int],
        k: int,
    ) -> np.ndarray:
        sums, cnts = self._offset_parts(Y_eval - prior_eval, probe_sets)
        out = prior_eval.copy()
        out[:, np.asarray(target_idx, dtype=np.int64)] += (sums / (cnts + self.lam_by_k_[k]))[:, None]
        return out


def fit_ablation_models(
    m1: OmicsRidgeM1,
    Y_oof: np.ndarray,
    Y_tr: np.ndarray,
    subj_tr: np.ndarray,
    probe_pool_idx: list[int],
    target_idx: list[int],
    fit_drugs: list[int],
    k_values: tuple[int, ...],
    inner_seed: int,
) -> dict[str, Any]:
    """M3_no_omics (M2a), M3_perm_train_cov ve v2 M3_offset_only modellerini yalnız eğitim katında fit eder."""
    m2a = CovarianceFewShotM2a().fit(Y_tr, subj_tr, probe_pool_idx, target_idx, k_values=k_values, inner_seed=inner_seed)
    # M3_perm_train_cov: eğitimde M1 OOF kalıntılarını her sütunda bağımsız permüte et
    rng_cov = np.random.default_rng(inner_seed + 4444)
    E_oof = Y_tr - Y_oof
    E_perm = E_oof.copy()
    for col in fit_drugs:
        obs_r = np.flatnonzero(np.isfinite(E_perm[:, col]))
        if len(obs_r) > 1:
            E_perm[obs_r, col] = E_perm[rng_cov.permutation(obs_r), col]
    m3_perm_cov = HybridFewShotM3().fit(
        m1, Y_oof, Y_oof + E_perm, subj_tr, probe_pool_idx, target_idx, k_values=k_values, inner_seed=inner_seed,
    )
    offset_only = OffsetOnlyM3().fit(E_oof, Y_tr, probe_pool_idx, target_idx, k_values, seed=inner_seed + 5555)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fit dışı (tümü-NaN) sütunlar -> NaN, kullanılmaz
        train_resid_sd = np.nanstd(E_oof, axis=0)
    return {"M2a": m2a, "M3_perm_cov": m3_perm_cov, "M3_offset_only": offset_only, "train_resid_sd": train_resid_sd}


def run_ablations_for_k(
    ds: Any,
    fitted_splits: list[tuple],
    k: int,
    target_idx: list[int],
    static_probes_by_split: list[list[np.ndarray]],
    rand_probes_by_rep_split: list[list[list[np.ndarray]]],
    random50_node: dict[str, Any],
    evaluate: Callable[[np.ndarray], dict[str, Any]],
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bir k için (a) dondurulmuş 5 ablasyonu StaticDOpt ve Random50 panellerinde, (b) v2 duyarlılık
    ablasyonlarını (offset-only, ofset-sabit standardize permütasyon x20) üretir."""
    n_rows, n_drugs = ds.Y_auc.shape

    def predict_all(fn: Callable[[int, tuple], np.ndarray]) -> np.ndarray:
        pred = np.full((n_rows, n_drugs), np.nan, dtype=np.float64)
        for f_idx, split in enumerate(fitted_splits):
            pred[split[0]] = fn(f_idx, split)
        return pred

    def m3(split, Y, probes):
        te_idx, fit_obj, _, m1_pred_te, _ = split
        return fit_obj["M3"].predict(ds.X_rna[te_idx], ds.X_mut[te_idx], Y, probes, target_idx, k=k, prior_eval=m1_pred_te)

    def ablation_pred(name: str, f_idx: int, split: tuple, probes: list[np.ndarray], perm_seeds: tuple[int, int]):
        te_idx, fit_obj, eval_mask_te, m1_pred_te, _ = split
        Y_te = ds.Y_auc[te_idx]
        if name == "M3_full":
            return m3(split, Y_te, probes)
        if name == "M3_no_omics":
            return fit_obj["M2a"].predict(Y_te, probes, target_idx, k=k)
        if name == "M3_perm_within_patient":
            return m3(split, _permute_within_patient_probes(Y_te, m1_pred_te, probes, seed=perm_seeds[0]), probes)
        if name == "M3_perm_across_patients":
            Y_mod = _permute_across_patient_probes(Y_te, m1_pred_te, probes, eval_mask_te, seed=perm_seeds[1])
            return m3(split, Y_mod, probes)
        if name == "M3_perm_train_cov":
            return fit_obj["M3_perm_cov"].predict(ds.X_rna[te_idx], ds.X_mut[te_idx], Y_te, probes, target_idx, k=k, prior_eval=m1_pred_te)
        if name == "M3_offset_only":
            return fit_obj["M3_offset_only"].predict(Y_te, m1_pred_te, probes, target_idx, k=k)
        raise ValueError(f"Bilinmeyen ablasyon: {name}")

    # (a) StaticDOpt panelleri (v1 tohumlarıyla birebir)
    abl_static = {
        name: evaluate(predict_all(lambda f, sp, n=name: ablation_pred(
            n, f, sp, static_probes_by_split[f], (seed_offset + k * 19 + f, seed_offset + k * 23 + f))))
        for name in ABLATION_NAMES
    }

    # (b) Random 50x panelleri (M3_full = Random50 politikası; v1 tohumlarıyla birebir)
    def rand_node(name: str) -> dict[str, Any]:
        return average_metric_nodes([
            evaluate(predict_all(lambda f, sp, n=name, r=r: ablation_pred(
                n, f, sp, rand_probes_by_rep_split[r][f],
                (seed_offset * 10 + k * 19 + f * 100 + r, seed_offset * 10 + k * 23 + f * 100 + r))))
            for r in range(len(rand_probes_by_rep_split))
        ])

    abl_rand = {name: random50_node if name == "M3_full" else rand_node(name) for name in ABLATION_NAMES}

    # (c) v2 duyarlılık: offset-only (iki panel kaynağı) ve ofset-sabit standardize permütasyon (StaticDOpt, 20 çekiliş)
    def offset_fixed_draw(d: int) -> dict[str, Any]:
        def fn(f_idx, split):
            te_idx, fit_obj, _, m1_pred_te, _ = split
            Y_mod = permute_offset_fixed_standardized(
                ds.Y_auc[te_idx], m1_pred_te, static_probes_by_split[f_idx], fit_obj["train_resid_sd"],
                seed=seed_offset * 1000 + k * 37 + f_idx * 101 + d * 7919,
            )
            return m3(split, Y_mod, static_probes_by_split[f_idx])
        return evaluate(predict_all(fn))

    draws = [offset_fixed_draw(d) for d in range(N_OFFSET_FIXED_DRAWS)]
    abl_v2 = {
        "StaticDOpt/M3_offset_only": evaluate(predict_all(lambda f, sp: ablation_pred(
            "M3_offset_only", f, sp, static_probes_by_split[f], (0, 0)))),
        "Random50/M3_offset_only": rand_node("M3_offset_only"),
        "StaticDOpt/M3_perm_offset_fixed": average_metric_nodes(draws),
        "draws": draws,
    }
    return {"StaticDOpt": abl_static, "Random50": abl_rand}, abl_v2
