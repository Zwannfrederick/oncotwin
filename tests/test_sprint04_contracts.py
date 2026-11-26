"""Sprint 4 sızıntı/izolasyon sözleşmesi — doğrudan prob kimliği ve tahmin matrisi kontrolleri (DEV-OPUS-17, OPUS-020).

`test_sprint04.py` özet skorları sınar; bu dosya tüm politikalarda seçilen prob kimliklerini ve `Y_pred`'i
doğrudan karşılaştırır (Sol §8.1 talebi) ve SOL-09 için kabul testini içerir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.oncotwin.experiments.run_sprint04 as rs4
from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.splits import get_evaluable_mask, make_sprint04_splits
from src.oncotwin.models.panel_selection import compute_contender_weights
from tests.test_sprint02 import _synthetic_cohort
from tests.test_sprint04 import _to_dataset

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

POLICIES = ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED")
TARGETS = [0, 1, 2, 3, 4]


def _with_y(ds: BeatAMLDataset, Y: np.ndarray) -> BeatAMLDataset:
    return BeatAMLDataset(
        meta=ds.meta, Y_auc=Y, X_rna=ds.X_rna, X_mut=ds.X_mut, drug_names=ds.drug_names,
        gene_names=ds.gene_names, mut_feature_names=ds.mut_feature_names,
        w12_all_subjects=ds.w12_all_subjects, both_all_subjects=ds.both_all_subjects,
    )


def _setup(seed: int = 29):
    ds = _to_dataset(_synthetic_cohort(n_patients=60, n_genes=80, n_drugs=12, seed=seed))
    part = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=2026)["id_60_20_20"]
    fit = rs4._fit_sprint04_fold(ds, part["train_tune"], TARGETS, k_budgets=(2,), inner_seed=2026, min_train_obs=10)
    return ds, part, fit


def _panels_and_preds(ds: BeatAMLDataset, te_idx: np.ndarray, fit: dict, policy: str, k: int):
    sub = ds.subset_by_positions(te_idx)
    m1 = fit["M1"].predict(sub.X_rna, sub.X_mut, n_drugs=ds.Y_auc.shape[1])
    cov_cc = fit["cov_ref"][np.ix_(fit["target_local"], fit["target_local"])]
    cont = compute_contender_weights(m1[:, TARGETS], cov_cc, top_k=3, n_mc=500, seed=2028)
    mask = get_evaluable_mask(sub.Y_auc, TARGETS, fit["probe_pool_idx"], 3, 3)
    uni = np.full(len(TARGETS), 1.0 / len(TARGETS))
    probes = rs4._select_archive_panels(sub.Y_auc, fit, cont, policy, k, mask, uni, set(TARGETS))
    y_pred, _ = rs4._predict_arm(sub, probes, fit, m1, TARGETS, k=k, complete_mask=mask)
    return [p.tolist() for p in probes], y_pred


# (b) Gizli test hedef AUC'leri permüte edilip gürültülendiğinde tüm politikalarda prob kimlikleri ve Y_pred aynı.
@pytest.mark.parametrize("policy", POLICIES)
def test_archive_probes_and_predictions_ignore_hidden_test_targets(policy: str) -> None:
    ds, part, fit = _setup()
    te = part["test"]
    Y = ds.Y_auc.copy()
    rng = np.random.default_rng(777)
    for j in TARGETS:
        fin = np.flatnonzero(np.isfinite(Y[te, j]))
        Y[te[fin], j] = rng.permutation(Y[te[fin], j]) + rng.normal(0.0, 500.0, size=len(fin))
    p0, y0 = _panels_and_preds(ds, te, fit, policy, 2)
    p1, y1 = _panels_and_preds(_with_y(ds, Y), te, fit, policy, 2)
    assert p0 == p1
    np.testing.assert_array_equal(y0, y1)


# (d) Kilitli katalog: test mevcudiyet maskesi değişince ön-seçilen panel değişmez (adaptif dahil);
#     eksik prob geri doldurulmaz (tam olmayan hastaya boş panel).
@pytest.mark.parametrize("policy", POLICIES)
def test_locked_panels_ignore_test_mask_and_never_backfill(policy: str) -> None:
    ds, part, fit = _setup()
    te = part["test"]
    sub = ds.subset_by_positions(te)
    m1 = fit["M1"].predict(sub.X_rna, sub.X_mut, n_drugs=ds.Y_auc.shape[1])
    cov_cc = fit["cov_ref"][np.ix_(fit["target_local"], fit["target_local"])]
    cont = compute_contender_weights(m1[:, TARGETS], cov_cc, top_k=3, n_mc=500, seed=2028)
    uni = np.full(len(TARGETS), 1.0 / len(TARGETS))
    locked = rs4._choose_locked_panels(fit, cont, policy, 2, uni, set(TARGETS))

    Y_drop = sub.Y_auc.copy()
    Y_drop[::2, fit["probe_pool_idx"]] = np.nan  # çift satırlarda tüm problar ölçülmemiş
    for Y_eval in (sub.Y_auc, Y_drop):
        probes, complete, _ = rs4._open_archive_mask_for_locked(locked, Y_eval, 2)
        for i, p in enumerate(probes):
            if complete[i]:
                assert p.tolist() == locked[i].tolist()
            else:
                assert len(p) == 0
    assert not np.any(rs4._open_archive_mask_for_locked(locked, Y_drop, 2)[1][::2])


def _clone_cal_into_test(ds: BeatAMLDataset, part: dict) -> tuple[BeatAMLDataset, dict]:
    """Kalibrasyon hastalarının omik ve prob-mevcudiyet ikizlerini yeni hasta kimlikleriyle test kümesi yapar."""
    cal = part["calibration"]
    n = len(ds.meta)
    meta_clone = ds.meta.iloc[cal].copy()
    meta_clone["dbgap_subject_id"] = ds.meta["dbgap_subject_id"].max() + 1 + np.arange(len(cal))
    meta = pd.concat([ds.meta, meta_clone], ignore_index=True)
    ds2 = BeatAMLDataset(
        meta=meta, Y_auc=np.vstack([ds.Y_auc, ds.Y_auc[cal]]), X_rna=np.vstack([ds.X_rna, ds.X_rna[cal]]),
        X_mut=np.vstack([ds.X_mut, ds.X_mut[cal]]), drug_names=ds.drug_names, gene_names=ds.gene_names,
        mut_feature_names=ds.mut_feature_names, w12_all_subjects=ds.w12_all_subjects,
        both_all_subjects=ds.both_all_subjects,
    )
    return ds2, {"train_tune": part["train_tune"], "calibration": cal, "test": n + np.arange(len(cal))}


# SOL-09 / OPUS-017 kabul testi (DEV-OPUS-13): aynı omik profil + aynı mevcudiyet maskesi -> kalibrasyonda ve
# testte aynı panel. DEV-OPUS-13 ile kapatıldı.
def test_selector_is_same_fixed_function_for_calibration_and_test(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = _to_dataset(_synthetic_cohort(n_patients=90, n_genes=80, n_drugs=30, seed=31))
    part = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=2026)["id_60_20_20"]
    ds2, split = _clone_cal_into_test(ds, part)
    cal_ids = tuple(ds2.meta["dbgap_subject_id"].to_numpy()[split["calibration"]])
    te_ids = tuple(ds2.meta["dbgap_subject_id"].to_numpy()[split["test"]])

    calls: dict[tuple, list] = {cal_ids: [], te_ids: []}
    orig = rs4._predict_arm

    def spy(ds_sub, probes, *args, **kwargs):
        ids = tuple(ds_sub.meta["dbgap_subject_id"].to_numpy())
        if ids in calls:
            calls[ids].append([p.tolist() for p in probes])
        return orig(ds_sub, probes, *args, **kwargs)

    weights: dict[bytes, list[np.ndarray]] = {}
    orig_w = rs4.compute_contender_weights

    def spy_w(prior, *args, **kwargs):
        w = orig_w(prior, *args, **kwargs)
        for row, w_row in zip(prior, w):
            weights.setdefault(row.tobytes(), []).append(w_row)
        return w

    monkeypatch.setattr(rs4, "_predict_arm", spy)
    monkeypatch.setattr(rs4, "compute_contender_weights", spy_w)
    rs4.run_regime_sprint04(
        ds2, split, list(range(10)), regime_name="clone", seed=2026, min_train_obs=10, min_targets=3,
        min_probes=3, k_budgets=(5,), inner_n_splits=3, n_boot=20,
    )
    assert len(calls[cal_ids]) == len(calls[te_ids]) > 0
    for cal_panels, te_panels in zip(calls[cal_ids], calls[te_ids]):
        assert cal_panels == te_panels
    # Panel çevrilmese bile seçicinin girdisi (hasta ağırlıkları) ikizlerde birebir aynı olmalı.
    twins = [ws for ws in weights.values() if len(ws) >= 2]
    assert twins
    for ws in twins:
        for w in ws[1:]:
            np.testing.assert_array_equal(w, ws[0])
