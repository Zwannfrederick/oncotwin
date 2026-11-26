"""Sprint 2 sızıntı, protokol uyumu, AUC yönü ve model tutarlılığı testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oncotwin.data.dataset import select_index_specimens
from src.oncotwin.eval.metrics import evaluate_predictions, ndcg_at_k
from src.oncotwin.eval.splits import (
    make_clean_wave_split,
    make_patient_disjoint_kfold,
    select_train_probe_pool,
)
from src.oncotwin.models.baselines import (
    CovarianceFewShotM2a,
    HybridFewShotM3,
    LowRankSVDM2b,
    MeanEffectM0,
    OmicsRidgeM1,
)


def _synthetic_cohort(n_patients: int = 40, n_genes: int = 30, n_drugs: int = 12, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    subjects = np.arange(1, n_patients + 1)
    cohorts = np.where(np.arange(n_patients) < 25, "Waves1+2", "Waves3+4")
    meta = pd.DataFrame(
        {
            "dbgap_subject_id": subjects,
            "dbgap_dnaseq_sample": [f"D{i}" for i in subjects],
            "dbgap_rnaseq_sample": [f"R{i}" for i in subjects],
            "cohort": cohorts,
            "timeOfSampleCollectionRelativeToInclusion": np.zeros(n_patients),
            "diseaseStageAtSpecimenCollection": ["Initial Diagnosis"] * n_patients,
            "specimenType": ["Bone Marrow Aspirate"] * n_patients,
            "specimen_key": [f"D{i}|R{i}" for i in subjects],
        }
    )
    X_rna = rng.normal(0.0, 1.0, size=(n_patients, n_genes)).astype(np.float32)
    X_mut = rng.integers(0, 2, size=(n_patients, 4)).astype(np.float32)
    # Güçlü korele iki bloklu kalıntı yapısı
    z_lat = rng.normal(0.0, 25.0, size=(n_patients, 2))
    loadings = rng.normal(0.0, 1.0, size=(2, n_drugs))
    # İlk 4 ve sonraki 8 ilaç aynı gizil faktörlerden beslensin
    loadings[0, :6] = 1.2
    loadings[1, 6:] = 1.2
    b_d = np.linspace(120.0, 220.0, n_drugs)
    Y = b_d[None, :] + 0.5 * X_rna[:, :1] * 15.0 + z_lat @ loadings + rng.normal(0.0, 3.0, size=(n_patients, n_drugs))
    drug_names = [f"Drug_{j}" for j in range(n_drugs)]
    return {
        "meta": meta,
        "X_rna": X_rna,
        "X_mut": X_mut,
        "Y": Y,
        "drug_names": drug_names,
    }


def test_patient_disjoint_splits_no_leakage() -> None:
    meta = pd.DataFrame(
        {
            "dbgap_subject_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
            "cohort": [
                "Waves1+2", "Waves1+2", "Waves1+2", "Waves1+2", "Waves1+2",
                "Waves3+4", "Waves3+4", "Waves3+4", "Both", "Waves3+4", "Waves1+2", "Waves3+4",
            ],
        }
    )
    folds = make_patient_disjoint_kfold(meta, n_splits=3, seed=42)
    for tr, te in folds:
        tr_s = set(meta.iloc[tr]["dbgap_subject_id"])
        te_s = set(meta.iloc[te]["dbgap_subject_id"])
        assert tr_s.isdisjoint(te_s)

    w12_all = {1, 2, 3, 4, 10}
    both_all = {8}
    tr_w, te_w = make_clean_wave_split(meta, w12_all, both_all)
    te_subj = set(meta.iloc[te_w]["dbgap_subject_id"])
    assert 10 not in te_subj  # Hem W1+2 hem W3+4'te görünen hasta temizlenmeli
    assert 8 not in te_subj   # Both hastası temizlenmeli
    assert te_subj == {5, 6, 7, 9}


def test_index_specimen_deterministic_tiebreak() -> None:
    meta = pd.DataFrame(
        {
            "dbgap_subject_id": [2248, 2248, 3001, 3001],
            "dbgap_dnaseq_sample": [None, "BA2308D", "D1", "D2"],
            "dbgap_rnaseq_sample": ["BA2413R", "BA2308R", "R1", "R2"],
            "timeOfSampleCollectionRelativeToInclusion": [0.0, 0.0, 120.0, 0.0],
            "diseaseStageAtSpecimenCollection": ["Residual", "Residual", "Relapse", "Initial Diagnosis"],
            "specimenType": ["Bone Marrow Aspirate", "Peripheral Blood", "Bone Marrow Aspirate", "Bone Marrow Aspirate"],
            "specimen_key": ["NONE|BA2413R", "BA2308D|BA2308R", "D1|R1", "D2|R2"],
        }
    )
    chosen = select_index_specimens(meta)
    assert len(chosen) == 2
    row_2248 = chosen[chosen["dbgap_subject_id"] == 2248].iloc[0]
    row_3001 = chosen[chosen["dbgap_subject_id"] == 3001].iloc[0]
    assert row_2248["specimen_key"] == "BA2308D|BA2308R"  # DNA+RNA dolu olan seçilmeli
    assert row_3001["specimen_key"] == "D2|R2"            # Daha erken zaman/tanı seçilmeli


def test_ndcg_auc_direction() -> None:
    # Düşük AUC = en duyarlı = en yüksek alaka
    y_true = np.array([50.0, 80.0, 120.0, 200.0, 250.0])
    perfect_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    worst_pred = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    const_pred = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

    assert np.isclose(ndcg_at_k(y_true, perfect_pred, k=3), 1.0)
    assert ndcg_at_k(y_true, worst_pred, k=3) < 0.3
    assert ndcg_at_k(y_true, const_pred, k=3) < 0.75


def test_zero_outside_train_statistics_leakage() -> None:
    d = _synthetic_cohort(n_patients=36, n_genes=25, n_drugs=12, seed=11)
    tr = np.arange(0, 26)
    te = np.arange(26, 36)
    target_idx = [0, 1, 2, 3, 4]
    probe_pool = select_train_probe_pool(d["Y"][tr], d["drug_names"], [d["drug_names"][i] for i in target_idx], min_train_obs=10)
    subj_tr = d["meta"].iloc[tr]["dbgap_subject_id"].to_numpy()
    probes_te = [np.array(probe_pool[:3], dtype=np.int64) for _ in range(len(te))]

    def _run_with_perturbed_other_test(perturb: bool):
        Xr_te = d["X_rna"][te].copy()
        Xm_te = d["X_mut"][te].copy()
        Y_te = d["Y"][te].copy()
        if perturb:
            # Testteki son 9 hastanın tüm özellik ve AUC'lerini uçuk biçimde boz; hasta 0 sabit kalsın
            Xr_te[1:] += 999.0
            Xm_te[1:] = 1.0 - Xm_te[1:]
            Y_te[1:] -= 500.0

        m0 = MeanEffectM0().fit(d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(3,))
        m1, oof = OmicsRidgeM1(top_k_genes=10).fit_and_oof(
            d["X_rna"][tr], d["X_mut"][tr], d["Y"][tr], subj_tr, list(range(12)), target_idx
        )
        m2a = CovarianceFewShotM2a().fit(d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(3,))
        m2b = LowRankSVDM2b(rank_grid=(2, 3)).fit(d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(3,))
        m3 = HybridFewShotM3().fit(m1, oof, d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(3,))

        p0 = m0.predict(Y_te, probes_te, k=3)[0]
        p1 = m1.predict(Xr_te, Xm_te, n_drugs=12)[0]
        p2a = m2a.predict(Y_te, probes_te, target_idx, k=3)[0]
        p2b = m2b.predict(Y_te, probes_te, target_idx, k=3)[0]
        p3 = m3.predict(Xr_te, Xm_te, Y_te, probes_te, target_idx, k=3)[0]
        return (
            m1.fold_model_.gene_idx.copy(),
            m1.fold_model_.rna_mean.copy(),
            m2a.cov_by_k_[3].copy(),
            p0,
            p1,
            p2a,
            p2b,
            p3,
        )

    ref = _run_with_perturbed_other_test(False)
    alt = _run_with_perturbed_other_test(True)
    for a, b in zip(ref, alt):
        assert np.array_equal(a, b)


def test_fair_m0_few_shot_and_m3_identity() -> None:
    d = _synthetic_cohort(n_patients=50, n_genes=20, n_drugs=14, seed=21)
    tr = np.arange(0, 35)
    te = np.arange(35, 50)
    target_idx = [0, 1, 2, 3, 4, 5]
    probe_pool = list(range(6, 14))
    subj_tr = d["meta"].iloc[tr]["dbgap_subject_id"].to_numpy()

    m0 = MeanEffectM0().fit(d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(5,))
    m1, oof = OmicsRidgeM1(top_k_genes=15).fit_and_oof(
        d["X_rna"][tr], d["X_mut"][tr], d["Y"][tr], subj_tr, list(range(14)), target_idx
    )
    m3 = HybridFewShotM3().fit(m1, oof, d["Y"][tr], subj_tr, probe_pool, target_idx, k_values=(5,))

    probes_k0 = [np.array([], dtype=np.int64) for _ in range(len(te))]
    probes_k5 = [np.array(probe_pool[:5], dtype=np.int64) for _ in range(len(te))]

    p_m1 = m1.predict(d["X_rna"][te], d["X_mut"][te], n_drugs=14)
    p_m3_k0 = m3.predict(d["X_rna"][te], d["X_mut"][te], d["Y"][te], probes_k0, target_idx, k=0)
    assert np.allclose(p_m1, p_m3_k0, atol=1e-12)

    p_m0_k5 = m0.predict(d["Y"][te], probes_k5, k=5)
    ev_m0_k5 = evaluate_predictions(d["Y"][te], p_m0_k5, target_idx, m0.mu_, m0.b_d_, min_targets_per_patient=5)
    assert np.isclose(ev_m0_k5["within_patient_residual_spearman"], 0.0, atol=1e-12)

    p_m3_k5 = m3.predict(d["X_rna"][te], d["X_mut"][te], d["Y"][te], probes_k5, target_idx, k=5)
    ev_m3_k5 = evaluate_predictions(d["Y"][te], p_m3_k5, target_idx, m0.mu_, m0.b_d_, min_targets_per_patient=5)
    ev_m1 = evaluate_predictions(d["Y"][te], p_m1, target_idx, m0.mu_, m0.b_d_, min_targets_per_patient=5)
    assert ev_m3_k5["rmse"] < ev_m0_k5["rmse"]
    assert ev_m3_k5["rmse"] < ev_m1["rmse"]


def test_sprint02_results_consistency() -> None:
    res_path = Path("data/processed/sprint02_results.json")
    if not res_path.exists():
        pytest.skip("sprint02_results.json henüz üretilmedi.")

    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert res["permutation_leakage_check"]["clean_wave_perm"]["passed"] is True
    assert res["permutation_leakage_check"]["cv_seed42_perm"]["passed"] is True

    for regime in ("cv_5fold_3seeds", "wave_ood_index"):
        m0_k0 = res[regime]["metrics"]["0"]["M0"]
        assert np.isclose(m0_k0["within_patient_residual_spearman"], 0.0, atol=1e-9)
        assert np.isclose(m0_k0["per_drug_spearman"], 0.0, atol=1e-9)
        for k_str in ("3", "5", "8"):
            m0_k = res[regime]["metrics"][k_str]["M0"]
            assert np.isclose(m0_k["within_patient_residual_spearman"], 0.0, atol=1e-9)
