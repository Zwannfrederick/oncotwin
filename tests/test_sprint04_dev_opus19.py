"""Sprint 4 DEV-OPUS-19 sözleşme ve regresyon testleri (Addendum 19 §3 (a)-(c))."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.splits import make_sprint04_splits
from src.oncotwin.experiments.run_sprint04 import _fit_sprint04_fold
from src.oncotwin.experiments.sprint04_dev_opus19 import (
    compute_final_fit_reference_scores,
    summarize_reference_threshold,
)


def _make_synthetic_dataset(seed: int = 19, n_pat: int = 80) -> BeatAMLDataset:
    rng = np.random.default_rng(seed)
    n_targets, n_probes, n_genes = 6, 12, 24
    n_drugs = n_targets + n_probes
    subj_ids = np.arange(5000, 5000 + n_pat, dtype=np.int64)
    cohorts = np.array(["Waves1+2" if i < 60 else "Waves3+4" for i in range(n_pat)])
    meta = pd.DataFrame({
        "dbgap_subject_id": subj_ids,
        "lab_id": [f"L{i}" for i in range(n_pat)],
        "cohort": cohorts,
    })
    X_rna = rng.normal(0.0, 1.0, size=(n_pat, n_genes))
    X_mut = (rng.uniform(0.0, 1.0, size=(n_pat, 4)) < 0.25).astype(np.float64)
    latent = rng.normal(0.0, 18.0, size=(n_pat, 3))
    load = rng.normal(0.0, 1.0, size=(n_drugs, 3))
    Y = 140.0 + latent @ load.T + 15.0 * (X_rna[:, :1] * load[:, :1].T) + rng.normal(0.0, 6.0, size=(n_pat, n_drugs))
    # Hedeflerde hafif eksiklik ve problarda çeşitlilik oluştur (hem tam hem eksik panel vakaları)
    for i in range(n_pat):
        if i % 9 == 0:
            Y[i, n_targets + (i % n_probes)] = np.nan
        if i % 11 == 0:
            Y[i, n_targets:] = np.nan  # eksik panel / teknik çekinme vakası
    drugs = [f"TGT_{j}" for j in range(n_targets)] + [f"PRB_{j}" for j in range(n_probes)]
    return BeatAMLDataset(
        meta=meta,
        Y_auc=Y,
        X_rna=X_rna,
        X_mut=X_mut,
        drug_names=drugs,
        gene_names=[f"G{g}" for g in range(n_genes)],
        mut_feature_names=[f"MUT{m}" for m in range(4)],
        w12_all_subjects=set(int(s) for s, c in zip(subj_ids, cohorts) if c == "Waves1+2"),
        both_all_subjects=set(),
    )


def test_dev_opus19_contract_c_ties_constant_empty_and_missing_panel() -> None:
    """Sözleşme (c): bağ [1,1,2,2,2], sabit skor, boş referans ve eksik panel paydası."""
    tie_scores = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 9.0, 9.0], dtype=np.float64)
    tie_valid = np.array([True, True, True, True, True, False, False], dtype=bool)
    res_tie = summarize_reference_threshold(tie_scores, tie_valid, n_target_eligible=7, target_accept=0.80)
    assert res_tie["status"] == "ok"
    assert res_tie["is_constant_score"] is False
    assert res_tie["tau_final"] == 2.0
    assert res_tie["rank_1based"] == 5
    assert res_tie["n_ref"] == 5
    assert res_tie["n_unique_scores"] == 2
    assert res_tie["n_ref_below_tau"] == 2
    assert res_tie["n_ref_equal_tau"] == 3
    assert res_tie["n_ref_accepted"] == 5
    assert res_tie["ref_accept_rate_among_ref"] == 1.0
    assert abs(res_tie["ref_accept_rate_among_target_eligible"] - 5.0 / 7.0) <= 1e-12
    assert res_tie["n_ref_technical_abstain"] == 2

    const_scores = np.array([3.5, 3.5, 3.5, 3.5], dtype=np.float64)
    res_const = summarize_reference_threshold(const_scores, np.ones(4, dtype=bool), n_target_eligible=4)
    assert res_const["status"] == "sabit skor — ayırt edici değil"
    assert res_const["is_constant_score"] is True
    assert res_const["n_unique_scores"] == 1
    assert res_const["n_ref_below_tau"] == 0
    assert res_const["n_ref_equal_tau"] == 4
    assert res_const["n_ref_accepted"] == 4

    empty_scores = np.array([np.nan, 1.0, np.inf], dtype=np.float64)
    empty_valid = np.array([True, False, True], dtype=bool)
    res_empty = summarize_reference_threshold(empty_scores, empty_valid, n_target_eligible=5)
    assert res_empty["status"] == "hesaplanamadı (sonlu referans skor yok)"
    assert not math.isfinite(res_empty["tau_final"])
    assert res_empty["n_ref"] == 0
    assert res_empty["n_ref_accepted"] == 0
    assert res_empty["n_ref_technical_abstain"] == 5


def test_dev_opus19_contract_b_frozen_model_invariant_to_auc_and_cal_test_edits() -> None:
    """Sözleşme (b): dondurulmuş fit sonrası sayısal AUC ve cal/test değişimleri τ_final'i değiştirmez."""
    ds = _make_synthetic_dataset(seed=2026, n_pat=80)
    splits = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=2026)
    tr_idx = splits["id_60_20_20"]["train_tune"]
    target_idx = list(range(6))

    fit_frozen = _fit_sprint04_fold(
        ds, tr_idx, target_idx, k_budgets=(2, 4), inner_seed=2026, min_train_obs=15, top_k_genes=16
    )
    ds_tr = ds.subset_by_positions(tr_idx)
    avail_mask_tr = np.isfinite(ds_tr.Y_auc)

    for src in ("oof", "insample"):
        ref_base = compute_final_fit_reference_scores(
            fit_frozen,
            avail_mask_tr,
            ds_tr.X_rna,
            ds_tr.X_mut,
            target_idx,
            k_budgets=(2, 4),
            min_targets=3,
            min_probes=4,
            adaptive_m1_source=src,
        )

        # Sayısal AUC değerlerini maske sabit kalarak boz, cal/test satırlarını silinmiş bir alt kopya gibi maskele
        Y_pert = np.where(avail_mask_tr, -ds_tr.Y_auc + 999.0, np.nan)
        avail_mask_pert = np.isfinite(Y_pert)
        assert np.array_equal(avail_mask_pert, avail_mask_tr)

        ref_pert = compute_final_fit_reference_scores(
            fit_frozen,
            avail_mask_pert,
            ds_tr.X_rna,
            ds_tr.X_mut,
            target_idx,
            k_budgets=(2, 4),
            min_targets=3,
            min_probes=4,
            adaptive_m1_source=src,
        )

        # Kalibrasyon ve test satırlarının tamamen çıkarılması da referans eşiklerini değiştirmez.
        fit_no_cal_test = _fit_sprint04_fold(
            ds_tr,
            np.arange(len(ds_tr.meta), dtype=np.int64),
            target_idx,
            k_budgets=(2, 4),
            inner_seed=2026,
            min_train_obs=15,
            top_k_genes=16,
        )
        ref_no_ct = compute_final_fit_reference_scores(
            fit_no_cal_test,
            avail_mask_tr,
            ds_tr.X_rna,
            ds_tr.X_mut,
            target_idx,
            k_budgets=(2, 4),
            min_targets=3,
            min_probes=4,
            adaptive_m1_source=src,
        )

        for scen in ("archive_available", "locked_train_catalog"):
            assert ref_base[scen]["M0_k0"]["threshold_assigned"] is False
            assert ref_base[scen]["M1_k0"]["threshold_assigned"] is False
            for k in (2, 4):
                for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
                    arm = f"{pol}_k{k}"
                    b, p, nc = ref_base[scen][arm], ref_pert[scen][arm], ref_no_ct[scen][arm]
                    assert b["status"] == p["status"] == nc["status"]
                    assert b["n_ref"] == p["n_ref"] == nc["n_ref"]
                    assert b["n_unique_scores"] == p["n_unique_scores"] == nc["n_unique_scores"]
                    assert b["n_ref_accepted"] == p["n_ref_accepted"] == nc["n_ref_accepted"]
                    if math.isfinite(b["tau_final"]):
                        assert b["tau_final"] == p["tau_final"]
                        assert abs(b["tau_final"] - nc["tau_final"]) <= 1e-12
                    if scen == "locked_train_catalog" and pol in ("StaticDOpt", "EqualWeightVarRed") and b["n_ref"] > 0:
                        assert b["status"] == "sabit skor — ayırt edici değil"
                        assert b["is_constant_score"] is True


def test_dev_opus19_contract_a_v2_outputs_unchanged_and_consistency() -> None:
    """Sözleşme (a) & v2 tutarlılığı: OOF kıyasları v2 ile ≤1e-12, ana hücre 84/91 ve 65/148, kilitli sabit etiketli."""
    v2_path = Path("data/processed/sprint04_v2_results.json")
    op19_path = Path("data/processed/sprint04_v2_dev_opus19_results.json")
    op19_md = Path("docs/sprints/sprint-04-v2-dev-opus19-sensitivity.md")
    assert v2_path.exists() and op19_path.exists() and op19_md.exists()

    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    op19 = json.loads(op19_path.read_text(encoding="utf-8"))

    assert op19["metadata"]["v2_unchanged_contract_a"] is True
    assert op19["metadata"]["user_approval_date"] == "2026-11-22"

    for rn in ("id_60_20_20", "wave_ood"):
        for scen in ("archive_available", "locked_train_catalog"):
            arms19 = op19["regimes"][rn][scen]["arms"]
            arms_v2 = v2["regimes"][rn]["scenarios"][scen]["arms"]
            for z_arm in ("M0_k0", "M1_k0"):
                assert arms19[z_arm]["threshold_assigned"] is False

            for k in (2, 5, 8):
                for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
                    arm = f"{pol}_k{k}"
                    an19 = arms19[arm]
                    v2_m3 = arms_v2[arm]["abstention"]["m3_post_scale"]
                    assert abs(an19["v2_aurc_regret1_unchanged"]["aurc"] - v2_m3["risk_coverage_regret1"]["aurc"]) <= 1e-12

                    oof19 = an19["oof_threshold_eval"]
                    oofv2 = v2_m3["threshold_eval"]
                    assert oof19["status"] == oofv2["status"]
                    assert oof19["n_accepted"] == oofv2["n_accepted"]
                    assert oof19["n_available"] == oofv2["n_available"]
                    if oof19["accepted_mean_regret1"] is not None:
                        assert abs(oof19["accepted_mean_regret1"] - oofv2["accepted_mean_regret1"]) <= 1e-12

                    if scen == "locked_train_catalog" and pol in ("StaticDOpt", "EqualWeightVarRed"):
                        assert an19["reference"]["status"] == "sabit skor — ayırt edici değil"
                        assert an19["final_fit_threshold_eval"]["descriptive_only"] is True
                    if pol == "AdaptiveOED":
                        assert "insample_m1_sensitivity" in an19

    id_main = op19["regimes"]["id_60_20_20"]["archive_available"]["arms"]["StaticDOpt_k5"]["final_fit_threshold_eval"]
    w_main = op19["regimes"]["wave_ood"]["archive_available"]["arms"]["StaticDOpt_k5"]["final_fit_threshold_eval"]
    assert id_main["n_accepted"] == 84 and id_main["n_available"] == 91
    assert w_main["n_accepted"] == 65 and w_main["n_available"] == 148
