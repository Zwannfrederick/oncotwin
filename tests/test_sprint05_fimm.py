"""Sprint 5 Kulvar C (FIMM / Malani MD5, katalog, hasta gruplama ve sıfır sızıntı) testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.oncotwin.data.fetch_fimm import DEFAULT_FIMM_DIR, FIMM_EXPECTED_MD5, FIMM_ZIP_NAME, verify_fimm_zip
from src.oncotwin.data.fimm_dataset import (
    DEFAULT_CATALOG_AUDIT_JSON,
    load_fimm_aml_dataset,
    select_fimm_index_specimen_positions,
    within_sample_rank_normalize,
)
from src.oncotwin.experiments.run_sprint05_fimm import _build_single_patient_probe_z_matrix


def test_fimm_zip_md5_and_catalog_audit_verified() -> None:
    zip_path = DEFAULT_FIMM_DIR / FIMM_ZIP_NAME
    info = verify_fimm_zip(zip_path, expected_md5=FIMM_EXPECTED_MD5)
    assert info["md5"] == FIMM_EXPECTED_MD5
    assert info["n_members"] >= 12

    assert DEFAULT_CATALOG_AUDIT_JSON.exists()
    audit = json.loads(DEFAULT_CATALOG_AUDIT_JSON.read_text(encoding="utf-8"))
    sc = audit["sample_counts"]
    assert sc["file7_rna_columns_total"] == 167  # 163 AML + 4 sağlıklı CD34+ (Opus §3.9)
    assert sc["matched_aml_rna_sdss_specimens"] == 132
    assert sc["matched_aml_rna_sdss_unique_patients"] == 109
    assert sc["matched_index_specimens_selected"] == 109
    assert audit["compound_overlap"]["matched_unique_compounds"] == 102
    assert audit["compound_overlap"]["common_targets_count"] == 23
    assert audit["compound_overlap"]["unmatched_targets"] == ["Gilteritinib"]
    assert audit["gene_overlap"]["common_ensembl_genes"] == 14688


def test_within_sample_rank_normalize_is_strictly_single_patient() -> None:
    """Hasta i'nin sıra-normalize RNA profili diğer hastaların değiştirilmesinden bağımsızdır."""
    rng = np.random.default_rng(77)
    X = rng.normal(3.0, 2.0, size=(8, 50))
    R1 = within_sample_rank_normalize(X)
    X_pert = X.copy()
    X_pert[1:] = X_pert[1:] * 50.0 - 123.0
    R2 = within_sample_rank_normalize(X_pert)
    assert np.allclose(R1[0], R2[0], atol=0.0)
    # Tek-hasta ölçek/öteleme değişmezliği: monoton pozitif dönüşüm aynı sıralamayı üretir
    X_mono = X.copy()
    X_mono[0] = 3.7 * X_mono[0] + 19.2
    R3 = within_sample_rank_normalize(X_mono)
    assert np.allclose(R1[0], R3[0], atol=0.0)


def test_fimm_dataset_patient_grouping_and_direction_alignment() -> None:
    """132 -> 109 benzersiz hasta seçimi ve -sDSS yön uyumu (yüksek sDSS -> düşük Y_aligned)."""
    audit = json.loads(DEFAULT_CATALOG_AUDIT_JSON.read_text(encoding="utf-8"))
    targets = audit["compound_overlap"]["common_targets"]
    ds, _ = load_fimm_aml_dataset(targets)
    assert len(ds.meta) == 132
    assert ds.meta["Patient_ID"].nunique() == 109
    assert np.allclose(ds.Y_aligned[np.isfinite(ds.Y_sdss)], -ds.Y_sdss[np.isfinite(ds.Y_sdss)])

    idx = select_fimm_index_specimen_positions(ds.meta)
    ds_idx = ds.subset_by_positions(idx)
    assert len(ds_idx.meta) == 109
    assert ds_idx.meta["Patient_ID"].nunique() == 109
    # Deterministik tekrarlanabilirlik
    assert np.array_equal(idx, select_fimm_index_specimen_positions(ds.meta))


def test_c1_single_patient_probe_z_zero_leakage_and_sprint05_fimm_consistency() -> None:
    """C1 prob z-skoru gizli hedeflerden ve diğer hastalardan tam izole; C1/C2 çıktıları ayrı ve tutarlıdır."""
    rng = np.random.default_rng(2026)
    Y = rng.normal(-15.0, 8.0, size=(6, 12))
    probes = [np.asarray([6, 7, 8, 9, 10], dtype=np.int64) for _ in range(6)]
    Z1 = _build_single_patient_probe_z_matrix(Y, probes)

    # Hedef sütunlar (0..5) ve seçilmeyen prob (11) kesinlikle NaN kalmalı
    assert np.all(np.isnan(Z1[:, :6]))
    assert np.all(np.isnan(Z1[:, 11]))

    # Gizli hedefleri ve diğer hastaları aşırı değiştirince hasta 0'ın z-skoru bit-düzeyinde değişmez
    Y_mut = Y.copy()
    Y_mut[:, :6] += 1000.0
    Y_mut[1:] = Y_mut[1:] * 40.0 - 500.0
    Z2 = _build_single_patient_probe_z_matrix(Y_mut, probes)
    assert np.allclose(Z1[0, probes[0]], Z2[0, probes[0]], atol=0.0)

    # Üretilen Sprint 5 Kulvar C JSON & NPZ doğrulaması
    s5 = json.loads(Path("data/processed/sprint05_results.json").read_text(encoding="utf-8"))
    tc = s5["track_c_fimm"]
    assert tc["common_targets_count"] == 23
    assert tc["common_probes_count"] == 64
    assert tc["common_genes_count"] == 14688

    c1 = tc["experiment_c1_frozen_transfer"]
    c2 = tc["experiment_c2_within_fimm_replication"]
    assert c1["conformal_intervals_transferred"] is False
    assert "Donmuş" in c1["label"]
    assert "Yöntem Tekrarı" in c2["label"]
    assert c1["cohorts"]["index_109_patients"]["n_specimens"] == 109
    assert c1["cohorts"]["index_109_patients"]["arms"]["C1_M3_StaticDOpt_k5"]["n_eval_units"] == 109
    assert c1["cohorts"]["all_132_specimens_clustered"]["n_specimens"] == 132
    assert c1["cohorts"]["all_132_specimens_clustered"]["arms"]["C1_M3_StaticDOpt_k5"]["n_eval_units"] == 109
    assert c2["n_eval"] == 109

    z = np.load(Path("data/processed/sprint05_vectors.npz"))
    assert "track_c/C1/index_109_patients/C1_M3_StaticDOpt_k5/patient_ndcg3" in z.files
    assert "track_c/C2/index_109_patients/M3_StaticDOpt_k5/patient_rmse_sq" in z.files
    assert len(z["track_c/C1/index_109_patients/C1_M3_StaticDOpt_k5/patient_ndcg3"]) == 109
    assert len(z["track_c/C2/index_109_patients/M3_StaticDOpt_k5/patient_rmse_sq"]) == 109
