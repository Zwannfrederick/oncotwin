"""Sprint 1 birim ve veri tutarlılığı testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oncotwin.data.audit_beataml import (
    analyze_subject_specimen_leakage,
    decompose_two_way_mean_effect,
)
from src.oncotwin.data.fetch_beataml import (
    check_not_lfs_pointer,
    download_file,
    sha256_file,
)


def test_sha256_and_lfs_pointer_rejection(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.txt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        check_not_lfs_pointer(pointer)

    src = tmp_path / "src.bin"
    payload = b"beataml-test-payload-123"
    src.write_bytes(payload)
    expected_sha = sha256_file(src)

    dest = tmp_path / "out" / "data.bin"
    res1 = download_file(src.as_uri(), dest, expected_sha256=expected_sha, expected_size=len(payload))
    assert res1["status"] == "downloaded"
    assert dest.read_bytes() == payload

    # İkinci çağrı idempotent cache hit dönmeli
    res2 = download_file(src.as_uri(), dest, expected_sha256=expected_sha, expected_size=len(payload))
    assert res2["status"] == "cached"

    # Uzak LFS pointer gelirse hata fırlatmalı ve dest bozulmamalı
    dest_bad = tmp_path / "out" / "bad.bin"
    with pytest.raises(ValueError, match="Git LFS pointer"):
        download_file(pointer.as_uri(), dest_bad)
    assert not dest_bad.exists()


def test_decompose_two_way_mean_effect_synthetic_masked() -> None:
    rng = np.random.default_rng(42)
    b_p = np.array([-30.0, -10.0, 0.0, 15.0, 25.0])[:, None]
    b_d = np.array([-40.0, -5.0, 10.0, 35.0])[None, :]
    pure_additive = 150.0 + b_p + b_d
    # %25 rastgele eksiklik maskesi
    masked = pure_additive.copy()
    masked[0, 3] = np.nan
    masked[2, 1] = np.nan
    masked[3, 0] = np.nan
    masked[4, 2] = np.nan

    res_clean = decompose_two_way_mean_effect(masked)
    assert res_clean["two_way_m0"]["converged"] is True
    assert res_clean["two_way_m0"]["r2"] > 0.999999
    assert res_clean["two_way_m0"]["residual_var"] < 1e-8

    # Bilinen etkileşim/gürültü eklendiğinde varyans dengesi korunmalı
    noise = rng.normal(0.0, 8.0, size=pure_additive.shape)
    noisy_masked = np.where(np.isfinite(masked), pure_additive + noise, np.nan)
    res_noisy = decompose_two_way_mean_effect(noisy_masked)
    explained_plus_res = res_noisy["total_var"] * (1.0 - res_noisy["two_way_m0"]["r2"])
    assert np.isclose(explained_plus_res, res_noisy["two_way_m0"]["residual_var"], rtol=1e-6)
    assert res_noisy["two_way_m0"]["r2"] > res_noisy["drug_only"]["r2"]


def test_subject_specimen_leakage_detection() -> None:
    df = pd.DataFrame(
        {
            "dbgap_subject_id": [101, 101, 102, 103, 104],
            "dbgap_dnaseq_sample": ["D1", "D2", "D3", None, "D5"],
            "dbgap_rnaseq_sample": ["R1", None, "R3", "R4", "R5"],
            "cohort": ["Waves1+2", "Waves3+4", "Waves1+2", "Both", "Waves3+4"],
        }
    )
    res = analyze_subject_specimen_leakage(df)
    assert res["n_patients"] == 4
    assert res["n_specimens"] == 5
    assert res["specimens_per_patient"]["patients_with_multiple_specimens"] == 1
    assert res["cross_wave_patient_overlap"]["waves12_and_waves34"] == 1


def test_sprint01_audit_summary_consistency() -> None:
    summary_path = Path("data/processed/sprint01_audit_summary.json")
    if not summary_path.exists():
        pytest.skip("sprint01_audit_summary.json henüz üretilmedi.")

    s = json.loads(summary_path.read_text(encoding="utf-8"))
    leak = s["leakage_and_units"]
    assert leak["n_patients"] <= leak["n_specimens"]
    assert leak["n_specimens"] == leak["n_unique_specimen_keys"]

    for key in ("all_166", "train_test_95", "cross_wave_ge100", "clinical_candidates_C"):
        sub = s["matrix_subsets"][key]
        assert 0.0 <= sub["missing_rate"] <= 1.0
        assert np.isclose(sub["fill_rate"] + sub["missing_rate"], 1.0, atol=1e-9)
        m0 = sub["m0_variance_decomposition"]
        recon_res_var = m0["total_var"] * (1.0 - m0["two_way_m0"]["r2"])
        assert np.isclose(recon_res_var, m0["two_way_m0"]["residual_var"], rtol=1e-6)
