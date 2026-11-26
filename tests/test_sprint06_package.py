"""Sprint 6 tekrarlanabilir araştırma paketi, OPUS-032 hijyeni ve release manifest testleri."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.oncotwin.eval import rank_conformal, uncertainty
from src.oncotwin.experiments.verify_package import (
    DEFAULT_MANIFEST_PATH,
    verify_release_package,
    verify_sha256_manifest,
    write_sha256_manifest,
)


def test_opus032_public_conformal_quantile_helper() -> None:
    """OPUS-032: finite_sample_conformal_quantile public olmalı ve rank_conformal bunu kullanmalı."""
    assert hasattr(uncertainty, "finite_sample_conformal_quantile")
    assert not hasattr(uncertainty, "_finite_sample_conformal_quantile")
    assert rank_conformal.finite_sample_conformal_quantile is uncertainty.finite_sample_conformal_quantile

    q, r, pub = uncertainty.finite_sample_conformal_quantile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), alpha=0.10)
    assert pub is False and not math.isfinite(q) and r == 6

    q2, r2, pub2 = uncertainty.finite_sample_conformal_quantile(np.arange(1.0, 20.0), alpha=0.10)
    assert pub2 is True and r2 == 18 and abs(q2 - 18.0) <= 1e-12


def test_manifest_verification_detects_tampering_and_missing_files(tmp_path: Path) -> None:
    """SHA-256 manifest doğrulaması sağlam dosyada geçmeli, 1 bayt değişimde veya eksikte hata vermelidir."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("oncotwin-a\n", encoding="utf-8")
    f2.write_text("oncotwin-b\n", encoding="utf-8")
    manifest = tmp_path / "manifest.sha256"

    write_sha256_manifest([f1, f2], manifest, base_dir=tmp_path)
    ok_res = verify_sha256_manifest(manifest, base_dir=tmp_path)
    assert ok_res["status"] == "PASS" and ok_res["checked"] == 2 and ok_res["failures"] == []

    # 1 bayt değişim yakalanmalı
    f1.write_text("oncotwin-A\n", encoding="utf-8")
    tamper_res = verify_sha256_manifest(manifest, base_dir=tmp_path)
    assert tamper_res["status"] == "FAIL" and any("a.txt" in msg for msg in tamper_res["failures"])

    # Eksik dosya yakalanmalı
    f2.unlink()
    miss_res = verify_sha256_manifest(manifest, base_dir=tmp_path)
    assert miss_res["status"] == "FAIL" and len(miss_res["failures"]) == 2


def test_release_package_integrity_and_cross_sprint_invariants() -> None:
    """Sprint 6 paket dosyaları, release_manifest.sha256 ve Sprint 1-5 + OPUS-028 değişmezleri doğrulanır."""
    for req_file in (
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("CITATION.cff"),
        Path("docs/METHODS_REPORT.md"),
        DEFAULT_MANIFEST_PATH,
    ):
        assert req_file.exists(), f"Eksik paket dosyası: {req_file}"

    res = verify_release_package(check_raw=True)
    assert res["status"] == "PASS", f"Paket doğrulama hatası: {res['errors']}"
    assert res["raw_data"]["beataml_checked"] == 7
    assert res["raw_data"]["fimm_md5_verified"] is True
    assert res["manifest"]["checked"] >= 30
    assert res["invariants"]["checked"] >= 10
    assert res["errors"] == []
