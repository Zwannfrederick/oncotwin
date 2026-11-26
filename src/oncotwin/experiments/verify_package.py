"""OncoTwin v1 tek-komut tekrarlanabilirlik, ham veri bütünlüğü ve Sprint 1-5 + OPUS-028 doğrulayıcısı."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.fetch_beataml import (
    BEATAML_FILES,
    DEFAULT_OUT_DIR as DEFAULT_BEATAML_DIR,
    check_not_lfs_pointer,
    sha256_file,
)
from src.oncotwin.data.fetch_fimm import (
    DEFAULT_FIMM_DIR,
    FIMM_EXPECTED_MD5,
    FIMM_ZIP_NAME,
    verify_fimm_zip,
)

DEFAULT_MANIFEST_PATH = Path("data/processed/release_manifest.sha256")

FROZEN_RELEASE_FILES: tuple[str, ...] = (
    # Dondurulmuş protokoller
    "docs/PROTOCOL.md",
    "docs/PROTOCOL_SPRINT03.md",
    "docs/PROTOCOL_SPRINT04.md",
    "docs/PROTOCOL_SPRINT04_ADDENDUM_19.md",
    "docs/PROTOCOL_SPRINT05.md",
    # Sprint 1-5 işlenmiş JSON/NPZ artefaktları (tarihsel v1 + v2 + Ek 19)
    "data/processed/sprint01_audit_summary.json",
    "data/processed/sprint02_results.json",
    "data/processed/sprint03_results.json",
    "data/processed/sprint03_v2_results.json",
    "data/processed/sprint03_v2_vectors.npz",
    "data/processed/sprint03_v2_reanalysis.json",
    "data/processed/sprint04_split_manifest.json",
    "data/processed/sprint04_results.json",
    "data/processed/sprint04_vectors.npz",
    "data/processed/sprint04_v2_split_manifest.json",
    "data/processed/sprint04_v2_results.json",
    "data/processed/sprint04_v2_vectors.npz",
    "data/processed/sprint04_v2_dev_opus19_results.json",
    "data/processed/sprint05_fimm_catalog_audit.json",
    "data/processed/sprint05_results.json",
    "data/processed/sprint05_vectors.npz",
    # Tarihsel sprint raporları
    "docs/sprints/sprint-01-data-audit.md",
    "docs/sprints/sprint-02-baselines.md",
    "docs/sprints/sprint-03-panel-selection.md",
    "docs/sprints/sprint-03-v2-panel-selection.md",
    "docs/sprints/sprint-03-v2-reanalysis.md",
    "docs/sprints/sprint-04-conformal-abstain.md",
    "docs/sprints/sprint-04-v2-conformal-abstain.md",
    "docs/sprints/sprint-04-v2-dev-opus19-sensitivity.md",
    "docs/sprints/sprint-05-m4-shortlist-fimm.md",
    # OPUS-028 Beat AML prob-ölçeği duyarlılık artefaktları (Sol)
    "audits/sol-sprint06/opus028_beataml_probe_scale.py",
    "audits/sol-sprint06/opus028_beataml_probe_scale.json",
)


def write_sha256_manifest(
    paths: list[Path],
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    base_dir: Path = Path("."),
) -> Path:
    """Verilen dosyalar için sha256sum uyumlu manifest dosyası yazar."""
    base_resolved = base_dir.resolve()
    lines: list[str] = []
    for p in paths:
        full = p if p.is_absolute() else (base_resolved / p)
        if not full.exists():
            raise FileNotFoundError(f"Manifest dosyası eksik: {full}")
        rel = full.resolve().relative_to(base_resolved).as_posix()
        lines.append(f"{sha256_file(full)}  {rel}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def verify_sha256_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    base_dir: Path = Path("."),
) -> dict[str, Any]:
    """sha256sum biçimli manifest dosyasındaki tüm kayıtları doğrular."""
    if not manifest_path.exists():
        return {"status": "FAIL", "checked": 0, "failures": [f"Manifest bulunamadı: {manifest_path}"]}
    base_resolved = base_dir.resolve()
    failures: list[str] = []
    checked = 0
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(None, 1)
        rel_clean = rel_path.lstrip("*").strip()
        target = base_resolved / rel_clean
        checked += 1
        if not target.exists():
            failures.append(f"Eksik dosya: {rel_clean}")
            continue
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            failures.append(f"SHA-256 uyuşmazlığı: {rel_clean}")
    return {"status": "PASS" if not failures and checked > 0 else "FAIL", "checked": checked, "failures": failures}


def verify_raw_datasets(
    beataml_dir: Path = DEFAULT_BEATAML_DIR,
    fimm_dir: Path = DEFAULT_FIMM_DIR,
) -> dict[str, Any]:
    """Beat AML 2.0 (SHA-256 + LFS dışı) ve FIMM Zenodo 7370747 (MD5) ham veri bütünlüğünü doğrular."""
    failures: list[str] = []
    checked_ba = 0
    for fname, spec in BEATAML_FILES.items():
        path = beataml_dir / fname
        if not path.exists():
            failures.append(f"Eksik Beat AML ham dosyası: {path}")
            continue
        try:
            check_not_lfs_pointer(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"LFS pointer hatası ({fname}): {exc}")
            continue
        expected = spec.get("sha256")
        if expected is not None and sha256_file(path) != expected:
            failures.append(f"Beat AML SHA-256 uyuşmazlığı: {fname}")
        checked_ba += 1

    fimm_ok = False
    fimm_zip = fimm_dir / FIMM_ZIP_NAME
    if not fimm_zip.exists():
        failures.append(f"Eksik FIMM ZIP arşivi: {fimm_zip}")
    else:
        try:
            verify_fimm_zip(fimm_zip, expected_md5=FIMM_EXPECTED_MD5)
            fimm_ok = True
        except Exception as exc:  # noqa: BLE001
            failures.append(f"FIMM ZIP/MD5 hatası: {exc}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "beataml_checked": checked_ba,
        "fimm_md5_verified": fimm_ok,
        "failures": failures,
    }


def verify_cross_sprint_invariants(base_dir: Path = Path(".")) -> dict[str, Any]:
    """Sprint 1-5 ve OPUS-028 JSON/NPZ artefaktları arasındaki temel bilimsel değişmezleri kontrol eder."""
    failures: list[str] = []
    checked = 0

    def _chk(cond: bool, msg: str) -> None:
        nonlocal checked
        checked += 1
        if not cond:
            failures.append(msg)

    s1 = json.loads((base_dir / "data/processed/sprint01_audit_summary.json").read_text(encoding="utf-8"))
    s2 = json.loads((base_dir / "data/processed/sprint02_results.json").read_text(encoding="utf-8"))
    s3 = json.loads((base_dir / "data/processed/sprint03_v2_results.json").read_text(encoding="utf-8"))
    s3r = json.loads((base_dir / "data/processed/sprint03_v2_reanalysis.json").read_text(encoding="utf-8"))
    s4 = json.loads((base_dir / "data/processed/sprint04_v2_results.json").read_text(encoding="utf-8"))
    s4_19 = json.loads((base_dir / "data/processed/sprint04_v2_dev_opus19_results.json").read_text(encoding="utf-8"))
    s5 = json.loads((base_dir / "data/processed/sprint05_results.json").read_text(encoding="utf-8"))
    o28 = json.loads((base_dir / "audits/sol-sprint06/opus028_beataml_probe_scale.json").read_text(encoding="utf-8"))

    _chk(s1["modality_overlap"]["drug_and_norm_exp_patients"] == 479, "Sprint 1 eşleşmiş hasta != 479")
    _chk(s2["gate1_decision"]["status"] == "PASS", "Sprint 2 Kapı 1 != PASS")
    _chk(s3["gate2_decision"]["status"] == "PASS", "Sprint 3 v2 Kapı 2 != PASS")
    _chk(
        s3r["cv_5fold_3seeds"]["label"] == "INSUFFICIENT_EVIDENCE"
        and s3r["wave_ood"]["label"] == "INSUFFICIENT_EVIDENCE",
        "Sprint 3 v2 Holm yeniden analiz etiketi uyuşmuyor",
    )
    _chk(
        s4["regimes"]["wave_ood"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["conformal"]["n_cal"] == 70,
        "Sprint 4 v2 W3+4 arşiv n_cal != 70 (SOL-14)",
    )
    _chk(
        s4_19["regimes"]["id_60_20_20"]["archive_available"]["arms"]["StaticDOpt_k5"]["final_fit_threshold_eval"]["n_accepted"] == 84
        and s4_19["regimes"]["wave_ood"]["archive_available"]["arms"]["StaticDOpt_k5"]["final_fit_threshold_eval"]["n_accepted"] == 65,
        "Sprint 4 Ek 19 ana hücre test kabul sayıları (84, 65) uyuşmuyor",
    )
    _chk(s5["track_a_m4"]["gate3"]["decision"] == "NO-GO (v2 ODE ERTELENİR)", "Sprint 5 Kapı 3 != NO-GO")
    _chk(s5["track_a_m4"]["drug_family_graph"]["n_perm_draws"] == 20, "Sprint 5 M4 permütasyon sayısı != 20")

    # Kulvar B EqualWidthAUC == Sprint 4 v2 <= 1e-12 ve tek aday paydaları (SOL-13 / OPUS-027)
    b_id = s5["track_b_shortlist"]["regimes"]["id_60_20_20"]["arms"]["StaticDOpt_k5"]["methods"]
    c4_id = s4["regimes"]["id_60_20_20"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["conformal"]
    _chk(
        abs(b_id["EqualWidthAUC"]["test"]["shortlist_mean_size_obs"] - c4_id["mean_shortlist_size_obs"]) <= 1e-12,
        "Kulvar B EqualWidthAUC Sprint 4 v2 ile eşleşmiyor",
    )
    _chk(
        s5["track_c_fimm"]["experiment_c1_frozen_transfer"]["conformal_intervals_transferred"] is False,
        "Kulvar C konformal aralık aktarım bayrağı False değil",
    )
    _chk(
        s5["track_c_fimm"]["common_targets_count"] == 23
        and s5["track_c_fimm"]["common_probes_count"] == 64
        and s5["track_c_fimm"]["experiment_c2_within_fimm_replication"]["n_eval"] == 109,
        "Kulvar C FIMM paydaları (23 hedef, 64 prob, 109 hasta) uyuşmuyor",
    )
    _chk(
        o28["n_eval"] == 91
        and o28["budgets"]["5"]["hidden_response_perturbation_invariant"] is True
        and o28["budgets"]["2"]["n_probe_vectors_with_abs_values_all_one"] == 91,
        "OPUS-028 Beat AML prob ölçeği duyarlılık değişmezleri uyuşmuyor",
    )

    # NPZ arşivlerinin okunabilirliği, dizi sayıları ve SOL-13/OPUS-027 ham paydaları
    z5 = np.load(base_dir / "data/processed/sprint05_vectors.npz")
    _chk(len(z5.files) == 1014, f"sprint05_vectors.npz dizi sayısı != 1014 ({len(z5.files)})")
    for reg, exp_rc, exp_sing in (("id_60_20_20", (85, 91), (5, 9)), ("wave_ood", (137, 148), (19, 25))):
        rc_ok = z5[f"track_b/{reg}/StaticDOpt_k5/RankConformal/patient_shortlist_contains_top1"]
        tg_ok = z5[f"track_b/{reg}/StaticDOpt_k5/Top1GapConformal/patient_shortlist_contains_top1"]
        tg_def = z5[f"track_b/{reg}/StaticDOpt_k5/Top1GapConformal/patient_definite_top1"]
        sing = np.isfinite(tg_def) & (tg_def == 1.0)
        _chk(
            (int(np.nansum(rc_ok)), int(np.isfinite(rc_ok).sum())) == exp_rc
            and (int(np.nansum(tg_ok[sing])), int(sing.sum())) == exp_sing,
            f"Kulvar B ({reg}) RankConformal veya Top1Gap tekil paydaları uyuşmuyor",
        )
    z4 = np.load(base_dir / "data/processed/sprint04_v2_vectors.npz")
    _chk(len(z4.files) == 582, f"sprint04_v2_vectors.npz dizi sayısı != 582 ({len(z4.files)})")

    return {"status": "PASS" if not failures else "FAIL", "checked": checked, "failures": failures}


def verify_release_package(
    base_dir: Path = Path("."),
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    check_raw: bool = True,
    run_tests: bool = False,
) -> dict[str, Any]:
    """Ham veriler, dondurulmuş release manifesti, bilimsel değişmezler ve isteğe bağlı pytest paketini doğrular."""
    errors: list[str] = []
    raw_res: dict[str, Any] = {"status": "SKIPPED", "beataml_checked": 0, "fimm_md5_verified": False, "failures": []}
    if check_raw:
        raw_res = verify_raw_datasets(base_dir / DEFAULT_BEATAML_DIR, base_dir / DEFAULT_FIMM_DIR)
        errors.extend(raw_res["failures"])

    man_res = verify_sha256_manifest(base_dir / manifest_path if not manifest_path.is_absolute() else manifest_path, base_dir=base_dir)
    errors.extend(man_res["failures"])

    inv_res = verify_cross_sprint_invariants(base_dir=base_dir)
    errors.extend(inv_res["failures"])

    test_res: dict[str, Any] = {"status": "SKIPPED", "returncode": None}
    if run_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        test_res = {"status": "PASS" if proc.returncode == 0 else "FAIL", "returncode": proc.returncode, "summary": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""}
        if proc.returncode != 0:
            errors.append(f"pytest başarısız (kod {proc.returncode}): {proc.stdout[-300:]}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "raw_data": raw_res,
        "manifest": man_res,
        "invariants": inv_res,
        "tests": test_res,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoTwin v1 tekrarlanabilirlik ve bütünlük doğrulayıcısı.")
    parser.add_argument("--write-manifest", action="store_true", help="Dondurulmuş artefaktlar için release_manifest.sha256 üretir.")
    parser.add_argument("--skip-raw", action="store_true", help="Ham veri (Beat AML / FIMM) hash kontrolünü atlar.")
    parser.add_argument("--run-tests", action="store_true", help="Doğrulama sonunda pytest paketini de koşturur.")
    args = parser.parse_args()

    if args.write_manifest:
        out = write_sha256_manifest([Path(p) for p in FROZEN_RELEASE_FILES], DEFAULT_MANIFEST_PATH)
        print(f"Release manifest yazıldı ({len(FROZEN_RELEASE_FILES)} dosya) -> {out}")
        return

    res = verify_release_package(check_raw=not args.skip_raw, run_tests=args.run_tests)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if res["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
