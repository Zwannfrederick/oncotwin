"""FIMM / Malani (Cancer Discov 2022, Zenodo 7370747) AML veri paketini indirir ve MD5 doğrular."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "7370747"
FIMM_ZIP_NAME = "Functional_Precision_Medicine_Tumor_Board_AML.zip"
FIMM_ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{FIMM_ZIP_NAME}/content"
)
FIMM_EXPECTED_MD5 = "3db5280e20f315d56d6bf07dd6a9e241"
DEFAULT_FIMM_DIR = Path("data/raw/fimm")


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Dosyanın MD5 özetini parçalı okuma ile hesaplar."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_fimm_zip(zip_path: Path, expected_md5: str = FIMM_EXPECTED_MD5) -> dict[str, object]:
    """ZIP bütünlüğünü ve Zenodo MD5 özetini doğrular."""
    if not zip_path.exists():
        raise FileNotFoundError(f"FIMM ZIP dosyası yok: {zip_path}")
    actual_md5 = md5_file(zip_path)
    if actual_md5 != expected_md5:
        raise ValueError(f"FIMM ZIP MD5 uyuşmazlığı: beklenen={expected_md5}, gelen={actual_md5}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"Bozuk ZIP girdisi: {bad}")
        members = zf.namelist()
    return {
        "path": str(zip_path),
        "size_bytes": int(zip_path.stat().st_size),
        "md5": actual_md5,
        "n_members": len(members),
        "members": members,
    }


def extract_fimm_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Path-traversal korumasıyla ZIP içeriğini hedef dizine çıkarır."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (extract_dir / info.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Güvensiz ZIP yolu reddedildi: {info.filename}")
        zf.extractall(extract_dir)
    return extract_dir


def fetch_fimm(out_dir: Path = DEFAULT_FIMM_DIR, timeout: int = 300) -> dict[str, object]:
    """FIMM/Malani Zenodo paketini indirir (varsa MD5 doğrular) ve çıkarır."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / FIMM_ZIP_NAME
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".zip.part")
        req = urllib.request.Request(FIMM_ZIP_URL, headers={"User-Agent": "OncoTwin-Research/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        try:
            verify_fimm_zip(tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(zip_path)

    info = verify_fimm_zip(zip_path)
    extract_dir = out_dir / "extracted"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_fimm_zip(zip_path, extract_dir)
    info["extract_dir"] = str(extract_dir)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="FIMM / Malani Zenodo 7370747 AML veri indirici.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_FIMM_DIR)
    args = parser.parse_args()
    info = fetch_fimm(args.out_dir)
    print(f"FIMM indirildi & doğrulandı: {info['path']} ({info['size_bytes']} B, md5={info['md5']}, {info['n_members']} dosya)")


if __name__ == "__main__":
    main()
