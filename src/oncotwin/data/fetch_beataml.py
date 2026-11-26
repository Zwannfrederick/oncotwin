"""Beat AML 2.0 resmi harmonize dosyalarını Git LFS medya uçlarından indirir ve doğrular."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/biodev/beataml2.0_data/main"
MEDIA_BASE = "https://media.githubusercontent.com/media/biodev/beataml2.0_data/main"
LFS_POINTER_PREFIX = b"version https://git-lfs"
DEFAULT_OUT_DIR = Path("data/raw/beataml2")

BEATAML_FILES: dict[str, dict[str, object]] = {
    "LICENSE.txt": {
        "url": f"{RAW_BASE}/LICENSE.txt",
        "size": None,
        "sha256": None,
        "is_expression": False,
    },
    "beataml_wv1to4_clinical.xlsx": {
        "url": f"{MEDIA_BASE}/beataml_wv1to4_clinical.xlsx",
        "size": 488278,
        "sha256": "bc692f647f93945e1cf883271af5501bf75c8af3e681676241093c198ed167ad",
        "is_expression": False,
    },
    "beataml_waves1to4_sample_mapping.xlsx": {
        "url": f"{MEDIA_BASE}/beataml_waves1to4_sample_mapping.xlsx",
        "size": 72038,
        "sha256": "5581140cedd3348ec78bed878c27aa12c9538a5fab115324b91ca56aa138f0fc",
        "is_expression": False,
    },
    "beataml_drug_families.xlsx": {
        "url": f"{MEDIA_BASE}/beataml_drug_families.xlsx",
        "size": 59330,
        "sha256": "636e27515bc868194686689c635657293b334741dd64b0253e3d1d0b4adc35a0",
        "is_expression": False,
    },
    "beataml_probit_curve_fits_v4_dbgap.txt": {
        "url": f"{MEDIA_BASE}/beataml_probit_curve_fits_v4_dbgap.txt",
        "size": 19047078,
        "sha256": "d4bc5f0d91f66314107411e0f2511adc987e29df5b83d9b03df56d3d12928314",
        "is_expression": False,
    },
    "beataml_wes_wv1to4_mutations_dbgap.txt": {
        "url": f"{MEDIA_BASE}/beataml_wes_wv1to4_mutations_dbgap.txt",
        "size": 3668996,
        "sha256": "5a5a5eb8f492b1385aebe85c490b9333f65590f09391a7c1951b04dd1dba1680",
        "is_expression": False,
    },
    "beataml_waves1to4_norm_exp_dbgap.txt": {
        "url": f"{MEDIA_BASE}/beataml_waves1to4_norm_exp_dbgap.txt",
        "size": 281337180,
        "sha256": "d5745b9dbf46dba866a3c7370bb0ba73b363ecdd21e01cc1d916b4e3021e6f87",
        "is_expression": True,
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Dosyanın SHA-256 özetini parçalı okuma ile hesaplar."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_not_lfs_pointer(path: Path) -> None:
    """Dosyanın gerçek veri yerine Git LFS pointer metni olmadığını doğrular."""
    with path.open("rb") as fh:
        head = fh.read(len(LFS_POINTER_PREFIX))
    if head.startswith(LFS_POINTER_PREFIX):
        raise ValueError(
            f"{path} gerçek veri yerine Git LFS pointer metni içeriyor; "
            "media.githubusercontent.com uç noktası kullanılmalıdır."
        )


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    timeout: int = 300,
) -> dict[str, object]:
    """Tek bir dosyayı indirir, LFS pointer / boyut / SHA-256 doğrulaması yapar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        check_not_lfs_pointer(dest)
        actual_size = dest.stat().st_size
        actual_sha = sha256_file(dest)
        size_ok = expected_size is None or actual_size == expected_size
        sha_ok = expected_sha256 is None or actual_sha == expected_sha256
        if size_ok and sha_ok:
            return {
                "file": dest.name,
                "status": "cached",
                "size": actual_size,
                "sha256": actual_sha,
            }

    part_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "OncoTwin-v1-Fetcher/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, part_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)

        check_not_lfs_pointer(part_path)
        actual_size = part_path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                f"{dest.name} boyut uyuşmazlığı: beklenen={expected_size}, gelen={actual_size}"
            )

        actual_sha = sha256_file(part_path)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(
                f"{dest.name} SHA-256 uyuşmazlığı: beklenen={expected_sha256}, gelen={actual_sha}"
            )

        part_path.replace(dest)
    except Exception:
        if part_path.exists():
            part_path.unlink()
        raise

    return {
        "file": dest.name,
        "status": "downloaded",
        "size": actual_size,
        "sha256": actual_sha,
    }


def fetch_all(out_dir: Path = DEFAULT_OUT_DIR, include_expression: bool = True) -> list[dict[str, object]]:
    """Tüm Beat AML 2.0 dosyalarını indirir ve doğrulama özetini döndürür."""
    results: list[dict[str, object]] = []
    for name, meta in BEATAML_FILES.items():
        if not include_expression and meta["is_expression"]:
            continue
        res = download_file(
            url=str(meta["url"]),
            dest=out_dir / name,
            expected_sha256=meta["sha256"],  # type: ignore[arg-type]
            expected_size=meta["size"],  # type: ignore[arg-type]
        )
        results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Beat AML 2.0 harmonize veri setini indir.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--skip-expression",
        action="store_true",
        help="281 MB RNA-seq norm_exp dosyasını atla.",
    )
    args = parser.parse_args()
    results = fetch_all(args.out_dir, include_expression=not args.skip_expression)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
