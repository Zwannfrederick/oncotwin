"""Deterministik makale figürleri ve fetch_fimm 5xx tekrar deneme sözleşme testleri."""

from __future__ import annotations

import io
import urllib.error
import zipfile
from pathlib import Path

import pytest

from src.oncotwin.data import fetch_fimm as ff

PINNED_MATPLOTLIB_VERSION = "3.11.2"


def _build_valid_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample.txt", "ok\n")
    return buf.getvalue()


def test_fetch_fimm_retries_5xx_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R-1: Geçici 504 hatasında 3 denemeye kadar artan beklemeyle yeniden dener ve başarıyla tamamlar."""
    payload = _build_valid_zip_bytes()
    expected_md5 = ff.hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(ff, "FIMM_EXPECTED_MD5", expected_md5)

    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=300):  # noqa: ANN001, ARG001
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(ff.FIMM_ZIP_URL, 504, "Gateway Timeout", hdrs=None, fp=None)
        return io.BytesIO(payload)

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ff.time, "sleep", lambda sec: sleeps.append(float(sec)))

    info = ff.fetch_fimm(out_dir=tmp_path, max_attempts=3, retry_backoff_sec=0.5)
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]
    assert info["md5"] == expected_md5
    assert (tmp_path / ff.FIMM_ZIP_NAME).exists()
    assert not (tmp_path / ff.FIMM_ZIP_NAME).with_suffix(".zip.part").exists()


def test_fetch_fimm_does_not_retry_4xx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx istemci hatasında tekrar denenmez ve yarım .part dosyası bırakılmaz."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=300):  # noqa: ANN001, ARG001
        calls["n"] += 1
        raise urllib.error.HTTPError(ff.FIMM_ZIP_URL, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ff.time, "sleep", lambda _sec: (_ for _ in ()).throw(AssertionError("4xx için sleep olmamalı")))

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        ff.fetch_fimm(out_dir=tmp_path, max_attempts=3, retry_backoff_sec=0.1)
    assert exc_info.value.code == 404
    assert calls["n"] == 1
    assert not (tmp_path / ff.FIMM_ZIP_NAME).with_suffix(".zip.part").exists()


def test_generate_all_figures_byte_exact_determinism(tmp_path: Path) -> None:
    """Opus §3.12 #3 & §3.13: İki ardışık koşu tüm PNG/PDF/SVG çıktılarında bayt-bayt aynı SHA-256 üretir."""
    mpl = pytest.importorskip("matplotlib")
    from src.oncotwin.experiments.make_figures import generate_all_figures

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    h1 = generate_all_figures(out_dir=out1)
    h2 = generate_all_figures(out_dir=out2)

    assert len(h1) == 12, f"4 figür × 3 biçim = 12 dosya bekleniyordu, {len(h1)} geldi"
    assert h1 == h2, f"Determinizm bozuldu: {h1} != {h2}"

    # Sabitlenen matplotlib sürümünde depodaki figürlerle birebir bayt eşleşmesini doğrula
    repo_fig_dir = Path("docs/manuscript/figures")
    if repo_fig_dir.exists() and mpl.__version__ == PINNED_MATPLOTLIB_VERSION:
        for fname, expected_sha in h1.items():
            f = repo_fig_dir / fname
            assert f.exists(), f"Depoda eksik figür dosyası: {f}"
            actual_sha = ff.hashlib.sha256(f.read_bytes()).hexdigest()
            assert actual_sha == expected_sha, f"Depodaki figür güncel değil: {f}"
