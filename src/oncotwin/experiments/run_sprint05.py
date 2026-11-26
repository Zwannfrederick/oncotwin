"""Sprint 5 ana orkestratörü: Kulvar A (M4 + Kapı 3), Kulvar B (Sıra-Konformal + AUGRC), Kulvar C (FIMM)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.data.fimm_dataset import DEFAULT_CATALOG_AUDIT_JSON, run_fimm_catalog_audit
from src.oncotwin.eval.uncertainty import json_sanitize
from src.oncotwin.experiments.run_sprint05_fimm import run_track_c_fimm
from src.oncotwin.experiments.run_sprint05_m4 import run_track_a_m4
from src.oncotwin.experiments.run_sprint05_shortlist import run_track_b_shortlist
from src.oncotwin.experiments.sprint05_report import render_sprint05_md

PROTOCOL_S05_PATH = Path("docs/PROTOCOL_SPRINT05.md")
DEFAULT_OUT_JSON = Path("data/processed/sprint05_results.json")
DEFAULT_OUT_NPZ = Path("data/processed/sprint05_vectors.npz")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-05-m4-shortlist-fimm.md")


def run_sprint05(
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_npz: Path = DEFAULT_OUT_NPZ,
) -> dict[str, Any]:
    """Dondurulmuş PROTOCOL_SPRINT05.md altında tüm Sprint 5 kulvarlarını koşturur ve kaydeder."""
    if not PROTOCOL_S05_PATH.exists():
        raise FileNotFoundError(f"Önce protokol dondurulmalı: {PROTOCOL_S05_PATH}")
    if not DEFAULT_CATALOG_AUDIT_JSON.exists():
        run_fimm_catalog_audit(DEFAULT_CATALOG_AUDIT_JSON)

    t0 = time.perf_counter()
    ds_all = load_beataml_aml_dataset()
    ds_idx = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    target_idx = [ds_idx.drug_names.index(d) for d in CLINICAL_CANDIDATE_DRUGS]

    track_a, vecs_a = run_track_a_m4(ds_idx, target_idx)
    track_b, vecs_b = run_track_b_shortlist(ds_idx, target_idx)
    track_c, vecs_c = run_track_c_fimm(ds_idx)

    res: dict[str, Any] = {
        "protocol_path": str(PROTOCOL_S05_PATH),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "track_a_m4": track_a,
        "track_b_shortlist": track_b,
        "track_c_fimm": track_c,
    }
    clean = json_sanitize(res)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    npz_payload = {**vecs_a, **vecs_b, **vecs_c}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **npz_payload)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_sprint05_md(clean), encoding="utf-8")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoTwin Sprint 5: M4 (Kapı 3), Sıra-Konformal Kısa Liste, FIMM.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    args = parser.parse_args()
    res = run_sprint05(args.out_json, args.out_md, args.out_npz)
    g3 = res["track_a_m4"]["gate3"]["decision"]
    print(f"Sprint 5 tamamlandı ({res['elapsed_seconds']:.1f} s, Kapı 3: {g3}) -> {args.out_md}")


if __name__ == "__main__":
    main()
