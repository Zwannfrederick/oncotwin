"""DEV-OPUS-08: Sprint 3 v2 hasta vektörlerinden simetrik, Holm düzeltmeli Adaptive-vs-StaticDOpt yeniden analizi.

Yalnız `sprint03_v2_vectors.npz` okur (model yeniden fit edilmez). Sol §3.2/3 kuralı; keşifsel.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.eval.metrics import paired_bootstrap_ci
from src.oncotwin.experiments.sprint03_gate import HOLM_FAMILY, adaptive_vs_static_symmetric_holm

DEFAULT_IN_NPZ = Path("data/processed/sprint03_v2_vectors.npz")
DEFAULT_OUT_JSON = Path("data/processed/sprint03_v2_reanalysis.json")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-03-v2-reanalysis.md")
N_BOOT = 10000
METRIC_VECTORS = {"rmse_diff": "patient_rmse_sq", "ndcg3_diff": "patient_ndcg3", "regret1_diff": "patient_regret1"}
REGIMES = {"cv_5fold_3seeds": "cv", "wave_ood": "wave"}


def reanalyze_adaptive_vs_static(vectors: Mapping[str, np.ndarray], n_boot: int = N_BOOT) -> dict[str, Any]:
    """Her rejimde 9 eşleştirilmiş testi (k={3,5,8} x RMSE/NDCG@3/Regret@1) üretip simetrik Holm etiketini döndürür."""
    rmse_fn = lambda v: float(np.sqrt(np.mean(v)))
    out: dict[str, Any] = {}
    for name, prefix in REGIMES.items():
        panel: dict[str, Any] = {}
        for k, metric in HOLM_FAMILY:
            vk = METRIC_VECTORS[metric]
            a = vectors[f"{prefix}/by_k/{k}/AdaptiveOED/{vk}"]
            b = vectors[f"{prefix}/by_k/{k}/StaticDOpt/{vk}"]
            panel.setdefault(f"k{k}_AdaptiveOED_vs_StaticDOpt", {})[metric] = paired_bootstrap_ci(
                a, b, n_boot=n_boot, agg_fn=rmse_fn if metric == "rmse_diff" else None
            )
        out[name] = adaptive_vs_static_symmetric_holm({"panel": panel})
    out["n_boot"] = n_boot
    out["regime_roles"] = {"cv_5fold_3seeds": "birincil aile (keşifsel)", "wave_ood": "keşifsel stres testi"}
    return out


def render_reanalysis_md(res: dict[str, Any]) -> str:
    lines = [
        "# Sprint 3 v2 — Adaptive vs Static-D-Optimal Simetrik Holm Yeniden Analizi (Keşifsel)",
        "",
        "- **Kaynak:** `data/processed/sprint03_v2_vectors.npz` (hasta düzeyi vektörler; model yeniden fit edilmedi).",
        "- **Kural (Sol §3.2/3, Sprint 3 sonuçları görüldükten sonra tanımlandı):** aynı hastalar ve hedefler üzerinde "
        "`k={3,5,8}` × `{RMSE, NDCG@3, Regret@1}` = 9 iki yönlü eşleştirilmiş bootstrap testi "
        f"(B={res['n_boot']}, seed 2026), Holm düzeltmesi α=0.05. Fark = AdaptiveOED − StaticDOpt.",
        "- **Etiketler:** düzeltilmiş anlamlı test yok → `INSUFFICIENT_EVIDENCE` (yetersiz kanıt); iki yönde anlamlı → "
        "`MIXED` (karışık); tek yön → `ADAPTIVE_FAVORED` / `STATIC_FAVORED`. Pratik eşdeğerlik iddiası içermez.",
        "- v1 `ADAPTIVE_WINS` etiketi (`sprint03_results.json`) değiştirilmedi; bu analiz onun yerine geçmez, yanına konur.",
        "",
    ]
    for name, title in (("cv_5fold_3seeds", "ID 5-Fold CV × 3 seed"), ("wave_ood", "Temiz W1+2 → W3+4 (stres testi)")):
        r = res[name]
        lines += [
            f"## {title}: **`{r['label']}`**",
            "",
            "| $k$ | Metrik | Fark [%95 GA] | p (ham) | p (Holm) | Anlamlı | Lehine |",
            "|---:|---|---|---:|---:|:---:|---|",
        ]
        for t in r["tests"]:
            lines.append(
                f"| {t['k']} | `{t['metric'].replace('_diff', '')}` | {t['diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
                f"{t['p_raw']:.4f} | {t['p_holm']:.4f} | {'✓' if t['significant'] else '—'} | {t['favors']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint 3 v2 simetrik Holm Adaptive-vs-Static yeniden analizi.")
    parser.add_argument("--in-npz", type=Path, default=DEFAULT_IN_NPZ)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    with np.load(args.in_npz) as z:
        res = reanalyze_adaptive_vs_static({k: z[k] for k in z.files})
    args.out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_reanalysis_md(res), encoding="utf-8")
    print(json.dumps({n: res[n]["label"] for n in REGIMES}, indent=2))


if __name__ == "__main__":
    main()
