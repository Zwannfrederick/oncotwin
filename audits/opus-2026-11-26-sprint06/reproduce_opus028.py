"""Opus — Sol'un OPUS-028 deneyinin bağımsız yeniden üretimi (salt okunur).

1. Sol'un betiğini değiştirmeden çalıştırır ve arşiv JSON'u ile birebir karşılaştırır.
2. Betiğin `paired_bootstrap_ci` çağrılarındaki hasta vektörlerini yakalar ve aralıkları
   bağımsız bir bootstrap uygulamasıyla (farklı tohum, 4000 tekrar) yeniden hesaplar.
Kullanım: PYTHONPATH=. python3 audits/opus-2026-11-26-sprint06/reproduce_opus028.py OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from src.oncotwin.eval.uncertainty import json_sanitize

SOL_SCRIPT = Path("audits/sol-sprint06/opus028_beataml_probe_scale.py")
SOL_JSON = Path("audits/sol-sprint06/opus028_beataml_probe_scale.json")


def _independent_ci(a: np.ndarray, b: np.ndarray, seed: int = 99, n_boot: int = 4000) -> dict:
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n": int(m.sum())}


def main(out_path: str) -> None:
    spec = importlib.util.spec_from_file_location("sol_opus028", SOL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    orig = mod.paired_bootstrap_ci

    def spy(a, b, **kw):
        captured.append((np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()))
        return orig(a, b, **kw)

    mod.paired_bootstrap_ci = spy
    res = json_sanitize(mod.run())
    archived = json.loads(SOL_JSON.read_text())
    report = {"archive_json_identical": res == archived, "independent_ci": {}}

    i = 0
    for k, node in res["budgets"].items():
        for pair, metrics in node["paired_diffs"].items():
            for metric, ci in metrics.items():
                a, b = captured[i]
                i += 1
                ind = _independent_ci(a, b)
                report["independent_ci"][f"k{k}/{pair}/{metric}"] = {
                    "sol": [round(ci["diff"], 4), round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
                    "opus": [round(ind["diff"], 4), round(ind["ci_low"], 4), round(ind["ci_high"], 4)],
                    "same_sign_of_ci_vs_zero": (ci["ci_low"] > 0) == (ind["ci_low"] > 0)
                    and (ci["ci_high"] < 0) == (ind["ci_high"] < 0),
                }
    report["n_captured"] = i
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "independent_ci"}))
    for k, v in report["independent_ci"].items():
        print(k, v)


if __name__ == "__main__":
    main(sys.argv[1])
