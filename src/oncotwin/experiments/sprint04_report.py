"""Sprint 4 Markdown rapor üreticisi (`docs/sprints/sprint-04-conformal-abstain.md`)."""

from __future__ import annotations

import math
from typing import Any


def _is_fin(val: Any) -> bool:
    return val is not None and isinstance(val, (int, float)) and math.isfinite(float(val))


def _fmt(val: Any, digits: int = 3, suffix: str = "") -> str:
    if not _is_fin(val):
        return "hesaplanamadı"
    return f"{float(val):.{digits}f}{suffix}"


def _pct(val: Any, digits: int = 1) -> str:
    if not _is_fin(val):
        return "hesaplanamadı"
    return f"%{float(val) * 100.0:.{digits}f}"


def _fmt_ci(ci: dict[str, Any] | None, digits: int = 3) -> str:
    if not ci or not _is_fin(ci.get("diff")):
        return f"hesaplanamadı (n={ci.get('n_paired', 0) if ci else 0})"
    d = float(ci["diff"])
    lo = float(ci["ci_low"])
    hi = float(ci["ci_high"])
    return f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] (p={float(ci['p_two_sided']):.3f}, n={ci['n_paired']})"


def _arm_order(arms_dict: dict[str, Any]) -> list[str]:
    ordered = ["M0_k0", "M1_k0"]
    for k in (2, 5, 8):
        for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
            name = f"{pol}_k{k}"
            if name in arms_dict:
                ordered.append(name)
    return [a for a in ordered if a in arms_dict]


def _render_archive_conformal_table(arms: dict[str, Any]) -> list[str]:
    lines = [
        "| Kol | `n_eval` | `RMSE` ↓ | `NDCG@3` ↑ | `Regret@1` ↓ | Sonlu Aralık | $2\\hat{q}$ | Eşzamanlı Kapsam (%90 hed.) | İlaç Ampirik Kapsam | İlaç Marjinal (Shokrzadeh stili) | Kısa Liste (gözl.) | Gerçek 1. Listede | Kesin-1 Oranı |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _arm_order(arms):
        node = arms[name]
        tm = node["test_metrics"]
        cf = node["conformal"]
        pub = "✅" if cf.get("finite_interval_published") else "❌ (∞)"
        width = _fmt(cf.get("width_sim"), 1) if cf.get("finite_interval_published") else "∞"
        lines.append(
            f"| `{name}` | {tm['n_eval']} | {_fmt(tm['rmse'], 2)} | {_fmt(tm['ndcg_at_3'], 3)} | "
            f"{_fmt(tm['regret_at_1'], 1)} | {pub} | {width} | {_pct(cf.get('simultaneous_coverage'))} | "
            f"{_pct(cf.get('per_drug_empirical_coverage_sim'))} | {_pct(cf.get('per_drug_marginal_conformal_coverage'))} | "
            f"{_fmt(cf.get('mean_shortlist_size_obs'), 2)} | {_pct(cf.get('shortlist_contains_true_top1_rate'))} | "
            f"{_pct(cf.get('definite_top1_rate_obs'))} |"
        )
    return lines


def _render_locked_table(arms: dict[str, Any]) -> list[str]:
    lines = [
        "| Kol | Hedef-Uygun `n` | Tam Panel `n` | Bulunabilirlik | Teknik Çekinme | `RMSE` (tam-panel) | `Regret@1` (tam-panel) | Eşzamanlı Kapsam (`cal n`) | Kısa Liste (gözl.) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in _arm_order(arms):
        node = arms[name]
        av = node["availability"]
        tm = node["test_metrics"]
        cf = node["conformal"]
        cov_str = (
            f"{_pct(cf.get('simultaneous_coverage'))} (n_cal={cf.get('n_cal', 0)})"
            if cf.get("finite_interval_published") and _is_fin(cf.get("simultaneous_coverage"))
            else f"hesaplanamadı (n_cal={cf.get('n_cal', 0)})"
        )
        lines.append(
            f"| `{name}` | {av['n_target_eligible_test']} | {av['n_complete_test']} | "
            f"{_pct(av['panel_availability_rate_test'])} | {_pct(av['technical_abstain_rate_test'])} | "
            f"{_fmt(tm.get('rmse'), 2)} | {_fmt(tm.get('regret_at_1'), 1)} | {cov_str} | "
            f"{_fmt(cf.get('mean_shortlist_size_obs'), 2)} |"
        )
    return lines


def _render_abstention_table(arms: dict[str, Any], include_cap: bool = False) -> list[str]:
    hdr_cap = " | Ortak-Cap AURC" if include_cap else ""
    sep_cap = "---:|" if include_cap else ""
    lines = [
        f"| Kol | Skor | Kilitli $\\tau_{{0.80}}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC`{hdr_cap} |",
        f"|---|---|---:|---:|---:|---:|---:|---:|---:|{sep_cap}",
    ]
    for name in _arm_order(arms):
        node = arms[name]
        for s_name in ("omics_pca_dist", "m3_post_scale"):
            ab = node["abstention"][s_name]
            th = ab["threshold_eval"]
            rc = ab["risk_coverage_regret1"]
            acc_str = (
                f"{_pct(th.get('acceptance_rate_among_available'))} / {_pct(th.get('acceptance_rate_total'))}"
                if th.get("status") == "ok"
                else "hesaplanamadı"
            )
            cap_col = f" | {_fmt(rc.get('aurc_capped'), 2)}" if include_cap else ""
            const_note = " (sabit skor)" if s_name == "m3_post_scale" and name.endswith("_k0") else ""
            lines.append(
                f"| `{name}` | `{s_name}`{const_note} | {_fmt(ab.get('tau_locked'), 3)} | {acc_str} | "
                f"{_fmt(th.get('accepted_mean_regret1'), 1)} | {_fmt(th.get('abstained_mean_regret1'), 1)} | "
                f"{_fmt(rc.get('aurc'), 2)} | {_fmt(rc.get('e_aurc'), 2)} | {_fmt(rc.get('random_aurc'), 2)}{cap_col} |"
            )
    return lines


def _render_locked_common_comparisons(comp_lock: dict[str, Any]) -> list[str]:
    lines = [
        "**Ortak Tam-Panel Hasta Maskesinde Eşleştirilmiş Kıyaslar (`n_paired < 5 → hesaplanamadı`):**",
        "",
        "| Karşılaştırma | `n_paired` | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `AURC(A)` / `AURC(B)` (ortak `n`) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cname, node in comp_lock.get("policy_pairs", {}).items():
        n_p = int(node.get("n_paired", 0))
        if node.get("status") != "ok":
            lines.append(f"| `{cname}` | {n_p} | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |")
        else:
            aurc_ab = f"{_fmt(node.get('aurc_omics_a'), 2)} / {_fmt(node.get('aurc_omics_b'), 2)}"
            lines.append(
                f"| `{cname}` | {n_p} | {_fmt_ci(node.get('rmse_diff'), 2)} | {_fmt_ci(node.get('regret1_diff'), 2)} | "
                f"{_fmt_ci(node.get('ndcg3_diff'), 3)} | {aurc_ab} |"
            )
    lines.extend(["", "**Üçlü Ortak Tam-Panel Kesişimi (`n_paired < 5 → hesaplanamadı`):**", ""])
    lines.append("| Bütçe | `n_paired` | `StaticDOpt` (RMSE / Reg1 / AURC) | `EqualWeightVarRed` (RMSE / Reg1 / AURC) | `AdaptiveOED` (RMSE / Reg1 / AURC) |")
    lines.append("|---|---:|---:|---:|---:|")
    for k_key, node in comp_lock.get("three_way_common", {}).items():
        n_p = int(node.get("n_paired", 0))
        if node.get("status") != "ok":
            lines.append(f"| `{k_key}` | {n_p} | hesaplanamadı | hesaplanamadı | hesaplanamadı |")
        else:
            a = node["arms"]
            fmt_p = lambda p: f"{_fmt(a[p]['rmse'], 2)} / {_fmt(a[p]['regret_at_1'], 1)} / {_fmt(a[p]['aurc_omics_regret1'], 2)}"
            lines.append(f"| `{k_key}` | {n_p} | {fmt_p('StaticDOpt')} | {fmt_p('EqualWeightVarRed')} | {fmt_p('AdaptiveOED')} |")
    return lines


def render_sprint04_md(res: dict[str, Any]) -> str:
    id_res = res["regimes"]["id_60_20_20"]
    wv_res = res["regimes"]["wave_ood"]
    id_arch = id_res["scenarios"]["archive_available"]
    id_lock = id_res["scenarios"]["locked_train_catalog"]
    wv_arch = wv_res["scenarios"]["archive_available"]
    wv_lock = wv_res["scenarios"]["locked_train_catalog"]
    pnotes = res.get("protocol_notes", {})

    lines: list[str] = [
        "# Sprint 4 v2 — Hasta-Birimli Split-Conformal Kalibrasyon, Kısa Aday Listesi ve OOD Çekinme (`DEV-OPUS-13..16` / `SOL-09..12`)",
        "",
        f"- **Çalışma Süresi:** `{res['elapsed_seconds']:.1f} s`",
        f"- **Protokol & v2 Sürüm Notu:** [PROTOCOL_SPRINT04.md](../PROTOCOL_SPRINT04.md) korunmuştur; tarihsel v1 kayıtları (`sprint04_results.json`, [sprint-04-conformal-abstain.md](sprint-04-conformal-abstain.md)) değiştirilmemiştir. Bu v2 koşusu `{pnotes.get('version', 'v2')}` düzeltmelerini içerir:",
        f"  - **`SOL-09` (`DEV-OPUS-13`):** {pnotes.get('sol_09_fix', 'Fold başına tek kilitli contender_seed.')}",
        f"  - **`SOL-10` (`DEV-OPUS-14`):** {pnotes.get('sol_10_fix', 'Ortak tam-panel hasta maskesinde kıyas.')}",
        f"  - **`SOL-11` (`DEV-OPUS-15`):** {pnotes.get('sol_11_fix', 'q=inf durumunda NaN/hesaplanamadı.')}",
        f"  - **`SOL-12` (`DEV-OPUS-16`):** {pnotes.get('sol_12_fix', 'M0(0) ek taban kolu eklendi.')}",
        f"  - **`DEV-OPUS-19` Durumu:** {pnotes.get('pending_addendum_19', 'Sol A1 netleştirmesi bekleniyor; v2 içinde uygulanmadı.')}",
        f"- **Sabitler:** Seed `{res['seed']}`, nominal $\\alpha = {res['alpha']:.2f}$ → %90, doğrulama kabul hedefi %80.",
        "- **Kapsam & Etiketleme Beyanı:**",
        "  - `id_60_20_20`: Beat AML 2.0'ın önceki sprintlerde toplu analize giren indeks hastalarının 60/20/20 hasta-bazlı ayrımıdır (*yeniden kullanılan kohort* kapsamıyla sınırlıdır).",
        "  - `wave_ood` (`Waves1+2 → Waves3+4`): `Waves3+4` Sprint 2–3'te incelendiği için tüm `Waves3+4` sonuçları **keşifsel OOD stres testi** olarak etiketlenmiştir; OOD üzerinde konformal kapsam garantisi iddia edilmez.",
        "  - **Plaka Kataloğu Durumu:** Beat AML 2.0 içinde bağımsız belgelenmiş prospektif plaka-kataloğu manifesti bulunmadığından *belgelenmiş plaka kataloğu* senaryosu **`hesaplanamadı`** olarak kaydedilmiştir. `locked_train_catalog` yalnızca `train_tune` kataloğuna dayalı ayrı simülasyon olarak raporlanır.",
        "  - **Kabul Edilen Alt Grup Sınırı:** Çekinme eşiği altında kabul edilen alt grup için ayrı teorik kapsam garantisi verilmez; kabul edilen eşzamanlı kapsam yalnızca tanısaldır.",
        "  - **Literatür Konumlandırması:** Shokrzadeh & Shokrzadeh (bioRxiv Haziran 2026, `2026.06.07.730728v1`) ilaç-bazlı marjinal split-conformal makro kapsam raporlamaktadır; OncoTwin ise sabit prob bütçesinde **hasta-birimli eşzamanlı kapsama**, **kısa aday listesi daraltması** ve **OOD çekinme (`AURC`)** ölçmektedir (karşılaştırma için ilaç-bazlı marjinal sütunu ayrı verilmiştir).",
        "",
        "---",
        "",
        "## 1. ID 60/20/20 (`id_60_20_20`, Yeniden Kullanılan Kohort) — `archive_available` Ana Senaryo",
        "",
    ]
    lines.extend(_render_archive_conformal_table(id_arch["arms"]))
    lines.extend(["", "### 1.1 ID `archive_available` Eşleştirilmiş Bootstrap Kıyasları (Seed 2026, B=1000)", ""])
    comp_id = id_res["comparisons"]["archive_available"]
    lines.append("| Karşılaştırma | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `ΔKısa Liste Boyutu` |")
    lines.append("|---|---:|---:|---:|---:|")
    for cname, cb in comp_id["policy_pairs"].items():
        lines.append(
            f"| `{cname}` | {_fmt_ci(cb.get('rmse_diff'), 2)} | {_fmt_ci(cb.get('regret1_diff'), 2)} | "
            f"{_fmt_ci(cb.get('ndcg3_diff'), 3)} | {_fmt_ci(cb.get('shortlist_size_diff'), 2)} |"
        )

    lines.extend([
        "",
        "### 1.2 ID `archive_available` Seçici Risk–Kapsam (`AURC`) ve Kilitli %80 Eşik Çekinme Özeti",
        "",
    ])
    lines.extend(_render_abstention_table(id_arch["arms"], include_cap=False))
    lines.extend(["", "**ID `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları (`omics_pca_dist` vs karşılaştırıcılar):**", ""])
    for kname, bci in comp_id["aurc_pairs"].items():
        lines.append(f"- `{kname}`: `{_fmt_ci(bci, 2)}`")

    lines.extend([
        "",
        "---",
        "",
        "## 2. ID `locked_train_catalog` (Salt Eğitim Kataloğu Simülasyonu — Geri Doldurmasız)",
        "",
        "> **Not:** Aşağıdaki tablodaki her satır o kolun **kendi tam-panel hastaları** üzerinde betimleyicidir; farklı hasta kümelerine sahip satırlar politika üstünlüğü kıyası için kullanılmaz. Eşleştirilmiş kıyaslar §2.2'dedir.",
        "",
    ])
    lines.extend(_render_locked_table(id_lock["arms"]))
    lines.extend(["", "### 2.1 ID `locked_train_catalog` Kendi Tam-Panel Alt Kümesinde Betimleyici Çekinme ve `AURC`", ""])
    lines.extend(_render_abstention_table(id_lock["arms"], include_cap=False))
    lines.extend(["", "### 2.2 ID `locked_train_catalog` Ortak Tam-Panel Eşleştirilmiş Kıyaslar", ""])
    lines.extend(_render_locked_common_comparisons(id_res["comparisons"]["locked_train_catalog"]))

    lines.extend([
        "",
        "---",
        "",
        "## 3. Temiz `Waves1+2 → Waves3+4` Keşifsel OOD Stres Testi (`archive_available` & `locked_train_catalog`)",
        "",
        "> **Not:** Aşağıdaki tüm `Waves3+4` değerleri keşifsel stres testi kapsamındadır; `Waves3+4` hastaları eğitim, eşik kilitleme veya kalibrasyona girmemiştir.",
        "",
        "### 3.1 `Waves3+4` — `archive_available` Konformal, Kısa Liste ve Eşzamanlı Kapsam Kayması",
        "",
    ])
    lines.extend(_render_archive_conformal_table(wv_arch["arms"]))
    lines.extend(["", "### 3.2 `Waves3+4` — `archive_available` Kilitli %80 Eşik Çekinme ve `AURC`", ""])
    lines.extend(_render_abstention_table(wv_arch["arms"], include_cap=False))
    lines.extend(["", "**`Waves3+4` `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları:**", ""])
    for kname, bci in wv_res["comparisons"]["archive_available"]["aurc_pairs"].items():
        lines.append(f"- `{kname}`: `{_fmt_ci(bci, 2)}`")

    lines.extend(["", "### 3.3 `Waves3+4` — `locked_train_catalog` Bulunabilirlik ve Tam-Panel Durumu", ""])
    lines.extend(_render_locked_table(wv_lock["arms"]))
    lines.extend(["", "### 3.4 `Waves3+4` — `locked_train_catalog` Kendi Tam-Panelinde Betimleyici Risk–Kapsam ve Ortak Kıyas", ""])
    lines.extend(_render_abstention_table(wv_lock["arms"], include_cap=False))
    lines.append("")
    lines.extend(_render_locked_common_comparisons(wv_res["comparisons"]["locked_train_catalog"]))
    lines.append("")
    return "\n".join(lines)
