"""Sprint 4 v2 DEV-OPUS-19 keşifsel eşik duyarlılık modülü (nihai fit ölçeğinde m3_post_scale)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.eval.splits import get_evaluable_mask, make_sprint04_splits
from src.oncotwin.eval.uncertainty import (
    evaluate_locked_threshold_abstention,
    json_sanitize,
    m3_conditional_posterior_scale,
)
from src.oncotwin.experiments.run_sprint04 import (
    K_BUDGETS,
    MIN_EVAL_PROBES_ARCHIVE,
    MIN_EVAL_TARGETS,
    MIN_TRAIN_PROBE_OBS,
    SPRINT04_SEED,
    TARGET_ACCEPT_RATE,
    _choose_locked_panels,
    _fit_sprint04_fold,
    _open_archive_mask_for_locked,
    _select_archive_panels,
)
from src.oncotwin.models.panel_selection import compute_contender_weights

PROTOCOL_ADDENDUM_19 = Path("docs/PROTOCOL_SPRINT04_ADDENDUM_19.md")
V2_RESULTS_JSON = Path("data/processed/sprint04_v2_results.json")
V2_VECTORS_NPZ = Path("data/processed/sprint04_v2_vectors.npz")
V2_MANIFEST_JSON = Path("data/processed/sprint04_v2_split_manifest.json")
DEFAULT_OUT_JSON = Path("data/processed/sprint04_v2_dev_opus19_results.json")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-04-v2-dev-opus19-sensitivity.md")


def summarize_reference_threshold(
    ref_scores: np.ndarray,
    valid_mask: np.ndarray,
    n_target_eligible: int,
    target_accept: float = TARGET_ACCEPT_RATE,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Sonlu referans skorları üzerinden Addendum 19 §2.4–§3 izlenebilirlik özetini üretir."""
    sc = np.asarray(ref_scores, dtype=np.float64)
    vm = np.asarray(valid_mask, dtype=bool)
    vals = sc[vm & np.isfinite(sc)]
    n_ref = int(len(vals))
    n_tgt = int(n_target_eligible)
    n_tech_abst = max(0, n_tgt - n_ref)

    if n_ref == 0:
        return {
            "status": "hesaplanamadı (sonlu referans skor yok)",
            "is_constant_score": False,
            "n_ref": 0,
            "n_ref_target_eligible": n_tgt,
            "n_ref_technical_abstain": n_tech_abst,
            "n_unique_scores": 0,
            "rank_1based": None,
            "tau_final": float("nan"),
            "n_ref_below_tau": 0,
            "n_ref_equal_tau": 0,
            "n_ref_accepted": 0,
            "ref_accept_rate_among_ref": float("nan"),
            "ref_accept_rate_among_target_eligible": float("nan"),
        }

    sorted_vals = np.sort(vals)
    rank_1based = int(1 + math.ceil(target_accept * (n_ref - 1)))
    tau_final = float(np.quantile(vals, target_accept, method="higher"))
    n_unique = 1 + int(np.sum(np.diff(sorted_vals) > tie_tol))
    below = int(np.sum(vals < tau_final))
    equal = int(np.sum(vals == tau_final))
    accepted = int(np.sum(vals <= tau_final))
    is_const = bool(n_unique == 1)

    return {
        "status": "sabit skor — ayırt edici değil" if is_const else "ok",
        "is_constant_score": is_const,
        "n_ref": n_ref,
        "n_ref_target_eligible": n_tgt,
        "n_ref_technical_abstain": n_tech_abst,
        "n_unique_scores": n_unique,
        "rank_1based": rank_1based,
        "tau_final": tau_final,
        "n_ref_below_tau": below,
        "n_ref_equal_tau": equal,
        "n_ref_accepted": accepted,
        "ref_accept_rate_among_ref": accepted / float(n_ref),
        "ref_accept_rate_among_target_eligible": (accepted / float(n_tgt)) if n_tgt > 0 else float("nan"),
    }


def compute_final_fit_reference_scores(
    fit_obj: dict[str, Any],
    avail_mask_tr: np.ndarray,
    X_rna_tr: np.ndarray,
    X_mut_tr: np.ndarray,
    target_idx: list[int],
    k_budgets: tuple[int, ...] = K_BUDGETS,
    min_targets: int = MIN_EVAL_TARGETS,
    min_probes: int = MIN_EVAL_PROBES_ARCHIVE,
    target_accept: float = TARGET_ACCEPT_RATE,
    adaptive_m1_source: str = "oof",
) -> dict[str, dict[str, dict[str, Any]]]:
    """Dondurulmuş nihai fit ve yalnız ikili mevcudiyet maskesiyle referans τ_final değerlerini hesaplar."""
    if adaptive_m1_source not in ("oof", "insample"):
        raise ValueError(f"Geçersiz adaptive_m1_source: {adaptive_m1_source}")

    avail_bool = np.asarray(avail_mask_tr, dtype=bool)
    # Yapısal izolasyon: sayısal AUC'lere erişim olmaması için yalnız 0.0/NaN maske matrisi kurulur.
    Y_mask_only = np.where(avail_bool, 0.0, np.nan)
    arch_tr_mask = get_evaluable_mask(
        Y_mask_only, target_idx, fit_obj["probe_pool_idx"], min_targets=min_targets, min_probes=min_probes
    )
    tgt_tr_mask = np.sum(avail_bool[:, target_idx], axis=1) >= min_targets
    n_tgt_elig = int(np.sum(tgt_tr_mask))
    n_arch_elig = int(np.sum(arch_tr_mask))

    if adaptive_m1_source == "oof":
        m1_tr = fit_obj["M1"].Y_oof_
        if m1_tr is None or not np.all(np.isfinite(m1_tr[:, target_idx])):
            raise ValueError("Adaptive referans için sonlu M1 Y_oof_ mevcut değil.")
    else:
        m1_tr = fit_obj["M1"].predict(X_rna_tr, X_mut_tr, n_drugs=avail_bool.shape[1])

    cov_cc = fit_obj["cov_ref"][np.ix_(fit_obj["target_local"], fit_obj["target_local"])]
    cont_tr = compute_contender_weights(
        m1_tr[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=fit_obj["contender_seed"]
    )
    uniform_w = np.full(len(target_idx), 1.0 / len(target_idx), dtype=np.float64)
    target_set = set(target_idx)
    g2l = fit_obj["g2l"]
    cov_model = fit_obj["M3"]._cov_model

    out: dict[str, dict[str, dict[str, Any]]] = {"archive_available": {}, "locked_train_catalog": {}}
    for scen in ("archive_available", "locked_train_catalog"):
        out[scen]["M0_k0"] = {
            "status": "skor uygulanamaz (k=0, M0 modeli)",
            "threshold_assigned": False,
            "tau_final": float("nan"),
            "n_ref_target_eligible": n_tgt_elig,
        }
        out[scen]["M1_k0"] = {
            "status": "sabit skor — yeni eşik atanmadı (k=0, prob yok)",
            "threshold_assigned": False,
            "tau_final": float("nan"),
            "n_ref_target_eligible": n_tgt_elig,
        }
        for k in k_budgets:
            cov_k = cov_model.cov_by_k_[k]
            _, lam_k = cov_model.best_params_by_k_[k]
            for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
                arm = f"{pol}_k{k}"
                if scen == "archive_available":
                    probes_tr = _select_archive_panels(
                        Y_mask_only, fit_obj, cont_tr, pol, k, arch_tr_mask, uniform_w, target_set
                    )
                    valid_ref = arch_tr_mask
                else:
                    locked_tr = _choose_locked_panels(fit_obj, cont_tr, pol, k, uniform_w, target_set)
                    probes_tr, comp_tr, _ = _open_archive_mask_for_locked(locked_tr, Y_mask_only, k)
                    valid_ref = tgt_tr_mask & comp_tr

                probes_loc = [np.asarray([g2l[int(g)] for g in s], dtype=np.int64) for s in probes_tr]
                sc_tr = m3_conditional_posterior_scale(cov_k, probes_loc, fit_obj["target_local"], lam=lam_k)
                sc_masked = sc_tr.copy()
                sc_masked[~valid_ref] = np.nan

                summary = summarize_reference_threshold(
                    sc_masked, valid_ref, n_target_eligible=n_tgt_elig, target_accept=target_accept
                )
                summary["threshold_assigned"] = True
                summary["adaptive_m1_source"] = adaptive_m1_source if pol == "AdaptiveOED" else "not_applicable"
                if scen == "archive_available":
                    summary["n_ref_archive_eval"] = n_arch_elig
                    summary["n_ref_excluded_probe_insufficient"] = max(0, n_tgt_elig - n_arch_elig)
                out[scen][arm] = summary
    return out


def _assert_eval_close(ev_calc: dict[str, Any], ev_v2: dict[str, Any], label: str) -> None:
    assert ev_calc["status"] == ev_v2["status"], f"{label}: status mismatch"
    for k, v2_val in ev_v2.items():
        if k == "status":
            continue
        c_val = ev_calc[k]
        if v2_val is None or (isinstance(c_val, float) and not math.isfinite(c_val)):
            assert v2_val is None and isinstance(c_val, float) and not math.isfinite(c_val), f"{label}.{k}"
        elif isinstance(v2_val, int):
            assert int(c_val) == v2_val, f"{label}.{k}: {c_val} != {v2_val}"
        else:
            assert abs(float(c_val) - float(v2_val)) <= 1e-12, f"{label}.{k}: {c_val} != {v2_val}"


def _eval_with_tau(
    z_v2: Any,
    rn: str,
    scen: str,
    arm: str,
    rc_mask: np.ndarray,
    tau_val: float,
    n_tgt_te: int,
) -> dict[str, Any]:
    sc = z_v2[f"{rn}/{scen}/{arm}/m3_post_scale"][rc_mask]
    reg = z_v2[f"{rn}/{scen}/{arm}/patient_regret1"][rc_mask]
    rsq = z_v2[f"{rn}/{scen}/{arm}/patient_rmse_sq"][rc_mask]
    cov = z_v2[f"{rn}/{scen}/{arm}/patient_covered_sim"][rc_mask]
    vm = z_v2[f"{rn}/{scen}/{arm}/valid_mask"][rc_mask]
    ev = evaluate_locked_threshold_abstention(sc, reg, rsq, cov, tau_locked=tau_val, available_mask=vm)
    ev["acceptance_rate_among_target_eligible"] = (
        ev["n_accepted"] / float(n_tgt_te) if n_tgt_te > 0 and ev["status"] == "ok" else float("nan")
    )
    return ev


def run_sprint04_dev_opus19(
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    v2_results_json: Path = V2_RESULTS_JSON,
    v2_vectors_npz: Path = V2_VECTORS_NPZ,
    v2_manifest_json: Path = V2_MANIFEST_JSON,
) -> dict[str, Any]:
    """Sprint 4 v2 üzerine DEV-OPUS-19 keşifsel duyarlılık analizini çalıştırır."""
    for req in (PROTOCOL_ADDENDUM_19, v2_results_json, v2_vectors_npz, v2_manifest_json):
        if not req.exists():
            raise FileNotFoundError(f"Gerekli dosya eksik: {req}")

    v2_res = json.loads(v2_results_json.read_text(encoding="utf-8"))
    v2_manifest = json.loads(v2_manifest_json.read_text(encoding="utf-8"))
    z_v2 = np.load(v2_vectors_npz)

    ds_all = load_beataml_aml_dataset()
    ds_idx = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    target_idx = [ds_idx.drug_names.index(d) for d in CLINICAL_CANDIDATE_DRUGS]
    splits = make_sprint04_splits(ds_idx.meta, ds_idx.w12_all_subjects, ds_idx.both_all_subjects, seed=SPRINT04_SEED)
    if splits["manifest"] != v2_manifest:
        raise ValueError("Bölme manifestosu sprint04_v2_split_manifest.json ile uyuşmuyor.")

    regimes_out: dict[str, Any] = {}
    for rn in ("id_60_20_20", "wave_ood"):
        tr_idx = splits[rn]["train_tune"]
        full_fit = _fit_sprint04_fold(
            ds_idx, tr_idx, target_idx, k_budgets=K_BUDGETS, inner_seed=SPRINT04_SEED, min_train_obs=MIN_TRAIN_PROBE_OBS
        )
        ds_tr = ds_idx.subset_by_positions(tr_idx)
        avail_tr = np.isfinite(ds_tr.Y_auc)
        ref_oof = compute_final_fit_reference_scores(
            full_fit, avail_tr, ds_tr.X_rna, ds_tr.X_mut, target_idx, adaptive_m1_source="oof"
        )
        ref_ins = compute_final_fit_reference_scores(
            full_fit, avail_tr, ds_tr.X_rna, ds_tr.X_mut, target_idx, adaptive_m1_source="insample"
        )

        arch_te_mask = z_v2[f"{rn}/test_archive_eval_mask"].astype(bool)
        tgt_te_mask = z_v2[f"{rn}/test_target_eligible_mask"].astype(bool)
        n_tgt_te = int(np.sum(tgt_te_mask))
        n_arch_te = int(np.sum(arch_te_mask))

        v2_reg = v2_res["regimes"][rn]
        scen_out: dict[str, Any] = {"archive_available": {"arms": {}}, "locked_train_catalog": {"arms": {}}}
        for scen in ("archive_available", "locked_train_catalog"):
            rc_mask = arch_te_mask if scen == "archive_available" else tgt_te_mask
            for arm in ("M0_k0", "M1_k0"):
                scen_out[scen]["arms"][arm] = {
                    "threshold_assigned": False,
                    "reference": ref_oof[scen][arm],
                    "note": "Addendum 19 §2.1: M0_k0 ve M1_k0 için yeni m3_post_scale eşiği atanmaz.",
                }
            for k in K_BUDGETS:
                for pol in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED"):
                    arm = f"{pol}_k{k}"
                    tau_oof = float(v2_reg["inner_oof_taus"][scen]["m3_post_scale"][arm])
                    v2_m3 = v2_reg["scenarios"][scen]["arms"][arm]["abstention"]["m3_post_scale"]
                    ev_oof = _eval_with_tau(z_v2, rn, scen, arm, rc_mask, tau_oof, n_tgt_te)
                    _assert_eval_close(
                        {kk: vv for kk, vv in ev_oof.items() if kk != "acceptance_rate_among_target_eligible"},
                        v2_m3["threshold_eval"],
                        f"{rn}.{scen}.{arm}.oof",
                    )

                    ref_node = ref_oof[scen][arm]
                    ev_final = _eval_with_tau(z_v2, rn, scen, arm, rc_mask, float(ref_node["tau_final"]), n_tgt_te)
                    ev_final["descriptive_only"] = bool(ref_node["is_constant_score"])

                    denoms: dict[str, int] = {
                        "n_target_eligible_test": n_tgt_te,
                        "n_complete_test": int(ev_final["n_available"]),
                        "n_technical_abstain_test": max(0, n_tgt_te - int(ev_final["n_available"])),
                    }
                    if scen == "archive_available":
                        denoms["n_archive_eval_test"] = n_arch_te
                        denoms["n_excluded_probe_insufficient_test"] = max(0, n_tgt_te - n_arch_te)

                    rc_v2 = v2_m3["risk_coverage_regret1"]
                    arm_node: dict[str, Any] = {
                        "threshold_assigned": True,
                        "denominators": denoms,
                        "reference": ref_node,
                        "oof_threshold_eval": ev_oof,
                        "final_fit_threshold_eval": ev_final,
                        "v2_aurc_regret1_unchanged": {
                            "aurc": rc_v2["aurc"],
                            "e_aurc": rc_v2["e_aurc"],
                            "random_aurc": rc_v2["random_aurc"],
                            "note": "Eşikten bağımsız; v2 değeri değişmez (Addendum 19 §3).",
                        },
                    }
                    if pol == "AdaptiveOED":
                        ref_i = ref_ins[scen][arm]
                        ev_i = _eval_with_tau(z_v2, rn, scen, arm, rc_mask, float(ref_i["tau_final"]), n_tgt_te)
                        ev_i["descriptive_only"] = bool(ref_i["is_constant_score"])
                        arm_node["insample_m1_sensitivity"] = {
                            "reference": ref_i,
                            "final_fit_threshold_eval": ev_i,
                            "delta_n_accepted_vs_oof": int(ev_i["n_accepted"] - ev_final["n_accepted"]),
                        }
                    scen_out[scen]["arms"][arm] = arm_node
        regimes_out[rn] = scen_out

    res = {
        "metadata": {
            "task_id": "DEV-OPUS-19",
            "addendum_path": str(PROTOCOL_ADDENDUM_19),
            "addendum_date": "2026-11-22",
            "user_approval_date": "2026-11-22",
            "user_approval_note": "Kullanıcı Addendum 19 (A4) ve Sprint 4 kapanış işlerini onayladı; Sprint 5 başlatılmadı.",
            "exploratory_status": "keşifsel duyarlılık analizi (ön kayıtlı değil; Sprint 4 sonuçları daha önce görüldü)",
            "a1_adaptive_reference_m1": "primary=m1_oof (fit_and_oof Y_oof_), auxiliary=m1_insample",
            "a2_locked_constant_note": "Kilitli StaticDOpt/EqualWeightVarRed tüm tam-panel hastalarda tek skor verir (sabit skor — ayırt edici değil).",
            "a3_prior_preview_note": "audits/opus-2026-11-22-sprint04/realdata_probe_s04.json (archive_available/StaticDOpt_k5: ID %92.3, W3+4 %43.9).",
            "v2_unchanged_contract_a": True,
        },
        "regimes": regimes_out,
    }
    clean = json_sanitize(res)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_dev_opus19_md(clean), encoding="utf-8")
    return clean


def _fmt(v: float | None, dec: int = 1, pct: bool = False) -> str:
    if v is None or not math.isfinite(float(v)):
        return "hesaplanamadı"
    return f"%{float(v) * 100.0:.{dec}f}" if pct else f"{float(v):.{dec}f}"


def render_dev_opus19_md(res: dict[str, Any]) -> str:
    """DEV-OPUS-19 eşik duyarlılık analizini Markdown raporu olarak biçimlendirir."""
    md = res["metadata"]
    lines = [
        "# Sprint 4 v2 — `DEV-OPUS-19` Eşik Duyarlılık Analizi (`m3_post_scale` Nihai Fit Ölçeği)",
        "",
        f"- **Protokol Eki:** [{md['addendum_path']}](../../{md['addendum_path']}) (`{md['addendum_date']}`, kullanıcı `A4` onayı: `{md['user_approval_date']}`).",
        f"- **Analiz Statüsü:** {md['exploratory_status']}. Üstünlük etiketi çıkarılmaz; `v2` tahminleri, kısa listeleri, `q_sim`, `omics_pca_dist` ve eşikten bağımsız `AURC` değerleri **değişmemiştir**.",
        f"- **`A1` (Adaptif `M1` Referansı):** `{md['a1_adaptive_reference_m1']}`.",
        f"- **`A2` (Kilitli Sabit Skor):** {md['a2_locked_constant_note']}",
        f"- **`A3` (Ön İzleme Şeffaflığı):** Ana hücre `archive_available / StaticDOpt_k5` için ID `%92.3` ve `Waves3+4` `%43.9` test kabulü ek öncesinde `{md['a3_prior_preview_note']}` çıktısında görülmüştür.",
        "- **`Waves3+4` Uyarısı:** Tüm `Waves3+4` sonuçları keşifsel OOD stres testidir; `M0_k0` ve `M1_k0` için yeni eşik atanmamıştır (§2.1).",
        "",
        "## 1. Ana İnceleme Hücresi Özeti (`archive_available / StaticDOpt_k5`)",
        "",
        "| Rejim | `n_ref` / `uniq` | $\\tau_{\\text{OOF}}$ → $\\tau_{\\text{final}}$ | Ref Kabul | Test `n_tgt` / `n_eval` | OOF Kabul (`Regret@1`) | Nihai-Fit Kabul (`Regret@1`) | Nihai-Fit Çek. (`Regret@1`) | `AURC` (değişmez) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rn, rlab in (("id_60_20_20", "ID 60/20/20"), ("wave_ood", "Waves3+4 (OOD stres)")):
        node = res["regimes"][rn]["archive_available"]["arms"]["StaticDOpt_k5"]
        ref, oof, fin = node["reference"], node["oof_threshold_eval"], node["final_fit_threshold_eval"]
        lines.append(
            f"| {rlab} | {ref['n_ref']} / {ref['n_unique_scores']} | "
            f"{oof['tau_locked']:.3f} → {ref['tau_final']:.3f} | "
            f"{ref['n_ref_accepted']}/{ref['n_ref']} ({_fmt(ref['ref_accept_rate_among_ref'], 1, True)}) | "
            f"{node['denominators']['n_target_eligible_test']} / {node['denominators']['n_archive_eval_test']} | "
            f"{oof['n_accepted']}/{oof['n_available']} ({_fmt(oof['acceptance_rate_among_available'], 1, True)}; {_fmt(oof['accepted_mean_regret1'])}) | "
            f"{fin['n_accepted']}/{fin['n_available']} ({_fmt(fin['acceptance_rate_among_available'], 1, True)}; {_fmt(fin['accepted_mean_regret1'])}) | "
            f"{fin['n_abstained_score']} ({_fmt(fin['abstained_mean_regret1'])}) | "
            f"{_fmt(node['v2_aurc_regret1_unchanged']['aurc'], 2)} |"
        )

    for rn, rtitle in (("id_60_20_20", "2. ID 60/20/20"), ("wave_ood", "3. Waves3+4 (Keşifsel OOD Stres Testi)")):
        lines.extend(["", f"## {rtitle}", ""])
        for scen in ("archive_available", "locked_train_catalog"):
            lines.extend([
                f"### {scen}",
                "",
                "| Kol | Referans Durumu | `n_ref` (`< τ` / `= τ` / `uniq`) | $\\tau_{\\text{OOF}}$ → $\\tau_{\\text{final}}$ | Test `n_tgt` / `n_comp` / `tek_çek` | OOF Kabul (`Reg1`) | Nihai Kabul (`Reg1`) | Nihai Puan Çek. (`Reg1`) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ])
            arms = res["regimes"][rn][scen]["arms"]
            for arm in ("M0_k0", "M1_k0") + tuple(f"{p}_k{k}" for k in K_BUDGETS for p in ("StaticDOpt", "EqualWeightVarRed", "AdaptiveOED")):
                an = arms[arm]
                if not an["threshold_assigned"]:
                    lines.append(f"| `{arm}` | {an['reference']['status']} | — | — | — | — | — | — |")
                    continue
                ref, oof, fin, dn = an["reference"], an["oof_threshold_eval"], an["final_fit_threshold_eval"], an["denominators"]
                lines.append(
                    f"| `{arm}` | {ref['status']} | {ref['n_ref']} ({ref['n_ref_below_tau']}/{ref['n_ref_equal_tau']}/{ref['n_unique_scores']}) | "
                    f"{_fmt(oof['tau_locked'], 3)} → {_fmt(ref['tau_final'], 3)} | "
                    f"{dn['n_target_eligible_test']} / {dn['n_complete_test']} / {dn['n_technical_abstain_test']} | "
                    f"{oof['n_accepted']}/{oof['n_available']} ({_fmt(oof['accepted_mean_regret1'])}) | "
                    f"{fin['n_accepted']}/{fin['n_available']} ({_fmt(fin['accepted_mean_regret1'])}) | "
                    f"{fin['n_abstained_score']} ({_fmt(fin['abstained_mean_regret1'])}) |"
                )
            lines.append("")

    lines.extend([
        "## 4. `A1` Duyarlılığı: `AdaptiveOED` Referansında `M1 OOF` vs `M1 Örneklem-İçi`",
        "",
        "| Rejim | Senaryo | Kol | `OOF` $\\tau_{\\text{final}}$ (`uniq`, test kabul) | `In-Sample` $\\tau_{\\text{final}}$ (`uniq`, test kabul) | `Δn_acc` |",
        "|---|---|---|---:|---:|---:|",
    ])
    for rn in ("id_60_20_20", "wave_ood"):
        for scen in ("archive_available", "locked_train_catalog"):
            for k in K_BUDGETS:
                arm = f"AdaptiveOED_k{k}"
                an = res["regimes"][rn][scen]["arms"][arm]
                ro, fo = an["reference"], an["final_fit_threshold_eval"]
                ri, fi = an["insample_m1_sensitivity"]["reference"], an["insample_m1_sensitivity"]["final_fit_threshold_eval"]
                d_acc = an["insample_m1_sensitivity"]["delta_n_accepted_vs_oof"]
                lines.append(
                    f"| {rn} | {scen} | `{arm}` | {_fmt(ro['tau_final'], 3)} ({ro['n_unique_scores']}, {fo['n_accepted']}/{fo['n_available']}) | "
                    f"{_fmt(ri['tau_final'], 3)} ({ri['n_unique_scores']}, {fi['n_accepted']}/{fi['n_available']}) | {d_acc:+d} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="OncoTwin Sprint 4 DEV-OPUS-19 m3_post_scale eşik duyarlılık analizi.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    res = run_sprint04_dev_opus19(out_json=args.out_json, out_md=args.out_md)
    print(f"DEV-OPUS-19 tamamlandı ({res['metadata']['user_approval_date']}) -> {args.out_md}")


if __name__ == "__main__":
    main()
