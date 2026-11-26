"""Opus Sprint 4 QA — gerçek veri tanı probu (salt okunur; üretim çıktılarına yazmaz).

Ölçülenler:
  P1  m3_post_scale eşik kayması: iç-OOF fitleri ile nihai fit arasında öncül (k=0) ve
      StaticDOpt_k5 posterior ölçeği; nihai fit ölçeğinde etiket kullanmadan train_tune
      üzerinde kilitlenen alternatif eşiğin test kabul oranı.
  P2  SOL-09 yeniden üretimi: AdaptiveOED_k5 arşiv paneli kalibrasyon tohumu (seed+1) ile
      test tohumu (seed+2) arasında kaç test hastasında değişiyor.
Kullanım: PYTHONPATH=. python3 audits/opus-2026-11-22-sprint04/realdata_probe_s04.py OUT.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.eval.splits import get_evaluable_mask, make_patient_disjoint_kfold, make_sprint04_splits
from src.oncotwin.experiments.run_sprint04 import (
    SPRINT04_SEED, K_BUDGETS, _fit_sprint04_fold, _predict_arm, _select_archive_panels,
)
from src.oncotwin.models.panel_selection import compute_contender_weights


def _arch_scale(ds, idx, fit, target_idx, policy, k, cont):
    sub = ds.subset_by_positions(idx)
    mask = get_evaluable_mask(sub.Y_auc, target_idx, fit["probe_pool_idx"], 5, 8)
    uni = np.full(len(target_idx), 1.0 / len(target_idx))
    probes = _select_archive_panels(sub.Y_auc, fit, cont, policy, k, mask, uni, set(target_idx))
    m1 = fit["M1"].predict(sub.X_rna, sub.X_mut, n_drugs=ds.Y_auc.shape[1])
    _, sc = _predict_arm(sub, probes, fit, m1, target_idx, k=k, complete_mask=mask)
    return sc[mask], probes, mask


def main(out_path: str) -> None:
    ds_all = load_beataml_aml_dataset()
    ds = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    target_idx = [ds.drug_names.index(d) for d in CLINICAL_CANDIDATE_DRUGS]
    splits = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=SPRINT04_SEED)
    res = json.load(open("data/processed/sprint04_results.json"))
    out: dict = {}
    for rn in ("id_60_20_20", "wave_ood"):
        part = splits[rn]
        tr = part["train_tune"]
        seed = SPRINT04_SEED
        # --- P1: iç fitlerin öncül ölçeği vs nihai fit
        meta_tr = ds.meta.iloc[tr].reset_index(drop=True)
        inner_prior, inner_k5 = [], []
        for f, (itr, iva) in enumerate(make_patient_disjoint_kfold(meta_tr, n_splits=5, seed=seed, strata_col="cohort")):
            fi = _fit_sprint04_fold(ds, tr[itr], target_idx, K_BUDGETS, inner_seed=seed + 100 + f, min_train_obs=100)
            inner_prior.append(float(np.sqrt(np.mean(np.diag(fi["cov_ref"])[fi["target_local"]]))))
            dummy = np.zeros((len(iva), len(target_idx)))
            sc, _, _ = _arch_scale(ds, tr[iva], fi, target_idx, "StaticDOpt", 5, dummy)
            inner_k5.append(sc)
        full = _fit_sprint04_fold(ds, tr, target_idx, K_BUDGETS, inner_seed=seed, min_train_obs=100)
        full_prior = float(np.sqrt(np.mean(np.diag(full["cov_ref"])[full["target_local"]])))
        dummy_tr = np.zeros((len(tr), len(target_idx)))
        sc_tr_full, _, _ = _arch_scale(ds, tr, full, target_idx, "StaticDOpt", 5, dummy_tr)
        te = part["test"]
        sc_te, _, _ = _arch_scale(ds, te, full, target_idx, "StaticDOpt", 5, np.zeros((len(te), len(target_idx))))
        tau_oof = res["regimes"][rn]["inner_oof_taus"]["archive_available"]["m3_post_scale"]["StaticDOpt_k5"]
        tau_alt = float(np.quantile(sc_tr_full, 0.80, method="higher"))
        inner_all = np.concatenate(inner_k5)
        # --- P2: SOL-09
        sub_te = ds.subset_by_positions(te)
        m1_te = full["M1"].predict(sub_te.X_rna, sub_te.X_mut, n_drugs=ds.Y_auc.shape[1])
        cov_cc = full["cov_ref"][np.ix_(full["target_local"], full["target_local"])]
        w_cal_seed = compute_contender_weights(m1_te[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=seed + 1)
        w_te_seed = compute_contender_weights(m1_te[:, target_idx], cov_cc, top_k=3, n_mc=500, seed=seed + 2)
        _, p_a, mask_te = _arch_scale(ds, te, full, target_idx, "AdaptiveOED", 5, w_cal_seed)
        _, p_b, _ = _arch_scale(ds, te, full, target_idx, "AdaptiveOED", 5, w_te_seed)
        changed = int(sum(set(p_a[i].tolist()) != set(p_b[i].tolist()) for i in np.flatnonzero(mask_te)))
        out[rn] = {
            "P1_prior_rms_sd_inner_folds": inner_prior,
            "P1_prior_rms_sd_full": full_prior,
            "P1_StaticDOpt_k5_inner_oof_median": float(np.median(inner_all)),
            "P1_StaticDOpt_k5_full_fit_train_tune_median": float(np.median(sc_tr_full)),
            "P1_StaticDOpt_k5_test_median": float(np.median(sc_te)),
            "P1_tau_locked_inner_oof": tau_oof,
            "P1_test_accept_at_tau_oof": float(np.mean(sc_te <= tau_oof)),
            "P1_tau_alt_full_fit_train_tune_label_free": tau_alt,
            "P1_test_accept_at_tau_alt": float(np.mean(sc_te <= tau_alt)),
            "P1_full_fit_score_mode_mass_test": float(np.max(np.unique(np.round(sc_te, 9), return_counts=True)[1]) / len(sc_te)),
            "P2_adaptive_k5_test_panels_changed_seed+1_vs_seed+2": changed,
            "P2_n_eval_test": int(mask_te.sum()),
        }
        print(rn, json.dumps(out[rn], indent=1))
    json.dump(out, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
