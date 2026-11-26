"""OPUS ayrı deney sürümü (opus-probe-v1): üretim JSON'larına YAZMAZ, çıktı yalnızca argv[1].

Seed 42, 5-fold CV (+ W1+2->W3+4), k=5:
  - M3_full vs perm_within vs offset-only (compound-symmetry) Gaussian, aynı StaticDOpt panelleri
  - Adaptif panel çeşitliliği: gerçek mevcudiyet maskesi vs tam mevcudiyet
  - OOF kalıntı varyansı vs test kalıntı varyansı (Sprint 4 kalibrasyon riski)
  - Zamanlama ve peak RSS
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

from src.oncotwin.data.audit_beataml import CLINICAL_CANDIDATE_DRUGS, DEFAULT_RAW_DIR
from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.eval.splits import get_evaluable_mask, make_clean_wave_split, make_patient_disjoint_kfold
from src.oncotwin.experiments.run_sprint03 import _fit_sprint03_split, _permute_within_patient_probes
from src.oncotwin.models.baselines import conditional_gaussian_predict_matrix
from src.oncotwin.models.panel_selection import (
    compute_contender_weights,
    select_adaptive_batch_oed,
    take_first_k_available,
)

K = 5
LAM_GRID = (50.0, 200.0, 500.0, 1500.0, 5000.0)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_split(ds, tr, te, tgt, seed, timings):
    t = time.perf_counter()
    fo = _fit_sprint03_split(ds, tr, tgt, inner_seed=seed, fit_ablations=False)
    timings.append(("fit_split", time.perf_counter() - t))
    Y_te = ds.Y_auc[te]
    emask = get_evaluable_mask(Y_te, tgt, fo["probe_pool_idx"], min_targets=5, min_probes=12)
    prior = fo["M1"].predict(ds.X_rna[te], ds.X_mut[te], n_drugs=Y_te.shape[1])
    m3 = fo["M3"]
    cm = m3._cov_model
    probes = [take_first_k_available(fo["static_dopt_global"], Y_te[i], K)[0] for i in range(len(te))]

    full = m3.predict(None, None, Y_te, probes, tgt, k=K, prior_eval=prior)
    Yp = _permute_within_patient_probes(Y_te, prior, probes, seed=seed + 1)
    within = m3.predict(None, None, Yp, probes, tgt, k=K, prior_eval=prior)

    # Offset-only: compound-symmetry kovaryans (ortak hasta ofseti + ilaç-özgü diag), aynı panel
    cov5 = cm.cov_by_k_[K]
    off = cov5[~np.eye(len(cov5), dtype=bool)]
    s_b2 = max(float(np.mean(off)), 0.0)
    cov_cs = s_b2 * np.ones_like(cov5) + np.diag(np.maximum(np.diag(cov5) - s_b2, 1e-4))
    t_loc = np.array([cm.global_to_local_[g] for g in tgt])
    p_loc = [np.array([cm.global_to_local_[g] for g in s], dtype=np.int64) for s in probes]
    R_act = (Y_te - prior)[:, cm.active_idx_]
    cs_by_lam = {}
    for lam in LAM_GRID:
        out = prior.copy()
        out[:, tgt] += conditional_gaussian_predict_matrix(R_act, cov_cs, p_loc, t_loc, lam=lam)
        cs_by_lam[lam] = out

    # Adaptif çeşitlilik: gerçek maske vs tam mevcudiyet
    w = compute_contender_weights(prior[:, tgt], cov5[np.ix_(t_loc, t_loc)], seed=seed)
    _, lam_k = cm.best_params_by_k_[K]
    pool = np.asarray(fo["probe_pool_idx"])
    all_loc = np.array([cm.global_to_local_[g] for g in pool])
    ad_real, ad_full = set(), set()
    for i in np.flatnonzero(emask):
        av = np.array([cm.global_to_local_[g] for g in pool[np.isfinite(Y_te[i, pool])]])
        ad_real.add(tuple(sorted(select_adaptive_batch_oed(cov5, av, t_loc, w[i], K, lam_k))))
        ad_full.add(tuple(sorted(select_adaptive_batch_oed(cov5, all_loc, t_loc, w[i], K, lam_k))))

    # OOF kalıntı varyansı (eğitim) vs gerçek test kalıntı varyansı (M1, hedefler)
    oof_var = np.diag(cov5)[t_loc]
    res_te = (Y_te - prior)[emask][:, tgt]
    te_var = np.nanmean(res_te ** 2, axis=0)
    return dict(emask=emask, Yt=Y_te[:, tgt], prior=prior[:, tgt], full=full[:, tgt], within=within[:, tgt],
                cs={lam: v[:, tgt] for lam, v in cs_by_lam.items()}, s_b2=s_b2,
                mean_diag=float(np.mean(np.diag(cov5))), n_uniq_real=len(ad_real), n_uniq_full=len(ad_full),
                var_ratio=float(np.nanmedian(te_var / oof_var)), lam_k=lam_k)


def summarise(parts):
    def rmse(key, lam=None):
        se, n = 0.0, 0
        pm = []
        for p in parts:
            pred = p["cs"][lam] if key == "cs" else p[key]
            e = (p["Yt"] - pred)[p["emask"]]
            pm.extend(np.nanmean(e ** 2, axis=1).tolist())
        return float(np.sqrt(np.nanmean(pm)))
    r = {"M1": rmse("prior"), "M3_full": rmse("full"), "perm_within": rmse("within")}
    r.update({f"offset_only_lam{int(l)}": rmse("cs", l) for l in LAM_GRID})
    best_cs = min(r[f"offset_only_lam{int(l)}"] for l in LAM_GRID)
    gain = r["M1"] - r["M3_full"]
    r["offset_only_oracle_best"] = best_cs
    r["offset_share_of_M3_gain(oracle)"] = (r["M1"] - best_cs) / gain
    r["perm_within_retained_gain"] = (r["M1"] - r["perm_within"]) / gain
    r["adaptive_unique_panels_real_mask"] = [p["n_uniq_real"] for p in parts]
    r["adaptive_unique_panels_full_avail"] = [p["n_uniq_full"] for p in parts]
    r["n_eval"] = [int(p["emask"].sum()) for p in parts]
    r["median_test_over_oof_resid_var"] = [round(p["var_ratio"], 3) for p in parts]
    r["s_b2_over_mean_diag"] = [round(p["s_b2"] / p["mean_diag"], 3) for p in parts]
    return r


def main(out_path: str) -> None:
    t0 = time.perf_counter()
    timings: list = []
    ds_all = load_beataml_aml_dataset(DEFAULT_RAW_DIR)
    timings.append(("load", time.perf_counter() - t0))
    rss_after_load = rss_mb()
    ds = ds_all.subset_by_positions(select_index_specimen_positions(ds_all.meta))
    dmap = {d: i for i, d in enumerate(ds.drug_names)}
    tgt = [dmap[d] for d in CLINICAL_CANDIDATE_DRUGS]
    subj = ds.meta["dbgap_subject_id"].to_numpy()

    tr_w, te_w = make_clean_wave_split(ds.meta, ds.w12_all_subjects, ds.both_all_subjects)
    avail_w34 = {d: int(np.isfinite(ds.Y_auc[te_w, dmap[d]]).sum()) for d in ("ABT-737", "XAV-939", "Sunitinib", "Bay 11-7085")}

    cv_parts = [run_split(ds, tr, te, tgt, 42 + f, timings) for f, (tr, te) in enumerate(make_patient_disjoint_kfold(ds.meta, 5, 42))]
    wave_parts = [run_split(ds, tr_w, te_w, tgt, 777, timings)]

    res = {
        "version": "opus-probe-v1",
        "rows": int(len(subj)), "unique_subjects": int(len(np.unique(subj))),
        "all_specimen_rows": int(len(ds_all.meta)),
        "w34_test_obs_counts": avail_w34,
        "cv_seed42_k5_staticdopt": summarise(cv_parts),
        "wave_k5_staticdopt": summarise(wave_parts),
        "timings_s": [(n, round(s, 2)) for n, s in timings],
        "peak_rss_mb_after_load": round(rss_after_load, 1),
        "peak_rss_mb_end": round(rss_mb(), 1),
        "X_rna_shape_dtype": [list(ds_all.X_rna.shape), str(ds_all.X_rna.dtype)],
        "total_s": round(time.perf_counter() - t0, 1),
    }
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
