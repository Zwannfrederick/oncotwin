"""Paired, retrospective Beat AML check of the C1 test-probe normalization."""

import json
from pathlib import Path

import numpy as np

from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.eval.metrics import paired_bootstrap_ci
from src.oncotwin.eval.splits import make_sprint04_splits
from src.oncotwin.eval.uncertainty import json_sanitize
from src.oncotwin.experiments import run_sprint05_fimm as c1

SEED = 2026
BUDGETS = (2, 5)
REFERENCE_K = 5
METRICS = {"ndcg_at_3": "patient_ndcg3", "regret_at_1_auc": "patient_regret1"}


def probe_views(y, panels, center, scale):
    rows = np.repeat(np.arange(len(y)), [len(panel) for panel in panels])
    cols = np.concatenate(panels)
    mask = np.zeros_like(y, dtype=bool)
    mask[rows, cols] = True
    probe_only = np.where(mask, y, np.nan)
    means = np.asarray([y[i, panel].mean() if len(panel) else 0.0 for i, panel in enumerate(panels)])
    return {
        "train_reference": (probe_only - center) / scale,
        "k_center": (probe_only - means[:, None]) / scale,
        "k_probe": c1._build_single_patient_probe_z_matrix(probe_only, panels),
        # Diagnostic only. Other test responses enter this normalization.
        "full_row_oracle": np.where(mask, c1._within_patient_zscore_rows(y), np.nan),
    }


def compare(a, b):
    out = {}
    for name, key in METRICS.items():
        ci = paired_bootstrap_ci(a["vectors"][key], b["vectors"][key], n_boot=2000, seed=SEED)
        out[name] = {k: v for k, v in ci.items() if k != "p_two_sided"}
    return out


def run():
    ds = load_beataml_aml_dataset()
    ds = ds.subset_by_positions(select_index_specimen_positions(ds.meta))
    regimes = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=SEED)
    frozen = json.loads(Path("data/processed/sprint04_v2_split_manifest.json").read_text())
    assert regimes["manifest"] == frozen
    part = regimes["id_60_20_20"]
    train, test = part["train_tune"], part["test"]
    assert len(train) == 284 and len(test) == 93
    # C1 schema only. FIMM outcomes and patient-level features are never used below.
    fimm, _ = c1.load_fimm_aml_dataset(ds.drug_names, [g.split(".")[0] for g in ds.gene_names])
    x_rna, x_mut, y = c1._align_beataml_to_fimm(ds, fimm)
    lookup = {drug: j for j, drug in enumerate(fimm.drug_names)}
    targets = [lookup[d] for d in c1.CLINICAL_CANDIDATE_DRUGS if d in lookup]
    pool_ba = c1.select_train_probe_pool(ds.Y_auc[train], ds.drug_names, c1.CLINICAL_CANDIDATE_DRUGS, min_train_obs=100)
    pool = [lookup[ds.drug_names[j]] for j in pool_ba if ds.drug_names[j] in lookup]
    assert len(targets) == 23 and len(pool) >= 8 and set(targets).isdisjoint(pool)

    z_train = c1._within_patient_zscore_rows(y[train])
    x_train = c1.within_sample_rank_normalize(x_rna[train])
    x_test = c1.within_sample_rank_normalize(x_rna[test])
    subject = ds.meta["dbgap_subject_id"].to_numpy()[train]
    m1, oof = c1.OmicsRidgeM1(top_k_genes=min(2000, x_train.shape[1])).fit_and_oof(
        x_train, x_mut[train], z_train, subject,
        fit_drug_idx=sorted(set(targets) | set(pool)), target_idx=targets, inner_seed=SEED,
    )
    m3 = c1.HybridFewShotM3(lam_grid=c1.C1_Z_LAM_GRID).fit(
        m1, oof, z_train, subject, pool, targets, k_values=BUDGETS, inner_seed=SEED,
    )
    cov = m3._cov_model
    local = cov.global_to_local_
    _, lam = cov.best_params_by_k_[REFERENCE_K]
    rank = cov.active_idx_[c1.select_static_d_optimal_ranking(
        cov.cov_by_k_[REFERENCE_K], np.asarray([local[j] for j in pool]),
        np.asarray([local[j] for j in targets]), lam=lam,
    )]
    prior = m1.predict(x_test, x_mut[test], n_drugs=y.shape[1])
    y_test = y[test]
    valid = c1.get_evaluable_mask(y_test, targets, pool, min_targets=5, min_probes=8)
    reference_rows = [row[np.isfinite(row)] for row in y[train] if np.isfinite(row).sum() >= 2]
    center = float(np.median([r.mean() for r in reference_rows]))
    scale = float(np.median([r.std() for r in reference_rows]))
    assert scale > 1e-6
    out = {
        "status": "retrospective_sensitivity",
        "seed": SEED,
        "n_train": len(train),
        "n_cal_unused": len(part["calibration"]),
        "n_test_split": len(test),
        "n_target_eligible": int((np.isfinite(y_test[:, targets]).sum(axis=1) >= 5).sum()),
        "n_eval": int(valid.sum()),
        "n_common_drugs": y.shape[1],
        "n_targets": len(targets),
        "n_train_only_probes": len(pool),
        "train_reference_center_auc": center,
        "train_reference_sd_auc": scale,
        "baseline": "M1_k0",
        "test_panel_mode": "archive_available",
        "budgets": {},
    }
    for k in BUDGETS:
        empty = np.empty(0, dtype=np.int64)
        panels = [c1.take_first_k_available(rank, row, k)[0] if valid[i] else empty for i, row in enumerate(y_test)]
        c1.require_full_budget(panels, valid, k, f"OPUS-028 k={k}")
        views = probe_views(y_test, panels, center, scale)
        hidden_mask = np.isfinite(y_test) & ~np.isfinite(views["train_reference"])
        perturbed = np.where(hidden_mask, 10000.0 - 10.0 * y_test, y_test)
        changed_views = probe_views(perturbed, panels, center, scale)
        assert np.isfinite(perturbed).tolist() == np.isfinite(y_test).tolist()
        preds = {"M1_k0": prior}
        for name, values in views.items():
            preds[name] = m3.predict(x_test, x_mut[test], values, panels, targets, k=k, prior_eval=prior)
            assert np.isfinite(preds[name][np.ix_(valid, targets)]).all()
            if name != "full_row_oracle":
                np.testing.assert_array_equal(values, changed_views[name])
                after = m3.predict(x_test, x_mut[test], changed_views[name], panels, targets, k=k, prior_eval=prior)
                np.testing.assert_array_equal(preds[name], after)
        assert not np.allclose(views["full_row_oracle"], changed_views["full_row_oracle"], equal_nan=True)
        series = {name: c1._eval_scale_free_ranking(y_test, pred, valid, targets) for name, pred in preds.items()}
        assert all(node["n_eval_units"] == int(valid.sum()) for node in series.values())
        differences = [
            ("k_probe", "train_reference"), ("k_center", "train_reference"),
            ("k_probe", "k_center"), ("k_probe", "M1_k0"), ("k_probe", "full_row_oracle"),
        ]
        probe_z = [views["k_probe"][i, s] for i, s in enumerate(panels) if valid[i]]
        top = rank[:k]
        out["budgets"][str(k)] = {
            "locked_prefix": [fimm.drug_names[j] for j in top],
            "locked_prefix_complete_among_evaluable": int(np.isfinite(y_test[valid][:, top]).all(axis=1).sum()),
            "n_probe_vectors_with_abs_values_all_one": sum(bool(np.allclose(np.abs(z), 1, rtol=0, atol=1e-12)) for z in probe_z),
            "n_probe_vectors_all_zero": sum(bool(np.all(z == 0)) for z in probe_z),
            "hidden_response_perturbation_invariant": True,
            "arms": {
                name: {"ndcg_at_3": node["ndcg_at_3"], "regret_at_1_auc": node["regret_at_1_sdss"]}
                for name, node in series.items()
            },
            "paired_diffs": {f"{a}_minus_{b}": compare(series[a], series[b]) for a, b in differences},
        }
    return out


if __name__ == "__main__":
    print(json.dumps(json_sanitize(run()), ensure_ascii=False, indent=2, allow_nan=False))
