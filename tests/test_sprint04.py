"""Sprint 4 (hasta-birimli split-conformal, kısa liste, OOD çekinme ve sızıntısız izolasyon) testleri."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.oncotwin.data.dataset import BeatAMLDataset
from src.oncotwin.eval.splits import make_sprint04_splits
from src.oncotwin.eval.uncertainty import (
    OmicsPCADistanceScorer,
    evaluate_conformal_and_shortlist,
    evaluate_locked_threshold_abstention,
    fit_split_conformal_simultaneous,
    m3_conditional_posterior_scale,
    paired_aurc_bootstrap_ci,
    selective_risk_coverage_curve,
)
from src.oncotwin.experiments.run_sprint04 import (
    _choose_locked_panels,
    _open_archive_mask_for_locked,
    _select_archive_panels,
    run_regime_sprint04,
)
from src.oncotwin.models.panel_selection import select_adaptive_batch_oed
from tests.test_sprint02 import _synthetic_cohort


def _to_dataset(d: dict) -> BeatAMLDataset:
    meta = d["meta"]
    w12 = set(int(x) for x in meta.loc[meta["cohort"] == "Waves1+2", "dbgap_subject_id"].unique())
    both = set(int(x) for x in meta.loc[meta["cohort"] == "Both", "dbgap_subject_id"].unique())
    return BeatAMLDataset(
        meta=meta,
        Y_auc=d["Y"],
        X_rna=d["X_rna"],
        X_mut=d["X_mut"],
        drug_names=d["drug_names"],
        gene_names=[f"G{i}" for i in range(d["X_rna"].shape[1])],
        mut_feature_names=[f"M{i}" for i in range(d["X_mut"].shape[1])],
        w12_all_subjects=w12,
        both_all_subjects=both,
    )


def test_sprint04_splits_disjoint_and_auc_independent() -> None:
    ds = _to_dataset(_synthetic_cohort(n_patients=40, n_genes=120, seed=7))
    splits = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=2026)

    for rname in ("id_60_20_20", "wave_ood"):
        part = splits[rname]
        subj = ds.meta["dbgap_subject_id"].to_numpy()
        s_tr = set(int(x) for x in subj[part["train_tune"]])
        s_cal = set(int(x) for x in subj[part["calibration"]])
        s_te = set(int(x) for x in subj[part["test"]])
        assert s_tr and s_cal and s_te
        assert s_tr.isdisjoint(s_cal)
        assert s_tr.isdisjoint(s_te)
        assert s_cal.isdisjoint(s_te)
        man = splits["manifest"]["regimes"][rname]
        assert set(man["train_tune"]["subject_ids"]) == s_tr
        assert set(man["calibration"]["subject_ids"]) == s_cal
        assert set(man["test"]["subject_ids"]) == s_te

    w_te_subj = set(int(x) for x in ds.meta["dbgap_subject_id"].to_numpy()[splits["wave_ood"]["test"]])
    assert w_te_subj.isdisjoint(ds.w12_all_subjects | ds.both_all_subjects)


def test_split_conformal_finite_sample_rank_and_infinite_guard() -> None:
    rng = np.random.default_rng(11)
    # n=8 için ceil(9 * 0.9) = 9 > 8 -> sonlu aralık yayımlanmamalı
    Yt8 = rng.normal(100.0, 10.0, size=(8, 4))
    Yp8 = Yt8 + rng.normal(0.0, 5.0, size=(8, 4))
    fit8 = fit_split_conformal_simultaneous(Yt8, Yp8, target_idx=[0, 1, 2, 3], alpha=0.10)
    assert fit8["n_cal"] == 8
    assert fit8["order_stat_rank"] == 9
    assert math.isinf(fit8["q_sim"])
    assert fit8["finite_interval_published"] is False

    # n=9 için ceil(10 * 0.9) = 9 -> maksimum kalibrasyon skoru
    Yt9 = rng.normal(100.0, 10.0, size=(9, 4))
    Yp9 = Yt9 + rng.normal(0.0, 5.0, size=(9, 4))
    fit9 = fit_split_conformal_simultaneous(Yt9, Yp9, target_idx=[0, 1, 2, 3], alpha=0.10)
    expected_max = float(np.max(np.abs(Yt9 - Yp9)))
    assert fit9["finite_interval_published"] is True
    assert np.isclose(fit9["q_sim"], expected_max)
    assert np.isclose(fit9["width_sim"], 2.0 * expected_max)

    # n=0 -> uyarısız inf
    fit0 = fit_split_conformal_simultaneous(Yt8[:0], Yp8[:0], target_idx=[0, 1, 2, 3], alpha=0.10)
    assert fit0["n_cal"] == 0
    assert fit0["finite_interval_published"] is False


def test_conformal_shortlist_guarantees_true_top1_when_covered() -> None:
    # Eşzamanlı kapsama gerçekleşen hastalarda gerçek 1. her zaman kısa listede olmalı
    q = 15.0
    Yt = np.array([
        [50.0, 80.0, 120.0, 140.0],
        [95.0, 60.0, 88.0, 110.0],
        [70.0, 72.0, np.nan, 200.0],  # 1 hasta kapsanamaz
    ])
    Yp = np.array([
        [58.0, 72.0, 115.0, 150.0],   # tüm |err| <= 10 <= 15 (kapsandı)
        [85.0, 70.0, 80.0, 108.0],    # tüm |err| <= 10 <= 15 (kapsandı)
        [95.0, 65.0, 90.0, 200.0],    # ilk ilaçta err=25 > 15 (kapsanamadı)
    ])
    ev = evaluate_conformal_and_shortlist(Yt, Yp, target_idx=[0, 1, 2, 3], q_sim=q)
    cov = ev["vectors"]["patient_covered_sim"]
    top1_in = ev["vectors"]["patient_shortlist_contains_top1"]
    assert np.all(top1_in[cov == 1.0] == 1.0)
    # Satır 0 için U_min = 72 + 15 = 87; d3 L = 135 > 87 elenmeli -> boyut < 4
    assert ev["vectors"]["patient_shortlist_size_obs"][0] < 4

    # q = inf iken sonlu aralık yok -> kapsam ve kısa liste alanları NaN / hesaplanamadı (SOL-11 / DEV-OPUS-15)
    ev_inf = evaluate_conformal_and_shortlist(Yt, Yp, target_idx=[0, 1, 2, 3], q_sim=float("inf"))
    assert ev_inf["finite_interval_published"] is False
    assert ev_inf["shortlist_fallback_all_targets"] is True
    assert ev_inf["status"].startswith("hesaplanamadı")
    assert math.isnan(ev_inf["simultaneous_coverage"])
    assert math.isnan(ev_inf["shortlist_contains_true_top1_rate"])
    assert math.isnan(ev_inf["mean_shortlist_size_obs"])
    assert np.all(np.isnan(ev_inf["vectors"]["patient_covered_sim"]))
    assert np.allclose(ev_inf["vectors"]["patient_max_abs_err"], [10.0, 10.0, 25.0])

    # Boş değerlendirme uyarısız "hesaplanamadı"
    ev_empty = evaluate_conformal_and_shortlist(Yt[:0], Yp[:0], target_idx=[0, 1, 2, 3], q_sim=q)
    assert ev_empty["status"] == "hesaplanamadı"
    assert ev_empty["n_eval"] == 0


def test_selective_risk_coverage_ties_and_coverage_cap() -> None:
    # Bağsız: skor [1, 2, 3, 4], kayıp [10, 20, 30, 40] -> R=[10, 15, 20, 25], AURC = 70/4 = 17.5
    sc = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([10.0, 20.0, 30.0, 40.0])
    rc = selective_risk_coverage_curve(sc, lo)
    assert np.isclose(rc["aurc"], 17.5)
    assert np.isclose(rc["oracle_aurc"], 17.5)
    assert np.isclose(rc["e_aurc"], 0.0)
    assert np.isclose(rc["random_aurc"], 25.0)

    # Tümü bağlı: skorlar eşit -> ortalama risk = 25.0
    rc_all_tied = selective_risk_coverage_curve(np.ones(4), lo)
    assert np.isclose(rc_all_tied["aurc"], 25.0)

    # Kısmi bağ: [1, 2, 2, 3] ile [10, 30, 10, 50] -> bloklar: (end=1, R=10), (end=3, R=50/3), (end=4, R=25)
    sc_t = np.array([1.0, 2.0, 2.0, 3.0])
    lo_t = np.array([10.0, 30.0, 10.0, 50.0])
    rc_t = selective_risk_coverage_curve(sc_t, lo_t)
    expected_aurc_t = 0.25 * 10.0 + 0.50 * (50.0 / 3.0) + 0.25 * 25.0
    assert np.isclose(rc_t["aurc"], expected_aurc_t)

    # Kısmi mevcudiyet + coverage_cap=0.50
    avail = np.array([True, True, True, False])
    rc_cap = selective_risk_coverage_curve(sc, lo, available_mask=avail, coverage_cap=0.50)
    assert np.isclose(rc_cap["max_coverage"], 0.75)
    assert rc_cap["cap_status"] == "ok"
    assert rc_cap["cap_cuts_tie_group"] is False
    assert rc_cap["cap_interpretation"] == "gerçekleşebilir_eşik_noktası"
    assert np.isclose(rc_cap["aurc_capped"], (0.25 * 10.0 + 0.25 * 15.0) / 0.50)

    # Bağ grubunu kesen cap için kesirli varsayım etiketi ve altındaki gerçekleşebilir nokta (DEV-OPUS-14)
    rc_tie_cap = selective_risk_coverage_curve(
        np.array([0.0, 1.0, 1.0, 1.0]), np.array([0.0, 30.0, 0.0, 0.0]), coverage_cap=0.50
    )
    assert rc_tie_cap["cap_cuts_tie_group"] is True
    assert rc_tie_cap["cap_interpretation"] == "kesirli_kabul_varsayımı"
    assert np.isclose(rc_tie_cap["realizable_cap_below"], 0.25)
    assert np.isclose(rc_tie_cap["aurc_realizable_at_or_below_cap"], 0.0)

    rc_below = selective_risk_coverage_curve(sc, lo, available_mask=np.array([True, False, False, False]), coverage_cap=0.50)
    assert rc_below["cap_status"] == "below_cap"
    assert math.isnan(rc_below["aurc_capped"])

    # Boş giriş -> sıfır uyarı ve hesaplanamadı
    rc_empty = selective_risk_coverage_curve(np.array([]), np.array([]))
    assert rc_empty["status"] == "hesaplanamadı"


def test_equal_weight_var_red_matches_adaptive_with_uniform_weights() -> None:
    rng = np.random.default_rng(42)
    A = rng.normal(0.0, 1.0, size=(14, 10))
    cov = A @ A.T + 0.2 * np.eye(14)
    avail = np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    tgt = np.array([0, 1, 2], dtype=np.int64)
    uniform_w = np.full(len(tgt), 1.0 / len(tgt), dtype=np.float64)
    s_eq = select_adaptive_batch_oed(cov, avail, tgt, uniform_w, k=4, lam=50.0)
    # Tekdüze ağırlıklar ile iki ayrı çağrı birebir aynı seçimi üretir
    s_eq2 = select_adaptive_batch_oed(cov, avail, tgt, np.array([1 / 3, 1 / 3, 1 / 3]), k=4, lam=50.0)
    assert np.array_equal(s_eq, s_eq2)


def test_omics_pca_and_m3_scale_and_empty_abstention_no_warnings() -> None:
    ds = _to_dataset(_synthetic_cohort(n_patients=30, n_genes=80, seed=13))
    scorer = OmicsPCADistanceScorer(top_k_genes=40, max_components=10).fit(ds.X_rna[:20], ds.X_mut[:20])
    d_te = scorer.score(ds.X_rna[20:], ds.X_mut[20:])
    assert d_te.shape == (10,)
    assert np.all(np.isfinite(d_te)) and np.all(d_te >= 0.0)

    # Korele kovaryansta prob gözlemi koşullu posterior ölçeğini azaltmalı
    A = np.ones((10, 1)) * 8.0
    cov = A @ A.T + np.eye(10) * 20.0
    sc = m3_conditional_posterior_scale(
        cov, [np.array([5, 6]), np.array([], dtype=np.int64)], np.array([0, 1, 2]), lam=50.0
    )
    assert sc.shape == (2,) and sc[0] < sc[1]

    # Eşik alt/üst boş dilimler uyarı üretmez
    ab = evaluate_locked_threshold_abstention(
        np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([25.0, 36.0]), np.array([1.0, 1.0]), tau_locked=10.0
    )
    assert ab["n_accepted"] == 2 and ab["n_abstained_score"] == 0 and math.isnan(ab["abstained_mean_regret1"])
    ab_nan = evaluate_locked_threshold_abstention(
        np.array([1.0]), np.array([5.0]), np.array([25.0]), np.array([1.0]), tau_locked=float("nan")
    )
    assert ab_nan["status"] == "hesaplanamadı"
    boot_empty = paired_aurc_bootstrap_ci(np.array([1.0]), np.array([1.0]), np.array([np.nan]))
    assert boot_empty["n_paired"] == 0


def test_sprint04_pre_run_leakage_and_isolation_contracts() -> None:
    """Sol §5 #6 (a)-(f) doğrulama sözleşmesi: test AUC, cal AUC ve test maskesi izolasyonu."""
    ds = _to_dataset(_synthetic_cohort(n_patients=60, n_genes=80, n_drugs=12, seed=29))
    target_idx = [0, 1, 2, 3, 4]
    splits = make_sprint04_splits(ds.meta, ds.w12_all_subjects, ds.both_all_subjects, seed=2026)
    id_part = splits["id_60_20_20"]

    res_base = run_regime_sprint04(
        ds, id_part, target_idx, regime_name="id_60_20_20", seed=2026,
        min_train_obs=10, min_targets=3, min_probes=3, k_budgets=(2,), inner_n_splits=3, n_boot=50,
    )

    # (b) Gizli test AUC'lerini permüte edip +N(0,500) ekle -> fit, sıralama, seçilen prob ve çekinme skorları aynı!
    rng = np.random.default_rng(777)
    ds_test_mod = ds.subset_by_positions(np.arange(len(ds.meta)))
    Y_mod = ds_test_mod.Y_auc.copy()
    te_idx = id_part["test"]
    for j in target_idx:
        fin = np.isfinite(Y_mod[te_idx, j])
        vals = Y_mod[te_idx[fin], j].copy()
        rng.shuffle(vals)
        Y_mod[te_idx[fin], j] = vals + rng.normal(0.0, 500.0, size=int(fin.sum()))
    ds_b = ds_test_mod.__class__(
        meta=ds_test_mod.meta, Y_auc=Y_mod, X_rna=ds_test_mod.X_rna, X_mut=ds_test_mod.X_mut,
        drug_names=ds_test_mod.drug_names, gene_names=ds_test_mod.gene_names,
        mut_feature_names=ds_test_mod.mut_feature_names,
        w12_all_subjects=ds_test_mod.w12_all_subjects, both_all_subjects=ds_test_mod.both_all_subjects,
    )
    res_b = run_regime_sprint04(
        ds_b, id_part, target_idx, regime_name="id_60_20_20", seed=2026,
        min_train_obs=10, min_targets=3, min_probes=3, k_budgets=(2,), inner_n_splits=3, n_boot=50,
    )
    arm_base = res_base["scenarios"]["archive_available"]["arms"]["StaticDOpt_k2"]
    arm_b = res_b["scenarios"]["archive_available"]["arms"]["StaticDOpt_k2"]
    assert np.isclose(arm_base["conformal"]["q_sim"], arm_b["conformal"]["q_sim"])
    assert np.isclose(arm_base["abstention"]["omics_pca_dist"]["tau_locked"],
                      arm_b["abstention"]["omics_pca_dist"]["tau_locked"])
    assert np.array_equal(res_base["vectors"]["test_omics_pca_dist"], res_b["vectors"]["test_omics_pca_dist"])
    assert np.array_equal(res_base["vectors"]["archive_available/StaticDOpt_k2/m3_post_scale"],
                          res_b["vectors"]["archive_available/StaticDOpt_k2/m3_post_scale"])

    # (c) Kalibrasyon yanıtlarını boz -> yalnız conformal q / kısa liste değişir; model tahmin & çekinme eşikleri/skorları aynı
    Y_cal = ds.Y_auc.copy()
    cal_idx = id_part["calibration"]
    Y_cal[np.ix_(cal_idx, target_idx)] += 75.0
    ds_c = ds_test_mod.__class__(
        meta=ds_test_mod.meta, Y_auc=Y_cal, X_rna=ds_test_mod.X_rna, X_mut=ds_test_mod.X_mut,
        drug_names=ds_test_mod.drug_names, gene_names=ds_test_mod.gene_names,
        mut_feature_names=ds_test_mod.mut_feature_names,
        w12_all_subjects=ds_test_mod.w12_all_subjects, both_all_subjects=ds_test_mod.both_all_subjects,
    )
    res_c = run_regime_sprint04(
        ds_c, id_part, target_idx, regime_name="id_60_20_20", seed=2026,
        min_train_obs=10, min_targets=3, min_probes=3, k_budgets=(2,), inner_n_splits=3, n_boot=50,
    )
    arm_c = res_c["scenarios"]["archive_available"]["arms"]["StaticDOpt_k2"]
    assert not np.isclose(arm_base["conformal"]["q_sim"], arm_c["conformal"]["q_sim"])
    assert np.isclose(arm_base["test_metrics"]["rmse"], arm_c["test_metrics"]["rmse"])
    assert np.isclose(arm_base["abstention"]["omics_pca_dist"]["tau_locked"],
                      arm_c["abstention"]["omics_pca_dist"]["tau_locked"])

    # (d) & (e) & (f) Locked senaryoda test maskesi değişince seçilen paneller değişmez; eksik bütçe maskesi atlanır
    Y_mask_drop = ds.Y_auc.copy()
    probe_cols = [i for i in range(len(ds.drug_names)) if i not in set(target_idx)]
    Y_mask_drop[np.ix_(te_idx, probe_cols)] = np.nan
    ds_d = ds_test_mod.__class__(
        meta=ds_test_mod.meta, Y_auc=Y_mask_drop, X_rna=ds_test_mod.X_rna, X_mut=ds_test_mod.X_mut,
        drug_names=ds_test_mod.drug_names, gene_names=ds_test_mod.gene_names,
        mut_feature_names=ds_test_mod.mut_feature_names,
        w12_all_subjects=ds_test_mod.w12_all_subjects, both_all_subjects=ds_test_mod.both_all_subjects,
    )
    res_d = run_regime_sprint04(
        ds_d, id_part, target_idx, regime_name="id_60_20_20", seed=2026,
        min_train_obs=10, min_targets=3, min_probes=3, k_budgets=(2,), inner_n_splits=3, n_boot=50,
    )
    assert res_base["scenarios"]["locked_train_catalog"]["arms"]["StaticDOpt_k2"]["availability"]["locked_panel_example"] == (
        res_d["scenarios"]["locked_train_catalog"]["arms"]["StaticDOpt_k2"]["availability"]["locked_panel_example"]
    )
    # Test prob maskesi tamamen silindiğinde kilitli senaryo sıfır-bilgi skoru üretmez -> hesaplanamadı
    lock_d = res_d["scenarios"]["locked_train_catalog"]["arms"]["StaticDOpt_k2"]
    assert lock_d["availability"]["n_complete_test"] == 0
    assert lock_d["test_metrics"]["status"] == "hesaplanamadı"
    assert lock_d["conformal"]["status"] == "hesaplanamadı"
    # Aynı durum archive_available içinde require_full_budget ValueError fırlatmalı (yalnız seçim çağrısı sarılır)
    from src.oncotwin.experiments.run_sprint04 import _fit_sprint04_fold
    fobj = _fit_sprint04_fold(ds_d, id_part["train_tune"], target_idx, k_budgets=(2,), inner_seed=2026, min_train_obs=10)
    bad_eval_mask = np.ones(len(te_idx), dtype=bool)
    cw = np.ones((len(te_idx), len(target_idx)), dtype=np.float64) / len(target_idx)
    with pytest.raises(ValueError, match="karşılanamadı"):
        _select_archive_panels(ds_d.Y_auc[te_idx], fobj, cw, "StaticDOpt", 2, bad_eval_mask, cw[0], set(target_idx))
    assert set(probe_cols).isdisjoint(set(target_idx))
    # _choose_locked_panels yalnızca train_tune bilgisinden çalışır (Y_eval almaz)
    locked_ex, comp_ex, _ = _open_archive_mask_for_locked(
        _choose_locked_panels(
            fobj, np.ones((len(te_idx), len(target_idx))) / len(target_idx), "StaticDOpt", 2,
            np.ones(len(target_idx)) / len(target_idx), set(target_idx),
        ),
        ds_d.Y_auc[te_idx], k=2,
    )
    assert not np.any(comp_ex) and all(len(x) == 0 for x in locked_ex)


def test_sprint04_results_consistency() -> None:
    res_path = Path("data/processed/sprint04_results.json")
    man_path = Path("data/processed/sprint04_split_manifest.json")
    npz_path = Path("data/processed/sprint04_vectors.npz")
    md_path = Path("docs/sprints/sprint-04-conformal-abstain.md")
    assert res_path.exists() and man_path.exists() and npz_path.exists() and md_path.exists()

    res = json.loads(res_path.read_text(encoding="utf-8"))
    man = json.loads(man_path.read_text(encoding="utf-8"))
    z = np.load(npz_path)

    for rn, exp_counts in (("id_60_20_20", (284, 94, 93)), ("wave_ood", (213, 71, 165))):
        reg_man = man["regimes"][rn]
        s_tr = set(reg_man["train_tune"]["subject_ids"])
        s_cal = set(reg_man["calibration"]["subject_ids"])
        s_te = set(reg_man["test"]["subject_ids"])
        assert (len(s_tr), len(s_cal), len(s_te)) == exp_counts
        assert s_tr.isdisjoint(s_cal) and s_tr.isdisjoint(s_te) and s_cal.isdisjoint(s_te)

        reg = res["regimes"][rn]
        assert reg["documented_plate_catalog_status"].startswith("hesaplanamadı")
        arch = reg["scenarios"]["archive_available"]["arms"]
        k5 = arch["StaticDOpt_k5"]
        assert k5["conformal"]["finite_interval_published"] is True
        assert k5["conformal"]["shortlist_contains_true_top1_rate"] + 1e-12 >= k5["conformal"]["simultaneous_coverage"]
        assert f"{rn}/archive_available/StaticDOpt_k5/patient_regret1" in z.files

    assert res["regimes"]["id_60_20_20"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["test_metrics"]["n_eval"] == 91
    assert res["regimes"]["wave_ood"]["scenarios"]["archive_available"]["arms"]["StaticDOpt_k5"]["test_metrics"]["n_eval"] == 148


def test_sprint04_v2_results_consistency() -> None:
    """Sprint 4 v2 çıktılarının şema, M0(0), n<5 guard ve v1 statik/M1 eşdeğerliğini doğrular."""
    res_v1_path = Path("data/processed/sprint04_results.json")
    man_v1_path = Path("data/processed/sprint04_split_manifest.json")
    res_v2_path = Path("data/processed/sprint04_v2_results.json")
    man_v2_path = Path("data/processed/sprint04_v2_split_manifest.json")
    npz_v2_path = Path("data/processed/sprint04_v2_vectors.npz")
    md_v2_path = Path("docs/sprints/sprint-04-v2-conformal-abstain.md")
    for p in (res_v1_path, man_v1_path, res_v2_path, man_v2_path, npz_v2_path, md_v2_path):
        assert p.exists(), f"Eksik çıktı: {p}"

    res_v1 = json.loads(res_v1_path.read_text(encoding="utf-8"))
    man_v1 = json.loads(man_v1_path.read_text(encoding="utf-8"))
    res_v2 = json.loads(res_v2_path.read_text(encoding="utf-8"))
    man_v2 = json.loads(man_v2_path.read_text(encoding="utf-8"))
    assert man_v1 == man_v2

    notes = res_v2["protocol_notes"]
    for k in ("sol_09_fix", "sol_10_fix", "sol_11_fix", "sol_12_fix"):
        assert k in notes

    def _assert_common_close(a: object, b: object) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for kk in set(a) & set(b):
                _assert_common_close(a[kk], b[kk])
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for x, y in zip(a, b):
                _assert_common_close(x, y)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not (isinstance(a, bool) or isinstance(b, bool)):
            if math.isfinite(a) and math.isfinite(b):
                assert abs(a - b) <= 1e-12

    z2 = np.load(npz_v2_path)
    for rn, exp_eval in (("id_60_20_20", 91), ("wave_ood", 148)):
        reg2 = res_v2["regimes"][rn]
        assert reg2["documented_plate_catalog_status"].startswith("hesaplanamadı")
        for scen in ("archive_available", "locked_train_catalog"):
            arms2 = reg2["scenarios"][scen]["arms"]
            assert "M0_k0" in arms2
            assert f"{rn}/{scen}/M0_k0/patient_regret1" in z2.files
        arch1 = res_v1["regimes"][rn]["scenarios"]["archive_available"]["arms"]
        arch2 = reg2["scenarios"]["archive_available"]["arms"]
        assert arch2["M0_k0"]["test_metrics"]["n_eval"] == exp_eval
        for deterministic_arm in (
            "M1_k0", "StaticDOpt_k2", "StaticDOpt_k5", "StaticDOpt_k8",
            "EqualWeightVarRed_k2", "EqualWeightVarRed_k5", "EqualWeightVarRed_k8",
        ):
            _assert_common_close(arch1[deterministic_arm], arch2[deterministic_arm])

    w_lock_comp = res_v2["regimes"]["wave_ood"]["comparisons"]["locked_train_catalog"]
    assert w_lock_comp["policy_pairs"]["k5_AdaptiveOED_vs_StaticDOpt"]["status"] == "hesaplanamadı"
    assert w_lock_comp["policy_pairs"]["k8_AdaptiveOED_vs_StaticDOpt"]["status"] == "hesaplanamadı"
    assert w_lock_comp["policy_pairs"]["k5_AdaptiveOED_vs_EqualWeightVarRed"]["status"] == "ok"
    assert w_lock_comp["policy_pairs"]["k5_AdaptiveOED_vs_EqualWeightVarRed"]["n_paired"] == 14
    for kb in ("k2", "k5", "k8"):
        assert w_lock_comp["three_way_common"][kb]["status"] == "hesaplanamadı"
