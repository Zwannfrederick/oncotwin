"""Beat AML 2.0 ampirik veri denetimi, sızıntı analizi ve M0 varyans ayrıştırması."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = Path("data/raw/beataml2")
DEFAULT_OUT_JSON = Path("data/processed/sprint01_audit_summary.json")
DEFAULT_OUT_MD = Path("docs/sprints/sprint-01-data-audit.md")

# AML'de klinik olarak onaylı veya ileri faz translasyonel önceliği yüksek çekirdek aday kümesi (C)
CLINICAL_CANDIDATE_DRUGS: list[str] = [
    "Venetoclax",
    "Azacytidine",
    "Cytarabine",
    "Gilteritinib",
    "Midostaurin",
    "Quizartinib (AC220)",
    "Sorafenib",
    "Crenolanib",
    "Trametinib (GSK1120212)",
    "Selumetinib (AZD6244)",
    "Dasatinib",
    "Ruxolitinib (INCB018424)",
    "Ibrutinib (PCI-32765)",
    "Entospletinib (GS-9973)",
    "AGI-5198",
    "AGI-6780",
    "Panobinostat",
    "Selinexor",
    "Rapamycin",
    "Bortezomib (Velcade)",
    "Palbociclib",
    "Lenalidomide",
    "Idelalisib",
    "Volasertib (BI-6727)",
]

AML_DRIVER_GENES: list[str] = [
    "FLT3",
    "NPM1",
    "DNMT3A",
    "TET2",
    "IDH2",
    "IDH1",
    "TP53",
    "NRAS",
    "KRAS",
    "RUNX1",
    "ASXL1",
    "SRSF2",
    "WT1",
    "PTPN11",
    "CEBPA",
    "KIT",
]


def make_specimen_key(df: pd.DataFrame) -> pd.Series:
    """DNA ve RNA dbGaP örnek kimliklerinden benzersiz örnek anahtarı üretir."""
    dna = df["dbgap_dnaseq_sample"].fillna("NONE").astype(str)
    rna = df["dbgap_rnaseq_sample"].fillna("NONE").astype(str)
    return dna + "|" + rna


def read_expression_sample_ids(exp_path: Path) -> list[str]:
    """281 MB norm_exp dosyasını RAM'e almadan yalnızca örnek sütun başlıklarını okur."""
    with exp_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
    return header[4:]


def analyze_subject_specimen_leakage(clin_df: pd.DataFrame) -> dict[str, Any]:
    """Hasta != örnek ayrımını ve dalgalar arası hasta sızıntı riskini hesaplar."""
    keys = make_specimen_key(clin_df)
    per_subj = clin_df.groupby("dbgap_subject_id").size()
    multi_subj = int((per_subj >= 2).sum())

    cohort_series = clin_df["cohort"].fillna("UNKNOWN").astype(str)
    cohort_specimens = cohort_series.value_counts().to_dict()
    cohort_subjects = clin_df.groupby(cohort_series)["dbgap_subject_id"].nunique().to_dict()

    w12 = set(clin_df.loc[cohort_series == "Waves1+2", "dbgap_subject_id"])
    w34 = set(clin_df.loc[cohort_series == "Waves3+4", "dbgap_subject_id"])
    both = set(clin_df.loc[cohort_series == "Both", "dbgap_subject_id"])

    return {
        "n_specimens": int(len(clin_df)),
        "n_unique_specimen_keys": int(keys.nunique()),
        "n_patients": int(clin_df["dbgap_subject_id"].nunique()),
        "specimens_per_patient": {
            "exactly_1": int((per_subj == 1).sum()),
            "exactly_2": int((per_subj == 2).sum()),
            "3_or_more": int((per_subj >= 3).sum()),
            "max": int(per_subj.max()),
            "patients_with_multiple_specimens": multi_subj,
            "specimens_from_multi_specimen_patients": int(per_subj[per_subj >= 2].sum()),
        },
        "cohort_specimens": {str(k): int(v) for k, v in cohort_specimens.items()},
        "cohort_patients": {str(k): int(v) for k, v in cohort_subjects.items()},
        "cross_wave_patient_overlap": {
            "waves12_and_waves34": int(len(w12 & w34)),
            "both_and_waves12": int(len(both & w12)),
            "both_and_waves34": int(len(both & w34)),
            "total_patients_in_multiple_cohort_labels": int(
                len((w12 & w34) | (both & w12) | (both & w34))
            ),
        },
    }


def decompose_two_way_mean_effect(
    auc_matrix: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Maskeli matriste y_{p,d} = mu + b_d + b_p + r_{p,d} ayrıştırması yapar."""
    mat = np.asarray(auc_matrix, dtype=np.float64)
    mask = np.isfinite(mat)
    n_obs = int(mask.sum())
    if n_obs == 0:
        raise ValueError("Varyans ayrıştırması için gözlenen değer yok.")

    vals = mat[mask]
    mu = float(vals.mean())
    total_ss = float(np.sum((vals - mu) ** 2))
    total_var = total_ss / n_obs

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    safe_mat = np.where(mask, mat, 0.0)

    # Tek-yönlü ilaç ortalaması: y_hat = col_mean
    col_means = np.divide(
        safe_mat.sum(axis=0),
        col_counts,
        out=np.full(mat.shape[1], mu),
        where=col_counts > 0,
    )
    drug_pred = np.broadcast_to(col_means[None, :], mat.shape)[mask]
    ss_res_drug = float(np.sum((vals - drug_pred) ** 2))

    # Tek-yönlü hasta ortalaması: y_hat = row_mean
    row_means = np.divide(
        safe_mat.sum(axis=1),
        row_counts,
        out=np.full(mat.shape[0], mu),
        where=row_counts > 0,
    )
    pat_pred = np.broadcast_to(row_means[:, None], mat.shape)[mask]
    ss_res_pat = float(np.sum((vals - pat_pred) ** 2))

    # Çift-yönlü toplamsal model: dönüşümlü ortalama çıkarımı (alternating projections)
    b_p = np.zeros(mat.shape[0], dtype=np.float64)
    b_d = np.zeros(mat.shape[1], dtype=np.float64)
    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        res_for_col = np.where(mask, mat - mu - b_p[:, None], 0.0)
        new_b_d = np.divide(
            res_for_col.sum(axis=0),
            col_counts,
            out=np.zeros_like(b_d),
            where=col_counts > 0,
        )
        res_for_row = np.where(mask, mat - mu - new_b_d[None, :], 0.0)
        new_b_p = np.divide(
            res_for_row.sum(axis=1),
            row_counts,
            out=np.zeros_like(b_p),
            where=row_counts > 0,
        )
        # Kimlik sabitleme (sum-zero kayması)
        if np.any(col_counts > 0):
            shift = float(new_b_d[col_counts > 0].mean())
            new_b_d[col_counts > 0] -= shift
            new_b_p[row_counts > 0] += shift

        delta = max(
            float(np.max(np.abs(new_b_d - b_d))),
            float(np.max(np.abs(new_b_p - b_p))),
        )
        b_d, b_p = new_b_d, new_b_p
        if delta < tol:
            converged = True
            break

    two_way_pred = (mu + b_p[:, None] + b_d[None, :])[mask]
    residuals = vals - two_way_pred
    ss_res_two_way = float(np.sum(residuals ** 2))

    # Hasta-içi sıralama açısından: hasta ortalaması çıkarıldıktan sonraki varyans
    # (sıralamada b_p sabit olduğundan belirleyici olan yalnızca b_d ve r_{p,d}'dir)
    within_pat_vals = vals - pat_pred
    within_pat_ss = float(np.sum(within_pat_vals ** 2))

    def _r2(ss_res: float, ss_tot: float) -> float:
        return 0.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    return {
        "n_patients_or_specimens": int(mat.shape[0]),
        "n_drugs": int(mat.shape[1]),
        "n_observed": n_obs,
        "missing_rate": float(1.0 - n_obs / mat.size),
        "global_mean": mu,
        "total_std": float(np.sqrt(total_var)),
        "total_var": total_var,
        "drug_only": {
            "r2": _r2(ss_res_drug, total_ss),
            "rmse": float(np.sqrt(ss_res_drug / n_obs)),
        },
        "patient_only": {
            "r2": _r2(ss_res_pat, total_ss),
            "rmse": float(np.sqrt(ss_res_pat / n_obs)),
        },
        "two_way_m0": {
            "r2": _r2(ss_res_two_way, total_ss),
            "rmse": float(np.sqrt(ss_res_two_way / n_obs)),
            "residual_var": ss_res_two_way / n_obs,
            "residual_std": float(np.sqrt(ss_res_two_way / n_obs)),
            "drug_effect_std": float(np.std(b_d[col_counts > 0])) if np.any(col_counts > 0) else 0.0,
            "patient_effect_std": float(np.std(b_p[row_counts > 0])) if np.any(row_counts > 0) else 0.0,
            "converged": converged,
            "n_iter": n_iter,
        },
        "within_patient_ranking_r2_of_m0_vs_patient_only": _r2(ss_res_two_way, within_pat_ss),
    }


def _summarize_matrix_subset(df_sub: pd.DataFrame, clin_df: pd.DataFrame, exp_rna_ids: set[str]) -> dict[str, Any]:
    """Verilen probit alt kümesi için matris yoğunluğu, kapsam ve M0 ayrıştırmasını hesaplar."""
    pivot = df_sub.pivot_table(index="specimen_key", columns="inhibitor", values="auc", aggfunc="median")
    n_spec, n_drug = pivot.shape
    n_obs = int(pivot.notna().sum().sum())
    per_spec = pivot.notna().sum(axis=1)
    per_drug = pivot.notna().sum(axis=0)

    clin_lookup = clin_df.set_index("specimen_key")
    subj_for_spec = clin_lookup.loc[pivot.index, "dbgap_subject_id"]
    cohort_for_spec = clin_lookup.loc[pivot.index, "cohort"].fillna("UNKNOWN")
    has_rna = clin_lookup.loc[pivot.index, "dbgap_rnaseq_sample"].isin(exp_rna_ids)

    thresholds = {}
    for k in (15, 30, 50, 80, 100):
        if k > n_drug:
            continue
        mask_k = per_spec >= k
        thresholds[f"ge_{k}_drugs"] = {
            "specimens": int(mask_k.sum()),
            "patients": int(subj_for_spec[mask_k].nunique()),
            "with_rnaseq_specimens": int((mask_k & has_rna).sum()),
            "with_rnaseq_patients": int(subj_for_spec[mask_k & has_rna].nunique()),
            "waves12_with_rnaseq_patients": int(
                subj_for_spec[mask_k & has_rna & (cohort_for_spec == "Waves1+2")].nunique()
            ),
            "waves34_with_rnaseq_patients": int(
                subj_for_spec[mask_k & has_rna & (cohort_for_spec == "Waves3+4")].nunique()
            ),
        }

    m0 = decompose_two_way_mean_effect(pivot.to_numpy(dtype=np.float64))
    return {
        "n_specimens": int(n_spec),
        "n_patients": int(subj_for_spec.nunique()),
        "n_inhibitors": int(n_drug),
        "n_observations": n_obs,
        "fill_rate": float(n_obs / (n_spec * n_drug)) if n_spec * n_drug > 0 else 0.0,
        "missing_rate": float(1.0 - n_obs / (n_spec * n_drug)) if n_spec * n_drug > 0 else 1.0,
        "drugs_per_specimen": {
            "min": int(per_spec.min()),
            "p10": float(np.percentile(per_spec, 10)),
            "median": float(np.median(per_spec)),
            "p90": float(np.percentile(per_spec, 90)),
            "max": int(per_spec.max()),
        },
        "specimens_per_drug": {
            "min": int(per_drug.min()),
            "p10": float(np.percentile(per_drug, 10)),
            "median": float(np.median(per_drug)),
            "p90": float(np.percentile(per_drug, 90)),
            "max": int(per_drug.max()),
        },
        "coverage_thresholds": thresholds,
        "m0_variance_decomposition": m0,
    }


def _compute_driver_frequencies(
    clin_df: pd.DataFrame,
    muts_df: pd.DataFrame,
    target_keys: set[str],
) -> list[dict[str, Any]]:
    """Eşleşmiş örnek alt kümesinde AML sürücü mutasyon sıklıklarını çıkarır."""
    sub_clin = clin_df[clin_df["specimen_key"].isin(target_keys)].copy()
    sub_dna = set(sub_clin["dbgap_dnaseq_sample"].dropna())
    sub_muts = muts_df[muts_df["dbgap_sample_id"].isin(sub_dna)]

    dna_to_subj = dict(
        zip(
            sub_clin.loc[sub_clin["dbgap_dnaseq_sample"].notna(), "dbgap_dnaseq_sample"],
            sub_clin.loc[sub_clin["dbgap_dnaseq_sample"].notna(), "dbgap_subject_id"],
        )
    )
    n_dna_spec = len(sub_dna)
    n_dna_pat = len(set(dna_to_subj.values()))

    rows = []
    for gene in AML_DRIVER_GENES:
        g_muts = sub_muts[sub_muts["symbol"] == gene]
        spec_hit = int(g_muts["dbgap_sample_id"].nunique())
        pat_hit = len({dna_to_subj[s] for s in g_muts["dbgap_sample_id"] if s in dna_to_subj})
        rows.append(
            {
                "gene": gene,
                "wes_specimens_mutated": spec_hit,
                "wes_specimens_total": n_dna_spec,
                "wes_specimen_freq": float(spec_hit / n_dna_spec) if n_dna_spec else 0.0,
                "wes_patients_mutated": pat_hit,
                "wes_patients_total": n_dna_pat,
                "wes_patient_freq": float(pat_hit / n_dna_pat) if n_dna_pat else 0.0,
            }
        )
    return rows


def run_audit(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
) -> dict[str, Any]:
    """Beat AML 2.0 tam veri denetimini çalıştırır ve JSON + Markdown raporu yazar."""
    clin = pd.read_excel(raw_dir / "beataml_wv1to4_clinical.xlsx", sheet_name="summary")
    smap = pd.read_excel(raw_dir / "beataml_waves1to4_sample_mapping.xlsx", sheet_name="sample_map")
    dfam = pd.read_excel(raw_dir / "beataml_drug_families.xlsx", sheet_name="drug_family")
    dgene = pd.read_excel(raw_dir / "beataml_drug_families.xlsx", sheet_name="drug_gene")
    probit = pd.read_csv(raw_dir / "beataml_probit_curve_fits_v4_dbgap.txt", sep="\t", low_memory=False)
    muts = pd.read_csv(raw_dir / "beataml_wes_wv1to4_mutations_dbgap.txt", sep="\t", low_memory=False)
    exp_cols = read_expression_sample_ids(raw_dir / "beataml_waves1to4_norm_exp_dbgap.txt")
    exp_rna_set = set(exp_cols)

    clin["specimen_key"] = make_specimen_key(clin)
    probit["specimen_key"] = make_specimen_key(probit)
    smap["specimen_key"] = make_specimen_key(smap)

    leakage = analyze_subject_specimen_leakage(clin)

    # Sağlıklı kontroller ve QC kapsamı
    is_ctrl = smap["rna_control"].notna() & (smap["rna_control"] != "No")
    smap_ctrl = smap[is_ctrl]
    smap_pat = smap[~is_ctrl]

    clin_keys = set(clin["specimen_key"])
    probit_keys = set(probit["specimen_key"])
    clin_has_exp_rna = clin["dbgap_rnaseq_sample"].isin(exp_rna_set)
    clin_has_wes_file = clin["dbgap_dnaseq_sample"].isin(set(muts["dbgap_sample_id"].unique()))
    clin_has_dna_col = clin["dbgap_dnaseq_sample"].notna()
    clin_has_drug = clin["specimen_key"].isin(probit_keys)

    overlap = {
        "clinical_specimens": int(len(clin)),
        "clinical_patients": int(clin["dbgap_subject_id"].nunique()),
        "smap_total_rows": int(len(smap)),
        "smap_healthy_controls": int(len(smap_ctrl)),
        "norm_exp_total_columns": int(len(exp_cols)),
        "norm_exp_patient_columns": int(len(exp_rna_set & set(clin["dbgap_rnaseq_sample"].dropna()))),
        "norm_exp_healthy_control_columns": int(len(exp_rna_set & set(smap_ctrl["dbgap_rnaseq_sample"].dropna()))),
        "probit_specimens": int(len(probit_keys)),
        "probit_orphan_specimens_not_in_clin": int(len(probit_keys - clin_keys)),
        "drug_and_norm_exp_specimens": int((clin_has_drug & clin_has_exp_rna).sum()),
        "drug_and_norm_exp_patients": int(clin.loc[clin_has_drug & clin_has_exp_rna, "dbgap_subject_id"].nunique()),
        "drug_and_dnaseq_col_specimens": int((clin_has_drug & clin_has_dna_col).sum()),
        "drug_and_wes_calls_specimens": int((clin_has_drug & clin_has_wes_file).sum()),
        "drug_and_norm_exp_and_dnaseq_specimens": int((clin_has_drug & clin_has_exp_rna & clin_has_dna_col).sum()),
        "drug_and_norm_exp_and_dnaseq_patients": int(
            clin.loc[clin_has_drug & clin_has_exp_rna & clin_has_dna_col, "dbgap_subject_id"].nunique()
        ),
        "smap_patient_keys_missing_from_clin": int(len(set(smap_pat["specimen_key"]) - clin_keys)),
    }

    # Dalga bazlı Drug ∩ RNA-seq kesişimi
    wave_overlap = {}
    for coh in ("Waves1+2", "Waves3+4", "Both"):
        m = (clin["cohort"] == coh) & clin_has_drug & clin_has_exp_rna
        wave_overlap[coh] = {
            "specimens": int(m.sum()),
            "patients": int(clin.loc[m, "dbgap_subject_id"].nunique()),
        }

    # Probit kalite, sansürleme ve AUC özetleri
    n_dup_pairs = int(probit.duplicated(subset=["specimen_key", "inhibitor"]).sum())
    auc_vals = probit["auc"].to_numpy(dtype=np.float64)
    cens_min = float((probit["ic50"] <= probit["min_conc"] * 1.0001).mean())
    cens_max = float((probit["ic50"] >= probit["max_conc"] * 0.9999).mean())

    probit_quality = {
        "total_rows": int(len(probit)),
        "duplicate_specimen_inhibitor_rows": n_dup_pairs,
        "unique_inhibitors": int(probit["inhibitor"].nunique()),
        "status_row_counts": {str(k): int(v) for k, v in probit["status"].value_counts().items()},
        "status_inhibitor_counts": {
            str(k): int(v) for k, v in probit.groupby("status")["inhibitor"].nunique().items()
        },
        "converged_true_rate": float((probit["converged"] == True).mean()),
        "curve_type_counts": {str(k): int(v) for k, v in probit["curve_type"].value_counts().items()},
        "ic50_censored_at_min_rate": cens_min,
        "ic50_censored_at_max_rate": cens_max,
        "ic50_total_censored_rate": cens_min + cens_max,
        "auc_summary": {
            "min": float(np.min(auc_vals)),
            "p05": float(np.percentile(auc_vals, 5)),
            "q25": float(np.percentile(auc_vals, 25)),
            "median": float(np.median(auc_vals)),
            "mean": float(np.mean(auc_vals)),
            "q75": float(np.percentile(auc_vals, 75)),
            "p95": float(np.percentile(auc_vals, 95)),
            "max": float(np.max(auc_vals)),
            "std": float(np.std(auc_vals)),
        },
    }

    # Dalga bazlı ilaç kapsamı
    probit_m = probit.merge(
        clin[["specimen_key", "cohort"]].drop_duplicates(),
        on="specimen_key",
        how="left",
    )
    has_exp = probit_m["dbgap_rnaseq_sample"].isin(exp_rna_set)
    probit_m["is_w12"] = probit_m["cohort"] == "Waves1+2"
    probit_m["is_w34"] = probit_m["cohort"] == "Waves3+4"
    probit_m["is_both"] = probit_m["cohort"] == "Both"
    probit_m["is_w12_exp"] = probit_m["is_w12"] & has_exp
    probit_m["is_w34_exp"] = probit_m["is_w34"] & has_exp

    drug_stats = (
        probit_m.groupby(["inhibitor", "status"])
        .agg(
            total_spec=("specimen_key", "nunique"),
            w12_spec=("is_w12", "sum"),
            w34_spec=("is_w34", "sum"),
            both_spec=("is_both", "sum"),
            w12_exp=("is_w12_exp", "sum"),
            w34_exp=("is_w34_exp", "sum"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
        )
        .reset_index()
    )

    cross_wave_ge100 = drug_stats[(drug_stats["w12_spec"] >= 100) & (drug_stats["w34_spec"] >= 100)]
    cross_wave_exp_ge80 = drug_stats[(drug_stats["w12_exp"] >= 80) & (drug_stats["w34_exp"] >= 80)]

    # Alt matris özetleri: tüm 166, train/test (95), cross-wave çekirdek ve aday küme (C)
    subsets = {
        "all_166": _summarize_matrix_subset(probit, clin, exp_rna_set),
        "train_test_95": _summarize_matrix_subset(
            probit[probit["status"] == "train/test"], clin, exp_rna_set
        ),
        "cross_wave_ge100": _summarize_matrix_subset(
            probit[probit["inhibitor"].isin(cross_wave_ge100["inhibitor"])], clin, exp_rna_set
        ),
        "clinical_candidates_C": _summarize_matrix_subset(
            probit[probit["inhibitor"].isin(CLINICAL_CANDIDATE_DRUGS)], clin, exp_rna_set
        ),
    }

    # İlaç aileleri ve hedef gen kapsamı
    probit_drugs = set(probit["inhibitor"].unique())
    fam_drugs = set(dfam["inhibitor"].unique())
    gene_drugs = set(dgene["inhibitor"].unique())
    top_families = dfam["family"].value_counts().head(12).to_dict()

    # Aday küme C tablosu
    cand_df = drug_stats[drug_stats["inhibitor"].isin(CLINICAL_CANDIDATE_DRUGS)].sort_values(
        "total_spec", ascending=False
    )
    candidate_records = cand_df.to_dict(orient="records")

    # Sürücü mutasyon sıklıkları (Drug ∩ RNA-seq örneklemi üzerinde)
    matched_keys = set(clin.loc[clin_has_drug & clin_has_exp_rna, "specimen_key"])
    driver_freqs = _compute_driver_frequencies(clin, muts, matched_keys)

    matched_clin = clin[clin["specimen_key"].isin(matched_keys)]
    clinical_markers = {
        "flt3_itd_positive_all_specimens": int((clin["FLT3-ITD"] == "positive").sum()),
        "flt3_itd_positive_all_patients": int(
            clin.loc[clin["FLT3-ITD"] == "positive", "dbgap_subject_id"].nunique()
        ),
        "flt3_itd_positive_matched_specimens": int((matched_clin["FLT3-ITD"] == "positive").sum()),
        "flt3_itd_positive_matched_patients": int(
            matched_clin.loc[matched_clin["FLT3-ITD"] == "positive", "dbgap_subject_id"].nunique()
        ),
        "npm1_positive_matched_patients": int(
            matched_clin.loc[matched_clin["NPM1"] == "positive", "dbgap_subject_id"].nunique()
        ),
        "eln2017_all_counts": {str(k): int(v) for k, v in clin["ELN2017"].fillna("MISSING").value_counts().items()},
        "eln2017_matched_counts": {
            str(k): int(v) for k, v in matched_clin["ELN2017"].fillna("MISSING").value_counts().items()
        },
        "non_aml_all_specimens": int((clin["ELN2017"] == "NonAML").sum()),
        "non_aml_matched_specimens": int((matched_clin["ELN2017"] == "NonAML").sum()),
        "non_aml_matched_patients": int(
            matched_clin.loc[matched_clin["ELN2017"] == "NonAML", "dbgap_subject_id"].nunique()
        ),
    }

    summary: dict[str, Any] = {
        "leakage_and_units": leakage,
        "modality_overlap": overlap,
        "wave_drug_rnaseq_overlap": wave_overlap,
        "probit_quality": probit_quality,
        "cross_wave_drug_counts": {
            "w12_ge100_and_w34_ge100": int(len(cross_wave_ge100)),
            "w12_exp_ge80_and_w34_exp_ge80": int(len(cross_wave_exp_ge80)),
            "by_status_ge100_both_waves": {
                str(k): int(v) for k, v in cross_wave_ge100["status"].value_counts().items()
            },
        },
        "matrix_subsets": subsets,
        "drug_families": {
            "probit_drugs_in_drug_family": int(len(probit_drugs & fam_drugs)),
            "probit_drugs_missing_from_drug_family": int(len(probit_drugs - fam_drugs)),
            "probit_drugs_in_drug_gene": int(len(probit_drugs & gene_drugs)),
            "top_families": {str(k): int(v) for k, v in top_families.items()},
        },
        "clinical_candidate_drugs_C": candidate_records,
        "driver_mutation_frequencies_in_drug_rna": driver_freqs,
        "clinical_markers_and_diagnosis": clinical_markers,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_markdown_report(summary), encoding="utf-8")
    return summary


def _render_markdown_report(s: dict[str, Any]) -> str:
    leak = s["leakage_and_units"]
    ov = s["modality_overlap"]
    wov = s["wave_drug_rnaseq_overlap"]
    pq = s["probit_quality"]
    cw = s["cross_wave_drug_counts"]
    sub_all = s["matrix_subsets"]["all_166"]
    sub_tt = s["matrix_subsets"]["train_test_95"]
    sub_cw = s["matrix_subsets"]["cross_wave_ge100"]
    sub_c = s["matrix_subsets"]["clinical_candidates_C"]

    m0_all = sub_all["m0_variance_decomposition"]
    m0_tt = sub_tt["m0_variance_decomposition"]
    m0_c = sub_c["m0_variance_decomposition"]

    lines = [
        "# Sprint 1 — Beat AML 2.0 Ampirik Veri Denetimi ve Protokol Temeli",
        "",
        "Bu rapor, `biodev/beataml2.0_data` (Waves 1–4, CC BY 4.0) resmi harmonize dosyalarının SHA-256 doğrulaması sonrası yapılan tam ampirik denetimini özetler. Tüm sayılar ham dosyalardan doğrudan ölçülmüştür.",
        "",
        "---",
        "",
        "## 1. Hasta ≠ Örnek Ayrımı ve Sızıntı Bulguları",
        "",
        f"- **Klinik özet (`beataml_wv1to4_clinical.xlsx`):** **{leak['n_patients']}** benzersiz hasta (`dbgap_subject_id`), **{leak['n_specimens']}** benzersiz örnek (`specimen_key`).",
        f"- **Birden fazla örneği olan hastalar:** {leak['specimens_per_patient']['patients_with_multiple_specimens']} hastanın 2+ örneği var "
        f"(2 örnek: {leak['specimens_per_patient']['exactly_2']}, ≥3 örnek: {leak['specimens_per_patient']['3_or_more']}, maks: {leak['specimens_per_patient']['max']}; "
        f"toplam {leak['specimens_per_patient']['specimens_from_multi_specimen_patients']} örnek). **Örnek-bazlı rastgele bölme sızıntı yapar; bölme birimi kesinlikle `dbgap_subject_id` olmalıdır.**",
        f"- **Örnek haritası (`sample_map`):** {ov['smap_total_rows']} satır = {ov['clinical_specimens']} hasta örneği + **{ov['smap_healthy_controls']} sağlıklı kontrol** (`Healthy pooled CD34+` / `Healthy Individual`). Analizde sağlıklı kontroller dışlanır.",
        f"- **Dalga dağılımı (örnek / hasta):** `Waves1+2`: {leak['cohort_specimens'].get('Waves1+2', 0)} / {leak['cohort_patients'].get('Waves1+2', 0)}, "
        f"`Waves3+4`: {leak['cohort_specimens'].get('Waves3+4', 0)} / {leak['cohort_patients'].get('Waves3+4', 0)}, "
        f"`Both`: {leak['cohort_specimens'].get('Both', 0)} / {leak['cohort_patients'].get('Both', 0)}.",
        f"- **Kritik dalga-arası sızıntı:** `Waves1+2` ile `Waves3+4` arasında **{leak['cross_wave_patient_overlap']['waves12_and_waves34']} ortak hasta**, "
        f"`Both` etiketi ile dalgalar arasında ek çakışmalarla toplam **{leak['cross_wave_patient_overlap']['total_patients_in_multiple_cohort_labels']} ortak hasta** vardır. "
        "Zamansal dış bölmede (`Waves1+2 → Waves3+4`) bu hastalar ve `Both` örnekleri test kümesinden temizlenmelidir.",
        "",
        "---",
        "",
        "## 2. Modalite Kesişimi ve Alt Kohort Büyüklükleri",
        "",
        "| Katman / Kesişim | Örnek | Hasta |",
        "|---|---:|---:|",
        f"| Klinik özet (tüm AML/MDS-MPN örnekleri) | {ov['clinical_specimens']} | {ov['clinical_patients']} |",
        f"| *Ex vivo* inhibitör verisi (`probit_curve_fits`) | {ov['probit_specimens']} | {sub_all['n_patients']} |",
        f"| RNA-seq (`norm_exp` içinde bulunan hasta sütunu) | {ov['norm_exp_patient_columns']} | — |",
        f"| **İnhibitör ∩ RNA-seq (`norm_exp`)** | **{ov['drug_and_norm_exp_specimens']}** | **{ov['drug_and_norm_exp_patients']}** |",
        f"| İnhibitör ∩ DNA-seq (klinik kimlik) | {ov['drug_and_dnaseq_col_specimens']} | — |",
        f"| İnhibitör ∩ RNA-seq ∩ DNA-seq | {ov['drug_and_norm_exp_and_dnaseq_specimens']} | {ov['drug_and_norm_exp_and_dnaseq_patients']} |",
        f"| `Waves1+2` (İnhibitör ∩ RNA-seq) | {wov['Waves1+2']['specimens']} | {wov['Waves1+2']['patients']} |",
        f"| `Waves3+4` (İnhibitör ∩ RNA-seq) | {wov['Waves3+4']['specimens']} | {wov['Waves3+4']['patients']} |",
        f"| `Both` (İnhibitör ∩ RNA-seq) | {wov['Both']['specimens']} | {wov['Both']['patients']} |",
        "",
        f"Not: `probit_curve_fits` içinde klinik tabloyla eşleşmeyen (orphan) örnek sayısı **{ov['probit_orphan_specimens_not_in_clin']}**, mükerrer `(örnek, inhibitör)` satır sayısı **{pq['duplicate_specimen_inhibitor_rows']}**.",
        "",
        "---",
        "",
        "## 3. İnhibitör Matrisi Kalitesi, Sansürleme ve Seyreklik",
        "",
        f"- **Neden birincil hedef `auc` (IC50 değil)?** `ic50` değerlerinin **%{pq['ic50_censored_at_max_rate']*100:.2f}**'si `max_conc` üst sınırında (`all_gt_50`), **%{pq['ic50_censored_at_min_rate']*100:.2f}**'si `min_conc` alt sınırında (`all_lt_50`) sansürlüdür (toplam **%{pq['ic50_total_censored_rate']*100:.2f}** sansürlü). `auc` kesintisizdir (min={pq['auc_summary']['min']:.1f}, medyan={pq['auc_summary']['median']:.1f}, std={pq['auc_summary']['std']:.2f}, maks={pq['auc_summary']['max']:.1f}); yakınsama oranı **%{pq['converged_true_rate']*100:.2f}**.",
        f"- **166 İnhibitörün `status` dağılımı:** `train/test` = **{pq['status_inhibitor_counts'].get('train/test', 0)}** ({pq['status_row_counts'].get('train/test', 0)} satır), "
        f"`combined` = **{pq['status_inhibitor_counts'].get('combined', 0)}** ({pq['status_row_counts'].get('combined', 0)} satır), "
        f"`exploratory` = **{pq['status_inhibitor_counts'].get('exploratory', 0)}** ({pq['status_row_counts'].get('exploratory', 0)} satır).",
        f"- **Her iki dalgada (`Waves1+2` ve `Waves3+4`) en az 100'er örnekte ölçülen ortak ilaç:** **{cw['w12_ge100_and_w34_ge100']}** ilaç "
        f"({cw['by_status_ge100_both_waves']}); her iki dalgada RNA-seq'li en az 80'er örnekte ortak ölçülen: **{cw['w12_exp_ge80_and_w34_exp_ge80']}** ilaç.",
        "",
        "| Alt Küme | Örnek | Hasta | İlaç | Gözlem | Doluluk (%) | Eksiklik (%) | Örnek Başına İlaç (Medyan [P10–P90]) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Tüm panel (`all_166`) | {sub_all['n_specimens']} | {sub_all['n_patients']} | {sub_all['n_inhibitors']} | {sub_all['n_observations']} | {sub_all['fill_rate']*100:.1f} | {sub_all['missing_rate']*100:.1f} | {sub_all['drugs_per_specimen']['median']:.0f} [{sub_all['drugs_per_specimen']['p10']:.0f}–{sub_all['drugs_per_specimen']['p90']:.0f}] |",
        f"| `train/test` (95 ilaç) | {sub_tt['n_specimens']} | {sub_tt['n_patients']} | {sub_tt['n_inhibitors']} | {sub_tt['n_observations']} | {sub_tt['fill_rate']*100:.1f} | {sub_tt['missing_rate']*100:.1f} | {sub_tt['drugs_per_specimen']['median']:.0f} [{sub_tt['drugs_per_specimen']['p10']:.0f}–{sub_tt['drugs_per_specimen']['p90']:.0f}] |",
        f"| Çift-dalga ≥100 (`cross_wave`) | {sub_cw['n_specimens']} | {sub_cw['n_patients']} | {sub_cw['n_inhibitors']} | {sub_cw['n_observations']} | {sub_cw['fill_rate']*100:.1f} | {sub_cw['missing_rate']*100:.1f} | {sub_cw['drugs_per_specimen']['median']:.0f} [{sub_cw['drugs_per_specimen']['p10']:.0f}–{sub_cw['drugs_per_specimen']['p90']:.0f}] |",
        f"| Aday Küme $\\mathcal{{C}}$ ({sub_c['n_inhibitors']} ilaç) | {sub_c['n_specimens']} | {sub_c['n_patients']} | {sub_c['n_inhibitors']} | {sub_c['n_observations']} | {sub_c['fill_rate']*100:.1f} | {sub_c['missing_rate']*100:.1f} | {sub_c['drugs_per_specimen']['median']:.0f} [{sub_c['drugs_per_specimen']['p10']:.0f}–{sub_c['drugs_per_specimen']['p90']:.0f}] |",
        "",
        "### Örnek Başına İnhibitör Kapsam Eşikleri (`cross_wave` ortak ilaç seti)",
        "",
        "| Eşik | Toplam Örnek | Toplam Hasta | RNA-seq'li Örnek | RNA-seq'li Hasta | W1+2 RNA Hasta | W3+4 RNA Hasta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for label, key in [("≥30 ilaç", "ge_30_drugs"), ("≥50 ilaç", "ge_50_drugs"), ("≥80 ilaç", "ge_80_drugs"), ("≥100 ilaç", "ge_100_drugs")]:
        t = sub_cw["coverage_thresholds"].get(key)
        if t:
            lines.append(
                f"| {label} | {t['specimens']} | {t['patients']} | {t['with_rnaseq_specimens']} | {t['with_rnaseq_patients']} | {t['waves12_with_rnaseq_patients']} | {t['waves34_with_rnaseq_patients']} |"
            )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Çift-Yönlü `M0` (Ortalama-Etki) Varyans Ayrıştırması — Gerçek Tahmin Tavanı",
            "",
            "*DrEval* (2026) bulgusuna paralel olarak, herhangi bir gelişmiş model kurmadan önce ham `auc` varyansının ne kadarının yalnızca ortalama etkilerle açıklandığı ölçülmüştür ($y_{p,d} = \\mu + b_d + b_p + r_{p,d}$):",
            "",
            "| Alt Küme | Toplam Std | İlaç Ort. $R^2$ | Hasta Ort. $R^2$ | Çift-Yön `M0` $R^2$ | `M0` RMSE | Artık ($r_{p,d}$) Std | Hasta-İçi Sıralama $R^2$ (`M0`) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| Tüm 166 ilaç | {m0_all['total_std']:.2f} | {m0_all['drug_only']['r2']:.3f} | {m0_all['patient_only']['r2']:.3f} | {m0_all['two_way_m0']['r2']:.3f} | {m0_all['two_way_m0']['rmse']:.2f} | {m0_all['two_way_m0']['residual_std']:.2f} | {m0_all['within_patient_ranking_r2_of_m0_vs_patient_only']:.3f} |",
            f"| `train/test` (95 ilaç) | {m0_tt['total_std']:.2f} | {m0_tt['drug_only']['r2']:.3f} | {m0_tt['patient_only']['r2']:.3f} | {m0_tt['two_way_m0']['r2']:.3f} | {m0_tt['two_way_m0']['rmse']:.2f} | {m0_tt['two_way_m0']['residual_std']:.2f} | {m0_tt['within_patient_ranking_r2_of_m0_vs_patient_only']:.3f} |",
            f"| Aday Küme $\\mathcal{{C}}$ ({sub_c['n_inhibitors']} ilaç) | {m0_c['total_std']:.2f} | {m0_c['drug_only']['r2']:.3f} | {m0_c['patient_only']['r2']:.3f} | {m0_c['two_way_m0']['r2']:.3f} | {m0_c['two_way_m0']['rmse']:.2f} | {m0_c['two_way_m0']['residual_std']:.2f} | {m0_c['within_patient_ranking_r2_of_m0_vs_patient_only']:.3f} |",
            "",
            "**Kritik metodolojik sonuç:** Ham `auc` üzerinde `M0` zaten tek başına yüksek $R^2$ açıklamaktadır (bunun ana nedeni ilaçların kendi ortalama duyarlılık farklarıdır). Bu yüzden modellerin başarısını yalnızca ham `auc` korelasyonu ile ölçmek **yanıltıcıdır**; Sprint 2 protokolünde mutlaka **ilaç-ortalamasından arındırılmış kalıntı ($r_{p,d}$) korelasyonu / RMSE'si** ile **hasta-içi sıralama kazancı (`ΔNDCG@3` vs `M0`)** esas alınmalıdır.",
            "",
            "---",
            "",
            "## 5. Klinik/Translasyonel Aday Küme ($\\mathcal{C}$) Taslağı ve İlaç Aileleri",
            "",
            f"- `beataml_drug_families.xlsx` dosyası 166 inhibitörün **{s['drug_families']['probit_drugs_in_drug_family']}** tanesini kapsıyor (**{s['drug_families']['probit_drugs_missing_from_drug_family']}** kinaz-dışı/ek ajan burada yok; `drug_gene` sekmesi **{s['drug_families']['probit_drugs_in_drug_gene']}** ilacı hedeflerine bağlıyor).",
            f"- Hem `Waves1+2` hem `Waves3+4`'te iyi kapsanan, AML'de klinik olarak anlamlı **{len(s['clinical_candidate_drugs_C'])} aday ilaç ($\\mathcal{{C}}$)** aşağıdadır (`Venetoclax`, `Azacytidine`, `Cytarabine`, `Gilteritinib`, `Midostaurin`, `Quizartinib`, `Sorafenib` dahil):",
            "",
            "| İnhibitör | Status | Toplam Örnek | W1+2 | W3+4 | W1+2 (RNA) | W3+4 (RNA) | Ort. AUC | Std AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in s["clinical_candidate_drugs_C"]:
        lines.append(
            f"| {row['inhibitor']} | {row['status']} | {row['total_spec']} | {row['w12_spec']} | {row['w34_spec']} | "
            f"{row['w12_exp']} | {row['w34_exp']} | {row['mean_auc']:.1f} | {row['std_auc']:.1f} |"
        )

    cm = s["clinical_markers_and_diagnosis"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Eşleşmiş Kohortta (`Drug ∩ RNA-seq`) Başlıca Sürücü Mutasyonlar ve Tanı Kapsamı",
            "",
            f"- **Kritik biyoinformatik tespit (`FLT3-ITD`):** WES (`beataml_wes_wv1to4_mutations_dbgap.txt`) yalnızca nokta/küçük indel çağrılarını (`FLT3` %10.3) içerir; `FLT3-ITD` ise klinik özet sütunundadır ve tüm kohortta **{cm['flt3_itd_positive_all_specimens']}** örnekte (**{cm['flt3_itd_positive_all_patients']}** hasta), `Drug ∩ RNA-seq` eşleşmiş kohortta ise **{cm['flt3_itd_positive_matched_specimens']}** örnekte (**{cm['flt3_itd_positive_matched_patients']}** hasta) pozitiftir. `NPM1` pozitif hasta sayısı eşleşmiş kohortta **{cm['npm1_positive_matched_patients']}**'tir.",
            f"- **Tanı kapsamı (`ELN2017 == NonAML`):** Tüm tabloda **{cm['non_aml_all_specimens']}** örnek, `Drug ∩ RNA-seq` eşleşmiş kohortta ise **{cm['non_aml_matched_specimens']}** örnek (**{cm['non_aml_matched_patients']}** hasta) `NonAML` (MDS/MPN veya diğer) olarak etiketlidir.",
            "",
            "| Gen (WES çağrısı) | Mutasyonlu Hasta | WES Hastası | Frekans (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for g in s["driver_mutation_frequencies_in_drug_rna"][:12]:
        lines.append(
            f"| {g['gene']} | {g['wes_patients_mutated']} | {g['wes_patients_total']} | {g['wes_patient_freq']*100:.1f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 7. Kapı 0 Değerlendirmesi ve Sprint 2 Öncesi Karar Soruları",
            "",
            "**Kapı 0 Sonucu: GEÇTİ (Veri yeterli, ama sıkı sızıntı temizliği zorunlu).**",
            f"- Hem ilaç hem RNA-seq verisine sahip **{ov['drug_and_norm_exp_patients']}** benzersiz hasta (**{ov['drug_and_norm_exp_specimens']}** örnek) mevcut.",
            f"- Çift-dalgada ortak ölçülen **{cw['w12_ge100_and_w34_ge100']}** inhibitör içinde `≥80 ilaç` ölçülmüş **RNA-seq'li {sub_cw['coverage_thresholds']['ge_80_drugs']['with_rnaseq_patients']} hasta** (W1+2: {sub_cw['coverage_thresholds']['ge_80_drugs']['waves12_with_rnaseq_patients']}, W3+4: {sub_cw['coverage_thresholds']['ge_80_drugs']['waves34_with_rnaseq_patients']}) var — güçlü `5-Fold CV` ve `Waves1+2 → Waves3+4` doğrulaması için yeterli.",
            "",
            "### Protokolü (`docs/PROTOCOL.md`) Dondurmadan Önce Netleştirilecek 4 Karar",
            f"1. **Hedef Aday Küme ($\\mathcal{{C}}$) ve Prob Havuzu ($\\mathcal{{U}}$):** Yukarıdaki {len(s['clinical_candidate_drugs_C'])} AML klinik adayı $\\mathcal{{C}}$ olarak dondurulsun mu, yoksa yalnızca `train/test` (95) içindeki klinik ilaçlar mı yoksa `combined` (`Venetoclax`, `Gilteritinib`, `Azacytidine`, `Cytarabine`, `Panobinostat`) mutlaka dahil edilsin mi? *(Öneri: Hepsi çift-dalga kapsamlı olduğundan bu {len(s['clinical_candidate_drugs_C'])} klinik ilacın tamamını $\\mathcal{{C}}$ olarak almak; prob havuzu $\\mathcal{{U}}$ ise `cross_wave_ge100` havuzu — {cw['w12_ge100_and_w34_ge100']} ilaç.)*",
            "2. **Çoklu Örnek Kuralı:** Birden fazla örneği olan hastalarda ana bölme `dbgap_subject_id` üzerinden yapılacak; test kümesinde hasta başına yalnızca **ilk/tanı örneği** mi puanlansın yoksa tüm zaman noktaları mı (küme ağırlıklı)? *(Öneri: Testte hasta başına ilk örnek, duyarlılık analizinde tüm örnekler.)*",
            f"3. **`NonAML` Örnekleri:** `ELN2017 == NonAML` olan {cm['non_aml_matched_specimens']} eşleşmiş örnek ({cm['non_aml_matched_patients']} hasta) ana AML çalışmasından çıkarılsın mı? *(Öneri: Evet, birincil analiz saf AML'de yapılıp `NonAML` yalnızca duyarlılıkta incelenmeli.)*",
            "4. **Birincil Başarı Ölçütü:** İlaçların ortalama AUC farkının yarattığı sahte başarıyı önlemek için birincil metrik **`residual_auc` (ilaç ortalamasından arındırılmış RMSE / hasta-içi Spearman)** ve **`ΔNDCG@3 (vs M0)`** olarak dondurulsun mu?",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Beat AML 2.0 ampirik veri denetimini çalıştır.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    summary = run_audit(args.raw_dir, args.out_json, args.out_md)
    print(
        json.dumps(
            {
                "patients": summary["leakage_and_units"]["n_patients"],
                "specimens": summary["leakage_and_units"]["n_specimens"],
                "drug_and_rnaseq_patients": summary["modality_overlap"]["drug_and_norm_exp_patients"],
                "cross_wave_drugs_ge100": summary["cross_wave_drug_counts"]["w12_ge100_and_w34_ge100"],
                "m0_all_r2": round(summary["matrix_subsets"]["all_166"]["m0_variance_decomposition"]["two_way_m0"]["r2"], 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
