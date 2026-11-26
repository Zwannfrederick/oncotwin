"""Beat AML 2.0 saf AML çoklu-omik ve ex vivo AUC veri yükleyicisi (sızıntısız ham tensörler)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.oncotwin.data.audit_beataml import (
    AML_DRIVER_GENES,
    CLINICAL_CANDIDATE_DRUGS,
    DEFAULT_RAW_DIR,
    make_specimen_key,
    read_expression_sample_ids,
)

STAGE_PRIORITY: dict[str, int] = {
    "Initial Diagnosis": 0,
    "Relapse": 1,
    "Residual": 2,
    "Unknown": 3,
    "Remission": 4,
}

SPECIMEN_TYPE_PRIORITY: dict[str, int] = {
    "Bone Marrow Aspirate": 0,
    "Peripheral Blood": 1,
}

WES_DRIVER_GENES: list[str] = [g for g in AML_DRIVER_GENES if g not in {"FLT3", "NPM1"}]


@dataclass(frozen=True)
class BeatAMLDataset:
    """Eşleşmiş saf AML örnekleri için ham (ölçeklenmemiş) veri kapsayıcısı."""

    meta: pd.DataFrame
    Y_auc: np.ndarray
    X_rna: np.ndarray
    X_mut: np.ndarray
    drug_names: list[str]
    gene_names: list[str]
    mut_feature_names: list[str]
    w12_all_subjects: set[int]
    both_all_subjects: set[int]

    def subset_by_positions(self, positions: np.ndarray) -> BeatAMLDataset:
        """Verilen satır pozisyonlarına göre alt küme döndürür."""
        pos = np.asarray(positions, dtype=np.int64)
        return BeatAMLDataset(
            meta=self.meta.iloc[pos].reset_index(drop=True),
            Y_auc=self.Y_auc[pos],
            X_rna=self.X_rna[pos],
            X_mut=self.X_mut[pos],
            drug_names=self.drug_names,
            gene_names=self.gene_names,
            mut_feature_names=self.mut_feature_names,
            w12_all_subjects=self.w12_all_subjects,
            both_all_subjects=self.both_all_subjects,
        )


def select_index_specimen_positions(meta_df: pd.DataFrame) -> np.ndarray:
    """Protokol Bölüm 2 uyarınca her hasta için deterministik indeks örnek pozisyonlarını seçer.

    Sıralama önceliği (hiçbir ilaç AUC etiketi kullanılmaz):
    1. timeOfSampleCollectionRelativeToInclusion (NaN -> +inf)
    2. diseaseStageAtSpecimenCollection (Initial Diagnosis < Relapse < Residual < Unknown < Remission)
    3. dbgap_dnaseq_sample dolu olması (True < False)
    4. specimenType (Bone Marrow Aspirate < Peripheral Blood < diğer)
    5. specimen_key alfabetik artan
    """
    work = pd.DataFrame(
        {
            "_pos": np.arange(len(meta_df), dtype=np.int64),
            "dbgap_subject_id": meta_df["dbgap_subject_id"].to_numpy(),
            "_time": meta_df["timeOfSampleCollectionRelativeToInclusion"].fillna(np.inf).to_numpy(),
            "_stage": meta_df["diseaseStageAtSpecimenCollection"].map(STAGE_PRIORITY).fillna(9).to_numpy(),
            "_no_dna": meta_df["dbgap_dnaseq_sample"].isna().astype(np.int64).to_numpy(),
            "_stype": meta_df["specimenType"].map(SPECIMEN_TYPE_PRIORITY).fillna(2).to_numpy(),
            "_key": meta_df["specimen_key"].astype(str).to_numpy(),
        }
    )
    chosen = (
        work.sort_values(
            ["_time", "_stage", "_no_dna", "_stype", "_key"],
            kind="mergesort",
        )
        .drop_duplicates("dbgap_subject_id", keep="first")
        .sort_values("_pos")
    )
    return chosen["_pos"].to_numpy(dtype=np.int64)


def select_index_specimens(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Her hasta için tek bir deterministik indeks örnek alt tablosu döndürür."""
    pos = select_index_specimen_positions(meta_df)
    return meta_df.iloc[pos].reset_index(drop=True)


def load_beataml_aml_dataset(raw_dir: Path = DEFAULT_RAW_DIR) -> BeatAMLDataset:
    """Beat AML 2.0 dosyalarını okur, sağlıklı ve NonAML kayıtlarını dışlar."""
    clin = pd.read_excel(raw_dir / "beataml_wv1to4_clinical.xlsx", sheet_name="summary")
    smap = pd.read_excel(raw_dir / "beataml_waves1to4_sample_mapping.xlsx", sheet_name="sample_map")
    probit = pd.read_csv(raw_dir / "beataml_probit_curve_fits_v4_dbgap.txt", sep="\t", low_memory=False)
    muts = pd.read_csv(raw_dir / "beataml_wes_wv1to4_mutations_dbgap.txt", sep="\t", low_memory=False)

    clin["specimen_key"] = make_specimen_key(clin)
    probit["specimen_key"] = make_specimen_key(probit)

    # Tüm klinik satırlardan (942) dalga sızıntı kümesi — temiz Waves3+4 filtresi için
    w12_all_subjects = set(
        int(x) for x in clin.loc[clin["cohort"] == "Waves1+2", "dbgap_subject_id"].dropna().unique()
    )
    both_all_subjects = set(
        int(x) for x in clin.loc[clin["cohort"] == "Both", "dbgap_subject_id"].dropna().unique()
    )

    # Sağlıklı kontrolleri dışla
    is_ctrl = smap["rna_control"].notna() & (smap["rna_control"] != "No")
    ctrl_rna_ids = set(smap.loc[is_ctrl, "dbgap_rnaseq_sample"].dropna().astype(str))

    exp_path = raw_dir / "beataml_waves1to4_norm_exp_dbgap.txt"
    exp_sample_set = set(read_expression_sample_ids(exp_path)) - ctrl_rna_ids
    probit_keys = set(probit["specimen_key"].unique())

    has_drug = clin["specimen_key"].isin(probit_keys)
    has_rna = clin["dbgap_rnaseq_sample"].isin(exp_sample_set)
    is_aml = clin["ELN2017"].fillna("") != "NonAML"

    meta = clin.loc[has_drug & has_rna & is_aml].copy().reset_index(drop=True)
    meta["dbgap_subject_id"] = meta["dbgap_subject_id"].astype(int)

    # Hedef ilaçlar önce, sonra alfabetik tüm 166 ilaç
    all_drugs = sorted(probit["inhibitor"].unique().tolist())
    assert set(CLINICAL_CANDIDATE_DRUGS).issubset(set(all_drugs)), "Eksik aday ilaç var."

    probit_sub = probit[probit["specimen_key"].isin(set(meta["specimen_key"]))]
    pivot = probit_sub.pivot_table(
        index="specimen_key",
        columns="inhibitor",
        values="auc",
        aggfunc="median",
    )
    pivot = pivot.reindex(index=meta["specimen_key"], columns=all_drugs)
    Y_auc = pivot.to_numpy(dtype=np.float64)

    # Yalnızca eşleşmiş hasta RNA sütunlarını streaming/usecols ile oku
    rna_ids = meta["dbgap_rnaseq_sample"].astype(str).tolist()
    unique_rna_cols = sorted(set(rna_ids))
    exp_df = pd.read_csv(exp_path, sep="\t", usecols=["stable_id", "display_label"] + unique_rna_cols)
    gene_names = exp_df["stable_id"].astype(str).tolist()
    exp_vals = exp_df.set_index("stable_id")[rna_ids].to_numpy(dtype=np.float32).T
    assert exp_vals.shape == (len(meta), len(gene_names))

    # 17 ham mutasyon özelliği: FLT3-ITD, NPM1 (klinik kürasyon), 14 WES sürücü geni + has_wes
    wes_dna_set = set(muts["dbgap_sample_id"].dropna().astype(str).unique())
    mut_hits_by_gene: dict[str, set[str]] = {
        g: set(muts.loc[muts["symbol"] == g, "dbgap_sample_id"].astype(str).unique())
        for g in WES_DRIVER_GENES
    }
    dna_series = meta["dbgap_dnaseq_sample"].fillna("").astype(str)
    mut_cols: list[np.ndarray] = [
        (meta["FLT3-ITD"] == "positive").to_numpy(dtype=np.float32),
        (meta["NPM1"] == "positive").to_numpy(dtype=np.float32),
    ]
    mut_names = ["clin_FLT3_ITD", "clin_NPM1"]
    for g in WES_DRIVER_GENES:
        hits = mut_hits_by_gene[g]
        mut_cols.append(dna_series.isin(hits).to_numpy(dtype=np.float32))
        mut_names.append(f"wes_{g}")
    # DNA WES kapsam göstergesi (klinik DNA kimliği veya WES çağrısı varsa 1)
    has_dna = (meta["dbgap_dnaseq_sample"].notna() | dna_series.isin(wes_dna_set)).to_numpy(dtype=np.float32)
    mut_cols.append(has_dna)
    mut_names.append("has_wes")
    X_mut = np.column_stack(mut_cols).astype(np.float32)

    return BeatAMLDataset(
        meta=meta,
        Y_auc=Y_auc,
        X_rna=exp_vals,
        X_mut=X_mut,
        drug_names=all_drugs,
        gene_names=gene_names,
        mut_feature_names=mut_names,
        w12_all_subjects=w12_all_subjects,
        both_all_subjects=both_all_subjects,
    )
