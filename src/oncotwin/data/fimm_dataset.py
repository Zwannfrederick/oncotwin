"""FIMM / Malani (Zenodo 7370747) katalog/doz/QC denetimi ve hasta-gruplu veri yükleyici."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.oncotwin.data.audit_beataml import AML_DRIVER_GENES, CLINICAL_CANDIDATE_DRUGS
from src.oncotwin.data.dataset import load_beataml_aml_dataset, select_index_specimen_positions
from src.oncotwin.eval.splits import select_train_probe_pool

DEFAULT_FIMM_EXTRACTED = Path(
    "data/raw/fimm/extracted/Functional_Precision_Medicine_Tumor_Board_AML"
)
DEFAULT_BEATAML_DIR = Path("data/raw/beataml2")
DEFAULT_CATALOG_AUDIT_JSON = Path("data/processed/sprint05_fimm_catalog_audit.json")

# Belgelenmiş birebir eşanlamlı eşlemeleri (Sol §7 ile bağımsız doğrulandı)
EXPLICIT_BEATAML_TO_FIMM_ID: dict[str, str] = {
    "Azacytidine": "FIMM023825",
}

FIMM_STAGE_PRIORITY: dict[str, int] = {
    "Diagnosis": 0,
    "Relapse": 1,
    "Refractory": 2,
}
FIMM_TISSUE_PRIORITY: dict[str, int] = {
    "Bone marrow": 0,
    "Blood": 1,
}


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def match_beataml_compounds_to_fimm(
    beataml_drug_names: list[str],
    fimm_dir: Path = DEFAULT_FIMM_EXTRACTED,
    beataml_dir: Path = DEFAULT_BEATAML_DIR,
) -> dict[str, Any]:
    """Beat AML bileşiklerini FIMM File_2/File_3.1 kimlik, eşanlamlı ve doz aralığıyla eşler."""
    f2 = pd.read_excel(fimm_dir / "File_2_Drug_library_515D.xlsx")
    f3 = pd.read_excel(fimm_dir / "File_3.1_Drug_response_sDSS_164S_17Healthy.xlsx", sheet_name="sDSS")
    fimm_meta = f3[["Drug_ID", "Drug_name"]].merge(f2, left_on="Drug_ID", right_on="FIMM batch ID", how="left")

    probit = pd.read_csv(
        beataml_dir / "beataml_probit_curve_fits_v4_dbgap.txt",
        sep="\t",
        usecols=["inhibitor", "min_conc", "max_conc"],
    )
    ba_conc = probit.groupby("inhibitor")[["min_conc", "max_conc"]].median()

    token_to_ids: dict[str, set[str]] = {}
    for _, row in fimm_meta.iterrows():
        did = str(row["Drug_ID"])
        fields = [
            row["Drug_name"],
            row["Preferred name"],
            row["Non-proprietary name"],
            row["Res. code"],
            row["Trade names"],
        ]
        if pd.notna(row["Alias"]):
            fields.extend([x.strip() for x in re.split(r"[,;/]", str(row["Alias"]))])
        for val in fields:
            if pd.notna(val) and str(val).strip():
                tok = _norm_token(str(val))
                if tok:
                    token_to_ids.setdefault(tok, set()).add(did)

    fimm_by_id = fimm_meta.set_index("Drug_ID")
    records: list[dict[str, Any]] = []
    ba_to_fimm: dict[str, str] = {}
    ambiguous: list[dict[str, Any]] = []

    for ba_name in beataml_drug_names:
        if ba_name in EXPLICIT_BEATAML_TO_FIMM_ID:
            did = EXPLICIT_BEATAML_TO_FIMM_ID[ba_name]
            how = "explicit_synonym"
        else:
            parts = [ba_name]
            m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", ba_name)
            if m:
                parts.append(m.group(1).strip())
                parts.extend([x.strip() for x in re.split(r"[,;/]", m.group(2))])
            cands: set[str] = set()
            for p in parts:
                tok = _norm_token(p)
                if tok in token_to_ids:
                    cands |= token_to_ids[tok]
            if len(cands) == 1:
                did = next(iter(cands))
                how = "exact_or_alias_norm"
            elif len(cands) > 1:
                ambiguous.append({"beataml_name": ba_name, "candidate_fimm_ids": sorted(cands)})
                continue
            else:
                continue

        ba_to_fimm[ba_name] = did
        row = fimm_by_id.loc[did]
        f_lo = float(row["Low conc (nM)"]) if pd.notna(row["Low conc (nM)"]) else None
        f_hi = float(row["High conc (nM)"]) if pd.notna(row["High conc (nM)"]) else None
        b_lo = float(ba_conc.loc[ba_name, "min_conc"] * 1000.0) if ba_name in ba_conc.index else None
        b_hi = float(ba_conc.loc[ba_name, "max_conc"] * 1000.0) if ba_name in ba_conc.index else None
        hi_ratio = (f_hi / b_hi) if (f_hi and b_hi and b_hi > 0) else None
        records.append({
            "beataml_name": ba_name,
            "is_target": ba_name in set(CLINICAL_CANDIDATE_DRUGS),
            "fimm_drug_id": did,
            "fimm_drug_name": str(row["Drug_name"]),
            "match_rule": how,
            "mechanism_targets": str(row["Mechanism/Targets"]) if pd.notna(row["Mechanism/Targets"]) else None,
            "drug_class": str(row["Drug_Class"]) if pd.notna(row["Drug_Class"]) else None,
            "solvent": str(row["Solvent"]) if pd.notna(row["Solvent"]) else None,
            "fimm_low_nM": f_lo,
            "fimm_high_nM": f_hi,
            "beataml_median_low_nM": round(b_lo, 2) if b_lo is not None else None,
            "beataml_median_high_nM": round(b_hi, 2) if b_hi is not None else None,
            "high_conc_ratio_fimm_over_beataml": round(hi_ratio, 4) if hi_ratio is not None else None,
        })

    return {
        "ba_to_fimm_id": ba_to_fimm,
        "records": records,
        "ambiguous": ambiguous,
        "unmatched_targets": [d for d in CLINICAL_CANDIDATE_DRUGS if d not in ba_to_fimm],
    }


@dataclass(frozen=True)
class FIMMDataset:
    """Eşleşmiş FIMM/Malani AML örnekleri için ham sDSS, yön-uyumlu -sDSS ve omik kapsayıcısı."""

    meta: pd.DataFrame
    Y_sdss: np.ndarray
    Y_aligned: np.ndarray  # -Y_sdss: düşük = daha duyarlı (Beat AML AUC yönüyle uyumlu)
    X_rna: np.ndarray
    X_mut: np.ndarray
    drug_names: list[str]
    fimm_drug_ids: list[str]
    gene_ensembl_ids: list[str]
    mut_feature_names: list[str]

    def subset_by_positions(self, positions: np.ndarray) -> FIMMDataset:
        pos = np.asarray(positions, dtype=np.int64)
        return FIMMDataset(
            meta=self.meta.iloc[pos].reset_index(drop=True),
            Y_sdss=self.Y_sdss[pos],
            Y_aligned=self.Y_aligned[pos],
            X_rna=self.X_rna[pos],
            X_mut=self.X_mut[pos],
            drug_names=self.drug_names,
            fimm_drug_ids=self.fimm_drug_ids,
            gene_ensembl_ids=self.gene_ensembl_ids,
            mut_feature_names=self.mut_feature_names,
        )


def within_sample_rank_normalize(X_rna: np.ndarray) -> np.ndarray:
    """Her hastanın kendi gen profilini [0, 1] yüzdelik sırasına dönüştürür (sıfır kohort sızıntısı)."""
    X = np.asarray(X_rna, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)
    denom = float(max(1, X.shape[1] - 1))
    for i in range(X.shape[0]):
        order = np.argsort(X[i], kind="mergesort")
        ranks = np.empty(X.shape[1], dtype=np.float64)
        ranks[order] = np.arange(X.shape[1], dtype=np.float64) / denom
        out[i] = ranks
    return out


def select_fimm_index_specimen_positions(meta: pd.DataFrame) -> np.ndarray:
    """Her FIMM hastasından deterministik tek indeks örneği seçer (132 örnek -> 109 hasta)."""
    work = meta[["Patient_ID", "Sample_ID", "Disease status", "Tissue", "n_obs_matched_drugs"]].copy()
    work["pos"] = np.arange(len(work), dtype=np.int64)
    work["stage_pri"] = work["Disease status"].map(FIMM_STAGE_PRIORITY).fillna(99).astype(int)
    work["tissue_pri"] = work["Tissue"].map(FIMM_TISSUE_PRIORITY).fillna(99).astype(int)
    work = work.sort_values(
        by=["Patient_ID", "stage_pri", "tissue_pri", "n_obs_matched_drugs", "Sample_ID"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    chosen = work.drop_duplicates(subset=["Patient_ID"], keep="first")["pos"].to_numpy(dtype=np.int64)
    return np.sort(chosen)


def load_fimm_aml_dataset(
    beataml_drug_names: list[str],
    common_gene_ensembl_ids: list[str] | None = None,
    fimm_dir: Path = DEFAULT_FIMM_EXTRACTED,
    beataml_dir: Path = DEFAULT_BEATAML_DIR,
) -> tuple[FIMMDataset, dict[str, Any]]:
    """FIMM 132 eşleşmiş AML örneğini (109 hasta) Beat AML ortak bileşik/gen uzayında yükler."""
    f0 = pd.read_excel(fimm_dir / "File_0_Common_sample_annotation_252S.xlsx")
    f0["Sample_ID"] = f0["Sample_ID"].astype(str).str.strip()
    f0["Patient_ID"] = f0["Patient_ID"].astype(str).str.strip()
    f0["Disease status"] = f0["Disease status"].astype(str).str.strip()
    f0["Tissue"] = f0["Tissue"].astype(str).str.strip()

    f3 = pd.read_excel(fimm_dir / "File_3.1_Drug_response_sDSS_164S_17Healthy.xlsx", sheet_name="sDSS")
    f4 = pd.read_excel(fimm_dir / "File_4_DSRT_assay_details_164S_DM.xlsx")
    f4["Sample_ID"] = f4["Sample_ID"].astype(str).str.strip()
    f6 = pd.read_excel(fimm_dir / "File_6_Binary_mutation_225S_57G.xlsx")
    f7 = pd.read_csv(fimm_dir / "File_7_RNA_seq_CPM_163S_4Healthy.csv")
    f9 = pd.read_csv(fimm_dir / "File_9_RNA-seq_sample_annotation_167S.csv")

    f7_gene_col = f7.columns[0]
    ens_ids = f7[f7_gene_col].astype(str).str.split(".").str[0]
    keep_genes = ~ens_ids.duplicated(keep="first")
    f7_indexed = f7.loc[keep_genes].copy()
    f7_indexed.index = ens_ids.loc[keep_genes].to_numpy()

    aml_samples = set(f0["Sample_ID"])
    sdss_samples = set(f3.columns[2:])
    rna_samples = set(f7.columns[1:])
    matched_samples = sorted(aml_samples & sdss_samples & rna_samples)

    comp_match = match_beataml_compounds_to_fimm(beataml_drug_names, fimm_dir=fimm_dir, beataml_dir=beataml_dir)
    ba_to_fimm = comp_match["ba_to_fimm_id"]
    matched_ba_names = [d for d in beataml_drug_names if d in ba_to_fimm]
    matched_fimm_ids = [ba_to_fimm[d] for d in matched_ba_names]

    f3_by_id = f3.set_index("Drug_ID")
    Y_sdss = f3_by_id.loc[matched_fimm_ids, matched_samples].to_numpy(dtype=np.float64).T
    Y_aligned = -Y_sdss  # Yüksek sDSS = daha duyarlı -> negatif işaret ile düşük = daha duyarlı

    if common_gene_ensembl_ids is None:
        genes_use = sorted(f7_indexed.index.tolist())
    else:
        genes_use = [g for g in common_gene_ensembl_ids if g in f7_indexed.index]
    X_rna = f7_indexed.loc[genes_use, matched_samples].to_numpy(dtype=np.float64).T

    # Binary mutasyonlar (Beat AML 16 sürücü gen adlarıyla örtüşenler; eksik örnekler 0.0 + has_mut bayrağı)
    mut_gene_col = f6.columns[0]
    f6_indexed = f6.set_index(mut_gene_col)
    mut_genes = [g for g in AML_DRIVER_GENES if g in f6_indexed.index]
    X_mut = np.zeros((len(matched_samples), len(mut_genes)), dtype=np.float64)
    mut_cols = set(f6_indexed.columns)
    for r_i, sid in enumerate(matched_samples):
        if sid in mut_cols:
            vals = pd.to_numeric(f6_indexed.loc[mut_genes, sid], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            X_mut[r_i] = (vals > 0.0).astype(np.float64)

    meta = (
        f0[f0["Sample_ID"].isin(matched_samples)]
        .drop_duplicates(subset=["Sample_ID"])
        .set_index("Sample_ID")
        .loc[matched_samples]
        .reset_index()
    )
    meta = meta.merge(
        f4[["Sample_ID", "Blast_cell_percentage", "Medium", "% cell viability", "Serial_samples"]],
        on="Sample_ID",
        how="left",
    ).merge(f9[["Sample_ID", "Library", "Extraction_method"]], on="Sample_ID", how="left")
    meta["has_mutation_panel"] = meta["Sample_ID"].isin(mut_cols)
    meta["n_obs_matched_drugs"] = np.sum(np.isfinite(Y_sdss), axis=1).astype(int)
    # Patient_ID -> deterministik tamsayı hasta kimliği (patient-disjoint CV uyumluluğu)
    pat_codes, _ = pd.factorize(meta["Patient_ID"], sort=True)
    meta["patient_int_id"] = pat_codes.astype(np.int64)
    meta["cohort"] = meta["Disease status"].fillna("Unknown").astype(str)

    ds = FIMMDataset(
        meta=meta,
        Y_sdss=Y_sdss,
        Y_aligned=Y_aligned,
        X_rna=X_rna,
        X_mut=X_mut,
        drug_names=matched_ba_names,
        fimm_drug_ids=matched_fimm_ids,
        gene_ensembl_ids=genes_use,
        mut_feature_names=mut_genes,
    )
    return ds, comp_match


def run_fimm_catalog_audit(
    out_json: Path = DEFAULT_CATALOG_AUDIT_JSON,
    fimm_dir: Path = DEFAULT_FIMM_EXTRACTED,
    beataml_dir: Path = DEFAULT_BEATAML_DIR,
) -> dict[str, Any]:
    """Model fit etmeden FIMM/Beat AML katalog, kimyasal kimlik, doz ve QC denetimini kaydeder."""
    ds_ba_all = load_beataml_aml_dataset(beataml_dir)
    ds_ba = ds_ba_all.subset_by_positions(select_index_specimen_positions(ds_ba_all.meta))
    ba_genes_df = pd.read_csv(beataml_dir / "beataml_waves1to4_norm_exp_dbgap.txt", sep="\t", usecols=[0, 1])
    ba_ens_list = [g.split(".")[0] for g in ba_genes_df.iloc[:, 0].astype(str).tolist()]

    fimm_ds, comp_match = load_fimm_aml_dataset(
        ds_ba.drug_names, common_gene_ensembl_ids=ba_ens_list, fimm_dir=fimm_dir, beataml_dir=beataml_dir
    )
    idx_pos = select_fimm_index_specimen_positions(fimm_ds.meta)
    fimm_idx = fimm_ds.subset_by_positions(idx_pos)

    full_ba_probes = select_train_probe_pool(ds_ba.Y_auc, ds_ba.drug_names, CLINICAL_CANDIDATE_DRUGS, min_train_obs=100)
    full_ba_probe_names = [ds_ba.drug_names[i] for i in full_ba_probes]
    matched_set = set(fimm_ds.drug_names)
    common_targets = [d for d in CLINICAL_CANDIDATE_DRUGS if d in matched_set]
    common_probes = [d for d in full_ba_probe_names if d in matched_set]

    f0 = pd.read_excel(fimm_dir / "File_0_Common_sample_annotation_252S.xlsx")
    f3 = pd.read_excel(fimm_dir / "File_3.1_Drug_response_sDSS_164S_17Healthy.xlsx", sheet_name="sDSS")
    f7_cols = pd.read_csv(fimm_dir / "File_7_RNA_seq_CPM_163S_4Healthy.csv", nrows=0).columns.tolist()

    target_recs = [r for r in comp_match["records"] if r["is_target"]]
    conc_exact = sum(1 for r in target_recs if r["high_conc_ratio_fimm_over_beataml"] == 1.0)
    conc_lower = sum(1 for r in target_recs if (r["high_conc_ratio_fimm_over_beataml"] or 1.0) < 1.0)
    conc_higher = sum(1 for r in target_recs if (r["high_conc_ratio_fimm_over_beataml"] or 1.0) > 1.0)

    audit = {
        "source": "Malani et al. Cancer Discov 2022 (PMID 34789538) / Zenodo 7370747 (MD5 3db5280e20f315d56d6bf07dd6a9e241)",
        "sample_counts": {
            "file0_total_aml_specimens": int(len(f0)),
            "file0_unique_aml_patients": int(f0["Patient_ID"].nunique()),
            "file3_sdss_columns_total": int(len(f3.columns) - 2),
            "file3_sdss_compounds": int(len(f3)),
            "file7_rna_columns_total": int(len(f7_cols) - 1),
            "file7_rna_note": "163 AML + 4 sağlıklı CD34+ (Opus §3.9 düzeltmesi doğrulandı)",
            "matched_aml_rna_sdss_specimens": int(len(fimm_ds.meta)),
            "matched_aml_rna_sdss_unique_patients": int(fimm_ds.meta["Patient_ID"].nunique()),
            "matched_index_specimens_selected": int(len(fimm_idx.meta)),
            "matched_with_mutation_panel_specimens": int(fimm_ds.meta["has_mutation_panel"].sum()),
        },
        "assay_qc_comparison": {
            "beataml_assay": "72h MTS kolorimetrik canlılık; 7-nokta (~3x seyreltme), probit AUC (düşük = daha duyarlı)",
            "fimm_assay": "72h CellTiter-Glo (CTG) ATP lüminesans; 5-nokta (10x seyreltme), sağlıklı-kontrollü sDSS (yüksek = daha duyarlı)",
            "fimm_matched_medium_counts": {str(k): int(v) for k, v in fimm_ds.meta["Medium"].value_counts(dropna=False).items()},
            "fimm_matched_rna_library_counts": {str(k): int(v) for k, v in fimm_ds.meta["Library"].value_counts(dropna=False).items()},
            "fimm_matched_disease_status": {str(k): int(v) for k, v in fimm_ds.meta["Disease status"].value_counts(dropna=False).items()},
            "fimm_matched_tissue": {str(k): int(v) for k, v in fimm_ds.meta["Tissue"].value_counts(dropna=False).items()},
            "direction_alignment_rule": "y_fimm_aligned = -sDSS (düşük değer = daha yüksek duyarlılık; Beat AML AUC yönüyle eşlendi)",
        },
        "gene_overlap": {
            "beataml_ensembl_genes": int(len(set(ba_ens_list))),
            "common_ensembl_genes": int(len(fimm_ds.gene_ensembl_ids)),
            "common_mutation_driver_genes": fimm_ds.mut_feature_names,
        },
        "compound_overlap": {
            "beataml_total_drugs": int(len(ds_ba.drug_names)),
            "matched_unique_compounds": int(len(fimm_ds.drug_names)),
            "ambiguous_compounds": comp_match["ambiguous"],
            "common_targets_count": int(len(common_targets)),
            "common_targets": common_targets,
            "unmatched_targets": comp_match["unmatched_targets"],
            "common_probes_from_all471_pool_count": int(len(common_probes)),
            "target_high_conc_comparison": {
                "exact_match_1x": conc_exact,
                "fimm_lower_max_conc": conc_lower,
                "fimm_higher_max_conc": conc_higher,
            },
            "target_details": target_recs,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="FIMM/Malani ön-modelleme katalog, doz ve QC denetimi.")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_CATALOG_AUDIT_JSON)
    args = parser.parse_args()
    audit = run_fimm_catalog_audit(out_json=args.out_json)
    sc = audit["sample_counts"]
    co = audit["compound_overlap"]
    print(
        f"FIMM katalog denetimi tamamlandı -> {args.out_json}: "
        f"{sc['matched_aml_rna_sdss_specimens']} örnek / {sc['matched_aml_rna_sdss_unique_patients']} hasta, "
        f"{co['matched_unique_compounds']}/{co['beataml_total_drugs']} ilaç, "
        f"{co['common_targets_count']}/24 hedef, {audit['gene_overlap']['common_ensembl_genes']} ortak gen."
    )


if __name__ == "__main__":
    main()
