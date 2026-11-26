# Empirical Evaluation of Budget-Constrained Ex Vivo Drug Panel Selection and Patient-Level Conformal Shortlists in Primary Acute Myeloid Leukemia

**Author:** Muhammed Emin Çelik¹*  
¹ Independent Researcher · ORCID: https://orcid.org/0009-0003-9804-0640  
\* Correspondence / Code: https://github.com/Zwannfrederick/oncotwin  
**Article Type:** New Results (Empirical Computational Study — Research Use Only)  
**License (Manuscript):** CC BY-NC 4.0 · **License (Code):** PolyForm Noncommercial 1.0.0  

> **Scope and Non-Clinical Notice:** All results in this study come from retrospective *ex vivo* 72-hour drug sensitivity archives. Nothing in this manuscript supports clinical diagnosis, treatment selection, or patient care.

---

## Abstract

Testing 100+ inhibitors *ex vivo* on primary acute myeloid leukemia (AML) cells is expensive and often impossible when cell counts are low, yet predicting drug sensitivity from baseline RNA-seq and mutations alone leaves substantial error. Here we report pre-registered empirical results testing a simpler question: if a laboratory measures only a small fixed batch of *k* probe drugs (focusing on *k* = 5) on a patient's cells, how much does that improve ranking of the remaining clinically relevant AML drugs, how small can a calibrated candidate shortlist get, and what fails under distribution shift or cross-cohort transfer?

Across Beat AML 2.0 (471 unique index AML patients, 24 target drugs, patient-disjoint cross-validation and temporal `Waves 1+2 → Waves 3+4` holdout) and an independent Finnish AML cohort (FIMM/Malani, Zenodo 7370747; 132 specimens / 109 unique patients, 23 shared targets, 64 shared probes), five probes selected by training-only static D-optimal design (`StaticDOpt`) with a conditional Gaussian residual update (`M3`) consistently beat baseline omics (`M1`) and equal-budget random panels. On the Beat AML 60/20/20 holdout (*n* = 91), `StaticDOpt + M3` (*k* = 5) reduced target RMSE by −7.29 AUC [95% CI −9.45, −5.42] versus `M1`; in 5-fold cross-validation (*n* = 427), `StaticDOpt` beat 50× random panels by −2.37 AUC [−2.84, −1.89]. Patient-level rank calibration (`RankConformal`, nominal 90%) shrank uninformative equal-width AUC interval shortlists from 18.70 drugs (98.7% of observed targets) to 6.00 drugs while retaining the true best observed drug in 85/91 (93.4%) in-distribution patients and 137/148 (92.6%) temporal `Waves 3+4` patients. When frozen Beat AML models were transferred without refitting or cohort statistics to FIMM (`C1`, *n* = 109), five probes improved top-3 enrichment (`ΔNDCG@3` = +0.032 [+0.007, +0.058] vs `M1`; +0.033 [+0.013, +0.054] vs random) but did not significantly lower `Regret@1` (−0.23 sDSS [−1.26, +0.76]). Within-FIMM 5-fold cross-validation (`C2`, *n* = 109) replicated Beat AML gains (`ΔRMSE` = −1.00 sDSS [−1.26, −0.76] vs `M1`; −0.46 vs random).

We also report four negative findings: (1) adding a curated drug-family covariance prior (`M4`) gave zero gain over unstructured empirical shrinkage (`M3`: +0.011 AUC [−0.010, +0.033] in Beat AML; 0.00 sDSS in FIMM) and failed a 20-draw degree-preserving null control, triggering a pre-registered `NO-GO` against mechanistic ODE expansion; (2) patient-adaptive batch selection did not beat static D-optimal selection after symmetric Holm correction (`INSUFFICIENT_EVIDENCE`); (3) omics-distance out-of-distribution abstention failed on `Waves 3+4` (`AURC` 32.44 vs random 24.60; `AUGRC` 14.27 vs 12.30); and (4) adaptive margin shortlists (`Top1Gap`) yielded true-top-1 accuracy of only 5/9 (in-distribution) and 19/25 (`Waves 3+4`) in the singleton (`|S| = 1`) subgroup despite >86% marginal coverage. All code, frozen protocols, and verification hashes run on CPU with three runtime dependencies.

---

## 1. Introduction

Primary *ex vivo* drug sensitivity testing in acute myeloid leukemia (AML) measures how viable patient blast cells remain after short-term exposure to targeted and cytotoxic compounds, as demonstrated in Beat AML 2.0 (Bottomly et al., 2022) and the Finnish Functional Precision Medicine Tumor Board (FIMM/Malani et al., 2022). In practice, testing >100 drugs per specimen is costly and constrained by sample volume. Conversely, predicting drug response solely from baseline RNA sequencing and somatic mutations remains noisy, suffers from cross-cohort batch shifts, and is often overstated when benchmarks fail to control for drug-mean effects or repeated patient specimens (Bernett et al., 2026; Willoch-Allen et al., 2026).

This study started from a broader question about building mechanistic "digital twin" models for personalized cancer therapy, then deliberately narrowed its scope based on published identifiability and assay limits. Large kinetic ODE networks cannot be uniquely identified from static baseline omics and a handful of scalar viability measurements (Raue et al., 2009), and cell-line or PDX models transfer poorly to primary human samples (Guillen et al., 2022; Bernett et al., 2026). We therefore focused on a narrower, testable question in primary AML:

1. **Few-shot signal:** When a laboratory measures a single batch of *k* probe drugs on a patient's cells, how much does predicting the remaining clinically relevant AML inhibitors improve over drug-mean (`M0`) and baseline omics (`M1`) models?
2. **Panel design:** Does a training-selected batch panel (`StaticDOpt`, `StaticDiversity`, or `AdaptiveOED`) outperform equal-budget random panels, and does patient-adaptive selection justify its extra complexity?
3. **Pathway structure (`Gate 3`):** Does shrinking residual drug covariance toward curated drug-family annotations (`M4`) beat unstructured empirical shrinkage (`M3`) and a degree-preserving permuted family graph?
4. **Uncertainty and shortlists:** Can patient-level conformal calibration produce compact candidate shortlists that still contain the patient's best observed drug, and does out-of-distribution (OOD) abstention work across enrollment waves?
5. **External transfer:** How much of the ranking signal transfers frozen (without refitting or target-cohort statistics) from Beat AML (MTS assay, AUC) to FIMM (CTG assay, sDSS), versus refitting the same method within FIMM?

We do not claim a new statistical algorithm. Every model here uses standard regularized linear algebra, greedy D-optimal selection, and split-conformal order statistics. Our contribution is a leakage-controlled, pre-registered empirical evaluation across two primary AML cohorts—including explicit documentation of what failed.

---

## 2. Methods

### 2.1 Cohorts, Specimen Deduplication, and Target/Probe Disjointness

- **Beat AML 2.0 (Bottomly et al., 2022; CC BY 4.0):** Seven harmonized files were downloaded with SHA-256 verification from the official Beat AML 2.0 repository. Auditing revealed 942 clinical specimens across 805 unique patients (`dbgap_subject_id`), including 108 patients with multiple specimens (up to 5) and 18 patients spanning `Waves 1+2` and `Waves 3+4`. All splits were strictly patient-disjoint. Among 520 drug-plus-RNA specimens (479 unique patients), we retained one deterministic index specimen per patient (selecting the specimen with the most observed target drugs, breaking ties by total drug count and specimen key), yielding **471 unique index AML patients**, 166 total inhibitors, and **24 pre-specified clinically relevant AML target drugs** (lower probit AUC = higher sensitivity).
- **FIMM / Malani AML (Malani et al., 2022; Zenodo 7370747, CC BY 4.0):** The archive (`38,810,683` bytes, MD5 `3db5280e20f315d56d6bf07dd6a9e241`) contains 252 specimens from 186 AML patients (`File_0`), 515 drugs × 181 columns in `File_3.1` (164 AML + 17 healthy), and 167 RNA log2-CPM columns in `File_7` (163 AML + 4 healthy CD34+ controls). Intersecting AML RNA and sDSS yielded **132 specimens from 109 unique patients** (92 single-specimen, 17 multi-specimen), 14,688 shared Ensembl genes, and 15 shared driver mutation columns (`KIT` absent in FIMM `File_6`). Exact 1-to-1 compound matching covered 102/166 Beat AML drugs, **23/24 targets** (`Gilteritinib` absent), and **64 shared probes** from Beat AML's training probe pool. In 11 of 23 shared targets, maximum tested concentrations differed between cohorts (0.01× to 10×; e.g., Trametinib 40× lower and Rapamycin 100× lower in FIMM). To align direction with Beat AML AUC (lower = more sensitive), FIMM scores were negated (`y_aligned = -sDSS`).
- **Strict probe/target separation:** In every experiment, the candidate probe pool was chosen strictly inside the training fold (`min_train_obs = 100` in Beat AML; `50` in FIMM CV) and was strictly disjoint from the target drug set. Probes were never scored in evaluation metrics.

### 2.2 Pre-Registered Splits and Leakage Prevention

Protocols were frozen in `docs/PROTOCOL*.md` before each sprint's experiments were run:
1. **Beat AML ID 5-Fold CV × 3 seeds** (`n_eval = 448` in Sprint 2; `427` in Sprints 3 and 5 under `min_targets = 5, min_probes = 12`).
2. **Beat AML temporal OOD (`Waves 1+2 → Waves 3+4`):** All patients appearing in `Waves 1+2` or `Both` were excluded from the `Waves 3+4` test set (`n_eval = 148` in Sprint 2; `146` in Sprint 3/5 A1; `148` in Sprint 4/5 B). Because `Waves 3+4` was inspected in Sprints 2–3, it is labeled an **exploratory OOD stress test** in Sprints 4–5 and was never used for model or threshold selection.
3. **Beat AML 60/20/20 split (Sprints 4–5):** Patient-disjoint train/tune (`284`), calibration (`94`; `90` archive-eligible), and test (`93`; `91` archive-eligible) for ID; and `213` train/tune, `71` calibration (`70` archive-eligible; `SOL-14`), and `165` test (`148` archive-eligible) for `Waves 3+4`.
4. **Leakage controls:** Every gene filter (top 2,000 high-variance genes), z-score scaling, ridge hyperparameter, covariance matrix, and panel ranking was fit exclusively inside training folds. Adversarial tests confirmed that permuting or adding ±500 to hidden test target responses left probe selections and target predictions bit-for-bit unchanged.

### 2.3 Models, Batch Panel Selection, and Pathway Control (`M4`)

- **`M0` (Fair few-shot drug/patient mean):** `y_hat_{p,d} = mu + b_d + b_p^(S_p)` with shrunk patient offset from observed probes `S_p` (`k = 0` uses `mu + b_d` only).
- **`M1` (Baseline omics ridge):** Dual-space L2-regularized regression on top-2,000 training RNA features plus binary mutation/clinical indicators, predicting drug residuals over `mu + b_d` with inner patient-disjoint CV.
- **`M2a` / `M2b` (Omics-free few-shot):** Conditional Gaussian shrinkage (`M2a`) and low-rank SVD (`M2b`) on drug-mean residuals.
- **`M3` (Hybrid omics prior + conditional Gaussian update):** Uses `M1` out-of-fold (OOF) predictions on training patients to form residual matrix `R = Y - Y_hat_M1`, estimates a positive-semidefinite shrunk covariance `Sigma_hat = (1 - alpha) * S_psd + alpha * diag(S_psd)`, and updates unseen target drugs for a new patient from `k` observed probe residuals `r_S` via `delta_T = Sigma_{TS} (Sigma_{SS} + lam * I)^{-1} r_S`.
- **Batch panel selectors (single round, fixed budget *k*):** `Random` (50× Beat AML / 20× FIMM), `StaticDiversity` (greedy max-min correlation distance), `StaticDOpt` (greedy log-det / conditional target variance reduction using training `Sigma_hat`), `EqualWeightVarRed` (Sprint 4 control), and `AdaptiveOED` (patient-specific expected top-contender weights via locked per-fold Monte Carlo noise bank, `SOL-09`). In `archive_available` evaluation, the ranked panel selects the patient's first *k* observed probes; in Sprint 4 `locked_train_catalog`, missing prefix probes trigger technical abstention without backfill.
- **`M4` (Pathway/drug-family structured covariance & Gate 3):** Using `beataml_drug_families.xlsx` (`drug_family` sheet: 117/166 annotated drugs, 86 families, 395 edges; 15/24 targets annotated), we built a unit-diagonal cosine similarity Gram matrix `G` and replaced diagonal shrinkage with `T_bio = diag(sd) * ((1 - rho) * I + rho * G) * diag(sd)` (`rho in {0, 0.15, 0.30, 0.50, 0.70}`). When `rho = 0`, `M4` equals `M3` bit-for-bit. As a biological specificity control (`M4_perm_null`), we averaged 20 independent bipartite edge-swap permutations of the drug-family matrix that preserved every drug's exact degree and every family's exact size. Primary **Experiment A1** held patients, targets, and the `StaticDOpt(M3)` panel fixed; secondary **Experiment A2** also reselected the panel with `M4`. Pre-registered **Gate 3 (`AD-001`)** required `ΔRMSE <= -0.50` AUC (95% CI `< 0`, `p < 0.05`, no ranking degradation) against both `M3` and `M4_perm_null` on ID CV plus consistent negative direction on `Waves 3+4`.

### 2.4 Patient-Level Conformal Shortlists and Selective Risk (`AURC` / `AUGRC`)

All uncertainty calibration used the separate calibration split with patients as the exchangeable unit (Lei et al., 2018; Lee et al., 2023) at nominal coverage `1 - alpha = 0.90` (`alpha = 0.10`), using exact finite-sample order statistic `r = ceil((n_cal + 1)(1 - alpha))` (returning `hesaplanamadı`/NaN whenever `r > n_cal`):
- **`EqualWidthAUC` (Sprint 4 baseline):** Calibrates `q_sim` on each calibration patient's maximum absolute residual across observed targets, then retains every observed target whose lower interval overlaps the predicted best drug's upper interval (`y_hat_d - y_hat_(1) <= 2 * q_sim`).
- **`RankConformal` (primary shortlist):** Calibrates integer `k_hat` on the predicted rank of each calibration patient's true best observed drug (conservative worst rank under true ties), outputting the top `k_hat` predicted observed targets (including predicted ties). Scale-invariant.
- **`RelRankConformal`:** Same calibration on relative rank `r_i / n_obs,i`, taking `ceil(q_rel * n_obs)`.
- **`Top1GapConformal` (adaptive margin):** Calibrates `m_hat` on `y_hat_{d*_i} - min_d y_hat_d`, retaining all observed targets within `m_hat` AUC of the predicted best drug. Allows adaptive shortlist sizes including singletons (`|S| = 1`).
- **OOD abstention:** Evaluated omics PCA distance (`omics_pca_dist`) and `M3` conditional posterior standard deviation (`m3_post_scale`) via tie-aware Area Under the Risk–Coverage curve (`AURC`) and Area Under the Generalized Risk–Coverage curve (`AUGRC`, Traub et al., 2024; continuous `Regret@1` loss, where exact random `AUGRC = 0.5 * mean(Regret@1)`).

### 2.5 External FIMM Replication (`C1`, `C2`) and Probe-Scale Control (`OPUS-028`)

- **Experiment C1 (Frozen Beat AML transfer — zero FIMM refitting, zero FIMM cohort statistics):** Inspired by within-sample relative expression (Tercan, 2026), each patient's 14,688-gene RNA profile was rank-normalized independently within that single patient (`within_sample_rank_normalize`). Beat AML responses were row z-scored per training patient across observed shared drugs; `M0`, `M1`, `M2a`, `M3`, and the common-catalog `StaticDOpt` ranking were fit solely on Beat AML (471 patients). On FIMM, each test patient's `k` observed probes were z-scored using only those `k` values (targets and other patients strictly hidden). Beat AML conformal intervals were **not** transferred (`conformal_intervals_transferred = False`). Evaluated on 109 unique index patients (primary) and all 132 specimens clustered into 109 patient means (sensitivity).
- **Experiment C2 (Within-FIMM method replication — strictly separate from C1):** Patient-disjoint 5-fold CV × 3 seeds on the 109 unique FIMM patients directly on `-sDSS`, fitting gene filters, models (`M0`, `M1`, `M2a`, `M3`, `M4`, `M4_perm_null`), and `StaticDOpt` inside each FIMM fold. Note that FIMM `File_7` RNA values were batch-corrected upstream in the Zenodo release (`SOL-16`).
- **Beat AML probe-scale control (`OPUS-028`):** To isolate the effect of z-scoring only `k` probes versus full training rows, we tested four probe-scaling rules on the frozen Beat AML 60/20/20 split (`284` train, `91` evaluable test, 102 shared drugs, 23 targets, 62 shared probes) at `k = 2` and `k = 5`.

### 2.6 Statistical Inference

All metrics (`RMSE`, residual `R²`, within-patient tie-aware `NDCG@3`, `Recall@3`, `Regret@1`, `Regret@3`, and Spearman correlation) were aggregated at the patient level first, and paired differences used patient-clustered bootstrap confidence intervals (1,000 resamples in main runs; 2,000–4,000 in audit checks). Bootstrap intervals condition on fixed OOF/test predictions and do not include outer fold-refit variance.

### 2.7 Ethics, Data/Code Availability, and Human–AI Workflow Disclosure

- **Ethics:** This work analyzed only publicly available, de-identified secondary datasets (Beat AML 2.0 and Zenodo 7370747, both CC BY 4.0). No new human specimens were collected and no clinical interventions were performed.
- **Code and reproducibility:** All code, frozen protocols, unit/contract tests (88 tests passing with `RuntimeWarning` promoted to error), SHA-256 verification manifests (32 archived files), and the one-command verifier (`verify_package.py`) are available at https://github.com/Zwannfrederick/oncotwin under `PolyForm-Noncommercial-1.0.0` (code) and `CC BY-NC 4.0` (documents/figures). Runtime uses CPU only with three Python dependencies (`numpy==2.5.5`, `pandas==3.0.3`, `openpyxl==3.1.5`; verified in a clean virtual environment on Linux x86_64, Python 3.14.10, `88 passed in 15.6 s`, peak RSS 599 MB).
- **Human oversight and AI tool disclosure (above minimum requirements):** Muhammed Emin Çelik is the sole author and accepts complete scientific and ethical responsibility for this work. He defined the research question, enforced the scope reduction from kinetic ODEs/digital twins to *ex vivo* batch panel selection, specified patient-disjoint leakage rules and ablation controls, and personally approved each sprint decision gate (`Gates 0–3`, `Addendum 19`, and `v1` release). Three large language models were used strictly as software engineering and internal checking tools—not as authors: **Gemini 4 Neon** (pipeline/model implementation and packaging), **GPT-6.2 Sol** (protocol/leakage checks, claim-scoping audits, `OPUS-028` script, and drafting the scientific core of `METHODS_REPORT.md`), and **Claude Opus 5.5** (adversarial edge-case testing, memory/runtime profiling, clean-venv independent replay, and refactoring). Numerical matrices were not recomputed by hand; instead, outputs were verified through automated unit/contract tests and independent full-pipeline replays. **Crucially, internal checks performed by AI models are automated error-detection tools and must not be interpreted as independent academic peer review.**

---

## 3. Results

### 3.1 Five Ex Vivo Probes Substantially Outperform Baseline Omics and Random Panels (Beat AML)

In Beat AML 2.0, permutation leakage checks passed (`clean_wave_perm` per-drug Spearman `+0.013 [-0.052, +0.079]`). Adding five *ex vivo* probes via `M3` consistently lowered prediction error and improved within-patient ranking across every Beat AML evaluation split (Table 1; **[Figure 1]**).

**Table 1. Summary of Beat AML 2.0 prediction and panel selection results (`k = 5` reference).**

| Evaluation Regime | Evaluable Patients (*n*) | Arm / Comparison | RMSE (AUC) ↓ | NDCG@3 ↑ | Regret@1 (AUC) ↓ | Paired ΔRMSE [95% CI] |
|---|---:|---|---:|---:|---:|---:|
| Sprint 2 ID 5-Fold CV × 3 | 448 | `M0 (k=0)` | 55.85 | 0.785 | — | — |
| Sprint 2 ID 5-Fold CV × 3 | 448 | `M1 (k=0)` | 49.07 | 0.855 | — | — |
| Sprint 2 ID 5-Fold CV × 3 | 448 | `M3 (k=5, random)` vs `M1` | 44.78 | 0.869 | — | **−4.29 [−4.98, −3.60]** |
| Sprint 3 v2 ID 5-Fold CV × 3 | 427 | `Random50 (k=5)` | 45.11 | 0.871 | 24.0 | — |
| Sprint 3 v2 ID 5-Fold CV × 3 | 427 | `StaticDOpt (k=5)` vs `Random50` | **42.74** | 0.876 | 23.8 | **−2.37 [−2.84, −1.89]** |
| Sprint 3 v2 ID 5-Fold CV × 3 | 427 | `AdaptiveOED (k=5)` vs `StaticDOpt` | 42.98 | 0.881 | 21.6 | +0.24 [−0.33, +0.81] |
| Sprint 3 v2 `Waves 3+4` (OOD) | 146 | `StaticDOpt (k=5)` vs `Random50` | **50.81** | 0.871 | 28.9 | **−1.81 [−2.94, −0.65]** |
| Sprint 4 v2 ID 60/20/20 | 91 | `StaticDOpt (k=5)` vs `M1 (k=0)` | **39.84** | 0.887 | 13.7 | **−7.29 [−9.45, −5.42]** |
| Sprint 4 v2 `Waves 3+4` (OOD) | 148 | `StaticDOpt (k=5)` vs `M1 (k=0)` | **51.99** | 0.873 | 24.6 | **−4.18 [−5.86, −2.66]** |

Two caveats emerged from Sprint 3 audits:
1. **Static vs. Adaptive:** Across nine symmetric Holm-adjusted paired tests (`k in {3, 5, 8} × {RMSE, NDCG@3, Regret@1}`), `AdaptiveOED` vs `StaticDOpt` yielded **`INSUFFICIENT_EVIDENCE`** in both ID CV and `Waves 3+4` (where at `k = 8`, `AdaptiveOED` had lower RMSE `49.61` vs `50.56` but lower `NDCG@3` `0.868` vs `0.879`). We therefore retained `StaticDOpt` as the simpler working reference (`AD-008`).
2. **Archive availability vs. locked catalog:** On `Waves 3+4`, two top `Waves 1+2` probes (`XAV-939`: 0/146; `ABT-737`: 3/146) were nearly absent in later waves, reducing exact full-panel coverage under `locked_train_catalog` (`k = 5`) to 19/93 in ID and 1/163 in `Waves 3+4`. Reported archive performance therefore reflects backfill from the ordered ranking on measured drugs. Mechanism ablations confirmed that `M3` improved substantially over an offset-only model (`M3 − M3_offset_only`: −3.70 AUC [−4.48, −2.96] in ID; −2.46 [−3.89, −1.04] in `Waves 3+4`).

*[Figure 1 placeholder: Beat AML learning curves for `k = 0..12` comparing `M0`, `M1`, `Random50`, `StaticDOpt`, and `AdaptiveOED` on ID CV (`n = 427`) and `Waves 3+4` (`n = 146`). Source: `data/processed/sprint03_v2_results.json`.]*

### 3.2 Negative Finding 1: Pathway/Drug-Family Covariance (`M4`) Fails Gate 3

Holding patients, targets, and the `StaticDOpt(M3)` panel fixed (Experiment A1, `k = 5`), `M4` was active (inner CV selected `rho > 0` in 13/15 ID CV folds) yet produced **zero improvement** over unstructured `M3` or the 20-draw degree-preserving permuted family control (Table 2; **[Figure 2]**).

**Table 2. Gate 3 evaluation of pathway-structured covariance (`M4`) against `M3` and 20× degree-preserving null (`k = 5`, Experiment A1).**

| Cohort / Regime | *n* | `M3` RMSE | `M4` RMSE | `M4_perm_null` (20×) | `ΔRMSE (M4 − M3)` [95% CI] | `ΔRMSE (M4 − Perm)` [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Beat AML ID 5-Fold CV × 3 (Primary) | 427 | 42.91 | 42.92 | 42.90 | +0.011 [−0.010, +0.033] (*p* = 0.294) | +0.017 [−0.005, +0.039] (*p* = 0.128) |
| Beat AML ID 60/20/20 | 91 | 39.84 | 39.82 | 39.84 | −0.02 [−0.06, +0.01] (*p* = 0.262) | −0.02 [−0.05, +0.00] (*p* = 0.128) |
| Beat AML `Waves 3+4` S3 (Exploratory) | 146 | 50.64 | 50.62 | 50.63 | −0.02 [−0.05, +0.01] (*p* = 0.156) | −0.01 [−0.04, +0.02] (*p* = 0.438) |
| FIMM Within-Cohort CV (`C2`, sDSS) | 109 | 3.83 | 3.83 | 3.83 | −0.00 [−0.01, +0.01] (*p* = 0.444) | −0.00 [−0.01, +0.00] (*p* = 0.196) |

Reselecting panels with `M4` (Experiment A2, `k = 5`) also failed to improve over `M3` (ID CV `ΔRMSE` = −0.10 [−0.27, +0.08], `Regret@1` worsened 23.3 → 24.9; `Waves 3+4` degraded +0.16 [+0.01, +0.35], *p* = 0.044). Under pre-registered Gate 3 (`ΔRMSE <= -0.50`), the verdict was **`NO-GO`** and mechanistic Hill-ODE development was deferred (`AD-009`).

*[Figure 2 placeholder: Paired `ΔRMSE` forest plot for `M4 − M3`, `M4 − M4_perm_null`, and `M4_forced − M4_perm_forced` across Beat AML ID CV, ID 60/20/20, `Waves 3+4`, and FIMM `C2` against the −0.50 Gate 3 threshold. Source: `data/processed/sprint05_results.json`.]*

### 3.3 Patient-Level Rank-Calibrated Shortlists and OOD Abstention Limits

On Sprint 4 `v2` predictions (`StaticDOpt + M3, k = 5`), simultaneous split-conformal AUC intervals achieved 79/91 (86.8%; Wilson 95% CI 78.4–92.3%) coverage in ID and dropped to 110/148 (74.3%) on `Waves 3+4`, while interval widths (241.8 and 247.4 AUC) forced `EqualWidthAUC` shortlists to retain nearly every observed drug (Table 3). Patient-level rank calibration resolved shortlist inflation (**[Figure 3]**):

**Table 3. Conformal shortlist comparison at nominal 90% (`alpha = 0.10`) for `StaticDOpt + M3 (k = 5)` on observed targets.**

| Regime (*n*_cal / *n*_test) | Method | Calibrated Threshold | True Best in Shortlist | Mean Size (Median) | Mean Target Fraction | Singleton Rate (`|S| = 1`) | Singleton True-Best Accuracy | Mean Shortlist Regret@1 (AUC) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ID 60/20/20 (90 / 91) | `EqualWidthAUC` | `2q` = 241.8 | 91/91 (100.0%) | 18.70 (20.0) | 98.7% | 0.0% | — | 116.77 |
| ID 60/20/20 (90 / 91) | **`RankConformal`** | **`k_hat` = 6** | **85/91 (93.4%)** | **6.00 (6.0)** | **35.3%** | 0.0% | — | 66.03 |
| ID 60/20/20 (90 / 91) | `RelRankConformal` | `q_rel` = 0.429 | 85/91 (93.4%) | 8.64 (9.0) | 45.6% | 0.0% | — | 79.39 |
| ID 60/20/20 (90 / 91) | `Top1GapConformal` | `m_hat` = 74.2 | 84/91 (92.3%) | 5.45 (5.0) | 33.6% | 9.9% (9/91) | **5/9 (26.7–81.1%)** | 53.83 |
| `Waves 3+4` (70 / 148) | `EqualWidthAUC` | `2q` = 247.4 | 148/148 (100.0%) | 21.41 (22.0) | 98.7% | 0.0% | — | 130.90 |
| `Waves 3+4` (70 / 148) | **`RankConformal`** | **`k_hat` = 6** | **137/148 (92.6%)** | **6.00 (6.0)** | **29.1%** | 0.0% | — | 73.51 |
| `Waves 3+4` (70 / 148) | `Top1GapConformal` | `m_hat` = 63.6 | 128/148 (86.5%) | 4.76 (4.0) | 23.1% | 16.9% (25/148) | **19/25 (56.6–88.5%)** | 52.04 |

Three practical lessons stand out:
1. **`RankConformal` vs `EqualWidthAUC`:** `RankConformal` cut shortlist size by −12.70 drugs [−13.69, −11.69] (*p* = 0.002) in ID and −15.41 drugs in `Waves 3+4`, while staying above 92% true-best containment in both regimes because pure ranks are scale-invariant.
2. **Negative Finding 2 (Singleton reliability trap, `SOL-13`):** Although `Top1GapConformal` achieved 92.3% marginal coverage in ID and produced singletons in 9/91 patients, only **5/9** singletons contained the true best drug (Wilson 95% CI 26.7–81.1%; `Waves 3+4`: 19/25, 56.6–88.5%). Marginal shortlist coverage does not transfer to the singleton subgroup.
3. **Negative Finding 3 (OOD abstention failure):** On `Waves 3+4` (`StaticDOpt, k = 5`), omics PCA distance ranked patients worse than random (`AURC` 32.44 vs random 24.60; `AUGRC` 14.27 vs random 12.30). `m3_post_scale` ranked better than random (`AURC` 20.35; `AUGRC` 10.57), but its inner-OOF locked threshold accepted 91/91 ID and 146/148 `Waves 3+4` patients due to inner-vs-final scale shift. Exploratory refitting of the threshold on the final model's scale (`Addendum 19`) moved acceptance to 84/91 (ID) and 65/148 (`Waves 3+4`; accepted vs abstained `Regret@1`: 20.6 vs 27.7 AUC), but this post-hoc sensitivity does not establish prospective OOD risk control.

*[Figure 3 placeholder: Tradeoff between shortlist size, true-best containment, and mean shortlist Regret@1 across `EqualWidthAUC`, `RankConformal`, `RelRankConformal`, and `Top1GapConformal` on ID (`n = 91`) and `Waves 3+4` (`n = 148`), plus `AUGRC` curves. Source: `data/processed/sprint05_results.json` and `sprint05_vectors.npz`.]*

### 3.4 External FIMM Replication (`C1` Frozen Transfer vs `C2` Within-FIMM CV) and `OPUS-028` Control

In FIMM, the Beat AML-selected `StaticDOpt` prefix (`Flavopiridol, Sunitinib, XAV-939, CYT387, Tivozanib`) had high direct availability: 109/109 (100%) exact full panels at `k = 2` and 98/109 (89.9%) at `k = 5` and `k = 8` (11 patients received archive backfill at `k = 5, 8`). Table 4 separates frozen Beat AML transfer (`C1`) from within-FIMM refitting (`C2`) (**[Figure 4]**).

**Table 4. FIMM/Malani external evaluation on 109 unique index AML patients (23 shared targets, 64 shared probes; `-sDSS` alignment).**

| Experiment & Label | Arm | RMSE (sDSS) ↓ | NDCG@3 ↑ | Recall@3 ↑ | Within-Patient Spearman ↑ | Regret@1 (sDSS) ↓ |
|---|---|---:|---:|---:|---:|---:|
| **C1 (Frozen Beat AML transfer)** | `C1_M0_k0` | — | 0.571 | 0.310 | 0.186 | 11.23 |
| **C1 (Frozen Beat AML transfer)** | `C1_M1_rank_k0` | — | 0.651 | 0.372 | 0.258 | 7.41 |
| **C1 (Frozen Beat AML transfer)** | `C1_M3_StaticDOpt_k2` | — | 0.683 | 0.404 | 0.270 | 6.66 |
| **C1 (Frozen Beat AML transfer)** | **`C1_M3_StaticDOpt_k5`** | — | **0.683** | **0.408** | **0.280** | 7.18 |
| **C1 (Frozen Beat AML transfer)** | `C1_M3_Random20x_k5` | — | 0.649 | 0.362 | 0.281 | 7.44 |
| **C1 (Frozen Beat AML transfer)** | `C1_M3_StaticDOpt_k8` | — | 0.670 | 0.390 | 0.279 | 7.83 |
| **C2 (Within-FIMM 5-Fold CV × 3)** | `M0_k0` / `M0_k5` | 5.27 / 4.72 | 0.713 / 0.713 | 0.438 | 0.000 (res) | 6.89 |
| **C2 (Within-FIMM 5-Fold CV × 3)** | `M1_k0` | 4.83 | 0.765 | 0.486 | 0.325 (res) | 4.92 |
| **C2 (Within-FIMM 5-Fold CV × 3)** | `M2a_StaticDOpt_k5` | 4.06 | 0.797 | 0.510 | 0.453 (res) | 3.33 |
| **C2 (Within-FIMM 5-Fold CV × 3)** | **`M3_StaticDOpt_k5`** | **3.83** | **0.843** | **0.569** | **0.521 (res)** | **2.58** |
| **C2 (Within-FIMM 5-Fold CV × 3)** | `M3_Random20x_k5` | 4.29 | 0.802 | 0.523 | 0.420 (res) | 3.65 |

- **C1 frozen transfer (top-3 enrichment, not `Regret@1`):** At `k = 5`, frozen `C1_M3_StaticDOpt` beat `C1_M1_rank_k0` on `NDCG@3` (+0.032 [+0.007, +0.058], *p* = 0.016) and `Recall@3` (+0.037 [+0.006, +0.067]), and beat `C1_M3_Random20x_k5` on `NDCG@3` (+0.033 [+0.013, +0.054], *p* = 0.004). Results on all 132 specimens clustered into 109 patients were virtually identical (`NDCG@3` = 0.680; *p* = 0.008 vs `M1`, *p* = 0.002 vs random). However, within-patient Spearman crossed zero in independent audit bootstrap (+0.022 [−0.001, +0.045]), `Regret@1` differences were non-significant (−0.23 [−1.26, +0.76] vs `M1`; −0.26 [−0.93, +0.41] vs random), and moving from `k = 5` to `k = 8` degraded frozen transfer (`0.683 → 0.670`). Note that `Random20x` was run only for `M3, k = 5` (`SOL-16`).
- **C2 within-FIMM method replication:** Refitting inside FIMM cleanly replicated every Beat AML pattern: `M3_StaticDOpt_k5` beat `M1_k0` (`ΔRMSE` = −1.00 [−1.26, −0.76] sDSS, `ΔNDCG@3` = +0.077 [+0.052, +0.103], `ΔRegret@1` = −2.34 [−3.48, −1.28], all *p* = 0.002) and beat `M3_Random20x_k5` (`ΔRMSE` = −0.46 [−0.61, −0.32], `ΔNDCG@3` = +0.041 [+0.026, +0.056], `ΔRegret@1` = −1.07 [−1.60, −0.52], all *p* = 0.002).
- **Beat AML probe-scale control (`OPUS-028`, *n* = 91):** On Beat AML with the exact C1 pipeline (`k = 5`), single-patient *k*-probe z-scoring still beat `M1` on `NDCG@3` (+0.034 [+0.014, +0.055]; `Regret@1` −6.91 was borderline and seed-dependent: Sol 2,000-rep [−14.88, +0.21] vs Opus 4,000-rep [−14.34, −0.05]) and showed no statistically clear loss versus the full-row oracle scaling (`ΔNDCG@3` = −0.005 [−0.019, +0.007], `ΔRegret@1` = +2.2 [−3.5, +8.9]) or training-reference scaling (`−0.010 [−0.021, +0.001]`). At `k = 2`, however, 91/91 patients collapsed to `±1` vectors, losing magnitude information (`ΔNDCG@3` = −0.018 [−0.032, −0.005] vs full-row oracle; −0.016 [−0.031, −0.002] vs training reference).

*[Figure 4 placeholder: Two-panel FIMM summary showing (A) Experiment C1 frozen Beat AML transfer (`NDCG@3` and `Regret@1` across `M0`, `M1`, `Random20x`, and `StaticDOpt` `k = 2, 5, 8`) and (B) Experiment C2 within-FIMM CV replication (`RMSE` and `NDCG@3`). Source: `data/processed/sprint05_results.json`.]*

---

## 4. Discussion and Limitations

Our empirical results draw a clear boundary between what works and what fails in primary AML *ex vivo* prediction:

1. **Why 5 probes + `M3` works, and why `M4` did not:** Measuring five informative inhibitors directly captures patient-specific cellular state (proliferation rate, apoptosis priming, signaling dependence) that baseline RNA-seq misses. At *n* ≈ 340 (Beat AML) and even *n* ≈ 87 per fold (FIMM), empirical drug-drug residual covariance already captures shared target and off-target co-sensitivity better than manual drug-family labels—which leave 9/24 AML targets unannotated and miss polypharmacology. Deferring kinetic Hill-ODE modeling (`Gate 3: NO-GO`) avoided investing in unidentifiable models when a simple biological prior added zero signal.
2. **Comparison to related work:** Shokrzadeh & Shokrzadeh (2026) applied split-conformal regression to Beat AML (318 patients, 122 compounds, 90.7% macro-averaged coverage at nominal 90%), focusing on drug-level prediction intervals. Our work addresses a complementary problem: selecting a 5-drug probe batch and calibrating a **patient-level candidate shortlist** containing the best observed target, where scale-invariant `RankConformal` cut list length from 18.7 to 6.0 drugs. Tercan (2026) used within-sample relative gene expression (`kTSP`) for Beat AML → FIMM binary response transfer; our `C1` rank-normalized RNA representation drew on that idea, while single-patient probe z-scoring and shortlist calibration are distinct project choices.
3. **Key limitations (what this study does NOT prove):**
   - **Retrospective archive mask:** Shortlists and panel evaluations operate on drugs actually measured in the archives (`O_i` known at shortlist time; 11/109 FIMM patients backfilled at `k = 5`). Prospective validation requires a fixed, locked 5-probe assay plate with technical abstention tracked explicitly.
   - **Assay and dose mismatch across sites:** Beat AML (MTS, AUC) and FIMM (CTG, sDSS) differ in viability assay, medium, concentration series (11/23 targets differ in max dose), and response metric (Yadav et al., 2014; Willoch-Allen et al., 2026). This likely explains why frozen `C1` transfer showed top-3 enrichment (`NDCG@3`) without significant `Regret@1` gain and degraded at `k = 8`.
   - **Upstream FIMM RNA preprocessing (`SOL-16`):** FIMM `File_7` log2-CPM values were batch-corrected upstream in the Zenodo release; verifying truly single-sample preprocessing from raw counts (`File_8`) remains future work.
   - **No clinical endpoint validation:** Like Beat AML and FIMM drug screens themselves, these results measure 72-hour *ex vivo* single-agent viability. Translating to patient treatment requires prospective clinical designs with defined regimens and clinical endpoints (e.g., the prospective VenEx trial, Kytölä et al., *Blood* 2025).

In summary, on archived primary AML drug screens, a small fixed five-probe panel (`StaticDOpt + M3`) paired with scale-invariant rank-calibrated shortlists (`RankConformal`) offers a simple, CPU-only, reproducible baseline that outperforms omics alone and random panels across two independent cohorts—while curated pathway priors, adaptive OED, omics-distance OOD abstention, and singleton selection all showed clear empirical limits that should be respected in future prospective designs.

---

## References

1. Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. Conformal Risk Control. *ICLR* 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
2. Bernett J, Blumenthal DB, Grimm DG, Haselbeck F, Joeres R, Kalinina OV, List M. DrEval: a framework for rigorous evaluation of drug response prediction models. *Nature Communications* 2026;17:4238. https://doi.org/10.1038/s41467-026-72903-w
3. Bottomly D, Long N, Schultz AR, Kurtz SE, Tognon CE, Johnson K, et al. Integrative analysis of drug response and clinical outcome in acute myeloid leukemia. *Cancer Cell* 2022;40(8):850-864. https://doi.org/10.1016/j.ccell.2022.07.002
4. Guillen KP, Fujita M, Butterfield AJ, Scherer SD, Bailey MH, Chu Z, et al. A human breast cancer-derived xenograft and organoid platform for drug discovery and precision oncology. *Nature Cancer* 2022;3:232-250. https://doi.org/10.1038/s43018-022-00337-6
5. Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 1979;6(2):65-70. https://www.jstor.org/stable/4615733
6. Lee Y, Barber RF, Willett R. Distribution-free inference with hierarchical data. *arXiv* 2023;2306.06342. https://arxiv.org/abs/2306.06342
7. Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-free predictive inference for regression. *Journal of the American Statistical Association* 2018;113(523):1094-1111. https://doi.org/10.1080/01621459.2017.1307116
8. Liao W, Huang H, Jia C, Xi H, Zeng H, Wei H. Distribution-informed Efficient Conformal Prediction for Full Ranking. *arXiv* 2026;2601.23128. https://arxiv.org/abs/2601.23128
9. Malani D, Kumar A, Brück O, Kontro M, Yadav B, Hellesøy M, et al. Implementing a functional precision medicine tumor board for acute myeloid leukemia. *Cancer Discovery* 2022;12(2):388-401. https://doi.org/10.1158/2159-8290.CD-21-0410 (Dataset: Zenodo 7370747, https://zenodo.org/records/7370747)
10. Raue A, Kreutz C, Maiwald T, Bachmann J, Schilling M, Klingmüller U, Timmer J. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 2009;25(15):1923-1929. https://doi.org/10.1093/bioinformatics/btp358
11. Sale Y, Hüllermeier E. Conformal Poset Prediction for Label Ranking. *COPA / PMLR* 2026;329. https://proceedings.mlr.press/v329/sale26a.html
12. Shokrzadeh AJ, Shokrzadeh P. Calibrated Uncertainty Quantification for Patient-Level AML Drug Sensitivity Prediction Using Split Conformal Prediction. *bioRxiv* 2026.06.07.730728v1. https://www.biorxiv.org/content/10.64898/2026.06.07.730728v1
13. Tercan B. Robust predictors for drug response of patients with acute myeloid leukemia. *PLOS ONE* 2026;21(2):e0343422. https://doi.org/10.1371/journal.pone.0343422
14. Tibshirani RJ, Foygel Barber R, Candès EJ, Ramdas A. Conformal prediction under covariate shift. *NeurIPS* 2019.
15. Traub J, Bungert TJ, Lüth CT, Baumgartner M, Maier-Hein KH, Maier-Hein L, Jäger PF. Overcoming Common Flaws in the Evaluation of Selective Classification Systems. *NeurIPS* 2024 (Spotlight). https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract-Conference.html (arXiv:2407.01032)
16. Kytölä S, Vänttinen I, Ruokoranta T, Partanen A, Holopainen A, Saad J, et al. Ex vivo venetoclax sensitivity predicts clinical response in acute myeloid leukemia in the prospective VenEx trial. *Blood* 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC13122316/
17. Willoch-Allen et al. A multi-center study on the consistency of drug sensitivity testing in patients with acute myeloid leukemia. *npj Precision Oncology* 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC13408654/
18. Yadav B, Pemovska T, Szwajda A, Kulesskiy E, Kontro M, Karjalainen R, et al. Quantitative scoring of differential drug sensitivity for individually optimized anticancer therapies. *Scientific Reports* 2014;4:5193. https://doi.org/10.1038/srep05193
