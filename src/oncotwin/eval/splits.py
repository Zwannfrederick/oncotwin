"""Sızıntısız hasta-bazlı bölme ve eğitim-katı içi prob havuzu seçicileri."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_patient_disjoint_kfold(
    meta_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    strata_col: str = "cohort",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """cohort tabakalı, dbgap_subject_id düzeyinde sızıntısız K-Fold bölmeleri döndürür."""
    subjects = meta_df["dbgap_subject_id"].to_numpy()
    strata = meta_df[strata_col].fillna("UNKNOWN").astype(str).to_numpy()

    unique_subj, first_idx = np.unique(subjects, return_index=True)
    subj_strata = strata[first_idx]

    rng = np.random.default_rng(seed)
    fold_for_subj: dict[int, int] = {}

    for stratum in sorted(np.unique(subj_strata)):
        s_ids = unique_subj[subj_strata == stratum].copy()
        rng.shuffle(s_ids)
        for idx, sid in enumerate(s_ids):
            fold_for_subj[int(sid)] = idx % n_splits

    row_folds = np.array([fold_for_subj[int(sid)] for sid in subjects], dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        test_idx = np.flatnonzero(row_folds == fold)
        train_idx = np.flatnonzero(row_folds != fold)
        train_subj = set(int(s) for s in subjects[train_idx])
        test_subj = set(int(s) for s in subjects[test_idx])
        assert train_subj.isdisjoint(test_subj), "Hasta bazlı bölmede sızıntı tespit edildi!"
        splits.append((train_idx, test_idx))
    return splits


def make_clean_wave_split(
    meta_df: pd.DataFrame,
    w12_all_subjects: set[int],
    both_all_subjects: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Waves1+2 eğitim ve sızıntısız Waves3+4 test indekslerini döndürür."""
    subjects = meta_df["dbgap_subject_id"].to_numpy()
    cohort = meta_df["cohort"].fillna("UNKNOWN").astype(str).to_numpy()

    train_mask = cohort == "Waves1+2"
    excluded_from_test = w12_all_subjects | both_all_subjects
    is_clean_w34 = np.array([int(s) not in excluded_from_test for s in subjects], dtype=bool)
    test_mask = (cohort == "Waves3+4") & is_clean_w34

    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    train_subj = set(int(s) for s in subjects[train_idx])
    test_subj = set(int(s) for s in subjects[test_idx])
    assert train_subj.isdisjoint(test_subj), "Dalga bölmesinde hasta sızıntısı!"
    assert not (test_subj & both_all_subjects), "Test kümesinde 'Both' hastası olamaz!"
    return train_idx, test_idx


def select_train_probe_pool(
    Y_train: np.ndarray,
    drug_names: list[str],
    target_drugs: list[str],
    min_train_obs: int = 100,
) -> list[int]:
    """Yalnızca Y_train gözlem sayılarına bakarak C dışındaki prob havuzunu (U_tr) seçer."""
    target_set = set(target_drugs)
    obs_counts = np.sum(np.isfinite(Y_train), axis=0)
    return [
        idx
        for idx, name in enumerate(drug_names)
        if name not in target_set and int(obs_counts[idx]) >= min_train_obs
    ]


def get_evaluable_mask(
    Y_eval: np.ndarray,
    target_idx: list[int],
    probe_pool_idx: list[int],
    min_targets: int = 5,
    min_probes: int = 8,
) -> np.ndarray:
    """Hedef ve prob havuzunda yeterli ölçümü olan değerlendirilebilir test satırlarını işaretler."""
    n_targets = np.sum(np.isfinite(Y_eval[:, target_idx]), axis=1)
    n_probes = np.sum(np.isfinite(Y_eval[:, probe_pool_idx]), axis=1)
    return (n_targets >= min_targets) & (n_probes >= min_probes)


def sample_probe_sets(
    Y_eval: np.ndarray,
    probe_pool_idx: list[int],
    k: int,
    n_repeats: int = 10,
    seed: int = 42,
) -> list[list[np.ndarray]]:
    """Her tekrar ve her test hastası için gözlenen U_tr ilaçlarından deterministik k prob seçer."""
    n_eval = Y_eval.shape[0]
    if k <= 0:
        empty = np.array([], dtype=np.int64)
        return [[empty for _ in range(n_eval)] for _ in range(n_repeats)]

    pool_arr = np.asarray(probe_pool_idx, dtype=np.int64)
    rng = np.random.default_rng(seed)
    repeats: list[list[np.ndarray]] = []
    for _ in range(n_repeats):
        per_patient: list[np.ndarray] = []
        for i in range(n_eval):
            obs = pool_arr[np.isfinite(Y_eval[i, pool_arr])]
            if len(obs) <= k:
                per_patient.append(obs.copy())
            else:
                chosen = rng.choice(obs, size=k, replace=False)
                chosen.sort()
                per_patient.append(chosen)
        repeats.append(per_patient)
    return repeats


def _build_split_manifest_entry(meta_df: pd.DataFrame, idx: np.ndarray) -> dict[str, object]:
    sub_df = meta_df.iloc[idx]
    subj_ids = sorted(int(s) for s in sub_df["dbgap_subject_id"].unique())
    spec_keys = sub_df["specimen_key"].astype(str).tolist() if "specimen_key" in sub_df.columns else []
    cohort_counts = sub_df["cohort"].fillna("UNKNOWN").astype(str).value_counts().to_dict()
    return {
        "n_rows": int(len(idx)),
        "n_subjects": len(subj_ids),
        "subject_ids": subj_ids,
        "specimen_keys": spec_keys,
        "cohort_counts": {str(k): int(v) for k, v in cohort_counts.items()},
    }


def make_sprint04_splits(
    meta_df: pd.DataFrame,
    w12_all_subjects: set[int],
    both_all_subjects: set[int],
    seed: int = 2026,
) -> dict[str, object]:
    """Sprint 4 için hasta-ayrık (train_tune / calibration / test) ID ve W3+4 bölmelerini oluşturur.

    İlaç AUC değerlerini hiçbir aşamada okumaz; yalnızca meta ve hasta kimliği kullanır.
    """
    subjects = meta_df["dbgap_subject_id"].to_numpy()

    # 1. ID 60/20/20: 5-Fold içinde kat 0,1,2 -> train_tune; kat 3 -> calibration; kat 4 -> test
    folds5 = make_patient_disjoint_kfold(meta_df, n_splits=5, seed=seed, strata_col="cohort")
    id_train = np.sort(np.concatenate([folds5[0][1], folds5[1][1], folds5[2][1]])).astype(np.int64)
    id_cal = np.sort(folds5[3][1]).astype(np.int64)
    id_test = np.sort(folds5[4][1]).astype(np.int64)

    # 2. Clean W1+2 -> W3+4: W1+2 içinde 4-Fold (kat 0,1,2 -> train_tune; kat 3 -> calibration)
    w12_idx, w34_idx = make_clean_wave_split(meta_df, w12_all_subjects, both_all_subjects)
    w12_meta = meta_df.iloc[w12_idx].reset_index(drop=True)
    folds4 = make_patient_disjoint_kfold(w12_meta, n_splits=4, seed=seed, strata_col="cohort")
    w_train = np.sort(w12_idx[np.concatenate([folds4[0][1], folds4[1][1], folds4[2][1]])]).astype(np.int64)
    w_cal = np.sort(w12_idx[folds4[3][1]]).astype(np.int64)
    w_test = np.sort(w34_idx).astype(np.int64)

    regimes = {
        "id_60_20_20": {"train_tune": id_train, "calibration": id_cal, "test": id_test},
        "wave_ood": {"train_tune": w_train, "calibration": w_cal, "test": w_test},
    }
    manifest: dict[str, object] = {"seed": int(seed), "regimes": {}}
    for r_name, part in regimes.items():
        s_tr = set(int(s) for s in subjects[part["train_tune"]])
        s_cal = set(int(s) for s in subjects[part["calibration"]])
        s_te = set(int(s) for s in subjects[part["test"]])
        assert s_tr.isdisjoint(s_cal), f"{r_name}: train_tune/calibration hasta sızıntısı!"
        assert s_tr.isdisjoint(s_te), f"{r_name}: train_tune/test hasta sızıntısı!"
        assert s_cal.isdisjoint(s_te), f"{r_name}: calibration/test hasta sızıntısı!"
        manifest["regimes"][r_name] = {
            k: _build_split_manifest_entry(meta_df, idx_arr) for k, idx_arr in part.items()
        }

    return {"id_60_20_20": regimes["id_60_20_20"], "wave_ood": regimes["wave_ood"], "manifest": manifest}
