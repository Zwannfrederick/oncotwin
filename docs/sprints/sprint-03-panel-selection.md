# Sprint 3 — Toplu Panel Seçimi, `M3` Mekanizma Ablasyonları ve Kapı 2

Tüm deneyler dondurulmuş [PROTOCOL_SPRINT03.md](../PROTOCOL_SPRINT03.md) uyarınca **700.5 sn** içinde CPU üzerinde çalıştırılmıştır.

---

## 1. Birincil ID Rejimi — `Patient-Disjoint 5-Fold CV` (3 Seed, Eşit Bütçe $k \in \{1..12\}$)

| $k$ | Politika / Referans | `n_eval` | `RMSE` ↓ | `Res R²` ↑ | $\rho^{\text{res}}_{\text{pat}}$ ↑ | `NDCG@3` ↑ | `Recall@3` ↑ | `Regret@1` ↓ | `Regret@3` ↓ | Ek Bilgi |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `M0(0)` (ref) | 427 | 56.24 | +0.000 | +0.000 | 0.784 | 0.507 | 49.2 | 12.3 | Panel seçimi yok |
| 0 | **`M1` (omik ref)** | 427 | 49.27 | +0.241 | +0.413 | 0.856 | 0.577 | 27.0 | 7.5 | Panel seçimi yok |
| 1 | **Random50** | 427 | 47.98 | +0.279 | +0.428 | 0.860 | 0.582 | 26.4 | 6.8 | — |
| 1 | **StaticDiversity** | 427 | 45.58 | +0.346 | +0.465 | 0.868 | 0.596 | 25.7 | 6.3 | direct=94% (tam=94%) |
| 1 | **StaticDOpt** | 427 | 45.48 | +0.348 | +0.468 | 0.868 | 0.597 | 26.3 | 6.1 | direct=94% (tam=94%) |
| 1 | **AdaptiveOED** | 427 | 47.40 | +0.296 | +0.448 | 0.866 | 0.582 | 22.7 | 6.2 | uniq=10, Jacc=0.41 |
| 2 | **Random50** | 427 | 47.04 | +0.306 | +0.439 | 0.863 | 0.586 | 25.6 | 6.4 | — |
| 2 | **StaticDiversity** | 427 | 45.27 | +0.354 | +0.473 | 0.871 | 0.600 | 25.4 | 5.7 | direct=91% (tam=83%) |
| 2 | **StaticDOpt** | 427 | 44.44 | +0.376 | +0.487 | 0.873 | 0.600 | 24.6 | 6.0 | direct=93% (tam=87%) |
| 2 | **AdaptiveOED** | 427 | 45.19 | +0.354 | +0.477 | 0.872 | 0.592 | 22.8 | 5.5 | uniq=28, Jacc=0.27 |
| 3 | **Random50** | 427 | 46.29 | +0.327 | +0.450 | 0.866 | 0.591 | 25.2 | 6.2 | — |
| 3 | **StaticDiversity** | 427 | 45.15 | +0.355 | +0.478 | 0.871 | 0.598 | 25.2 | 5.7 | direct=90% (tam=73%) |
| 3 | **StaticDOpt** | 427 | 43.74 | +0.395 | +0.495 | 0.876 | 0.603 | 24.3 | 5.6 | direct=89% (tam=70%) |
| 3 | **AdaptiveOED** | 427 | 44.21 | +0.380 | +0.485 | 0.875 | 0.599 | 23.7 | 5.3 | uniq=78, Jacc=0.25 |
| 5 | **Random50** | 427 | 45.11 | +0.360 | +0.466 | 0.871 | 0.596 | 24.0 | 5.8 | — |
| 5 | **StaticDiversity** | 427 | 44.62 | +0.370 | +0.485 | 0.873 | 0.608 | 24.8 | 5.4 | direct=86% (tam=42%) |
| 5 | **StaticDOpt** | 427 | 42.74 | +0.422 | +0.511 | 0.876 | 0.604 | 23.8 | 5.9 | direct=85% (tam=47%) |
| 5 | **AdaptiveOED** | 427 | 42.98 | +0.414 | +0.503 | 0.881 | 0.609 | 21.6 | 5.2 | uniq=204, Jacc=0.26 |
| 8 | **Random50** | 427 | 43.91 | +0.393 | +0.486 | 0.875 | 0.602 | 23.2 | 5.6 | — |
| 8 | **StaticDiversity** | 427 | 43.79 | +0.392 | +0.496 | 0.873 | 0.600 | 23.6 | 5.6 | direct=86% (tam=25%) |
| 8 | **StaticDOpt** | 427 | 41.24 | +0.462 | +0.533 | 0.881 | 0.604 | 22.1 | 5.8 | direct=83% (tam=30%) |
| 8 | **AdaptiveOED** | 427 | 41.49 | +0.454 | +0.525 | 0.887 | 0.616 | 21.0 | 5.1 | uniq=317, Jacc=0.32 |
| 12 | **Random50** | 427 | 42.71 | +0.425 | +0.507 | 0.879 | 0.608 | 22.2 | 5.4 | — |
| 12 | **StaticDiversity** | 427 | 42.79 | +0.418 | +0.513 | 0.877 | 0.601 | 22.4 | 5.1 | direct=87% (tam=18%) |
| 12 | **StaticDOpt** | 427 | 40.24 | +0.488 | +0.546 | 0.886 | 0.614 | 20.8 | 5.3 | direct=82% (tam=19%) |
| 12 | **AdaptiveOED** | 427 | 40.41 | +0.482 | +0.537 | 0.890 | 0.623 | 20.1 | 5.0 | uniq=383, Jacc=0.36 |

### Eşleştirilmiş Hasta Bootstrap Farkları (CV — %95 GA)

| Karşılaştırma | `ΔRMSE` (negatif = iyi) | `ΔRecall@3` (pozitif = iyi) | `ΔNDCG@3` (pozitif = iyi) | `ΔRegret@1` (negatif = iyi) | `Δ(ρ^res)` (pozitif = iyi) |
|---|---:|---:|---:|---:|---:|
| `k1_StaticDiversity_vs_Random50` | -2.40 [-2.94, -1.95] | +0.014 [+0.004, +0.024] | +0.008 [+0.004, +0.013] | -0.7 [-2.6, +0.9] | +0.037 [+0.026, +0.048] |
| `k1_StaticDOpt_vs_Random50` | -2.51 [-3.03, -2.02] | +0.015 [+0.005, +0.026] | +0.009 [+0.004, +0.013] | -0.2 [-2.2, +1.8] | +0.040 [+0.028, +0.051] |
| `k1_AdaptiveOED_vs_Random50` | -0.59 [-1.08, -0.09] | -0.000 [-0.009, +0.009] | +0.006 [+0.001, +0.011] | -3.7 [-5.7, -1.8] | +0.020 [+0.009, +0.031] |
| `k1_AdaptiveOED_vs_StaticDOpt` | +1.92 [+1.26, +2.63] | -0.015 [-0.027, -0.003] | -0.002 [-0.009, +0.004] | -3.5 [-5.9, -1.1] | -0.020 [-0.033, -0.006] |
| `k2_StaticDiversity_vs_Random50` | -1.77 [-2.19, -1.38] | +0.014 [+0.005, +0.023] | +0.008 [+0.004, +0.012] | -0.2 [-2.0, +1.6] | +0.034 [+0.024, +0.044] |
| `k2_StaticDOpt_vs_Random50` | -2.60 [-3.19, -2.05] | +0.014 [+0.004, +0.024] | +0.009 [+0.005, +0.014] | -1.0 [-3.2, +1.1] | +0.048 [+0.036, +0.060] |
| `k2_AdaptiveOED_vs_Random50` | -1.85 [-2.45, -1.26] | +0.006 [-0.002, +0.014] | +0.008 [+0.004, +0.014] | -2.8 [-4.8, -0.9] | +0.038 [+0.028, +0.050] |
| `k2_AdaptiveOED_vs_StaticDOpt` | +0.75 [+0.23, +1.30] | -0.008 [-0.017, +0.001] | -0.001 [-0.005, +0.004] | -1.8 [-3.8, +0.0] | -0.010 [-0.020, -0.000] |
| `k3_StaticDiversity_vs_Random50` | -1.15 [-1.57, -0.73] | +0.008 [-0.001, +0.016] | +0.004 [+0.000, +0.009] | -0.0 [-1.8, +1.7] | +0.028 [+0.018, +0.038] |
| `k3_StaticDOpt_vs_Random50` | -2.56 [-3.06, -2.04] | +0.012 [+0.003, +0.022] | +0.009 [+0.005, +0.014] | -0.9 [-2.9, +1.1] | +0.045 [+0.033, +0.057] |
| `k3_AdaptiveOED_vs_Random50` | -2.09 [-2.78, -1.45] | +0.009 [-0.001, +0.018] | +0.008 [+0.003, +0.013] | -1.5 [-3.4, +0.6] | +0.035 [+0.024, +0.046] |
| `k3_AdaptiveOED_vs_StaticDOpt` | +0.47 [-0.16, +1.11] | -0.004 [-0.014, +0.007] | -0.001 [-0.006, +0.004] | -0.6 [-2.3, +1.3] | -0.010 [-0.021, -0.000] |
| `k5_StaticDiversity_vs_Random50` | -0.50 [-0.94, -0.06] | +0.012 [+0.003, +0.020] | +0.002 [-0.002, +0.006] | +0.8 [-1.2, +2.9] | +0.019 [+0.009, +0.029] |
| `k5_StaticDOpt_vs_Random50` | -2.37 [-2.84, -1.89] | +0.008 [-0.001, +0.017] | +0.005 [+0.001, +0.009] | -0.3 [-2.3, +1.7] | +0.045 [+0.034, +0.055] |
| `k5_AdaptiveOED_vs_Random50` | -2.13 [-2.76, -1.50] | +0.013 [+0.004, +0.022] | +0.011 [+0.006, +0.016] | -2.4 [-4.2, -0.5] | +0.037 [+0.026, +0.047] |
| `k5_AdaptiveOED_vs_StaticDOpt` | +0.24 [-0.33, +0.81] | +0.005 [-0.006, +0.016] | +0.006 [+0.001, +0.011] | -2.1 [-4.4, +0.1] | -0.008 [-0.019, +0.004] |
| `k8_StaticDiversity_vs_Random50` | -0.12 [-0.60, +0.33] | -0.002 [-0.011, +0.007] | -0.002 [-0.006, +0.003] | +0.4 [-1.6, +2.4] | +0.010 [-0.000, +0.020] |
| `k8_StaticDOpt_vs_Random50` | -2.67 [-3.16, -2.19] | +0.002 [-0.007, +0.011] | +0.006 [+0.001, +0.011] | -1.1 [-2.9, +0.6] | +0.047 [+0.037, +0.057] |
| `k8_AdaptiveOED_vs_Random50` | -2.42 [-2.95, -1.84] | +0.014 [+0.004, +0.024] | +0.013 [+0.008, +0.017] | -2.2 [-3.9, -0.4] | +0.039 [+0.029, +0.050] |
| `k8_AdaptiveOED_vs_StaticDOpt` | +0.25 [-0.21, +0.75] | +0.012 [+0.001, +0.024] | +0.006 [+0.001, +0.011] | -1.1 [-3.2, +0.9] | -0.008 [-0.019, +0.003] |
| `k12_StaticDiversity_vs_Random50` | +0.08 [-0.45, +0.54] | -0.007 [-0.017, +0.002] | -0.002 [-0.007, +0.003] | +0.3 [-2.0, +2.4] | +0.007 [-0.003, +0.016] |
| `k12_StaticDOpt_vs_Random50` | -2.47 [-2.94, -1.94] | +0.006 [-0.004, +0.016] | +0.007 [+0.002, +0.011] | -1.4 [-2.9, +0.3] | +0.040 [+0.030, +0.050] |
| `k12_AdaptiveOED_vs_Random50` | -2.30 [-2.84, -1.78] | +0.015 [+0.004, +0.025] | +0.011 [+0.006, +0.016] | -2.1 [-3.8, -0.3] | +0.031 [+0.021, +0.041] |
| `k12_AdaptiveOED_vs_StaticDOpt` | +0.17 [-0.28, +0.62] | +0.009 [-0.003, +0.020] | +0.004 [-0.001, +0.009] | -0.7 [-3.2, +1.6] | -0.009 [-0.020, +0.001] |

---

## 2. Zamansal OOD Rejimi — `Clean Waves1+2 → Waves3+4`

| $k$ | Politika / Referans | `n_eval` | `RMSE` ↓ | `Res R²` ↑ | $\rho^{\text{res}}_{\text{pat}}$ ↑ | `NDCG@3` ↑ | `Recall@3` ↑ | `Regret@1` ↓ | `Regret@3` ↓ | Ek Bilgi |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `M0(0)` (ref) | 146 | 62.19 | +0.000 | +0.000 | 0.797 | 0.499 | 54.7 | 14.2 | Panel seçimi yok |
| 0 | **`M1` (omik ref)** | 146 | 55.66 | +0.205 | +0.388 | 0.864 | 0.562 | 29.0 | 5.2 | Panel seçimi yok |
| 1 | **Random50** | 146 | 54.71 | +0.234 | +0.395 | 0.863 | 0.556 | 29.5 | 6.4 | — |
| 1 | **StaticDiversity** | 146 | 52.80 | +0.287 | +0.426 | 0.864 | 0.566 | 29.5 | 5.6 | direct=93% (tam=93%) |
| 1 | **StaticDOpt** | 146 | 53.23 | +0.277 | +0.424 | 0.863 | 0.566 | 31.9 | 5.6 | direct=93% (tam=93%) |
| 1 | **AdaptiveOED** | 146 | 53.91 | +0.256 | +0.412 | 0.864 | 0.564 | 31.4 | 6.2 | uniq=8, Jacc=0.43 |
| 2 | **Random50** | 146 | 54.11 | +0.252 | +0.403 | 0.863 | 0.553 | 29.4 | 6.5 | — |
| 2 | **StaticDiversity** | 146 | 52.39 | +0.298 | +0.427 | 0.865 | 0.575 | 31.2 | 5.3 | direct=95% (tam=91%) |
| 2 | **StaticDOpt** | 146 | 52.92 | +0.286 | +0.425 | 0.863 | 0.575 | 31.4 | 5.4 | direct=48% (tam=1%) |
| 2 | **AdaptiveOED** | 146 | 52.95 | +0.283 | +0.426 | 0.866 | 0.553 | 32.7 | 6.2 | uniq=14, Jacc=0.41 |
| 3 | **Random50** | 146 | 53.42 | +0.271 | +0.413 | 0.864 | 0.555 | 28.9 | 6.6 | — |
| 3 | **StaticDiversity** | 146 | 51.81 | +0.312 | +0.436 | 0.865 | 0.578 | 32.3 | 4.1 | direct=96% (tam=90%) |
| 3 | **StaticDOpt** | 146 | 51.57 | +0.326 | +0.458 | 0.871 | 0.583 | 30.2 | 4.7 | direct=63% (tam=0%) |
| 3 | **AdaptiveOED** | 146 | 52.68 | +0.291 | +0.433 | 0.865 | 0.559 | 27.2 | 6.5 | uniq=27, Jacc=0.45 |
| 5 | **Random50** | 146 | 52.62 | +0.294 | +0.421 | 0.864 | 0.553 | 29.4 | 7.0 | — |
| 5 | **StaticDiversity** | 146 | 51.73 | +0.317 | +0.454 | 0.862 | 0.562 | 34.3 | 5.2 | direct=77% (tam=0%) |
| 5 | **StaticDOpt** | 146 | 50.81 | +0.345 | +0.472 | 0.871 | 0.580 | 28.9 | 5.2 | direct=76% (tam=0%) |
| 5 | **AdaptiveOED** | 146 | 51.30 | +0.330 | +0.442 | 0.859 | 0.555 | 29.6 | 6.9 | uniq=49, Jacc=0.38 |
| 8 | **Random50** | 146 | 51.60 | +0.322 | +0.441 | 0.867 | 0.558 | 28.1 | 6.5 | — |
| 8 | **StaticDiversity** | 146 | 51.62 | +0.320 | +0.454 | 0.863 | 0.573 | 30.0 | 7.6 | direct=84% (tam=0%) |
| 8 | **StaticDOpt** | 146 | 50.56 | +0.352 | +0.480 | 0.879 | 0.579 | 24.6 | 5.7 | direct=71% (tam=0%) |
| 8 | **AdaptiveOED** | 146 | 49.61 | +0.373 | +0.466 | 0.868 | 0.567 | 29.1 | 6.3 | uniq=100, Jacc=0.38 |
| 12 | **Random50** | 146 | 50.65 | +0.347 | +0.459 | 0.872 | 0.564 | 26.7 | 6.2 | — |
| 12 | **StaticDiversity** | 146 | 50.30 | +0.354 | +0.464 | 0.874 | 0.577 | 26.8 | 5.6 | direct=88% (tam=0%) |
| 12 | **StaticDOpt** | 146 | 49.88 | +0.369 | +0.489 | 0.875 | 0.573 | 27.4 | 6.3 | direct=79% (tam=0%) |
| 12 | **AdaptiveOED** | 146 | 48.88 | +0.393 | +0.483 | 0.871 | 0.556 | 25.9 | 5.3 | uniq=128, Jacc=0.46 |

### Zamansal OOD Eşleştirilmiş Bootstrap Farkları (%95 GA)

| Karşılaştırma | `ΔRMSE` (negatif = iyi) | `ΔRecall@3` | `ΔNDCG@3` | `ΔRegret@1` (negatif = iyi) | `Δ(ρ^res)` |
|---|---:|---:|---:|---:|---:|
| `k1_StaticDOpt_vs_Random50` | -1.49 [-2.27, -0.70] | +0.011 [-0.009, +0.031] | +0.001 [-0.009, +0.009] | +2.4 [-2.6, +8.3] | +0.029 [+0.011, +0.047] |
| `k1_AdaptiveOED_vs_Random50` | -0.81 [-1.80, +0.24] | +0.008 [-0.016, +0.032] | +0.001 [-0.011, +0.013] | +2.0 [-3.7, +8.2] | +0.017 [-0.005, +0.040] |
| `k1_AdaptiveOED_vs_StaticDOpt` | +0.68 [-0.38, +1.80] | -0.002 [-0.030, +0.025] | +0.000 [-0.014, +0.014] | -0.5 [-8.3, +6.6] | -0.012 [-0.037, +0.014] |
| `k2_StaticDOpt_vs_Random50` | -1.18 [-2.42, -0.02] | +0.022 [+0.001, +0.043] | +0.000 [-0.011, +0.011] | +2.0 [-3.9, +8.2] | +0.022 [-0.002, +0.046] |
| `k2_AdaptiveOED_vs_Random50` | -1.16 [-2.12, -0.13] | -0.001 [-0.025, +0.023] | +0.003 [-0.010, +0.015] | +3.3 [-2.8, +10.2] | +0.024 [+0.002, +0.045] |
| `k2_AdaptiveOED_vs_StaticDOpt` | +0.03 [-1.35, +1.50] | -0.023 [-0.053, +0.009] | +0.002 [-0.014, +0.018] | +1.3 [-6.9, +9.2] | +0.002 [-0.028, +0.034] |
| `k3_StaticDOpt_vs_Random50` | -1.85 [-2.96, -0.79] | +0.029 [+0.006, +0.051] | +0.007 [-0.004, +0.018] | +1.3 [-3.9, +6.9] | +0.045 [+0.020, +0.072] |
| `k3_AdaptiveOED_vs_Random50` | -0.74 [-1.81, +0.47] | +0.005 [-0.017, +0.026] | +0.001 [-0.012, +0.013] | -1.7 [-6.8, +3.6] | +0.020 [-0.005, +0.042] |
| `k3_AdaptiveOED_vs_StaticDOpt` | +1.12 [-0.10, +2.53] | -0.024 [-0.049, +0.002] | -0.006 [-0.019, +0.007] | -3.0 [-9.3, +3.3] | -0.025 [-0.054, +0.000] |
| `k5_StaticDOpt_vs_Random50` | -1.81 [-2.94, -0.65] | +0.027 [+0.005, +0.049] | +0.007 [-0.005, +0.018] | -0.5 [-5.4, +5.1] | +0.051 [+0.029, +0.074] |
| `k5_AdaptiveOED_vs_Random50` | -1.32 [-2.29, -0.36] | +0.002 [-0.024, +0.024] | -0.005 [-0.018, +0.007] | +0.2 [-5.3, +5.3] | +0.021 [+0.001, +0.040] |
| `k5_AdaptiveOED_vs_StaticDOpt` | +0.49 [-0.68, +1.62] | -0.025 [-0.055, +0.002] | -0.012 [-0.025, +0.001] | +0.7 [-6.8, +6.6] | -0.030 [-0.051, -0.007] |
| `k8_StaticDOpt_vs_Random50` | -1.05 [-2.35, +0.30] | +0.021 [-0.000, +0.042] | +0.012 [+0.003, +0.020] | -3.5 [-8.3, +1.3] | +0.039 [+0.018, +0.061] |
| `k8_AdaptiveOED_vs_Random50` | -1.99 [-2.92, -1.10] | +0.009 [-0.017, +0.033] | +0.001 [-0.011, +0.012] | +1.0 [-4.5, +7.2] | +0.025 [+0.006, +0.046] |
| `k8_AdaptiveOED_vs_StaticDOpt` | -0.94 [-2.76, +0.68] | -0.012 [-0.043, +0.018] | -0.011 [-0.022, +0.000] | +4.5 [-2.5, +11.8] | -0.014 [-0.035, +0.008] |
| `k12_StaticDOpt_vs_Random50` | -0.76 [-2.02, +0.76] | +0.009 [-0.012, +0.029] | +0.004 [-0.006, +0.013] | +0.7 [-3.8, +5.3] | +0.030 [+0.014, +0.046] |
| `k12_AdaptiveOED_vs_Random50` | -1.77 [-2.74, -0.77] | -0.009 [-0.030, +0.013] | -0.001 [-0.012, +0.009] | -0.8 [-6.0, +4.7] | +0.023 [+0.004, +0.043] |
| `k12_AdaptiveOED_vs_StaticDOpt` | -1.00 [-3.11, +0.71] | -0.017 [-0.046, +0.008] | -0.005 [-0.018, +0.007] | -1.5 [-9.3, +7.0] | -0.007 [-0.027, +0.014] |

### Zamansal OOD `Random 50x` Yüzdelik Kıyaslaması & Tam-Panel (Sıfır Geri-Dolgu) Alt-Grubu

| $k$ | `Random50` Medyan RMSE [P10–P90] | `DOpt` Random Panellerin % Kaçını Geçiyor (RMSE / NDCG@3) | `Adaptive` RMSE Geçiş % | Tam-Panel (`DOpt`) Hasta Sayısı (RMSE: DOpt vs Rand50) |
|---:|---:|---:|---:|---|
| 1 | 54.67 [54.22–55.36] | %100 / %56 | %98 | — |
| 2 | 54.13 [53.42–54.77] | %100 / %50 | %100 | — |
| 3 | 53.49 [52.84–53.95] | %100 / %90 | %92 | — |
| 5 | 52.77 [51.82–53.39] | %100 / %90 | %96 | — |
| 8 | 51.62 [50.92–52.13] | %100 / %96 | %100 | — |
| 12 | 50.64 [50.01–51.18] | %98 / %72 | %100 | — |

---

## 3. `M3` Mekanizma Ablasyonları ($k \in \{3, 5, 8\}$)

| Rejim | Panel | $k$ | Model Koşulu | `RMSE` ↓ | `Res R²` ↑ | $\rho^{\text{res}}_{\text{pat}}$ ↑ | `Recall@3` ↑ | `Regret@1` ↓ | `ΔRMSE (M3_full - Koşul)` [%95 GA] |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| CV (3 Seed) | StaticDOpt | 3 | `M3_full` | 43.74 | +0.395 | +0.495 | 0.603 | 24.3 | referans |
| CV (3 Seed) | StaticDOpt | 3 | `M3_no_omics` | 47.78 | +0.273 | +0.337 | 0.541 | 35.7 | -4.04 [-4.91, -3.17] |
| CV (3 Seed) | StaticDOpt | 3 | `M3_perm_within_patient` | 49.11 | +0.245 | +0.386 | 0.567 | 30.5 | -5.37 [-6.51, -4.44] |
| CV (3 Seed) | StaticDOpt | 3 | `M3_perm_across_patients` | 54.37 | +0.079 | +0.358 | 0.548 | 30.2 | -10.63 [-11.87, -9.46] |
| CV (3 Seed) | StaticDOpt | 3 | `M3_perm_train_cov` | 49.26 | +0.242 | +0.414 | 0.576 | 27.2 | -5.52 [-6.49, -4.55] |
| CV (3 Seed) | StaticDOpt | 5 | `M3_full` | 42.74 | +0.422 | +0.511 | 0.604 | 23.8 | referans |
| CV (3 Seed) | StaticDOpt | 5 | `M3_no_omics` | 46.27 | +0.318 | +0.387 | 0.552 | 33.8 | -3.53 [-4.28, -2.77] |
| CV (3 Seed) | StaticDOpt | 5 | `M3_perm_within_patient` | 48.95 | +0.247 | +0.370 | 0.567 | 30.1 | -6.21 [-7.16, -5.27] |
| CV (3 Seed) | StaticDOpt | 5 | `M3_perm_across_patients` | 54.53 | +0.069 | +0.346 | 0.546 | 32.3 | -11.78 [-13.13, -10.63] |
| CV (3 Seed) | StaticDOpt | 5 | `M3_perm_train_cov` | 49.24 | +0.242 | +0.414 | 0.575 | 26.8 | -6.50 [-7.55, -5.54] |
| CV (3 Seed) | StaticDOpt | 8 | `M3_full` | 41.24 | +0.462 | +0.533 | 0.604 | 22.1 | referans |
| CV (3 Seed) | StaticDOpt | 8 | `M3_no_omics` | 43.90 | +0.388 | +0.448 | 0.568 | 29.3 | -2.66 [-3.33, -2.04] |
| CV (3 Seed) | StaticDOpt | 8 | `M3_perm_within_patient` | 48.93 | +0.247 | +0.358 | 0.550 | 30.9 | -7.69 [-8.88, -6.63] |
| CV (3 Seed) | StaticDOpt | 8 | `M3_perm_across_patients` | 55.67 | +0.029 | +0.335 | 0.540 | 32.6 | -14.43 [-15.71, -13.16] |
| CV (3 Seed) | StaticDOpt | 8 | `M3_perm_train_cov` | 49.26 | +0.241 | +0.413 | 0.575 | 27.1 | -8.02 [-9.24, -6.88] |
| CV (3 Seed) | Random50 | 3 | `M3_full` | 46.29 | +0.327 | +0.450 | 0.591 | 25.2 | referans |
| CV (3 Seed) | Random50 | 3 | `M3_no_omics` | 51.25 | +0.168 | +0.254 | 0.529 | 40.1 | -4.95 [-5.74, -4.21] |
| CV (3 Seed) | Random50 | 3 | `M3_perm_within_patient` | 48.07 | +0.276 | +0.410 | 0.577 | 27.7 | -1.77 [-2.00, -1.56] |
| CV (3 Seed) | Random50 | 3 | `M3_perm_across_patients` | 52.18 | +0.149 | +0.374 | 0.560 | 29.8 | -5.88 [-6.47, -5.32] |
| CV (3 Seed) | Random50 | 3 | `M3_perm_train_cov` | 49.28 | +0.241 | +0.412 | 0.576 | 27.0 | -2.99 [-3.62, -2.38] |
| CV (3 Seed) | Random50 | 5 | `M3_full` | 45.11 | +0.360 | +0.466 | 0.596 | 24.0 | referans |
| CV (3 Seed) | Random50 | 5 | `M3_no_omics` | 49.40 | +0.226 | +0.312 | 0.541 | 37.2 | -4.28 [-5.01, -3.60] |
| CV (3 Seed) | Random50 | 5 | `M3_perm_within_patient` | 47.99 | +0.279 | +0.402 | 0.575 | 28.1 | -2.87 [-3.26, -2.50] |
| CV (3 Seed) | Random50 | 5 | `M3_perm_across_patients` | 52.72 | +0.131 | +0.365 | 0.556 | 30.6 | -7.60 [-8.35, -6.94] |
| CV (3 Seed) | Random50 | 5 | `M3_perm_train_cov` | 49.29 | +0.241 | +0.412 | 0.576 | 27.1 | -4.17 [-4.93, -3.45] |
| CV (3 Seed) | Random50 | 8 | `M3_full` | 43.91 | +0.393 | +0.486 | 0.602 | 23.2 | referans |
| CV (3 Seed) | Random50 | 8 | `M3_no_omics` | 47.41 | +0.287 | +0.371 | 0.556 | 33.6 | -3.50 [-4.14, -2.88] |
| CV (3 Seed) | Random50 | 8 | `M3_perm_within_patient` | 48.13 | +0.274 | +0.390 | 0.571 | 29.0 | -4.21 [-4.76, -3.68] |
| CV (3 Seed) | Random50 | 8 | `M3_perm_across_patients` | 53.48 | +0.104 | +0.355 | 0.552 | 31.3 | -9.57 [-10.44, -8.78] |
| CV (3 Seed) | Random50 | 8 | `M3_perm_train_cov` | 49.30 | +0.241 | +0.412 | 0.576 | 27.0 | -5.38 [-6.30, -4.53] |
| Waves3+4 OOD | StaticDOpt | 3 | `M3_full` | 51.57 | +0.326 | +0.458 | 0.583 | 30.2 | referans |
| Waves3+4 OOD | StaticDOpt | 3 | `M3_no_omics` | 56.08 | +0.201 | +0.337 | 0.497 | 35.2 | -4.52 [-5.74, -3.38] |
| Waves3+4 OOD | StaticDOpt | 3 | `M3_perm_within_patient` | 57.27 | +0.155 | +0.317 | 0.537 | 33.2 | -5.71 [-7.89, -3.47] |
| Waves3+4 OOD | StaticDOpt | 3 | `M3_perm_across_patients` | 59.88 | +0.075 | +0.337 | 0.527 | 32.2 | -8.32 [-10.61, -5.91] |
| Waves3+4 OOD | StaticDOpt | 3 | `M3_perm_train_cov` | 55.63 | +0.206 | +0.389 | 0.568 | 29.5 | -4.07 [-5.72, -2.34] |
| Waves3+4 OOD | StaticDOpt | 5 | `M3_full` | 50.81 | +0.345 | +0.472 | 0.580 | 28.9 | referans |
| Waves3+4 OOD | StaticDOpt | 5 | `M3_no_omics` | 54.73 | +0.240 | +0.362 | 0.522 | 36.1 | -3.92 [-5.06, -2.83] |
| Waves3+4 OOD | StaticDOpt | 5 | `M3_perm_within_patient` | 57.85 | +0.142 | +0.291 | 0.540 | 38.1 | -7.04 [-8.90, -5.29] |
| Waves3+4 OOD | StaticDOpt | 5 | `M3_perm_across_patients` | 61.21 | +0.030 | +0.321 | 0.511 | 34.2 | -10.40 [-12.73, -7.97] |
| Waves3+4 OOD | StaticDOpt | 5 | `M3_perm_train_cov` | 55.70 | +0.204 | +0.389 | 0.571 | 28.5 | -4.89 [-6.57, -3.01] |
| Waves3+4 OOD | StaticDOpt | 8 | `M3_full` | 50.56 | +0.352 | +0.480 | 0.579 | 24.6 | referans |
| Waves3+4 OOD | StaticDOpt | 8 | `M3_no_omics` | 54.23 | +0.253 | +0.383 | 0.533 | 36.3 | -3.67 [-4.75, -2.69] |
| Waves3+4 OOD | StaticDOpt | 8 | `M3_perm_within_patient` | 56.93 | +0.172 | +0.322 | 0.516 | 35.0 | -6.38 [-8.12, -4.70] |
| Waves3+4 OOD | StaticDOpt | 8 | `M3_perm_across_patients` | 60.58 | +0.055 | +0.308 | 0.495 | 48.8 | -10.03 [-12.25, -7.58] |
| Waves3+4 OOD | StaticDOpt | 8 | `M3_perm_train_cov` | 55.79 | +0.202 | +0.386 | 0.562 | 32.6 | -5.23 [-7.03, -3.34] |
| Waves3+4 OOD | Random50 | 3 | `M3_full` | 53.42 | +0.271 | +0.413 | 0.555 | 28.9 | referans |
| Waves3+4 OOD | Random50 | 3 | `M3_no_omics` | 58.16 | +0.131 | +0.233 | 0.495 | 45.2 | -4.74 [-6.02, -3.55] |
| Waves3+4 OOD | Random50 | 3 | `M3_perm_within_patient` | 54.91 | +0.229 | +0.375 | 0.541 | 31.8 | -1.49 [-1.83, -1.15] |
| Waves3+4 OOD | Random50 | 3 | `M3_perm_across_patients` | 58.70 | +0.115 | +0.335 | 0.527 | 35.5 | -5.28 [-6.24, -4.23] |
| Waves3+4 OOD | Random50 | 3 | `M3_perm_train_cov` | 55.67 | +0.205 | +0.388 | 0.563 | 29.5 | -2.25 [-3.17, -1.27] |
| Waves3+4 OOD | Random50 | 5 | `M3_full` | 52.62 | +0.294 | +0.421 | 0.553 | 29.4 | referans |
| Waves3+4 OOD | Random50 | 5 | `M3_no_omics` | 56.64 | +0.178 | +0.281 | 0.502 | 42.7 | -4.01 [-5.22, -2.88] |
| Waves3+4 OOD | Random50 | 5 | `M3_perm_within_patient` | 55.16 | +0.222 | +0.360 | 0.530 | 33.2 | -2.53 [-2.92, -2.15] |
| Waves3+4 OOD | Random50 | 5 | `M3_perm_across_patients` | 59.71 | +0.083 | +0.321 | 0.514 | 36.5 | -7.09 [-8.21, -5.92] |
| Waves3+4 OOD | Random50 | 5 | `M3_perm_train_cov` | 55.67 | +0.205 | +0.388 | 0.562 | 29.2 | -3.04 [-4.11, -1.88] |
| Waves3+4 OOD | Random50 | 8 | `M3_full` | 51.60 | +0.322 | +0.441 | 0.558 | 28.1 | referans |
| Waves3+4 OOD | Random50 | 8 | `M3_no_omics` | 55.04 | +0.225 | +0.336 | 0.516 | 38.6 | -3.44 [-4.56, -2.37] |
| Waves3+4 OOD | Random50 | 8 | `M3_perm_within_patient` | 55.23 | +0.222 | +0.353 | 0.524 | 34.8 | -3.63 [-4.14, -3.11] |
| Waves3+4 OOD | Random50 | 8 | `M3_perm_across_patients` | 59.86 | +0.077 | +0.315 | 0.509 | 36.9 | -8.26 [-9.42, -7.02] |
| Waves3+4 OOD | Random50 | 8 | `M3_perm_train_cov` | 55.68 | +0.204 | +0.387 | 0.561 | 28.8 | -4.08 [-5.33, -2.83] |

---

## 4. `Waves1+2` Eğitiminden Öğrenilen İlk-12 Statik Mini-Panel

| Sıra | `Static-D-Optimal` İnhibitörü | Aile / Sınıf | `Static-Diversity` İnhibitörü | Aile / Sınıf |
|---:|---|---|---|---|
| 1 | **Sunitinib** | CAMK: Calcium/calmodulin-dependent protein kinases; Death-associated kinase (DAPK) family; Type III RTKs: PDGFR, CSFR, Kit, FLT3 receptor family; Type IV RTKs: VEGF (vascular endothelial growth factor) receptor family; Type XIV RTKs: RET | Sunitinib | CAMK: Calcium/calmodulin-dependent protein kinases; Death-associated kinase (DAPK) family; Type III RTKs: PDGFR, CSFR, Kit, FLT3 receptor family; Type IV RTKs: VEGF (vascular endothelial growth factor) receptor family; Type XIV RTKs: RET |
| 2 | **ABT-737** | B-cell lymphoma 2 (Bcl-2) protein family | SNS-032 (BMS-387032) | CMGC: Containing CDK, MAPK, GSK3, CLK families; Cyclin-dependent kinase (CDK) family; Cyclin-dependent kinase-like (CDKL) family |
| 3 | **Bay 11-7085** | Unlisted / Non-kinase | Vismodegib (GDC-0449) | Class Frizzled GPCRs; G protein-coupled receptors |
| 4 | **Flavopiridol** | CMGC: Containing CDK, MAPK, GSK3, CLK families; Cyclin-dependent kinase (CDK) family; Cyclin-dependent kinase-like (CDKL) family; RCK family | XAV-939 | Unlisted / Non-kinase |
| 5 | **INK-128** | Phosphatidyl inositol 3' kinase-related kinases (PIKK) family | JNJ-38877605 | Unlisted / Non-kinase |
| 6 | **DBZ** | Unlisted / Non-kinase | Elesclomol | Unlisted / Non-kinase |
| 7 | **Tivozanib (AV-951)** | Type IV RTKs: VEGF (vascular endothelial growth factor) receptor family | OTX-015 | Unlisted / Non-kinase |
| 8 | **XAV-939** | Unlisted / Non-kinase | Cediranib (AZD2171) | Type III RTKs: PDGFR, CSFR, Kit, FLT3 receptor family; Type IV RTKs: VEGF (vascular endothelial growth factor) receptor family; Type XVI RTKs: DDR (collagen receptor) family |
| 9 | **Roscovitine (CYC-202)** | AGC: Containing PKA, PKG, PKC families; CMGC: Containing CDK, MAPK, GSK3, CLK families; Cyclin-dependent kinase (CDK) family; Protein kinase C (PKC) family | JNJ-7706621 | CMGC: Containing CDK, MAPK, GSK3, CLK families; Cyclin-dependent kinase (CDK) family; Janus kinase (JakA) family |
| 10 | **Vargetef** | Src family; Type III RTKs: PDGFR, CSFR, Kit, FLT3 receptor family; Type IV RTKs: VEGF (vascular endothelial growth factor) receptor family | BMS-345541 | AGC: Containing PKA, PKG, PKC families; CAMK: Calcium/calmodulin-dependent protein kinases; CMGC: Containing CDK, MAPK, GSK3, CLK families; Cyclin-dependent kinase (CDK) family; DCAMKL family; Dual-specificity tyrosine-(Y)-phosphorylation regulated kinase (DYRK) family; IKK family; IRE family; Janus kinase (JakA) family; Mitogen-activated protein kinases (MAP kinases); Myosin Light Chain Kinase (MLCK) family; RSK family; STE20 family; STE: Homologs of yeast Sterile 7, Sterile 11, Sterile 20 kinases |
| 11 | **CYT387** | Janus kinase (JakA) family | CI-1040 (PD184352) | Aurora kinase (Aur) family; CAMK2 family; CAMK: Calcium/calmodulin-dependent protein kinases; STE7 family; STE: Homologs of yeast Sterile 7, Sterile 11, Sterile 20 kinases; Type III RTKs: PDGFR, CSFR, Kit, FLT3 receptor family |
| 12 | **GDC-0941** | Glycerophospholipid turnover; Phosphatidylinositol kinases | NF-kB Activation Inhibitor | IKK family |

---

## 5. Ardışık Bütçe Bilgi Değeri (VoI — `Static-D-Optimal` & `Adaptive-OED`, CV)

| Adım | `Static-D-Optimal` `ΔRMSE` [%95 GA] | `Adaptive-OED` `ΔRMSE` [%95 GA] |
|---|---:|---:|
| `k=1 → k=2` | -1.04 [-1.47, -0.59] | -2.21 [-2.76, -1.69] |
| `k=2 → k=3` | -0.70 [-1.04, -0.40] | -0.98 [-1.34, -0.68] |
| `k=3 → k=5` | -1.00 [-1.33, -0.67] | -1.22 [-1.50, -0.94] |
| `k=5 → k=8` | -1.50 [-1.94, -1.10] | -1.49 [-1.97, -1.06] |
| `k=8 → k=12` | -1.00 [-1.30, -0.70] | -1.08 [-1.48, -0.73] |

---

## 6. Önceden Kayıtlı Kapı 2 Kararı

- **Kapı 2 Sonucu:** **`PASS`**
- **`M3` Ablasyon Geçerliliği:** GEÇTİ
- **`Random 50x`'i Geçen Politikalar ($k \in {3,5,8}$):** `['StaticDiversity(k=3)', 'StaticDOpt(k=3)', 'AdaptiveOED(k=3)', 'StaticDiversity(k=5)', 'StaticDOpt(k=5)', 'AdaptiveOED(k=5)', 'StaticDOpt(k=8)', 'AdaptiveOED(k=8)']`
- **Sabit vs Adaptif Kararı:** **`ADAPTIVE_WINS`**
- **Doygunluk Eşiği ($k^*$):** **`k = 2`**
