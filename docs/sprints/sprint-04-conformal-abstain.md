# Sprint 4 — Hasta-Birimli Split-Conformal Kalibrasyon, Kısa Aday Listesi ve OOD Çekinme (`ABSTAIN` / `AURC`)

- **Çalışma Süresi:** `49.0 s`
- **Protokol:** [PROTOCOL_SPRINT04.md](../PROTOCOL_SPRINT04.md) (deneyden önce dondurulmuştur; seed `2026`, nominal $\alpha = 0.10$ → %90, doğrulama kabul hedefi %80).
- **Kapsam & Etiketleme Beyanı:**
  - `id_60_20_20`: Beat AML 2.0'ın önceki sprintlerde toplu analize giren indeks hastalarının 60/20/20 hasta-bazlı ayrımıdır (*yeniden kullanılan kohort* kapsamıyla sınırlıdır).
  - `wave_ood` (`Waves1+2 → Waves3+4`): `Waves3+4` Sprint 2–3'te incelendiği için tüm `Waves3+4` sonuçları **keşifsel OOD stres testi** olarak etiketlenmiştir; OOD üzerinde konformal kapsam garantisi iddia edilmez.
  - **Plaka Kataloğu Durumu:** Beat AML 2.0 içinde bağımsız belgelenmiş prospektif plaka-kataloğu manifesti bulunmadığından *belgelenmiş plaka kataloğu* senaryosu **`hesaplanamadı`** olarak kaydedilmiştir. Aşağıdaki `locked_train_catalog` kolonu yalnızca `train_tune` kataloğuna dayalı ayrı simülasyon olarak raporlanır.
  - **Kabul Edilen Alt Grup Sınırı:** Çekinme eşiği altında kabul edilen alt grup için ayrı teorik kapsam garantisi verilmez; kabul edilen eşzamanlı kapsam yalnızca tanısaldır.
  - **Literatür Konumlandırması:** Shokrzadeh & Shokrzadeh (bioRxiv Haziran 2026, `2026.06.07.730728v1`) ilaç-bazlı marjinal split-conformal makro kapsam raporlamaktadır; OncoTwin ise sabit prob bütçesinde **hasta-birimli eşzamanlı kapsama**, **kısa aday listesi daraltması** ve **OOD çekinme (`AURC`)** ölçmektedir (karşılaştırma için ilaç-bazlı marjinal sütunu ayrı verilmiştir).

---

## 1. ID 60/20/20 (`id_60_20_20`, Yeniden Kullanılan Kohort) — `archive_available` Ana Senaryo

| Kol | `n_eval` | `RMSE` ↓ | `NDCG@3` ↑ | `Regret@1` ↓ | Sonlu Aralık | $2\hat{q}$ | Eşzamanlı Kapsam (%90 hed.) | İlaç Ampirik Kapsam | İlaç Marjinal (Shokrzadeh stili) | Kısa Liste (gözl.) | Gerçek 1. Listede | Kesin-1 Oranı |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | 91 | 47.13 | 0.867 | 21.2 | ✅ | 290.1 | %91.2 | %99.4 | %91.8 | 19.00 | %100.0 | %0.0 |
| `StaticDOpt_k2` | 91 | 41.92 | 0.884 | 17.9 | ✅ | 287.4 | %93.4 | %99.3 | %92.4 | 19.00 | %100.0 | %0.0 |
| `EqualWeightVarRed_k2` | 91 | 41.64 | 0.882 | 18.8 | ✅ | 287.4 | %94.5 | %99.6 | %92.4 | 19.00 | %100.0 | %0.0 |
| `AdaptiveOED_k2` | 91 | 42.71 | 0.873 | 16.2 | ✅ | 289.4 | %96.7 | %99.7 | %92.2 | 19.00 | %100.0 | %0.0 |
| `StaticDOpt_k5` | 91 | 39.84 | 0.887 | 13.7 | ✅ | 241.8 | %86.8 | %98.9 | %92.4 | 18.70 | %100.0 | %0.0 |
| `EqualWeightVarRed_k5` | 91 | 39.54 | 0.888 | 20.1 | ✅ | 258.5 | %91.2 | %99.3 | %91.7 | 18.98 | %100.0 | %0.0 |
| `AdaptiveOED_k5` | 91 | 39.79 | 0.892 | 13.6 | ✅ | 264.9 | %94.5 | %99.6 | %92.4 | 18.87 | %100.0 | %0.0 |
| `StaticDOpt_k8` | 91 | 38.67 | 0.896 | 15.6 | ✅ | 234.1 | %87.9 | %99.0 | %92.3 | 18.59 | %100.0 | %0.0 |
| `EqualWeightVarRed_k8` | 91 | 38.12 | 0.898 | 14.0 | ✅ | 237.9 | %89.0 | %99.2 | %91.9 | 18.60 | %100.0 | %0.0 |
| `AdaptiveOED_k8` | 91 | 38.89 | 0.894 | 14.5 | ✅ | 253.4 | %93.4 | %99.5 | %91.6 | 18.84 | %100.0 | %0.0 |

### 1.1 ID `archive_available` Eşleştirilmiş Bootstrap Kıyasları (Seed 2026, B=1000)

| Karşılaştırma | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `ΔKısa Liste Boyutu` |
|---|---:|---:|---:|---:|
| `k2_AdaptiveOED_vs_EqualWeightVarRed` | +1.07 [+0.05, +2.26] (p=0.042, n=91) | -2.67 [-7.06, +1.73] (p=0.260, n=91) | -0.009 [-0.022, +0.005] (p=0.188, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k2_AdaptiveOED_vs_StaticDOpt` | +0.78 [-0.36, +2.12] (p=0.180, n=91) | -1.72 [-6.30, +3.20] (p=0.494, n=91) | -0.011 [-0.024, +0.003] (p=0.110, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k2_EqualWeightVarRed_vs_StaticDOpt` | -0.28 [-0.73, +0.06] (p=0.250, n=91) | +0.95 [-0.07, +2.90] (p=0.751, n=91) | -0.002 [-0.004, +0.000] (p=0.100, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k5_AdaptiveOED_vs_EqualWeightVarRed` | +0.24 [-0.84, +1.24] (p=0.681, n=91) | -6.51 [-13.01, -1.57] (p=0.004, n=91) | +0.003 [-0.011, +0.018] (p=0.613, n=91) | -0.11 [-0.30, +0.01] (p=0.170, n=91) |
| `k5_AdaptiveOED_vs_StaticDOpt` | -0.06 [-1.21, +1.07] (p=0.941, n=91) | -0.10 [-5.20, +4.94] (p=0.985, n=91) | +0.005 [-0.012, +0.020] (p=0.597, n=91) | +0.16 [-0.03, +0.42] (p=0.130, n=91) |
| `k5_EqualWeightVarRed_vs_StaticDOpt` | -0.30 [-1.13, +0.68] (p=0.545, n=91) | +6.42 [+0.84, +13.05] (p=0.026, n=91) | +0.001 [-0.013, +0.014] (p=0.893, n=91) | +0.27 [+0.05, +0.56] (p=0.006, n=91) |
| `k8_AdaptiveOED_vs_EqualWeightVarRed` | +0.77 [-0.07, +1.59] (p=0.078, n=91) | +0.53 [-2.83, +4.36] (p=0.791, n=91) | -0.004 [-0.018, +0.007] (p=0.577, n=91) | +0.23 [+0.03, +0.46] (p=0.024, n=91) |
| `k8_AdaptiveOED_vs_StaticDOpt` | +0.21 [-0.78, +1.13] (p=0.657, n=91) | -1.08 [-5.35, +3.35] (p=0.643, n=91) | -0.002 [-0.017, +0.014] (p=0.819, n=91) | +0.24 [+0.04, +0.48] (p=0.018, n=91) |
| `k8_EqualWeightVarRed_vs_StaticDOpt` | -0.56 [-1.31, +0.24] (p=0.178, n=91) | -1.62 [-7.64, +4.39] (p=0.607, n=91) | +0.002 [-0.009, +0.014] (p=0.725, n=91) | +0.01 [-0.19, +0.24] (p=1.000, n=91) |
| `StaticDOpt_k5_vs_M1_k0` | -7.29 [-9.45, -5.42] (p=0.002, n=91) | -7.46 [-13.51, -1.19] (p=0.020, n=91) | +0.020 [+0.003, +0.037] (p=0.022, n=91) | -0.30 [-0.58, -0.07] (p=0.006, n=91) |

### 1.2 ID `archive_available` Seçici Risk–Kapsam (`AURC`) ve Kilitli %80 Eşik Çekinme Özeti

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 20.0 | 23.7 | 15.10 | 11.86 | 21.17 |
| `M1_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 21.2 | hesaplanamadı | 21.17 | 17.93 | 21.17 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 16.2 | 21.5 | 11.27 | 8.77 | 17.90 |
| `StaticDOpt_k2` | `m3_post_scale` | 48.736 | %95.6 / %95.6 | 17.8 | 20.2 | 17.83 | 15.33 | 17.90 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 16.2 | 24.6 | 11.50 | 8.79 | 18.84 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 48.705 | %100.0 / %100.0 | 18.8 | hesaplanamadı | 17.88 | 15.17 | 18.84 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 14.2 | 20.3 | 11.18 | 8.98 | 16.17 |
| `AdaptiveOED_k2` | `m3_post_scale` | 49.253 | %79.1 / %79.1 | 15.7 | 17.9 | 14.03 | 11.84 | 16.17 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 11.5 | 18.4 | 8.83 | 7.14 | 13.71 |
| `StaticDOpt_k5` | `m3_post_scale` | 48.243 | %100.0 / %100.0 | 13.7 | hesaplanamadı | 9.46 | 7.78 | 13.71 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 17.3 | 26.1 | 13.70 | 11.03 | 20.12 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 47.717 | %100.0 / %100.0 | 20.1 | hesaplanamadı | 15.58 | 12.91 | 20.12 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 13.3 | 14.3 | 13.01 | 11.06 | 13.61 |
| `AdaptiveOED_k5` | `m3_post_scale` | 48.342 | %100.0 / %100.0 | 13.6 | hesaplanamadı | 13.19 | 11.24 | 13.61 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 12.6 | 22.0 | 11.73 | 9.58 | 15.63 |
| `StaticDOpt_k8` | `m3_post_scale` | 46.829 | %100.0 / %100.0 | 15.6 | hesaplanamadı | 12.88 | 10.73 | 15.63 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 12.2 | 17.8 | 11.30 | 9.45 | 14.01 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 46.612 | %97.8 / %97.8 | 14.0 | 15.8 | 11.95 | 10.10 | 14.01 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 13.1 | 17.7 | 12.23 | 10.29 | 14.55 |
| `AdaptiveOED_k8` | `m3_post_scale` | 47.107 | %100.0 / %100.0 | 14.5 | hesaplanamadı | 15.04 | 13.10 | 14.55 |

**ID `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları (`omics_pca_dist` vs karşılaştırıcılar):**

- `StaticDOpt_k2_omics_vs_m3_post_scale`: `-6.56 [-10.72, -2.13] (p=0.006, n=91)`
- `StaticDOpt_k2_omics_vs_random`: `-6.63 [-10.39, -2.51] (p=0.002, n=91)`
- `StaticDOpt_k5_omics_vs_m3_post_scale`: `-0.64 [-4.06, +2.93] (p=0.719, n=91)`
- `StaticDOpt_k5_omics_vs_random`: `-4.88 [-8.31, -1.22] (p=0.006, n=91)`
- `StaticDOpt_k8_omics_vs_m3_post_scale`: `-1.15 [-5.62, +3.21] (p=0.617, n=91)`
- `StaticDOpt_k8_omics_vs_random`: `-3.91 [-9.09, +0.85] (p=0.112, n=91)`
- `AdaptiveOED_k5_omics_vs_m3_post_scale`: `-0.18 [-7.20, +6.75] (p=0.933, n=91)`
- `AdaptiveOED_k5_omics_vs_random`: `-0.60 [-3.78, +3.31] (p=0.771, n=91)`

---

## 2. ID `locked_train_catalog` (Salt Eğitim Kataloğu Simülasyonu — Geri Doldurmasız)

| Kol | Hedef-Uygun `n` | Tam Panel `n` | Bulunabilirlik | Teknik Çekinme | `RMSE` (tam-panel) | `Regret@1` (tam-panel) | Eşzamanlı Kapsam (`cal n`) | Kısa Liste (gözl.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | 93 | 93 | %100.0 | %0.0 | 47.15 | 20.8 | %92.5 (n_cal=92) | 18.76 |
| `StaticDOpt_k2` | 93 | 83 | %89.2 | %10.8 | 40.44 | 17.8 | %95.2 (n_cal=80) | 19.33 |
| `EqualWeightVarRed_k2` | 93 | 83 | %89.2 | %10.8 | 40.44 | 17.8 | %95.2 (n_cal=80) | 19.33 |
| `AdaptiveOED_k2` | 93 | 82 | %88.2 | %11.8 | 40.55 | 13.3 | %97.6 (n_cal=80) | 19.28 |
| `StaticDOpt_k5` | 93 | 19 | %20.4 | %79.6 | 38.22 | 12.3 | %100.0 (n_cal=18) | 20.32 |
| `EqualWeightVarRed_k5` | 93 | 60 | %64.5 | %35.5 | 37.25 | 13.5 | %96.7 (n_cal=59) | 21.90 |
| `AdaptiveOED_k5` | 93 | 58 | %62.4 | %37.6 | 39.22 | 14.5 | %96.6 (n_cal=54) | 19.05 |
| `StaticDOpt_k8` | 93 | 19 | %20.4 | %79.6 | 36.86 | 13.0 | %100.0 (n_cal=17) | 20.32 |
| `EqualWeightVarRed_k8` | 93 | 56 | %60.2 | %39.8 | 35.83 | 10.8 | %89.3 (n_cal=52) | 21.59 |
| `AdaptiveOED_k8` | 93 | 41 | %44.1 | %55.9 | 37.02 | 17.8 | %100.0 (n_cal=39) | 18.90 |

### 2.1 ID `locked_train_catalog` OOD Çekinme ve `AURC` (Kapsam Tavanı Düzenli)

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` | Ortak-Cap AURC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | `omics_pca_dist` | 0.978 | %67.7 / %67.7 | 19.7 | 23.1 | 14.89 | 11.76 | 20.78 | 9.50 |
| `M1_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 20.8 | hesaplanamadı | 20.78 | 17.65 | 20.78 | 20.78 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.974 | %67.5 / %60.2 | 15.2 | 23.1 | 10.80 | 8.37 | 17.81 | 2.54 |
| `StaticDOpt_k2` | `m3_post_scale` | 48.633 | %100.0 / %89.2 | 17.8 | hesaplanamadı | 17.81 | 15.38 | 17.81 | 17.81 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.975 | %67.5 / %60.2 | 15.2 | 23.1 | 10.80 | 8.37 | 17.81 | 2.54 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 48.705 | %100.0 / %89.2 | 17.8 | hesaplanamadı | 17.81 | 15.38 | 17.81 | 17.81 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.975 | %69.5 / %61.3 | 11.4 | 17.7 | 9.13 | 7.42 | 13.33 | 2.27 |
| `AdaptiveOED_k2` | `m3_post_scale` | 49.288 | %79.3 / %69.9 | 13.4 | 13.0 | 11.78 | 10.07 | 13.33 | 8.42 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.974 | %78.9 / %16.1 | 11.7 | 14.3 | 10.53 | 8.53 | 12.27 | 10.53 |
| `StaticDOpt_k5` | `m3_post_scale` | 48.251 | %100.0 / %20.4 | 12.3 | hesaplanamadı | 12.27 | 10.26 | 12.27 | 12.27 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.974 | %73.3 / %47.3 | 14.5 | 10.9 | 12.39 | 10.54 | 13.53 | 7.63 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 47.717 | %100.0 / %64.5 | 13.5 | hesaplanamadı | 13.53 | 11.68 | 13.53 | 13.53 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.974 | %79.3 / %49.5 | 14.9 | 13.1 | 14.27 | 11.89 | 14.49 | 9.36 |
| `AdaptiveOED_k5` | `m3_post_scale` | 48.442 | %100.0 / %62.4 | 14.5 | hesaplanamadı | 10.68 | 8.30 | 14.49 | 3.42 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.974 | %78.9 / %16.1 | 12.6 | 14.3 | 10.96 | 8.63 | 12.97 | 10.96 |
| `StaticDOpt_k8` | `m3_post_scale` | 46.755 | %100.0 / %20.4 | 13.0 | hesaplanamadı | 12.97 | 10.63 | 12.97 | 12.97 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.968 | %73.2 / %44.1 | 10.6 | 11.6 | 9.58 | 8.19 | 10.85 | 6.98 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 46.025 | %100.0 / %60.2 | 10.8 | hesaplanamadı | 10.85 | 9.45 | 10.85 | 10.85 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.975 | %78.0 / %34.4 | 17.9 | 17.4 | 15.71 | 12.40 | 17.79 | 11.80 |
| `AdaptiveOED_k8` | `m3_post_scale` | 47.107 | %100.0 / %44.1 | 17.8 | hesaplanamadı | 18.04 | 14.73 | 17.79 | 16.17 |

---

## 3. Temiz `Waves1+2 → Waves3+4` Keşifsel OOD Stres Testi (`archive_available` & `locked_train_catalog`)

> **Not:** Aşağıdaki tüm `Waves3+4` değerleri keşifsel stres testi kapsamındadır; `Waves3+4` hastaları eğitim, eşik kilitleme veya kalibrasyona girmemiştir.

### 3.1 `Waves3+4` — `archive_available` Konformal, Kısa Liste ve Eşzamanlı Kapsam Kayması

| Kol | `n_eval` | `RMSE` ↓ | `NDCG@3` ↑ | `Regret@1` ↓ | Sonlu Aralık | $2\hat{q}$ | Eşzamanlı Kapsam (%90 hed.) | İlaç Ampirik Kapsam | İlaç Marjinal (Shokrzadeh stili) | Kısa Liste (gözl.) | Gerçek 1. Listede | Kesin-1 Oranı |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | 148 | 56.18 | 0.863 | 33.3 | ✅ | 278.9 | %78.4 | %97.8 | %86.6 | 21.72 | %100.0 | %0.0 |
| `StaticDOpt_k2` | 148 | 53.46 | 0.866 | 33.4 | ✅ | 270.7 | %75.7 | %97.4 | %86.1 | 21.69 | %100.0 | %0.0 |
| `EqualWeightVarRed_k2` | 148 | 51.78 | 0.864 | 32.3 | ✅ | 255.7 | %75.0 | %97.2 | %86.2 | 21.49 | %100.0 | %0.0 |
| `AdaptiveOED_k2` | 148 | 54.27 | 0.856 | 29.1 | ✅ | 263.7 | %76.4 | %97.4 | %86.3 | 21.53 | %100.0 | %0.0 |
| `StaticDOpt_k5` | 148 | 51.99 | 0.873 | 24.6 | ✅ | 247.4 | %74.3 | %97.2 | %86.6 | 21.41 | %100.0 | %0.0 |
| `EqualWeightVarRed_k5` | 148 | 50.81 | 0.876 | 23.1 | ✅ | 254.2 | %78.4 | %97.6 | %86.4 | 21.36 | %100.0 | %0.0 |
| `AdaptiveOED_k5` | 148 | 52.26 | 0.873 | 23.3 | ✅ | 261.3 | %77.0 | %97.5 | %86.7 | 21.55 | %100.0 | %0.0 |
| `StaticDOpt_k8` | 148 | 50.17 | 0.876 | 23.1 | ✅ | 244.6 | %75.7 | %97.6 | %87.1 | 21.28 | %100.0 | %0.0 |
| `EqualWeightVarRed_k8` | 148 | 50.35 | 0.874 | 25.2 | ✅ | 247.9 | %76.4 | %97.3 | %86.6 | 21.36 | %100.0 | %0.0 |
| `AdaptiveOED_k8` | 148 | 50.83 | 0.872 | 23.9 | ✅ | 256.9 | %77.0 | %97.6 | %87.3 | 21.47 | %100.0 | %0.0 |

### 3.2 `Waves3+4` — `archive_available` Kilitli %80 Eşik Çekinme ve `AURC`

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 36.2 | 25.4 | 30.47 | 25.12 | 33.32 |
| `M1_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 33.3 | hesaplanamadı | 33.32 | 27.97 | 33.32 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 37.7 | 21.7 | 34.63 | 29.51 | 33.37 |
| `StaticDOpt_k2` | `m3_post_scale` | 45.010 | %93.9 / %93.9 | 31.3 | 65.1 | 28.84 | 23.72 | 33.37 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 36.6 | 20.8 | 38.40 | 33.82 | 32.32 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 44.186 | %100.0 / %100.0 | 32.3 | hesaplanamadı | 29.47 | 24.89 | 32.32 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 31.3 | 23.1 | 34.63 | 30.57 | 29.10 |
| `AdaptiveOED_k2` | `m3_post_scale` | 45.379 | %85.1 / %85.1 | 30.8 | 19.6 | 31.10 | 27.04 | 29.10 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 28.8 | 13.3 | 32.44 | 29.06 | 24.60 |
| `StaticDOpt_k5` | `m3_post_scale` | 42.689 | %98.6 / %98.6 | 24.6 | 26.7 | 20.35 | 16.97 | 24.60 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 25.5 | 16.4 | 27.73 | 24.76 | 23.05 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 42.741 | %98.6 / %98.6 | 23.0 | 26.7 | 18.75 | 15.78 | 23.05 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 24.7 | 19.7 | 26.45 | 23.22 | 23.34 |
| `AdaptiveOED_k5` | `m3_post_scale` | 42.810 | %98.6 / %98.6 | 23.3 | 26.7 | 22.96 | 19.73 | 23.34 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 25.7 | 16.0 | 27.17 | 23.97 | 23.09 |
| `StaticDOpt_k8` | `m3_post_scale` | 43.254 | %98.6 / %98.6 | 23.0 | 26.7 | 20.61 | 17.41 | 23.09 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 27.4 | 19.1 | 28.46 | 24.98 | 25.15 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 43.121 | %98.6 / %98.6 | 25.1 | 26.7 | 22.03 | 18.55 | 25.15 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 23.8 | 24.2 | 19.70 | 16.54 | 23.89 |
| `AdaptiveOED_k8` | `m3_post_scale` | 43.373 | %100.0 / %100.0 | 23.9 | hesaplanamadı | 21.38 | 18.22 | 23.89 |

**`Waves3+4` `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları:**

- `StaticDOpt_k2_omics_vs_m3_post_scale`: `+5.78 [-2.62, +15.43] (p=0.176, n=148)`
- `StaticDOpt_k2_omics_vs_random`: `+1.26 [-5.67, +9.58] (p=0.789, n=148)`
- `StaticDOpt_k5_omics_vs_m3_post_scale`: `+12.09 [+2.12, +22.12] (p=0.024, n=148)`
- `StaticDOpt_k5_omics_vs_random`: `+7.84 [+0.14, +14.99] (p=0.046, n=148)`
- `StaticDOpt_k8_omics_vs_m3_post_scale`: `+6.57 [-3.90, +16.79] (p=0.252, n=148)`
- `StaticDOpt_k8_omics_vs_random`: `+4.09 [-3.53, +11.32] (p=0.340, n=148)`
- `AdaptiveOED_k5_omics_vs_m3_post_scale`: `+3.49 [-6.64, +13.62] (p=0.587, n=148)`
- `AdaptiveOED_k5_omics_vs_random`: `+3.11 [-4.70, +10.74] (p=0.478, n=148)`

### 3.3 `Waves3+4` — `locked_train_catalog` Bulunabilirlik ve Tam-Panel Durumu

| Kol | Hedef-Uygun `n` | Tam Panel `n` | Bulunabilirlik | Teknik Çekinme | `RMSE` (tam-panel) | `Regret@1` (tam-panel) | Eşzamanlı Kapsam (`cal n`) | Kısa Liste (gözl.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | 163 | 163 | %100.0 | %0.0 | 56.48 | 32.4 | %79.1 (n_cal=71) | 20.37 |
| `StaticDOpt_k2` | 163 | 3 | %1.8 | %98.2 | 50.04 | 63.7 | %100.0 (n_cal=51) | 9.67 |
| `EqualWeightVarRed_k2` | 163 | 131 | %80.4 | %19.6 | 52.30 | 29.7 | %75.6 (n_cal=62) | 21.91 |
| `AdaptiveOED_k2` | 163 | 52 | %31.9 | %68.1 | 55.27 | 34.0 | %69.2 (n_cal=62) | 21.54 |
| `StaticDOpt_k5` | 163 | 1 | %0.6 | %99.4 | 44.89 | 78.0 | %100.0 (n_cal=33) | 11.00 |
| `EqualWeightVarRed_k5` | 163 | 19 | %11.7 | %88.3 | 36.00 | 9.0 | %100.0 (n_cal=50) | 20.63 |
| `AdaptiveOED_k5` | 163 | 19 | %11.7 | %88.3 | 48.38 | 14.7 | %89.5 (n_cal=55) | 20.42 |
| `StaticDOpt_k8` | 163 | 1 | %0.6 | %99.4 | 42.97 | 78.0 | %100.0 (n_cal=33) | 11.00 |
| `EqualWeightVarRed_k8` | 163 | 16 | %9.8 | %90.2 | 35.12 | 6.1 | %100.0 (n_cal=35) | 22.62 |
| `AdaptiveOED_k8` | 163 | 1 | %0.6 | %99.4 | 29.85 | 0.0 | %100.0 (n_cal=47) | 24.00 |

### 3.4 `Waves3+4` — `locked_train_catalog` Risk–Kapsam Özeti

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` | Ortak-Cap AURC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M1_k0` | `omics_pca_dist` | 0.966 | %72.4 / %72.4 | 34.4 | 27.2 | 29.18 | 24.13 | 32.44 | 8.37 |
| `M1_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 32.4 | hesaplanamadı | 32.44 | 27.39 | 32.44 | 32.44 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.965 | %100.0 / %1.8 | 63.7 | hesaplanamadı | 40.92 | 7.52 | 63.71 | 0.00 |
| `StaticDOpt_k2` | `m3_post_scale` | 44.378 | %100.0 / %1.8 | 63.7 | hesaplanamadı | 63.71 | 30.30 | 63.71 | 63.71 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.965 | %74.0 / %59.5 | 33.1 | 20.0 | 33.50 | 29.46 | 29.70 | 131.12 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 44.186 | %100.0 / %80.4 | 29.7 | hesaplanamadı | 29.70 | 25.67 | 29.70 | 29.70 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.965 | %80.8 / %25.8 | 36.8 | 22.1 | 38.71 | 33.34 | 34.00 | 131.12 |
| `AdaptiveOED_k2` | `m3_post_scale` | 45.379 | %71.2 / %22.7 | 39.3 | 20.9 | 44.25 | 38.89 | 34.00 | 65.72 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.966 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 | 77.98 |
| `StaticDOpt_k5` | `m3_post_scale` | 41.967 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 | 77.98 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.961 | %84.2 / %9.8 | 10.5 | 1.0 | 5.76 | 4.95 | 8.97 | 0.00 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 42.741 | %100.0 / %11.7 | 9.0 | hesaplanamadı | 8.97 | 8.16 | 8.97 | 8.97 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.952 | %84.2 / %9.8 | 17.3 | 1.0 | 16.54 | 14.28 | 14.75 | 0.00 |
| `AdaptiveOED_k5` | `m3_post_scale` | 42.320 | %100.0 / %11.7 | 14.7 | hesaplanamadı | 14.01 | 11.75 | 14.75 | 13.52 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.966 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 | 77.98 |
| `StaticDOpt_k8` | `m3_post_scale` | 42.332 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 | 77.98 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.961 | %87.5 / %8.6 | 6.7 | 1.6 | 4.22 | 3.74 | 6.09 | 0.00 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 42.276 | %100.0 / %9.8 | 6.1 | hesaplanamadı | 6.09 | 5.61 | 6.09 | 6.09 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.939 | %100.0 / %0.6 | 0.0 | hesaplanamadı | 0.00 | 0.00 | 0.00 | 0.00 |
| `AdaptiveOED_k8` | `m3_post_scale` | 43.239 | %100.0 / %0.6 | 0.0 | hesaplanamadı | 0.00 | 0.00 | 0.00 | 0.00 |
