# Sprint 4 v2 — Hasta-Birimli Split-Conformal Kalibrasyon, Kısa Aday Listesi ve OOD Çekinme (`DEV-OPUS-13..16` / `SOL-09..12`)

- **Çalışma Süresi:** `48.6 s`
- **Protokol & v2 Sürüm Notu:** [PROTOCOL_SPRINT04.md](../PROTOCOL_SPRINT04.md) korunmuştur; tarihsel v1 kayıtları (`sprint04_results.json`, [sprint-04-conformal-abstain.md](sprint-04-conformal-abstain.md)) değiştirilmemiştir. Bu v2 koşusu `v2 (DEV-OPUS-13..16 / SOL-09..12)` düzeltmelerini içerir:
  - **`SOL-09` (`DEV-OPUS-13`):** Fold başına tek kilitli contender_seed (kalibrasyon ve test aynı sabit seçici fonksiyon).
  - **`SOL-10` (`DEV-OPUS-14`):** Kilitli senaryoda ortak tam-panel hasta maskesinde eşleştirilmiş kıyas (n<5 -> hesaplanamadı) ve bağ-kesen cap etiketi.
  - **`SOL-11` (`DEV-OPUS-15`):** Sonlu aralık yayımlanmadığında (q=inf) kapsam ve kısa liste alanları NaN/hesaplanamadı.
  - **`SOL-12` (`DEV-OPUS-16`):** PROTOCOL_SPRINT04.md §1.2 gereği M0(0) ek taban kolu v2'ye eklendi (v1 sapması kapatıldı).
  - **`DEV-OPUS-19` Durumu:** DEV-OPUS-19 (PROTOCOL_SPRINT04_ADDENDUM_19.md) Sol A1 netleştirmesi bekliyor; v2'de uygulanmadı.
- **Sabitler:** Seed `2026`, nominal $\alpha = 0.10$ → %90, doğrulama kabul hedefi %80.
- **Kapsam & Etiketleme Beyanı:**
  - `id_60_20_20`: Beat AML 2.0'ın önceki sprintlerde toplu analize giren indeks hastalarının 60/20/20 hasta-bazlı ayrımıdır (*yeniden kullanılan kohort* kapsamıyla sınırlıdır).
  - `wave_ood` (`Waves1+2 → Waves3+4`): `Waves3+4` Sprint 2–3'te incelendiği için tüm `Waves3+4` sonuçları **keşifsel OOD stres testi** olarak etiketlenmiştir; OOD üzerinde konformal kapsam garantisi iddia edilmez.
  - **Plaka Kataloğu Durumu:** Beat AML 2.0 içinde bağımsız belgelenmiş prospektif plaka-kataloğu manifesti bulunmadığından *belgelenmiş plaka kataloğu* senaryosu **`hesaplanamadı`** olarak kaydedilmiştir. `locked_train_catalog` yalnızca `train_tune` kataloğuna dayalı ayrı simülasyon olarak raporlanır.
  - **Kabul Edilen Alt Grup Sınırı:** Çekinme eşiği altında kabul edilen alt grup için ayrı teorik kapsam garantisi verilmez; kabul edilen eşzamanlı kapsam yalnızca tanısaldır.
  - **Literatür Konumlandırması:** Shokrzadeh & Shokrzadeh (bioRxiv Haziran 2026, `2026.06.07.730728v1`) ilaç-bazlı marjinal split-conformal makro kapsam raporlamaktadır; OncoTwin ise sabit prob bütçesinde **hasta-birimli eşzamanlı kapsama**, **kısa aday listesi daraltması** ve **OOD çekinme (`AURC`)** ölçmektedir (karşılaştırma için ilaç-bazlı marjinal sütunu ayrı verilmiştir).

---

## 1. ID 60/20/20 (`id_60_20_20`, Yeniden Kullanılan Kohort) — `archive_available` Ana Senaryo

| Kol | `n_eval` | `RMSE` ↓ | `NDCG@3` ↑ | `Regret@1` ↓ | Sonlu Aralık | $2\hat{q}$ | Eşzamanlı Kapsam (%90 hed.) | İlaç Ampirik Kapsam | İlaç Marjinal (Shokrzadeh stili) | Kısa Liste (gözl.) | Gerçek 1. Listede | Kesin-1 Oranı |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | 91 | 54.49 | 0.782 | 46.9 | ✅ | 303.7 | %84.6 | %98.6 | %91.5 | 19.00 | %100.0 | %0.0 |
| `M1_k0` | 91 | 47.13 | 0.867 | 21.2 | ✅ | 290.1 | %91.2 | %99.4 | %91.8 | 19.00 | %100.0 | %0.0 |
| `StaticDOpt_k2` | 91 | 41.92 | 0.884 | 17.9 | ✅ | 287.4 | %93.4 | %99.3 | %92.4 | 19.00 | %100.0 | %0.0 |
| `EqualWeightVarRed_k2` | 91 | 41.64 | 0.882 | 18.8 | ✅ | 287.4 | %94.5 | %99.6 | %92.4 | 19.00 | %100.0 | %0.0 |
| `AdaptiveOED_k2` | 91 | 42.75 | 0.875 | 15.3 | ✅ | 289.4 | %96.7 | %99.7 | %92.2 | 19.00 | %100.0 | %0.0 |
| `StaticDOpt_k5` | 91 | 39.84 | 0.887 | 13.7 | ✅ | 241.8 | %86.8 | %98.9 | %92.4 | 18.70 | %100.0 | %0.0 |
| `EqualWeightVarRed_k5` | 91 | 39.54 | 0.888 | 20.1 | ✅ | 258.5 | %91.2 | %99.3 | %91.7 | 18.98 | %100.0 | %0.0 |
| `AdaptiveOED_k5` | 91 | 39.55 | 0.889 | 13.5 | ✅ | 264.9 | %95.6 | %99.7 | %92.8 | 18.87 | %100.0 | %0.0 |
| `StaticDOpt_k8` | 91 | 38.67 | 0.896 | 15.6 | ✅ | 234.1 | %87.9 | %99.0 | %92.3 | 18.59 | %100.0 | %0.0 |
| `EqualWeightVarRed_k8` | 91 | 38.12 | 0.898 | 14.0 | ✅ | 237.9 | %89.0 | %99.2 | %91.9 | 18.60 | %100.0 | %0.0 |
| `AdaptiveOED_k8` | 91 | 38.77 | 0.895 | 14.3 | ✅ | 253.4 | %93.4 | %99.6 | %91.3 | 18.81 | %100.0 | %0.0 |

### 1.1 ID `archive_available` Eşleştirilmiş Bootstrap Kıyasları (Seed 2026, B=1000)

| Karşılaştırma | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `ΔKısa Liste Boyutu` |
|---|---:|---:|---:|---:|
| `k2_AdaptiveOED_vs_EqualWeightVarRed` | +1.11 [+0.07, +2.28] (p=0.036, n=91) | -3.56 [-8.34, +1.09] (p=0.158, n=91) | -0.007 [-0.021, +0.006] (p=0.256, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k2_AdaptiveOED_vs_StaticDOpt` | +0.82 [-0.32, +2.12] (p=0.182, n=91) | -2.61 [-7.04, +1.80] (p=0.258, n=91) | -0.009 [-0.023, +0.004] (p=0.164, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k2_EqualWeightVarRed_vs_StaticDOpt` | -0.28 [-0.73, +0.06] (p=0.250, n=91) | +0.95 [-0.07, +2.90] (p=0.751, n=91) | -0.002 [-0.004, +0.000] (p=0.100, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |
| `k5_AdaptiveOED_vs_EqualWeightVarRed` | +0.01 [-0.98, +0.93] (p=0.973, n=91) | -6.58 [-13.19, -1.50] (p=0.010, n=91) | +0.000 [-0.014, +0.016] (p=0.915, n=91) | -0.11 [-0.30, +0.01] (p=0.168, n=91) |
| `k5_AdaptiveOED_vs_StaticDOpt` | -0.29 [-1.44, +0.86] (p=0.647, n=91) | -0.16 [-5.21, +4.86] (p=0.941, n=91) | +0.002 [-0.014, +0.016] (p=0.845, n=91) | +0.16 [-0.04, +0.42] (p=0.132, n=91) |
| `k5_EqualWeightVarRed_vs_StaticDOpt` | -0.30 [-1.13, +0.68] (p=0.545, n=91) | +6.42 [+0.84, +13.05] (p=0.026, n=91) | +0.001 [-0.013, +0.014] (p=0.893, n=91) | +0.27 [+0.05, +0.56] (p=0.006, n=91) |
| `k8_AdaptiveOED_vs_EqualWeightVarRed` | +0.65 [-0.25, +1.54] (p=0.140, n=91) | +0.30 [-3.02, +3.98] (p=0.901, n=91) | -0.003 [-0.016, +0.007] (p=0.585, n=91) | +0.21 [+0.01, +0.45] (p=0.040, n=91) |
| `k8_AdaptiveOED_vs_StaticDOpt` | +0.10 [-0.96, +1.18] (p=0.799, n=91) | -1.32 [-5.57, +3.11] (p=0.555, n=91) | -0.001 [-0.016, +0.015] (p=0.867, n=91) | +0.22 [+0.00, +0.47] (p=0.058, n=91) |
| `k8_EqualWeightVarRed_vs_StaticDOpt` | -0.56 [-1.31, +0.24] (p=0.178, n=91) | -1.62 [-7.64, +4.39] (p=0.607, n=91) | +0.002 [-0.009, +0.014] (p=0.725, n=91) | +0.01 [-0.19, +0.24] (p=1.000, n=91) |
| `StaticDOpt_k5_vs_M1_k0` | -7.29 [-9.45, -5.42] (p=0.002, n=91) | -7.46 [-13.51, -1.19] (p=0.020, n=91) | +0.020 [+0.003, +0.037] (p=0.022, n=91) | -0.30 [-0.58, -0.07] (p=0.006, n=91) |
| `StaticDOpt_k5_vs_M0_k0` | -14.65 [-16.74, -12.51] (p=0.002, n=91) | -33.17 [-47.27, -19.40] (p=0.002, n=91) | +0.105 [+0.074, +0.137] (p=0.002, n=91) | -0.30 [-0.58, -0.07] (p=0.006, n=91) |
| `M1_k0_vs_M0_k0` | -7.36 [-9.55, -5.01] (p=0.002, n=91) | -25.71 [-39.19, -13.00] (p=0.002, n=91) | +0.085 [+0.056, +0.116] (p=0.002, n=91) | +0.00 [+0.00, +0.00] (p=1.000, n=91) |

### 1.2 ID `archive_available` Seçici Risk–Kapsam (`AURC`) ve Kilitli %80 Eşik Çekinme Özeti

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 43.6 | 53.8 | 35.47 | 27.25 | 46.88 |
| `M0_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 46.9 | hesaplanamadı | 46.88 | 38.66 | 46.88 |
| `M1_k0` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 20.0 | 23.7 | 15.10 | 11.86 | 21.17 |
| `M1_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 21.2 | hesaplanamadı | 21.17 | 17.93 | 21.17 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 16.2 | 21.5 | 11.27 | 8.77 | 17.90 |
| `StaticDOpt_k2` | `m3_post_scale` | 48.736 | %95.6 / %95.6 | 17.8 | 20.2 | 17.83 | 15.33 | 17.90 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 16.2 | 24.6 | 11.50 | 8.79 | 18.84 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 48.705 | %100.0 / %100.0 | 18.8 | hesaplanamadı | 17.88 | 15.17 | 18.84 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 14.2 | 17.5 | 10.94 | 8.90 | 15.28 |
| `AdaptiveOED_k2` | `m3_post_scale` | 49.387 | %83.5 / %83.5 | 13.9 | 22.3 | 14.44 | 12.40 | 15.28 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 11.5 | 18.4 | 8.83 | 7.14 | 13.71 |
| `StaticDOpt_k5` | `m3_post_scale` | 48.243 | %100.0 / %100.0 | 13.7 | hesaplanamadı | 9.46 | 7.78 | 13.71 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 17.3 | 26.1 | 13.70 | 11.03 | 20.12 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 47.717 | %100.0 / %100.0 | 20.1 | hesaplanamadı | 15.58 | 12.91 | 20.12 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 13.2 | 14.3 | 12.82 | 10.93 | 13.54 |
| `AdaptiveOED_k5` | `m3_post_scale` | 48.342 | %100.0 / %100.0 | 13.5 | hesaplanamadı | 11.04 | 9.15 | 13.54 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 12.6 | 22.0 | 11.73 | 9.58 | 15.63 |
| `StaticDOpt_k8` | `m3_post_scale` | 46.829 | %100.0 / %100.0 | 15.6 | hesaplanamadı | 12.88 | 10.73 | 15.63 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 12.2 | 17.8 | 11.30 | 9.45 | 14.01 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 46.612 | %97.8 / %97.8 | 14.0 | 15.8 | 11.95 | 10.10 | 14.01 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.975 | %68.1 / %68.1 | 13.3 | 16.4 | 12.33 | 10.45 | 14.31 |
| `AdaptiveOED_k8` | `m3_post_scale` | 47.107 | %100.0 / %100.0 | 14.3 | hesaplanamadı | 15.50 | 13.61 | 14.31 |

**ID `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları (`omics_pca_dist` vs karşılaştırıcılar):**

- `StaticDOpt_k2_omics_vs_m3_post_scale`: `-6.56 [-10.72, -2.13] (p=0.006, n=91)`
- `StaticDOpt_k2_omics_vs_random`: `-6.63 [-10.39, -2.51] (p=0.002, n=91)`
- `StaticDOpt_k5_omics_vs_m3_post_scale`: `-0.64 [-4.06, +2.93] (p=0.719, n=91)`
- `StaticDOpt_k5_omics_vs_random`: `-4.88 [-8.31, -1.22] (p=0.006, n=91)`
- `StaticDOpt_k8_omics_vs_m3_post_scale`: `-1.15 [-5.62, +3.21] (p=0.617, n=91)`
- `StaticDOpt_k8_omics_vs_random`: `-3.91 [-9.09, +0.85] (p=0.112, n=91)`
- `AdaptiveOED_k5_omics_vs_m3_post_scale`: `+1.78 [-3.05, +7.21] (p=0.472, n=91)`
- `AdaptiveOED_k5_omics_vs_random`: `-0.72 [-3.99, +3.13] (p=0.723, n=91)`

---

## 2. ID `locked_train_catalog` (Salt Eğitim Kataloğu Simülasyonu — Geri Doldurmasız)

> **Not:** Aşağıdaki tablodaki her satır o kolun **kendi tam-panel hastaları** üzerinde betimleyicidir; farklı hasta kümelerine sahip satırlar politika üstünlüğü kıyası için kullanılmaz. Eşleştirilmiş kıyaslar §2.2'dedir.

| Kol | Hedef-Uygun `n` | Tam Panel `n` | Bulunabilirlik | Teknik Çekinme | `RMSE` (tam-panel) | `Regret@1` (tam-panel) | Eşzamanlı Kapsam (`cal n`) | Kısa Liste (gözl.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | 93 | 93 | %100.0 | %0.0 | 54.61 | 47.9 | %84.9 (n_cal=92) | 18.76 |
| `M1_k0` | 93 | 93 | %100.0 | %0.0 | 47.15 | 20.8 | %92.5 (n_cal=92) | 18.76 |
| `StaticDOpt_k2` | 93 | 83 | %89.2 | %10.8 | 40.44 | 17.8 | %95.2 (n_cal=80) | 19.33 |
| `EqualWeightVarRed_k2` | 93 | 83 | %89.2 | %10.8 | 40.44 | 17.8 | %95.2 (n_cal=80) | 19.33 |
| `AdaptiveOED_k2` | 93 | 83 | %89.2 | %10.8 | 40.77 | 14.7 | %97.6 (n_cal=80) | 19.31 |
| `StaticDOpt_k5` | 93 | 19 | %20.4 | %79.6 | 38.22 | 12.3 | %100.0 (n_cal=18) | 20.32 |
| `EqualWeightVarRed_k5` | 93 | 60 | %64.5 | %35.5 | 37.25 | 13.5 | %96.7 (n_cal=59) | 21.90 |
| `AdaptiveOED_k5` | 93 | 57 | %61.3 | %38.7 | 39.25 | 14.8 | %96.5 (n_cal=54) | 18.88 |
| `StaticDOpt_k8` | 93 | 19 | %20.4 | %79.6 | 36.86 | 13.0 | %100.0 (n_cal=17) | 20.32 |
| `EqualWeightVarRed_k8` | 93 | 56 | %60.2 | %39.8 | 35.83 | 10.8 | %89.3 (n_cal=52) | 21.59 |
| `AdaptiveOED_k8` | 93 | 40 | %43.0 | %57.0 | 36.73 | 17.0 | %100.0 (n_cal=39) | 18.95 |

### 2.1 ID `locked_train_catalog` Kendi Tam-Panel Alt Kümesinde Betimleyici Çekinme ve `AURC`

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `omics_pca_dist` | 0.978 | %67.7 / %67.7 | 45.9 | 52.3 | 37.48 | 29.00 | 47.94 |
| `M0_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 47.9 | hesaplanamadı | 47.94 | 39.46 | 47.94 |
| `M1_k0` | `omics_pca_dist` | 0.978 | %67.7 / %67.7 | 19.7 | 23.1 | 14.89 | 11.76 | 20.78 |
| `M1_k0` | `m3_post_scale` | 52.822 | %100.0 / %100.0 | 20.8 | hesaplanamadı | 20.78 | 17.65 | 20.78 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.974 | %67.5 / %60.2 | 15.2 | 23.1 | 10.80 | 8.37 | 17.81 |
| `StaticDOpt_k2` | `m3_post_scale` | 48.633 | %100.0 / %89.2 | 17.8 | hesaplanamadı | 17.81 | 15.38 | 17.81 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.975 | %67.5 / %60.2 | 15.2 | 23.1 | 10.80 | 8.37 | 17.81 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 48.705 | %100.0 / %89.2 | 17.8 | hesaplanamadı | 17.81 | 15.38 | 17.81 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.978 | %69.9 / %62.4 | 13.3 | 18.0 | 10.49 | 8.58 | 14.69 |
| `AdaptiveOED_k2` | `m3_post_scale` | 49.414 | %81.9 / %73.1 | 13.0 | 22.3 | 13.49 | 11.58 | 14.69 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.974 | %78.9 / %16.1 | 11.7 | 14.3 | 10.53 | 8.53 | 12.27 |
| `StaticDOpt_k5` | `m3_post_scale` | 48.251 | %100.0 / %20.4 | 12.3 | hesaplanamadı | 12.27 | 10.26 | 12.27 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.974 | %73.3 / %47.3 | 14.5 | 10.9 | 12.39 | 10.54 | 13.53 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 47.717 | %100.0 / %64.5 | 13.5 | hesaplanamadı | 13.53 | 11.68 | 13.53 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.975 | %77.2 / %47.3 | 15.4 | 12.8 | 15.22 | 12.75 | 14.82 |
| `AdaptiveOED_k5` | `m3_post_scale` | 48.450 | %100.0 / %61.3 | 14.8 | hesaplanamadı | 12.35 | 9.88 | 14.82 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.974 | %78.9 / %16.1 | 12.6 | 14.3 | 10.96 | 8.63 | 12.97 |
| `StaticDOpt_k8` | `m3_post_scale` | 46.755 | %100.0 / %20.4 | 13.0 | hesaplanamadı | 12.97 | 10.63 | 12.97 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.968 | %73.2 / %44.1 | 10.6 | 11.6 | 9.58 | 8.19 | 10.85 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 46.025 | %100.0 / %60.2 | 10.8 | hesaplanamadı | 10.85 | 9.45 | 10.85 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.981 | %77.5 / %33.3 | 17.1 | 16.5 | 16.24 | 13.08 | 16.95 |
| `AdaptiveOED_k8` | `m3_post_scale` | 47.110 | %100.0 / %43.0 | 17.0 | hesaplanamadı | 18.07 | 14.91 | 16.95 |

### 2.2 ID `locked_train_catalog` Ortak Tam-Panel Eşleştirilmiş Kıyaslar

**Ortak Tam-Panel Hasta Maskesinde Eşleştirilmiş Kıyaslar (`n_paired < 5 → hesaplanamadı`):**

| Karşılaştırma | `n_paired` | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `AURC(A)` / `AURC(B)` (ortak `n`) |
|---|---:|---:|---:|---:|---:|
| `k2_AdaptiveOED_vs_EqualWeightVarRed` | 80 | +1.44 [+0.31, +2.71] (p=0.012, n=80) | -3.23 [-8.50, +0.48] (p=0.130, n=80) | -0.014 [-0.026, -0.001] (p=0.028, n=80) | 9.52 / 10.98 |
| `k2_AdaptiveOED_vs_StaticDOpt` | 80 | +1.44 [+0.31, +2.71] (p=0.012, n=80) | -3.23 [-8.50, +0.48] (p=0.130, n=80) | -0.014 [-0.026, -0.001] (p=0.028, n=80) | 9.52 / 10.98 |
| `k2_EqualWeightVarRed_vs_StaticDOpt` | 83 | +0.00 [+0.00, +0.00] (p=1.000, n=83) | +0.00 [+0.00, +0.00] (p=1.000, n=83) | +0.000 [+0.000, +0.000] (p=1.000, n=83) | 10.80 / 10.80 |
| `k5_AdaptiveOED_vs_EqualWeightVarRed` | 39 | +1.15 [-0.25, +2.69] (p=0.116, n=39) | -1.03 [-3.53, +1.65] (p=0.470, n=39) | -0.003 [-0.020, +0.014] (p=0.667, n=39) | 19.75 / 19.60 |
| `k5_AdaptiveOED_vs_StaticDOpt` | 18 | +0.25 [-2.79, +3.17] (p=0.909, n=18) | +0.93 [+0.00, +2.55] (p=0.252, n=18) | -0.017 [-0.063, +0.031] (p=0.476, n=18) | 12.30 / 11.23 |
| `k5_EqualWeightVarRed_vs_StaticDOpt` | 18 | -2.05 [-4.38, +0.23] (p=0.078, n=18) | +1.54 [+0.00, +3.81] (p=0.266, n=18) | -0.014 [-0.039, +0.007] (p=0.236, n=18) | 15.65 / 14.18 |
| `k8_AdaptiveOED_vs_EqualWeightVarRed` | 27 | +1.58 [+0.49, +2.75] (p=0.006, n=27) | +0.08 [+0.00, +0.24] (p=0.691, n=27) | -0.002 [-0.011, +0.008] (p=0.745, n=27) | 19.36 / 19.28 |
| `k8_AdaptiveOED_vs_StaticDOpt` | 17 | +0.64 [-1.51, +2.80] (p=0.583, n=17) | +0.98 [+0.00, +2.70] (p=0.232, n=17) | -0.005 [-0.035, +0.032] (p=0.731, n=17) | 13.87 / 12.60 |
| `k8_EqualWeightVarRed_vs_StaticDOpt` | 18 | -1.72 [-3.33, -0.24] (p=0.032, n=18) | +0.81 [+0.00, +2.43] (p=0.737, n=18) | +0.000 [-0.026, +0.030] (p=0.989, n=18) | 15.65 / 14.67 |

**Üçlü Ortak Tam-Panel Kesişimi (`n_paired < 5 → hesaplanamadı`):**

| Bütçe | `n_paired` | `StaticDOpt` (RMSE / Reg1 / AURC) | `EqualWeightVarRed` (RMSE / Reg1 / AURC) | `AdaptiveOED` (RMSE / Reg1 / AURC) |
|---|---:|---:|---:|---:|
| `k2` | 80 | 39.26 / 17.0 / 10.98 | 39.26 / 17.0 / 10.98 | 40.70 / 13.7 / 9.52 |
| `k5` | 17 | 37.87 / 13.7 / 15.17 | 35.82 / 15.3 / 16.89 | 38.28 / 14.7 / 16.44 |
| `k8` | 16 | 37.11 / 15.4 / 16.99 | 35.44 / 16.3 / 18.39 | 37.50 / 16.4 / 18.52 |

---

## 3. Temiz `Waves1+2 → Waves3+4` Keşifsel OOD Stres Testi (`archive_available` & `locked_train_catalog`)

> **Not:** Aşağıdaki tüm `Waves3+4` değerleri keşifsel stres testi kapsamındadır; `Waves3+4` hastaları eğitim, eşik kilitleme veya kalibrasyona girmemiştir.

### 3.1 `Waves3+4` — `archive_available` Konformal, Kısa Liste ve Eşzamanlı Kapsam Kayması

| Kol | `n_eval` | `RMSE` ↓ | `NDCG@3` ↑ | `Regret@1` ↓ | Sonlu Aralık | $2\hat{q}$ | Eşzamanlı Kapsam (%90 hed.) | İlaç Ampirik Kapsam | İlaç Marjinal (Shokrzadeh stili) | Kısa Liste (gözl.) | Gerçek 1. Listede | Kesin-1 Oranı |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | 148 | 62.05 | 0.801 | 54.6 | ✅ | 288.6 | %71.6 | %97.1 | %87.5 | 21.72 | %100.0 | %0.0 |
| `M1_k0` | 148 | 56.18 | 0.863 | 33.3 | ✅ | 278.9 | %78.4 | %97.8 | %86.6 | 21.72 | %100.0 | %0.0 |
| `StaticDOpt_k2` | 148 | 53.46 | 0.866 | 33.4 | ✅ | 270.7 | %75.7 | %97.4 | %86.1 | 21.69 | %100.0 | %0.0 |
| `EqualWeightVarRed_k2` | 148 | 51.78 | 0.864 | 32.3 | ✅ | 255.7 | %75.0 | %97.2 | %86.2 | 21.49 | %100.0 | %0.0 |
| `AdaptiveOED_k2` | 148 | 54.21 | 0.860 | 28.2 | ✅ | 263.7 | %77.0 | %97.4 | %86.9 | 21.50 | %100.0 | %0.0 |
| `StaticDOpt_k5` | 148 | 51.99 | 0.873 | 24.6 | ✅ | 247.4 | %74.3 | %97.2 | %86.6 | 21.41 | %100.0 | %0.0 |
| `EqualWeightVarRed_k5` | 148 | 50.81 | 0.876 | 23.1 | ✅ | 254.2 | %78.4 | %97.6 | %86.4 | 21.36 | %100.0 | %0.0 |
| `AdaptiveOED_k5` | 148 | 52.25 | 0.872 | 24.7 | ✅ | 261.3 | %77.0 | %97.5 | %86.9 | 21.56 | %100.0 | %0.0 |
| `StaticDOpt_k8` | 148 | 50.17 | 0.876 | 23.1 | ✅ | 244.6 | %75.7 | %97.6 | %87.1 | 21.28 | %100.0 | %0.0 |
| `EqualWeightVarRed_k8` | 148 | 50.35 | 0.874 | 25.2 | ✅ | 247.9 | %76.4 | %97.3 | %86.6 | 21.36 | %100.0 | %0.0 |
| `AdaptiveOED_k8` | 148 | 50.71 | 0.872 | 22.8 | ✅ | 256.9 | %77.7 | %97.7 | %87.5 | 21.45 | %100.0 | %0.0 |

### 3.2 `Waves3+4` — `archive_available` Kilitli %80 Eşik Çekinme ve `AURC`

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 52.7 | 59.8 | 53.95 | 44.67 | 54.59 |
| `M0_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 54.6 | hesaplanamadı | 54.59 | 45.31 | 54.59 |
| `M1_k0` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 36.2 | 25.4 | 30.47 | 25.12 | 33.32 |
| `M1_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 33.3 | hesaplanamadı | 33.32 | 27.97 | 33.32 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 37.7 | 21.7 | 34.63 | 29.51 | 33.37 |
| `StaticDOpt_k2` | `m3_post_scale` | 45.010 | %93.9 / %93.9 | 31.3 | 65.1 | 28.84 | 23.72 | 33.37 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 36.6 | 20.8 | 38.40 | 33.82 | 32.32 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 44.186 | %100.0 / %100.0 | 32.3 | hesaplanamadı | 29.47 | 24.89 | 32.32 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 29.7 | 24.2 | 31.82 | 27.85 | 28.24 |
| `AdaptiveOED_k2` | `m3_post_scale` | 45.379 | %84.5 / %84.5 | 30.0 | 18.5 | 30.21 | 26.24 | 28.24 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 28.8 | 13.3 | 32.44 | 29.06 | 24.60 |
| `StaticDOpt_k5` | `m3_post_scale` | 42.689 | %98.6 / %98.6 | 24.6 | 26.7 | 20.35 | 16.97 | 24.60 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 25.5 | 16.4 | 27.73 | 24.76 | 23.05 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 42.741 | %98.6 / %98.6 | 23.0 | 26.7 | 18.75 | 15.78 | 23.05 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 25.8 | 21.7 | 27.20 | 23.73 | 24.71 |
| `AdaptiveOED_k5` | `m3_post_scale` | 42.784 | %99.3 / %99.3 | 24.5 | 53.3 | 24.11 | 20.63 | 24.71 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 25.7 | 16.0 | 27.17 | 23.97 | 23.09 |
| `StaticDOpt_k8` | `m3_post_scale` | 43.254 | %98.6 / %98.6 | 23.0 | 26.7 | 20.61 | 17.41 | 23.09 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 27.4 | 19.1 | 28.46 | 24.98 | 25.15 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 43.121 | %98.6 / %98.6 | 25.1 | 26.7 | 22.03 | 18.55 | 25.15 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.966 | %73.0 / %73.0 | 23.4 | 21.2 | 19.19 | 16.23 | 22.82 |
| `AdaptiveOED_k8` | `m3_post_scale` | 43.321 | %100.0 / %100.0 | 22.8 | hesaplanamadı | 21.08 | 18.12 | 22.82 |

**`Waves3+4` `AURC(Regret@1)` Eşleştirilmiş Bootstrap Farkları:**

- `StaticDOpt_k2_omics_vs_m3_post_scale`: `+5.78 [-2.62, +15.43] (p=0.176, n=148)`
- `StaticDOpt_k2_omics_vs_random`: `+1.26 [-5.67, +9.58] (p=0.789, n=148)`
- `StaticDOpt_k5_omics_vs_m3_post_scale`: `+12.09 [+2.12, +22.12] (p=0.024, n=148)`
- `StaticDOpt_k5_omics_vs_random`: `+7.84 [+0.14, +14.99] (p=0.046, n=148)`
- `StaticDOpt_k8_omics_vs_m3_post_scale`: `+6.57 [-3.90, +16.79] (p=0.252, n=148)`
- `StaticDOpt_k8_omics_vs_random`: `+4.09 [-3.53, +11.32] (p=0.340, n=148)`
- `AdaptiveOED_k5_omics_vs_m3_post_scale`: `+3.10 [-7.44, +13.20] (p=0.607, n=148)`
- `AdaptiveOED_k5_omics_vs_random`: `+2.50 [-5.40, +9.97] (p=0.607, n=148)`

### 3.3 `Waves3+4` — `locked_train_catalog` Bulunabilirlik ve Tam-Panel Durumu

| Kol | Hedef-Uygun `n` | Tam Panel `n` | Bulunabilirlik | Teknik Çekinme | `RMSE` (tam-panel) | `Regret@1` (tam-panel) | Eşzamanlı Kapsam (`cal n`) | Kısa Liste (gözl.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | 163 | 163 | %100.0 | %0.0 | 62.38 | 56.3 | %71.8 (n_cal=71) | 20.37 |
| `M1_k0` | 163 | 163 | %100.0 | %0.0 | 56.48 | 32.4 | %79.1 (n_cal=71) | 20.37 |
| `StaticDOpt_k2` | 163 | 3 | %1.8 | %98.2 | 50.04 | 63.7 | %100.0 (n_cal=51) | 9.67 |
| `EqualWeightVarRed_k2` | 163 | 131 | %80.4 | %19.6 | 52.30 | 29.7 | %75.6 (n_cal=62) | 21.91 |
| `AdaptiveOED_k2` | 163 | 55 | %33.7 | %66.3 | 54.82 | 29.5 | %69.1 (n_cal=62) | 21.62 |
| `StaticDOpt_k5` | 163 | 1 | %0.6 | %99.4 | 44.89 | 78.0 | %100.0 (n_cal=33) | 11.00 |
| `EqualWeightVarRed_k5` | 163 | 19 | %11.7 | %88.3 | 36.00 | 9.0 | %100.0 (n_cal=50) | 20.63 |
| `AdaptiveOED_k5` | 163 | 21 | %12.9 | %87.1 | 49.12 | 10.9 | %85.7 (n_cal=55) | 21.29 |
| `StaticDOpt_k8` | 163 | 1 | %0.6 | %99.4 | 42.97 | 78.0 | %100.0 (n_cal=33) | 11.00 |
| `EqualWeightVarRed_k8` | 163 | 16 | %9.8 | %90.2 | 35.12 | 6.1 | %100.0 (n_cal=35) | 22.62 |
| `AdaptiveOED_k8` | 163 | 1 | %0.6 | %99.4 | 29.85 | 0.0 | %100.0 (n_cal=47) | 24.00 |

### 3.4 `Waves3+4` — `locked_train_catalog` Kendi Tam-Panelinde Betimleyici Risk–Kapsam ve Ortak Kıyas

| Kol | Skor | Kilitli $\tau_{0.80}$ | Kabul (mevc./toplam) | Kabul `Regret@1` | Çek. `Regret@1` | `AURC` (`Regret@1`) ↓ | `E-AURC` ↓ | `Random AURC` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `omics_pca_dist` | 0.966 | %72.4 / %72.4 | 55.1 | 59.4 | 55.33 | 45.40 | 56.29 |
| `M0_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 56.3 | hesaplanamadı | 56.29 | 46.37 | 56.29 |
| `M1_k0` | `omics_pca_dist` | 0.966 | %72.4 / %72.4 | 34.4 | 27.2 | 29.18 | 24.13 | 32.44 |
| `M1_k0` | `m3_post_scale` | 49.105 | %100.0 / %100.0 | 32.4 | hesaplanamadı | 32.44 | 27.39 | 32.44 |
| `StaticDOpt_k2` | `omics_pca_dist` | 0.965 | %100.0 / %1.8 | 63.7 | hesaplanamadı | 40.92 | 7.52 | 63.71 |
| `StaticDOpt_k2` | `m3_post_scale` | 44.378 | %100.0 / %1.8 | 63.7 | hesaplanamadı | 63.71 | 30.30 | 63.71 |
| `EqualWeightVarRed_k2` | `omics_pca_dist` | 0.965 | %74.0 / %59.5 | 33.1 | 20.0 | 33.50 | 29.46 | 29.70 |
| `EqualWeightVarRed_k2` | `m3_post_scale` | 44.186 | %100.0 / %80.4 | 29.7 | hesaplanamadı | 29.70 | 25.67 | 29.70 |
| `AdaptiveOED_k2` | `omics_pca_dist` | 0.965 | %78.2 / %26.4 | 32.6 | 18.5 | 29.55 | 24.99 | 29.55 |
| `AdaptiveOED_k2` | `m3_post_scale` | 45.379 | %65.5 / %22.1 | 33.3 | 22.4 | 39.47 | 34.91 | 29.55 |
| `StaticDOpt_k5` | `omics_pca_dist` | 0.966 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 |
| `StaticDOpt_k5` | `m3_post_scale` | 41.967 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 |
| `EqualWeightVarRed_k5` | `omics_pca_dist` | 0.961 | %84.2 / %9.8 | 10.5 | 1.0 | 5.76 | 4.95 | 8.97 |
| `EqualWeightVarRed_k5` | `m3_post_scale` | 42.741 | %100.0 / %11.7 | 9.0 | hesaplanamadı | 8.97 | 8.16 | 8.97 |
| `AdaptiveOED_k5` | `omics_pca_dist` | 0.961 | %81.0 / %10.4 | 12.0 | 6.3 | 14.25 | 12.77 | 10.91 |
| `AdaptiveOED_k5` | `m3_post_scale` | 42.761 | %100.0 / %12.9 | 10.9 | hesaplanamadı | 10.92 | 9.44 | 10.91 |
| `StaticDOpt_k8` | `omics_pca_dist` | 0.966 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 |
| `StaticDOpt_k8` | `m3_post_scale` | 42.332 | %100.0 / %0.6 | 78.0 | hesaplanamadı | 77.98 | 0.00 | 77.98 |
| `EqualWeightVarRed_k8` | `omics_pca_dist` | 0.961 | %87.5 / %8.6 | 6.7 | 1.6 | 4.22 | 3.74 | 6.09 |
| `EqualWeightVarRed_k8` | `m3_post_scale` | 42.276 | %100.0 / %9.8 | 6.1 | hesaplanamadı | 6.09 | 5.61 | 6.09 |
| `AdaptiveOED_k8` | `omics_pca_dist` | 0.942 | %100.0 / %0.6 | 0.0 | hesaplanamadı | 0.00 | 0.00 | 0.00 |
| `AdaptiveOED_k8` | `m3_post_scale` | 43.321 | %100.0 / %0.6 | 0.0 | hesaplanamadı | 0.00 | 0.00 | 0.00 |

**Ortak Tam-Panel Hasta Maskesinde Eşleştirilmiş Kıyaslar (`n_paired < 5 → hesaplanamadı`):**

| Karşılaştırma | `n_paired` | `ΔRMSE` (neg=iyi) | `ΔRegret@1` (neg=iyi) | `ΔNDCG@3` (poz=iyi) | `AURC(A)` / `AURC(B)` (ortak `n`) |
|---|---:|---:|---:|---:|---:|
| `k2_AdaptiveOED_vs_EqualWeightVarRed` | 52 | +1.95 [-0.86, +4.36] (p=0.166, n=52) | -0.71 [-4.38, +2.76] (p=0.645, n=52) | -0.009 [-0.027, +0.010] (p=0.378, n=52) | 29.85 / 30.08 |
| `k2_AdaptiveOED_vs_StaticDOpt` | 3 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k2_EqualWeightVarRed_vs_StaticDOpt` | 3 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k5_AdaptiveOED_vs_EqualWeightVarRed` | 14 | +0.56 [-2.94, +4.29] (p=0.797, n=14) | +2.25 [+0.00, +6.74] (p=0.693, n=14) | -0.004 [-0.034, +0.018] (p=0.871, n=14) | 9.91 / 4.86 |
| `k5_AdaptiveOED_vs_StaticDOpt` | 1 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k5_EqualWeightVarRed_vs_StaticDOpt` | 1 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k8_AdaptiveOED_vs_EqualWeightVarRed` | 1 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k8_AdaptiveOED_vs_StaticDOpt` | 0 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k8_EqualWeightVarRed_vs_StaticDOpt` | 1 | hesaplanamadı | hesaplanamadı | hesaplanamadı | hesaplanamadı |

**Üçlü Ortak Tam-Panel Kesişimi (`n_paired < 5 → hesaplanamadı`):**

| Bütçe | `n_paired` | `StaticDOpt` (RMSE / Reg1 / AURC) | `EqualWeightVarRed` (RMSE / Reg1 / AURC) | `AdaptiveOED` (RMSE / Reg1 / AURC) |
|---|---:|---:|---:|---:|
| `k2` | 3 | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k5` | 1 | hesaplanamadı | hesaplanamadı | hesaplanamadı |
| `k8` | 0 | hesaplanamadı | hesaplanamadı | hesaplanamadı |
