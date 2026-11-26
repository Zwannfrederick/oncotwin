# Sprint 5 — `M4` Yolak Kovaryansı (`Kapı 3`), Sıra-Kalibreli Konformal Kısa Liste ve FIMM/Malani Dış Tekrarı

- **Toplam Süre:** `244.1 s` (`Kulvar A: 188.7 s`, `Kulvar B: 13.0 s`, `Kulvar C: 39.5 s`)
- **Protokol:** [PROTOCOL_SPRINT05.md](../PROTOCOL_SPRINT05.md) deney koşusundan önce dondurulmuştur; tüm `sprint01..04` (`v1`, `v2`, `DEV-OPUS-19`) artefaktları korunmuştur.
- **Ön Kayıtlı Kapı 3 (`AD-001`) Hükmü:** **`NO-GO (v2 ODE ERTELENİR)`**
- **Bilimsel Sınırlar:** `Waves3+4` sonuçları yalnız keşifsel OOD stres testidir (yöntem/eşik seçimine girmez); FIMM `C1` (donmuş aktarım) ve `C2` (FIMM içi yöntem tekrarı) kesin biçimde ayrılmıştır; Beat AML konformal aralığı FIMM'e taşınmamıştır.

---

## 1. Kulvar A — `M4` Yolak / Hedef-Ailesi Yapılı Kovaryans & Kapı 3 (`AD-001`)

- **İlaç–Aile Grafı (`beataml_drug_families.xlsx`):** `117/166` anotasyonlu ilaç, `86` aile, `395` üyelik kenarı; derece & aile-boyutu korumalı permütasyon kontrolü `20` sabit çekiliş ortalaması.

### 1.1 Ön Kayıtlı Kapı 3 Değerlendirmesi (Deney A1, Sabit `StaticDOpt(M3)` Paneli, `k=5`)

| Kapı 3 Koşulu | Gözlenen Değer | Durum |
|---|---|---|
| 1. ID 5-Fold CV (`n=427`): `ΔRMSE(M4 − M3) ≤ −0.50`, GA`<0`, sıralama bozulmasız | `+0.01 [-0.01, +0.03] (p=0.294, n=427)` | `FAIL` |
| 2. ID 5-Fold CV (`n=427`): `ΔRMSE(M4 − M4_perm_null) ≤ −0.50`, GA`<0` | `+0.02 [-0.00, +0.04] (p=0.128, n=427)` | `FAIL` |
| 3. Keşifsel `Waves3+4` (`n=146`): `ΔRMSE(M4 − M3) < 0` ve `ΔRMSE(M4 − perm) < 0` | `vs M3: -0.02 [-0.05, +0.01] (p=0.156, n=146)` ; `vs Perm: -0.01 [-0.04, +0.02] (p=0.438, n=146)` | `PASS` |

### 1.2 Deney A1 (Birincil — Sabit `StaticDOpt(M3)` Paneli, Aynı Hastalar ve Hedefler)

| Rejim | `k` | `n` | `M3` RMSE / NDCG3 / Reg1 | `M4` (iç-CV $\rho\ge 0$) | `M4_forced` ($\rho>0$) | `M4_perm_null` (20×) | `%ρ>0` | `ΔRMSE (M4 − M3)` | `ΔRMSE (M4 − Perm)` | `ΔRMSE (Forced − PermF)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ID CV 5x3 (Birincil) | 5 | 427 | 42.91 / 0.877 / 23.3 | 42.92 / 0.876 / 23.4 | 42.92 / 0.876 / 23.4 | 42.90 / 0.877 / 23.3 | %87 | +0.01 [-0.01, +0.03] (p=0.294, n=427) | +0.02 [-0.00, +0.04] (p=0.128, n=427) | +0.02 [-0.00, +0.04] (p=0.110, n=427) |
| ID CV 5x3 (Birincil) | 2 | 427 | 44.60 / 0.874 / 24.3 | 44.58 / 0.874 / 24.3 | 44.59 / 0.874 / 24.2 | 44.60 / 0.874 / 24.4 | %87 | -0.02 [-0.04, +0.00] (p=0.054, n=427) | -0.01 [-0.03, +0.00] (p=0.126, n=427) | -0.01 [-0.03, +0.01] (p=0.184, n=427) |
| ID CV 5x3 (Birincil) | 8 | 427 | 41.38 / 0.881 / 22.5 | 41.38 / 0.881 / 22.4 | 41.38 / 0.881 / 22.4 | 41.38 / 0.881 / 22.6 | %87 | +0.00 [-0.02, +0.03] (p=0.877, n=427) | +0.01 [-0.02, +0.03] (p=0.729, n=427) | +0.01 [-0.02, +0.03] (p=0.701, n=427) |
| ID 60/20/20 (Holdout) | 5 | 91 | 39.84 / 0.887 / 13.7 | 39.82 / 0.888 / 13.7 | 39.82 / 0.888 / 13.7 | 39.84 / 0.888 / 13.6 | %100 | -0.02 [-0.06, +0.01] (p=0.262, n=91) | -0.02 [-0.05, +0.00] (p=0.128, n=91) | -0.02 [-0.05, +0.01] (p=0.132, n=91) |
| ID 60/20/20 (Holdout) | 2 | 91 | 41.92 / 0.884 / 17.9 | 41.96 / 0.882 / 17.9 | 41.96 / 0.882 / 17.9 | 41.92 / 0.884 / 17.9 | %100 | +0.04 [-0.10, +0.18] (p=0.555, n=91) | +0.04 [-0.11, +0.19] (p=0.579, n=91) | +0.04 [-0.11, +0.20] (p=0.595, n=91) |
| ID 60/20/20 (Holdout) | 8 | 91 | 38.67 / 0.896 / 15.6 | 38.65 / 0.896 / 15.8 | 38.65 / 0.896 / 15.8 | 38.67 / 0.895 / 15.6 | %100 | -0.02 [-0.08, +0.04] (p=0.519, n=91) | -0.01 [-0.07, +0.04] (p=0.551, n=91) | -0.01 [-0.06, +0.04] (p=0.589, n=91) |
| W3+4 S3 (Keşifsel OOD) | 5 | 146 | 50.64 / 0.876 / 27.7 | 50.62 / 0.876 / 27.7 | 50.62 / 0.876 / 27.7 | 50.63 / 0.876 / 27.6 | %100 | -0.02 [-0.05, +0.01] (p=0.156, n=146) | -0.01 [-0.04, +0.02] (p=0.438, n=146) | -0.01 [-0.04, +0.02] (p=0.440, n=146) |
| W3+4 S3 (Keşifsel OOD) | 2 | 146 | 52.16 / 0.870 / 29.5 | 52.16 / 0.870 / 29.5 | 52.16 / 0.870 / 29.5 | 52.16 / 0.870 / 29.6 | %0 | +0.00 [+0.00, +0.00] (p=1.000, n=146) | +0.01 [-0.00, +0.01] (p=0.144, n=146) | +0.00 [-0.01, +0.01] (p=0.651, n=146) |
| W3+4 S3 (Keşifsel OOD) | 8 | 146 | 50.16 / 0.876 / 28.3 | 50.15 / 0.875 / 28.3 | 50.15 / 0.875 / 28.3 | 50.16 / 0.876 / 28.3 | %100 | -0.01 [-0.02, -0.00] (p=0.044, n=146) | -0.01 [-0.02, +0.00] (p=0.092, n=146) | -0.01 [-0.02, +0.00] (p=0.134, n=146) |
| W3+4 S4 (Keşifsel OOD) | 5 | 148 | 51.99 / 0.873 / 24.6 | 51.99 / 0.873 / 24.6 | 51.99 / 0.873 / 24.6 | 51.98 / 0.873 / 24.9 | %0 | +0.00 [+0.00, +0.00] (p=1.000, n=148) | +0.01 [+0.00, +0.03] (p=0.044, n=148) | +0.01 [-0.00, +0.02] (p=0.244, n=148) |
| W3+4 S4 (Keşifsel OOD) | 2 | 148 | 53.46 / 0.866 / 33.4 | 53.46 / 0.867 / 33.4 | 53.46 / 0.867 / 33.4 | 53.46 / 0.866 / 33.6 | %100 | -0.00 [-0.02, +0.01] (p=0.571, n=148) | -0.00 [-0.02, +0.01] (p=0.655, n=148) | -0.00 [-0.02, +0.01] (p=0.639, n=148) |
| W3+4 S4 (Keşifsel OOD) | 8 | 148 | 50.17 / 0.876 / 23.1 | 50.03 / 0.875 / 23.9 | 50.03 / 0.875 / 23.9 | 50.14 / 0.876 / 23.0 | %100 | -0.14 [-0.22, -0.07] (p=0.002, n=148) | -0.12 [-0.19, -0.05] (p=0.002, n=148) | -0.12 [-0.19, -0.05] (p=0.002, n=148) |

### 1.3 Deney A2 (İkincil — `M4` Kovaryansıyla Yeniden Seçilmiş `StaticDOpt(M4)` Paneli, `k=5`)

| Rejim | Top-5 Panel Jaccard (`M4` vs `M3`) | `M4_reselected` RMSE / NDCG3 / Reg1 | `ΔRMSE (M4_resel − M3)` | `ΔRMSE (M4_resel − M4_fixed)` |
|---|---:|---:|---:|---:|
| ID CV 5x3 (Birincil) | 0.769 | 42.81 / 0.876 / 24.9 | -0.10 [-0.27, +0.08] (p=0.290, n=427) | -0.11 [-0.29, +0.06] (p=0.238, n=427) |
| ID 60/20/20 (Holdout) | 1.000 | 39.78 / 0.889 / 14.4 | -0.06 [-0.21, +0.06] (p=0.394, n=91) | -0.04 [-0.18, +0.08] (p=0.571, n=91) |
| W3+4 S3 (Keşifsel OOD) | 0.667 | 50.80 / 0.876 / 27.6 | +0.16 [+0.01, +0.35] (p=0.044, n=146) | +0.18 [+0.03, +0.36] (p=0.008, n=146) |
| W3+4 S4 (Keşifsel OOD) | 1.000 | 51.99 / 0.873 / 24.6 | +0.00 [+0.00, +0.00] (p=1.000, n=148) | +0.00 [+0.00, +0.00] (p=1.000, n=148) |

---

## 2. Kulvar B — Hasta-Düzeyi Sıra-Kalibreli Konformal Kısa Liste ve `AUGRC` (Nominal %90, $\alpha=0.10$)

> W3+4 hiçbir yöntem veya eşik seçiminde kullanılmamıştır (yalnız keşifsel stres testi).

### 2.1 ID 60/20/20 (Birincil) (`n_cal=90`, `n_test=91`)

| Kol | Yöntem | Kalibreli Eşik | Gerçek #1 Listede | Ort. Boyut (Med.) | Gözl. Oran | Tekil Oran (`|S|=1`) | En İyi AUC Kaybı (`Best Reg1`) | Ort. AUC Kaybı (`Mean Reg1`) | En Kötü AUC Kaybı (`Worst Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `EqualWidthAUC` | 2q=303.7 | %100.0 | 19.00 (21.0) | %100.0 | %0.0 | 0.00 | 117.97 | 200.18 |
| `M0_k0` | `RankConformal` | k̂=9 | %98.9 | 8.95 (9.0) | %52.1 | %0.0 | 0.27 | 90.25 | 172.41 |
| `M0_k0` | `RelRankConformal` | q_rel=0.571 | %96.7 | 11.25 (12.0) | %59.6 | %0.0 | 0.73 | 97.06 | 176.90 |
| `M0_k0` | `Top1GapConformal` | m̂=65.0 | %97.8 | 8.49 (9.0) | %48.6 | %0.0 | 0.58 | 88.94 | 171.77 |
| `M1_k0` | `EqualWidthAUC` | 2q=290.1 | %100.0 | 19.00 (21.0) | %100.0 | %0.0 | 0.00 | 117.97 | 200.18 |
| `M1_k0` | `RankConformal` | k̂=6 | %92.3 | 6.00 (6.0) | %35.3 | %0.0 | 2.06 | 70.17 | 131.07 |
| `M1_k0` | `RelRankConformal` | q_rel=0.444 | %93.4 | 8.81 (10.0) | %46.6 | %0.0 | 1.70 | 83.38 | 154.64 |
| `M1_k0` | `Top1GapConformal` | m̂=71.1 | %90.1 | 5.70 (6.0) | %34.8 | %8.8 | 3.66 | 58.90 | 113.78 |
| `StaticDOpt_k2` | `EqualWidthAUC` | 2q=287.4 | %100.0 | 19.00 (21.0) | %100.0 | %0.0 | 0.00 | 117.97 | 200.18 |
| `StaticDOpt_k2` | `RankConformal` | k̂=6 | %93.4 | 6.00 (6.0) | %35.3 | %0.0 | 1.67 | 67.19 | 128.63 |
| `StaticDOpt_k2` | `RelRankConformal` | q_rel=0.444 | %93.4 | 8.81 (10.0) | %46.6 | %0.0 | 1.31 | 80.75 | 149.12 |
| `StaticDOpt_k2` | `Top1GapConformal` | m̂=74.1 | %91.2 | 5.52 (5.0) | %34.1 | %12.1 | 2.68 | 54.42 | 105.46 |
| `StaticDOpt_k5` | `EqualWidthAUC` | 2q=241.8 | %100.0 | 18.70 (20.0) | %98.7 | %0.0 | 0.00 | 116.77 | 198.89 |
| `StaticDOpt_k5` | `RankConformal` | k̂=6 | %93.4 | 6.00 (6.0) | %35.3 | %0.0 | 1.99 | 66.03 | 127.39 |
| `StaticDOpt_k5` | `RelRankConformal` | q_rel=0.429 | %93.4 | 8.64 (9.0) | %45.6 | %0.0 | 1.31 | 79.39 | 147.57 |
| `StaticDOpt_k5` | `Top1GapConformal` | m̂=74.2 | %92.3 | 5.45 (5.0) | %33.6 | %9.9 | 2.31 | 53.83 | 107.56 |
| `StaticDOpt_k8` | `EqualWidthAUC` | 2q=234.1 | %100.0 | 18.59 (20.0) | %98.2 | %0.0 | 0.00 | 116.35 | 198.85 |
| `StaticDOpt_k8` | `RankConformal` | k̂=6 | %92.3 | 6.00 (6.0) | %35.3 | %0.0 | 2.06 | 65.49 | 126.65 |
| `StaticDOpt_k8` | `RelRankConformal` | q_rel=0.450 | %93.4 | 8.89 (10.0) | %47.2 | %0.0 | 1.31 | 80.16 | 147.09 |
| `StaticDOpt_k8` | `Top1GapConformal` | m̂=70.5 | %93.4 | 5.19 (5.0) | %32.0 | %9.9 | 2.30 | 49.69 | 100.65 |
| `AdaptiveOED_k5` | `EqualWidthAUC` | 2q=264.9 | %100.0 | 18.87 (21.0) | %99.4 | %0.0 | 0.00 | 117.45 | 199.85 |
| `AdaptiveOED_k5` | `RankConformal` | k̂=8 | %95.6 | 7.97 (8.0) | %46.6 | %0.0 | 1.13 | 76.69 | 146.04 |
| `AdaptiveOED_k5` | `RelRankConformal` | q_rel=0.476 | %95.6 | 9.56 (10.0) | %50.7 | %0.0 | 0.92 | 83.74 | 154.53 |
| `AdaptiveOED_k5` | `Top1GapConformal` | m̂=76.8 | %90.1 | 5.70 (6.0) | %35.1 | %8.8 | 2.45 | 54.31 | 107.22 |

**`StaticDOpt_k5` (ID 60/20/20 (Birincil)) Eşleştirilmiş Kısa Liste Farkları (Bootstrap B=1000):**

- `RankConformal − EqualWidthAUC`: Boyut `-12.70 [-13.69, -11.69] (p=0.002, n=91)`, Gerçek #1 `-6.6% [-12.1, -2.2] (p=0.004, n=91)`, Ort. Regret `-50.74 [-55.76, -45.85] (p=0.002, n=91)`, En İyi Regret `+1.99 [+0.47, +4.20] (p=0.004, n=91)`
- `Top1GapConformal − EqualWidthAUC`: Boyut `-13.25 [-14.44, -11.95] (p=0.002, n=91)`, Gerçek #1 `-7.7% [-14.3, -2.2] (p=0.006, n=91)`, Tekil Oran `+9.9% [+4.4, +16.5] (p=0.002, n=91)`, Ort. Regret `-62.94 [-71.50, -54.11] (p=0.002, n=91)`
- `Top1GapConformal − RankConformal`: Boyut `-0.55 [-1.15, +0.13] (p=0.116, n=91)`, Gerçek #1 `-1.1% [-6.6, +4.4] (p=0.857, n=91)`, Ort. Regret `-12.20 [-18.82, -5.96] (p=0.002, n=91)`

### 2.2 Waves3+4 (Keşifsel OOD Stres Testi) (`n_cal=70`, `n_test=148`)

| Kol | Yöntem | Kalibreli Eşik | Gerçek #1 Listede | Ort. Boyut (Med.) | Gözl. Oran | Tekil Oran (`|S|=1`) | En İyi AUC Kaybı (`Best Reg1`) | Ort. AUC Kaybı (`Mean Reg1`) | En Kötü AUC Kaybı (`Worst Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_k0` | `EqualWidthAUC` | 2q=288.6 | %100.0 | 21.72 (22.5) | %100.0 | %0.0 | 0.00 | 132.17 | 230.89 |
| `M0_k0` | `RankConformal` | k̂=8 | %93.2 | 7.97 (8.0) | %38.4 | %0.0 | 1.01 | 93.35 | 188.94 |
| `M0_k0` | `RelRankConformal` | q_rel=0.455 | %95.9 | 10.15 (10.5) | %47.0 | %0.0 | 0.74 | 99.60 | 196.19 |
| `M0_k0` | `Top1GapConformal` | m̂=68.4 | %91.9 | 7.82 (8.0) | %37.4 | %0.0 | 1.43 | 93.24 | 188.04 |
| `M1_k0` | `EqualWidthAUC` | 2q=278.9 | %100.0 | 21.72 (22.5) | %100.0 | %0.0 | 0.00 | 132.17 | 230.89 |
| `M1_k0` | `RankConformal` | k̂=6 | %91.9 | 6.00 (6.0) | %29.1 | %0.0 | 1.79 | 77.30 | 153.19 |
| `M1_k0` | `RelRankConformal` | q_rel=0.571 | %98.0 | 12.80 (13.5) | %58.9 | %0.0 | 0.41 | 104.22 | 194.39 |
| `M1_k0` | `Top1GapConformal` | m̂=69.5 | %91.9 | 5.97 (6.0) | %28.8 | %2.0 | 2.26 | 68.85 | 137.26 |
| `StaticDOpt_k2` | `EqualWidthAUC` | 2q=270.7 | %100.0 | 21.69 (22.5) | %99.9 | %0.0 | 0.00 | 132.02 | 230.63 |
| `StaticDOpt_k2` | `RankConformal` | k̂=6 | %93.9 | 6.00 (6.0) | %29.1 | %0.0 | 1.20 | 72.39 | 144.92 |
| `StaticDOpt_k2` | `RelRankConformal` | q_rel=0.429 | %95.9 | 9.75 (10.0) | %44.9 | %0.0 | 0.78 | 91.67 | 181.07 |
| `StaticDOpt_k2` | `Top1GapConformal` | m̂=74.4 | %89.9 | 6.10 (6.0) | %29.3 | %6.8 | 2.80 | 63.75 | 128.78 |
| `StaticDOpt_k5` | `EqualWidthAUC` | 2q=247.4 | %100.0 | 21.41 (22.0) | %98.7 | %0.0 | 0.00 | 130.90 | 229.08 |
| `StaticDOpt_k5` | `RankConformal` | k̂=6 | %92.6 | 6.00 (6.0) | %29.1 | %0.0 | 1.92 | 73.51 | 145.70 |
| `StaticDOpt_k5` | `RelRankConformal` | q_rel=0.400 | %95.3 | 9.13 (9.5) | %42.1 | %0.0 | 1.23 | 88.44 | 174.86 |
| `StaticDOpt_k5` | `Top1GapConformal` | m̂=63.6 | %86.5 | 4.76 (4.0) | %23.1 | %16.9 | 3.39 | 52.04 | 105.45 |
| `StaticDOpt_k8` | `EqualWidthAUC` | 2q=244.6 | %100.0 | 21.28 (22.0) | %98.1 | %0.0 | 0.00 | 130.34 | 228.54 |
| `StaticDOpt_k8` | `RankConformal` | k̂=6 | %93.2 | 6.00 (6.0) | %29.1 | %0.0 | 1.82 | 72.82 | 144.80 |
| `StaticDOpt_k8` | `RelRankConformal` | q_rel=0.450 | %94.6 | 10.11 (10.5) | %46.8 | %0.0 | 1.31 | 92.04 | 181.11 |
| `StaticDOpt_k8` | `Top1GapConformal` | m̂=63.2 | %85.8 | 4.54 (4.0) | %22.1 | %16.2 | 4.01 | 49.19 | 97.51 |
| `AdaptiveOED_k5` | `EqualWidthAUC` | 2q=261.3 | %100.0 | 21.56 (22.0) | %99.3 | %0.0 | 0.00 | 131.57 | 229.82 |
| `AdaptiveOED_k5` | `RankConformal` | k̂=6 | %91.2 | 6.00 (6.0) | %29.1 | %0.0 | 2.54 | 74.43 | 145.71 |
| `AdaptiveOED_k5` | `RelRankConformal` | q_rel=0.500 | %97.3 | 11.06 (11.5) | %51.1 | %0.0 | 0.73 | 97.63 | 186.84 |
| `AdaptiveOED_k5` | `Top1GapConformal` | m̂=68.4 | %89.2 | 5.13 (5.0) | %24.7 | %12.8 | 2.56 | 55.13 | 110.10 |

**`StaticDOpt_k5` (Waves3+4 (Keşifsel OOD Stres Testi)) Eşleştirilmiş Kısa Liste Farkları (Bootstrap B=1000):**

- `RankConformal − EqualWidthAUC`: Boyut `-15.41 [-15.92, -14.88] (p=0.002, n=148)`, Gerçek #1 `-7.4% [-11.5, -3.4] (p=0.002, n=148)`, Ort. Regret `-57.39 [-61.89, -53.10] (p=0.002, n=148)`, En İyi Regret `+1.92 [+0.69, +3.47] (p=0.002, n=148)`
- `Top1GapConformal − EqualWidthAUC`: Boyut `-16.65 [-17.26, -15.94] (p=0.002, n=148)`, Gerçek #1 `-13.5% [-18.9, -8.1] (p=0.002, n=148)`, Tekil Oran `+16.9% [+11.5, +23.0] (p=0.002, n=148)`, Ort. Regret `-78.86 [-86.16, -71.54] (p=0.002, n=148)`
- `Top1GapConformal − RankConformal`: Boyut `-1.24 [-1.71, -0.73] (p=0.002, n=148)`, Gerçek #1 `-6.1% [-10.8, -1.4] (p=0.018, n=148)`, Ort. Regret `-21.47 [-26.90, -16.09] (p=0.002, n=148)`

### 2.3 Ek Seçici Risk Duyarlılığı: `AUGRC` (Traub et al., NeurIPS 2024, `arXiv:2407.01032`) vs `AURC`

| Rejim | Kol | `omics` AUGRC / Rand (`E-AUGRC`) | `m3_post` AUGRC / Rand (`E-AUGRC`) | `omics` v2 AURC / Rand | `m3_post` v2 AURC / Rand |
|---|---|---:|---:|---:|---:|
| ID 60/20/20 | `M0_k0` | 20.49 / 23.44 (13.61) | 23.44 / 23.44 (16.56) | 35.47 / 46.88 | 46.88 / 46.88 |
| ID 60/20/20 | `M1_k0` | 8.87 / 10.58 (6.10) | 10.58 / 10.58 (7.82) | 15.10 / 21.17 | 21.17 / 21.17 |
| ID 60/20/20 | `StaticDOpt_k2` | 7.27 / 8.95 (5.13) | 8.93 / 8.95 (6.78) | 11.27 / 17.90 | 17.83 / 17.90 |
| ID 60/20/20 | `StaticDOpt_k5` | 5.59 / 6.85 (4.13) | 5.31 / 6.85 (3.85) | 8.83 / 13.71 | 9.46 / 13.71 |
| ID 60/20/20 | `StaticDOpt_k8` | 6.44 / 7.82 (4.61) | 6.46 / 7.82 (4.62) | 11.73 / 15.63 | 12.88 / 15.63 |
| ID 60/20/20 | `AdaptiveOED_k5` | 6.86 / 6.77 (5.24) | 6.14 / 6.77 (4.52) | 12.82 / 13.54 | 11.04 / 13.54 |
| W3+4 OOD | `M0_k0` | 26.16 / 27.30 (18.24) | 27.30 / 27.30 (19.38) | 53.95 / 54.59 | 54.59 / 54.59 |
| W3+4 OOD | `M1_k0` | 16.74 / 16.66 (12.17) | 16.66 / 16.66 (12.09) | 30.47 / 33.32 | 33.32 / 33.32 |
| W3+4 OOD | `StaticDOpt_k2` | 17.94 / 16.69 (13.54) | 14.38 / 16.69 (9.98) | 34.63 / 33.37 | 28.84 / 33.37 |
| W3+4 OOD | `StaticDOpt_k5` | 14.27 / 12.30 (11.34) | 10.57 / 12.30 (7.64) | 32.44 / 24.60 | 20.35 / 24.60 |
| W3+4 OOD | `StaticDOpt_k8` | 12.47 / 11.54 (9.69) | 10.63 / 11.54 (7.85) | 27.17 / 23.09 | 20.61 / 23.09 |
| W3+4 OOD | `AdaptiveOED_k5` | 12.59 / 12.35 (9.58) | 12.02 / 12.35 (9.01) | 27.20 / 24.71 | 24.11 / 24.71 |

---

## 3. Kulvar C — FIMM / Malani Dış Sıralama Tekrarı (`Zenodo 7370747`)

- **Ön-Modelleme Katalog & QC Denetimi (`sprint05_fimm_catalog_audit.json`):** `132` eşleşmiş AML örneği / `109` benzersiz hasta (`File_7 = 167` RNA [`163 AML + 4 CD34+`]), `23/24` ortak hedef (`Gilteritinib` yok), `64` ortak Beat AML probu, `14688` ortak Ensembl gen; yön uyumu `y_aligned = -sDSS`; Beat AML konformal aralığı **taşınmamıştır** (`conformal_intervals_transferred = False`).
- **Beat AML Eğitiminde Yeniden Seçilen `StaticDOpt` Ortak Katalog Top-8:** `Flavopiridol, Sunitinib, XAV-939, CYT387, Tivozanib (AV-951), AZD1152-HQPA (AZD2811), Linifanib (ABT-869), BI-2536`

### 3.1 Deney C1 — Donmuş Beat AML Model Aktarımı (Sıfır FIMM Yeniden Fit, Sıfır FIMM Kohort İstatistiği)

#### Birincil: 109 Benzersiz İndeks Hasta (`n_spec=109/109`)

| Kol | `n_units` | `NDCG@3` ↑ | `Recall@3` ↑ | `Within-Patient Spearman` ↑ | `Regret@1 (sDSS)` ↓ | `Regret@3 (sDSS)` ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `C1_M0_k0` | 109 | 0.571 | 0.310 | 0.186 | 11.23 | 5.47 |
| `C1_M1_rank_k0` | 109 | 0.651 | 0.372 | 0.258 | 7.41 | 3.26 |
| `C1_M2a_StaticDOpt_k2` | 109 | 0.598 | 0.350 | 0.256 | 10.34 | 4.49 |
| `C1_M3_StaticDOpt_k2` | 109 | 0.683 | 0.404 | 0.270 | 6.66 | 3.08 |
| `C1_M2a_StaticDOpt_k5` | 109 | 0.617 | 0.333 | 0.238 | 8.47 | 4.33 |
| `C1_M3_StaticDOpt_k5` | 109 | 0.683 | 0.408 | 0.280 | 7.18 | 3.12 |
| `C1_M3_Random20x_k5` | 109 | 0.649 | 0.362 | 0.281 | 7.44 | 3.40 |
| `C1_M2a_StaticDOpt_k8` | 109 | 0.580 | 0.287 | 0.222 | 9.27 | 4.74 |
| `C1_M3_StaticDOpt_k8` | 109 | 0.670 | 0.390 | 0.279 | 7.83 | 3.03 |

- `C1_M3_k5_vs_M1_k0`: `ΔNDCG@3 = +0.032 [+0.007, +0.058] (p=0.016, n=109)`, `ΔSpearman = +0.022 [+0.000, +0.045] (p=0.048, n=109)`, `ΔRegret@1 (sDSS) = -0.23 [-1.26, +0.76] (p=0.643, n=109)`
- `C1_M3_k5_vs_M0_k0`: `ΔNDCG@3 = +0.111 [+0.069, +0.152] (p=0.002, n=109)`, `ΔSpearman = +0.094 [+0.065, +0.129] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -4.05 [-5.55, -2.62] (p=0.002, n=109)`
- `C1_M3_Static_k5_vs_Random_k5`: `ΔNDCG@3 = +0.033 [+0.013, +0.054] (p=0.004, n=109)`, `ΔSpearman = -0.001 [-0.020, +0.017] (p=0.883, n=109)`, `ΔRegret@1 (sDSS) = -0.26 [-0.93, +0.41] (p=0.492, n=109)`
- `C1_M2a_k5_vs_M0_k0`: `ΔNDCG@3 = +0.045 [+0.003, +0.087] (p=0.040, n=109)`, `ΔSpearman = +0.053 [+0.017, +0.084] (p=0.004, n=109)`, `ΔRegret@1 (sDSS) = -2.76 [-4.12, -1.49] (p=0.002, n=109)`
- `C1_M1_rank_k0_vs_M0_k0`: `ΔNDCG@3 = +0.079 [+0.043, +0.117] (p=0.002, n=109)`, `ΔSpearman = +0.073 [+0.047, +0.101] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -3.81 [-5.30, -2.38] (p=0.002, n=109)`

#### Duyarlılık: Tüm 132 Örnek (109 Hasta Kümeli) (`n_spec=132/132`)

| Kol | `n_units` | `NDCG@3` ↑ | `Recall@3` ↑ | `Within-Patient Spearman` ↑ | `Regret@1 (sDSS)` ↓ | `Regret@3 (sDSS)` ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `C1_M0_k0` | 109 | 0.572 | 0.312 | 0.187 | 11.26 | 5.61 |
| `C1_M1_rank_k0` | 109 | 0.649 | 0.371 | 0.257 | 7.66 | 3.32 |
| `C1_M2a_StaticDOpt_k2` | 109 | 0.598 | 0.347 | 0.255 | 10.47 | 4.40 |
| `C1_M3_StaticDOpt_k2` | 109 | 0.681 | 0.403 | 0.265 | 7.00 | 3.02 |
| `C1_M2a_StaticDOpt_k5` | 109 | 0.619 | 0.342 | 0.242 | 8.59 | 4.22 |
| `C1_M3_StaticDOpt_k5` | 109 | 0.680 | 0.407 | 0.277 | 7.43 | 3.06 |
| `C1_M3_Random20x_k5` | 109 | 0.650 | 0.366 | 0.283 | 7.58 | 3.46 |
| `C1_M2a_StaticDOpt_k8` | 109 | 0.583 | 0.290 | 0.221 | 9.44 | 4.66 |
| `C1_M3_StaticDOpt_k8` | 109 | 0.670 | 0.391 | 0.274 | 7.99 | 2.98 |

- `C1_M3_k5_vs_M1_k0`: `ΔNDCG@3 = +0.031 [+0.009, +0.055] (p=0.008, n=109)`, `ΔSpearman = +0.019 [-0.000, +0.040] (p=0.058, n=109)`, `ΔRegret@1 (sDSS) = -0.22 [-1.15, +0.68] (p=0.629, n=109)`
- `C1_M3_k5_vs_M0_k0`: `ΔNDCG@3 = +0.109 [+0.069, +0.149] (p=0.002, n=109)`, `ΔSpearman = +0.090 [+0.062, +0.124] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -3.83 [-5.29, -2.45] (p=0.002, n=109)`
- `C1_M3_Static_k5_vs_Random_k5`: `ΔNDCG@3 = +0.030 [+0.012, +0.048] (p=0.002, n=109)`, `ΔSpearman = -0.006 [-0.023, +0.010] (p=0.509, n=109)`, `ΔRegret@1 (sDSS) = -0.15 [-0.77, +0.50] (p=0.683, n=109)`
- `C1_M2a_k5_vs_M0_k0`: `ΔNDCG@3 = +0.047 [+0.007, +0.088] (p=0.026, n=109)`, `ΔSpearman = +0.055 [+0.024, +0.084] (p=0.004, n=109)`, `ΔRegret@1 (sDSS) = -2.68 [-3.98, -1.45] (p=0.002, n=109)`
- `C1_M1_rank_k0_vs_M0_k0`: `ΔNDCG@3 = +0.077 [+0.041, +0.116] (p=0.002, n=109)`, `ΔSpearman = +0.071 [+0.045, +0.098] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -3.61 [-5.09, -2.17] (p=0.002, n=109)`

### 3.2 Deney C2 — FIMM İçi Yöntem Tekrarı (5-Kat Hasta-Ayrık CV x 3 Tohum; Donmuş Aktarımdan Ayrıdır)

- **Değerlendirilen Hasta Sayısı:** `n_eval = 109/109` (tüm istatistikler, gen filtreleri ve `StaticDOpt` paneli her katta yalnızca o katın FIMM eğitim hastalarından öğrenilmiştir; `-sDSS` ölçeği).

| Kol | `RMSE (sDSS)` ↓ | `Residual R²` ↑ | `NDCG@3` ↑ | `Recall@3` ↑ | `Regret@1 (sDSS)` ↓ | `Within-Pat Res Spearman` ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `M0_k0` | 5.27 | 0.000 | 0.713 | 0.438 | 6.89 | 0.000 |
| `M0_k5` | 4.72 | 0.196 | 0.713 | 0.438 | 6.89 | 0.000 |
| `M1_k0` | 4.83 | 0.154 | 0.765 | 0.486 | 4.92 | 0.325 |
| `M2a_StaticDOpt_k2` | 4.43 | 0.294 | 0.768 | 0.482 | 4.06 | 0.327 |
| `M2a_StaticDOpt_k5` | 4.06 | 0.408 | 0.797 | 0.510 | 3.33 | 0.453 |
| `M2a_StaticDOpt_k8` | 3.90 | 0.455 | 0.821 | 0.551 | 2.82 | 0.512 |
| `M3_StaticDOpt_k2` | 4.17 | 0.371 | 0.802 | 0.529 | 3.34 | 0.422 |
| `M3_StaticDOpt_k5` | 3.83 | 0.473 | 0.843 | 0.569 | 2.58 | 0.521 |
| `M3_StaticDOpt_k8` | 3.74 | 0.500 | 0.862 | 0.607 | 2.32 | 0.552 |
| `M3_Random20x_k5` | 4.29 | 0.334 | 0.802 | 0.523 | 3.65 | 0.420 |
| `M4_StaticDOpt_k5` | 3.83 | 0.474 | 0.844 | 0.572 | 2.53 | 0.522 |
| `M4_perm_null_k5` | 3.83 | 0.473 | 0.842 | 0.569 | 2.56 | 0.520 |

- `M3_Static_k5_vs_M1_k0`: `ΔRMSE (sDSS) = -1.00 [-1.26, -0.76] (p=0.002, n=109)`, `ΔNDCG@3 = +0.077 [+0.052, +0.103] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -2.34 [-3.48, -1.28] (p=0.002, n=109)`
- `M3_Static_k5_vs_M0_k0`: `ΔRMSE (sDSS) = -1.43 [-1.77, -1.13] (p=0.002, n=109)`, `ΔNDCG@3 = +0.130 [+0.101, +0.162] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -4.32 [-5.63, -3.11] (p=0.002, n=109)`
- `M3_Static_k5_vs_M0_k5`: `ΔRMSE (sDSS) = -0.89 [-1.13, -0.68] (p=0.002, n=109)`, `ΔNDCG@3 = +0.130 [+0.101, +0.162] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -4.32 [-5.63, -3.11] (p=0.002, n=109)`
- `M3_Static_k5_vs_Random20x_k5`: `ΔRMSE (sDSS) = -0.46 [-0.61, -0.32] (p=0.002, n=109)`, `ΔNDCG@3 = +0.041 [+0.026, +0.056] (p=0.002, n=109)`, `ΔRegret@1 (sDSS) = -1.07 [-1.60, -0.52] (p=0.002, n=109)`
- `M4_Static_k5_vs_M3_Static_k5`: `ΔRMSE (sDSS) = -0.00 [-0.01, +0.01] (p=0.444, n=109)`, `ΔNDCG@3 = +0.002 [-0.002, +0.006] (p=0.360, n=109)`, `ΔRegret@1 (sDSS) = -0.05 [-0.26, +0.12] (p=0.649, n=109)`
- `M4_Static_k5_vs_M4_perm_null_k5`: `ΔRMSE (sDSS) = -0.00 [-0.01, +0.00] (p=0.196, n=109)`, `ΔNDCG@3 = +0.002 [-0.001, +0.005] (p=0.212, n=109)`, `ΔRegret@1 (sDSS) = -0.02 [-0.19, +0.11] (p=0.811, n=109)`
