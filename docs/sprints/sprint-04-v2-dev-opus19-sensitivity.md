# Sprint 4 v2 — `DEV-OPUS-19` Eşik Duyarlılık Analizi (`m3_post_scale` Nihai Fit Ölçeği)

- **Protokol Eki:** [docs/PROTOCOL_SPRINT04_ADDENDUM_19.md](../../docs/PROTOCOL_SPRINT04_ADDENDUM_19.md) (`2026-11-22`, kullanıcı `A4` onayı: `2026-11-22`).
- **Analiz Statüsü:** keşifsel duyarlılık analizi (ön kayıtlı değil; Sprint 4 sonuçları daha önce görüldü). Üstünlük etiketi çıkarılmaz; `v2` tahminleri, kısa listeleri, `q_sim`, `omics_pca_dist` ve eşikten bağımsız `AURC` değerleri **değişmemiştir**.
- **`A1` (Adaptif `M1` Referansı):** `primary=m1_oof (fit_and_oof Y_oof_), auxiliary=m1_insample`.
- **`A2` (Kilitli Sabit Skor):** Kilitli StaticDOpt/EqualWeightVarRed tüm tam-panel hastalarda tek skor verir (sabit skor — ayırt edici değil).
- **`A3` (Ön İzleme Şeffaflığı):** Ana hücre `archive_available / StaticDOpt_k5` için ID `%92.3` ve `Waves3+4` `%43.9` test kabulü ek öncesinde `audits/opus-2026-11-22-sprint04/realdata_probe_s04.json (archive_available/StaticDOpt_k5: ID %92.3, W3+4 %43.9).` çıktısında görülmüştür.
- **`Waves3+4` Uyarısı:** Tüm `Waves3+4` sonuçları keşifsel OOD stres testidir; `M0_k0` ve `M1_k0` için yeni eşik atanmamıştır (§2.1).

## 1. Ana İnceleme Hücresi Özeti (`archive_available / StaticDOpt_k5`)

| Rejim | `n_ref` / `uniq` | $\tau_{\text{OOF}}$ → $\tau_{\text{final}}$ | Ref Kabul | Test `n_tgt` / `n_eval` | OOF Kabul (`Regret@1`) | Nihai-Fit Kabul (`Regret@1`) | Nihai-Fit Çek. (`Regret@1`) | `AURC` (değişmez) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ID 60/20/20 | 267 / 34 | 48.243 → 45.946 | 231/267 (%86.5) | 93 / 91 | 91/91 (%100.0; 13.7) | 84/91 (%92.3; 12.8) | 7 (24.2) | 9.46 |
| Waves3+4 (OOD stres) | 208 / 21 | 42.689 → 41.461 | 179/208 (%86.1) | 163 / 148 | 146/148 (%98.6; 24.6) | 65/148 (%43.9; 20.6) | 83 (27.7) | 20.35 |

## 2. ID 60/20/20

### archive_available

| Kol | Referans Durumu | `n_ref` (`< τ` / `= τ` / `uniq`) | $\tau_{\text{OOF}}$ → $\tau_{\text{final}}$ | Test `n_tgt` / `n_comp` / `tek_çek` | OOF Kabul (`Reg1`) | Nihai Kabul (`Reg1`) | Nihai Puan Çek. (`Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `M0_k0` | skor uygulanamaz (k=0, M0 modeli) | — | — | — | — | — | — |
| `M1_k0` | sabit skor — yeni eşik atanmadı (k=0, prob yok) | — | — | — | — | — | — |
| `StaticDOpt_k2` | ok | 267 (0/223/8) | 48.736 → 48.094 | 93 / 91 / 2 | 87/91 (17.8) | 83/91 (17.8) | 8 (18.9) |
| `EqualWeightVarRed_k2` | ok | 267 (0/223/5) | 48.705 → 48.094 | 93 / 91 / 2 | 91/91 (18.8) | 83/91 (17.8) | 8 (29.6) |
| `AdaptiveOED_k2` | ok | 267 (140/85/18) | 49.387 → 48.951 | 93 / 91 / 2 | 76/91 (13.9) | 73/91 (14.3) | 18 (19.1) |
| `StaticDOpt_k5` | ok | 267 (180/51/34) | 48.243 → 45.946 | 93 / 91 / 2 | 91/91 (13.7) | 84/91 (12.8) | 7 (24.2) |
| `EqualWeightVarRed_k5` | ok | 267 (172/56/21) | 47.717 → 45.243 | 93 / 91 / 2 | 91/91 (20.1) | 82/91 (20.6) | 9 (15.8) |
| `AdaptiveOED_k5` | ok | 267 (203/28/76) | 48.342 → 46.178 | 93 / 91 / 2 | 91/91 (13.5) | 82/91 (13.4) | 9 (15.2) |
| `StaticDOpt_k8` | ok | 267 (172/49/48) | 46.829 → 45.538 | 93 / 91 / 2 | 91/91 (15.6) | 81/91 (14.5) | 10 (24.6) |
| `EqualWeightVarRed_k8` | ok | 267 (175/50/41) | 46.612 → 45.045 | 93 / 91 / 2 | 89/91 (14.0) | 85/91 (14.3) | 6 (10.1) |
| `AdaptiveOED_k8` | ok | 267 (213/1/147) | 47.107 → 45.793 | 93 / 91 / 2 | 91/91 (14.3) | 72/91 (15.9) | 19 (8.2) |

### locked_train_catalog

| Kol | Referans Durumu | `n_ref` (`< τ` / `= τ` / `uniq`) | $\tau_{\text{OOF}}$ → $\tau_{\text{final}}$ | Test `n_tgt` / `n_comp` / `tek_çek` | OOF Kabul (`Reg1`) | Nihai Kabul (`Reg1`) | Nihai Puan Çek. (`Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `M0_k0` | skor uygulanamaz (k=0, M0 modeli) | — | — | — | — | — | — |
| `M1_k0` | sabit skor — yeni eşik atanmadı (k=0, prob yok) | — | — | — | — | — | — |
| `StaticDOpt_k2` | sabit skor — ayırt edici değil | 223 (0/223/1) | 48.633 → 48.094 | 93 / 83 / 10 | 83/83 (17.8) | 83/83 (17.8) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k2` | sabit skor — ayırt edici değil | 223 (0/223/1) | 48.705 → 48.094 | 93 / 83 / 10 | 83/83 (17.8) | 83/83 (17.8) | 0 (hesaplanamadı) |
| `AdaptiveOED_k2` | ok | 234 (113/83/12) | 49.414 → 48.951 | 93 / 83 / 10 | 68/83 (13.0) | 66/83 (13.3) | 17 (20.3) |
| `StaticDOpt_k5` | sabit skor — ayırt edici değil | 51 (0/51/1) | 48.251 → 45.898 | 93 / 19 / 74 | 19/19 (12.3) | 19/19 (12.3) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k5` | sabit skor — ayırt edici değil | 150 (0/150/1) | 47.717 → 45.076 | 93 / 60 / 33 | 60/60 (13.5) | 60/60 (13.5) | 0 (hesaplanamadı) |
| `AdaptiveOED_k5` | ok | 180 (143/28/41) | 48.450 → 46.178 | 93 / 57 / 36 | 57/57 (14.8) | 54/57 (14.7) | 3 (17.3) |
| `StaticDOpt_k8` | sabit skor — ayırt edici değil | 51 (0/51/1) | 46.755 → 45.252 | 93 / 19 / 74 | 19/19 (13.0) | 19/19 (13.0) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k8` | sabit skor — ayırt edici değil | 133 (0/133/1) | 46.025 → 44.877 | 93 / 56 / 37 | 56/56 (10.8) | 56/56 (10.8) | 0 (hesaplanamadı) |
| `AdaptiveOED_k8` | ok | 123 (98/2/51) | 47.110 → 45.778 | 93 / 40 / 53 | 40/40 (17.0) | 30/40 (21.3) | 10 (4.1) |


## 3. Waves3+4 (Keşifsel OOD Stres Testi)

### archive_available

| Kol | Referans Durumu | `n_ref` (`< τ` / `= τ` / `uniq`) | $\tau_{\text{OOF}}$ → $\tau_{\text{final}}$ | Test `n_tgt` / `n_comp` / `tek_çek` | OOF Kabul (`Reg1`) | Nihai Kabul (`Reg1`) | Nihai Puan Çek. (`Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `M0_k0` | skor uygulanamaz (k=0, M0 modeli) | — | — | — | — | — | — |
| `M1_k0` | sabit skor — yeni eşik atanmadı (k=0, prob yok) | — | — | — | — | — | — |
| `StaticDOpt_k2` | ok | 208 (59/139/8) | 45.010 → 43.957 | 163 / 148 / 15 | 139/148 (31.3) | 139/148 (31.3) | 9 (65.1) |
| `EqualWeightVarRed_k2` | ok | 208 (5/185/5) | 44.186 → 42.964 | 163 / 148 / 15 | 148/148 (32.3) | 132/148 (29.5) | 16 (55.8) |
| `AdaptiveOED_k2` | ok | 208 (157/16/14) | 45.379 → 43.708 | 163 / 148 / 15 | 125/148 (30.0) | 112/148 (31.1) | 36 (19.2) |
| `StaticDOpt_k5` | ok | 208 (136/43/21) | 42.689 → 41.461 | 163 / 148 / 15 | 146/148 (24.6) | 65/148 (20.6) | 83 (27.7) |
| `EqualWeightVarRed_k5` | ok | 208 (166/7/16) | 42.741 → 40.925 | 163 / 148 / 15 | 146/148 (23.0) | 24/148 (18.1) | 124 (24.0) |
| `AdaptiveOED_k5` | ok | 208 (158/9/24) | 42.784 → 41.471 | 163 / 148 / 15 | 147/148 (24.5) | 103/148 (23.6) | 45 (27.2) |
| `StaticDOpt_k8` | ok | 208 (165/2/31) | 43.254 → 41.210 | 163 / 148 / 15 | 146/148 (23.0) | 22/148 (23.8) | 126 (23.0) |
| `EqualWeightVarRed_k8` | ok | 208 (134/41/23) | 43.121 → 40.930 | 163 / 148 / 15 | 146/148 (25.1) | 24/148 (19.9) | 124 (26.2) |
| `AdaptiveOED_k8` | ok | 208 (166/1/66) | 43.321 → 41.318 | 163 / 148 / 15 | 148/148 (22.8) | 80/148 (23.3) | 68 (22.2) |

### locked_train_catalog

| Kol | Referans Durumu | `n_ref` (`< τ` / `= τ` / `uniq`) | $\tau_{\text{OOF}}$ → $\tau_{\text{final}}$ | Test `n_tgt` / `n_comp` / `tek_çek` | OOF Kabul (`Reg1`) | Nihai Kabul (`Reg1`) | Nihai Puan Çek. (`Reg1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `M0_k0` | skor uygulanamaz (k=0, M0 modeli) | — | — | — | — | — | — |
| `M1_k0` | sabit skor — yeni eşik atanmadı (k=0, prob yok) | — | — | — | — | — | — |
| `StaticDOpt_k2` | sabit skor — ayırt edici değil | 139 (0/139/1) | 44.378 → 43.957 | 163 / 3 / 160 | 3/3 (63.7) | 3/3 (63.7) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k2` | sabit skor — ayırt edici değil | 185 (0/185/1) | 44.186 → 42.964 | 163 / 131 / 32 | 131/131 (29.7) | 131/131 (29.7) | 0 (hesaplanamadı) |
| `AdaptiveOED_k2` | ok | 179 (143/5/11) | 45.379 → 43.497 | 163 / 55 / 108 | 36/55 (33.3) | 26/55 (38.5) | 29 (21.5) |
| `StaticDOpt_k5` | sabit skor — ayırt edici değil | 89 (0/89/1) | 41.967 → 41.264 | 163 / 1 / 162 | 1/1 (78.0) | 1/1 (78.0) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k5` | sabit skor — ayırt edici değil | 165 (0/165/1) | 42.741 → 40.905 | 163 / 19 / 144 | 19/19 (9.0) | 19/19 (9.0) | 0 (hesaplanamadı) |
| `AdaptiveOED_k5` | ok | 167 (127/11/11) | 42.761 → 41.111 | 163 / 21 / 142 | 21/21 (10.9) | 20/21 (10.4) | 1 (21.9) |
| `StaticDOpt_k8` | sabit skor — ayırt edici değil | 85 (0/85/1) | 42.332 → 40.989 | 163 / 1 / 162 | 1/1 (78.0) | 1/1 (78.0) | 0 (hesaplanamadı) |
| `EqualWeightVarRed_k8` | sabit skor — ayırt edici değil | 120 (0/120/1) | 42.276 → 40.804 | 163 / 16 / 147 | 16/16 (6.1) | 16/16 (6.1) | 0 (hesaplanamadı) |
| `AdaptiveOED_k8` | ok | 149 (119/5/33) | 43.321 → 41.215 | 163 / 1 / 162 | 1/1 (0.0) | 1/1 (0.0) | 0 (hesaplanamadı) |

## 4. `A1` Duyarlılığı: `AdaptiveOED` Referansında `M1 OOF` vs `M1 Örneklem-İçi`

| Rejim | Senaryo | Kol | `OOF` $\tau_{\text{final}}$ (`uniq`, test kabul) | `In-Sample` $\tau_{\text{final}}$ (`uniq`, test kabul) | `Δn_acc` |
|---|---|---|---:|---:|---:|
| id_60_20_20 | archive_available | `AdaptiveOED_k2` | 48.951 (18, 73/91) | 48.951 (18, 73/91) | +0 |
| id_60_20_20 | archive_available | `AdaptiveOED_k5` | 46.178 (76, 82/91) | 46.178 (86, 82/91) | +0 |
| id_60_20_20 | archive_available | `AdaptiveOED_k8` | 45.793 (147, 72/91) | 45.879 (167, 80/91) | +8 |
| id_60_20_20 | locked_train_catalog | `AdaptiveOED_k2` | 48.951 (12, 66/83) | 48.951 (14, 66/83) | +0 |
| id_60_20_20 | locked_train_catalog | `AdaptiveOED_k5` | 46.178 (41, 54/57) | 46.178 (46, 54/57) | +0 |
| id_60_20_20 | locked_train_catalog | `AdaptiveOED_k8` | 45.778 (51, 30/40) | 45.869 (62, 32/40) | +2 |
| wave_ood | archive_available | `AdaptiveOED_k2` | 43.708 (14, 112/148) | 43.714 (16, 114/148) | +2 |
| wave_ood | archive_available | `AdaptiveOED_k5` | 41.471 (24, 103/148) | 41.496 (36, 103/148) | +0 |
| wave_ood | archive_available | `AdaptiveOED_k8` | 41.318 (66, 80/148) | 41.351 (83, 81/148) | +1 |
| wave_ood | locked_train_catalog | `AdaptiveOED_k2` | 43.497 (11, 26/55) | 43.951 (15, 36/55) | +10 |
| wave_ood | locked_train_catalog | `AdaptiveOED_k5` | 41.111 (11, 20/21) | 41.471 (21, 21/21) | +1 |
| wave_ood | locked_train_catalog | `AdaptiveOED_k8` | 41.215 (33, 1/1) | 41.268 (47, 1/1) | +0 |
