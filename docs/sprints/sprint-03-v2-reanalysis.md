# Sprint 3 v2 — Adaptive vs Static-D-Optimal Simetrik Holm Yeniden Analizi (Keşifsel)

- **Kaynak:** `data/processed/sprint03_v2_vectors.npz` (hasta düzeyi vektörler; model yeniden fit edilmedi).
- **Kural (Sol §3.2/3, Sprint 3 sonuçları görüldükten sonra tanımlandı):** aynı hastalar ve hedefler üzerinde `k={3,5,8}` × `{RMSE, NDCG@3, Regret@1}` = 9 iki yönlü eşleştirilmiş bootstrap testi (B=10000, seed 2026), Holm düzeltmesi α=0.05. Fark = AdaptiveOED − StaticDOpt.
- **Etiketler:** düzeltilmiş anlamlı test yok → `INSUFFICIENT_EVIDENCE` (yetersiz kanıt); iki yönde anlamlı → `MIXED` (karışık); tek yön → `ADAPTIVE_FAVORED` / `STATIC_FAVORED`. Pratik eşdeğerlik iddiası içermez.
- v1 `ADAPTIVE_WINS` etiketi (`sprint03_results.json`) değiştirilmedi; bu analiz onun yerine geçmez, yanına konur.

## ID 5-Fold CV × 3 seed: **`INSUFFICIENT_EVIDENCE`**

| $k$ | Metrik | Fark [%95 GA] | p (ham) | p (Holm) | Anlamlı | Lehine |
|---:|---|---|---:|---:|:---:|---|
| 3 | `rmse` | +0.4669 [-0.1486, +1.0675] | 0.1328 | 0.7967 | — | StaticDOpt |
| 3 | `ndcg3` | -0.0011 [-0.0059, +0.0036] | 0.6601 | 1.0000 | — | StaticDOpt |
| 3 | `regret1` | -0.5724 [-2.3075, +1.2587] | 0.5093 | 1.0000 | — | AdaptiveOED |
| 5 | `rmse` | +0.2406 [-0.3264, +0.8007] | 0.3958 | 1.0000 | — | StaticDOpt |
| 5 | `ndcg3` | +0.0059 [+0.0007, +0.0113] | 0.0258 | 0.2064 | — | AdaptiveOED |
| 5 | `regret1` | -2.1383 [-4.3522, +0.0575] | 0.0566 | 0.3962 | — | AdaptiveOED |
| 8 | `rmse` | +0.2531 [-0.2068, +0.7367] | 0.2880 | 1.0000 | — | StaticDOpt |
| 8 | `ndcg3` | +0.0064 [+0.0014, +0.0116] | 0.0124 | 0.1116 | — | AdaptiveOED |
| 8 | `regret1` | -1.1266 [-3.2678, +0.9541] | 0.3004 | 1.0000 | — | AdaptiveOED |

## Temiz W1+2 → W3+4 (stres testi): **`INSUFFICIENT_EVIDENCE`**

| $k$ | Metrik | Fark [%95 GA] | p (ham) | p (Holm) | Anlamlı | Lehine |
|---:|---|---|---:|---:|:---:|---|
| 3 | `rmse` | +1.1159 [-0.2343, +2.5331] | 0.1102 | 0.7713 | — | StaticDOpt |
| 3 | `ndcg3` | -0.0058 [-0.0195, +0.0077] | 0.4050 | 1.0000 | — | StaticDOpt |
| 3 | `regret1` | -2.9791 [-9.4436, +3.3435] | 0.3624 | 1.0000 | — | AdaptiveOED |
| 5 | `rmse` | +0.4874 [-0.6680, +1.6850] | 0.4168 | 1.0000 | — | StaticDOpt |
| 5 | `ndcg3` | -0.0121 [-0.0253, +0.0008] | 0.0660 | 0.5489 | — | StaticDOpt |
| 5 | `regret1` | +0.6722 [-6.5395, +7.3032] | 0.8545 | 1.0000 | — | StaticDOpt |
| 8 | `rmse` | -0.9429 [-2.8872, +0.6481] | 0.3218 | 1.0000 | — | AdaptiveOED |
| 8 | `ndcg3` | -0.0105 [-0.0219, +0.0004] | 0.0610 | 0.5489 | — | StaticDOpt |
| 8 | `regret1` | +4.4776 [-2.3581, +11.8534] | 0.2114 | 1.0000 | — | StaticDOpt |
