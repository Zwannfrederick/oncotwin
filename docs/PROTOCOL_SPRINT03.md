# OncoTwin v1 — Sprint 3 Protokol Eki (`PROTOCOL_SPRINT03.md`)

**Durum:** DONDURULMUŞ (Sprint 3 deneyleri çalıştırılmadan önce sabitlenmiştir; [PROTOCOL.md](PROTOCOL.md) belgesi değiştirilmemiştir).

---

## 1. Kapsam, Eşit Bütçe ve Sıfır Test-AUC Sızıntısı Kuralları

1. **Bütçeler ve `k=0` Kuralı:**
   - Panel seçimi bütçeleri: $k \in \{1, 2, 3, 5, 8, 12\}$.
   - **`k = 0` için panel seçimi yapılmaz:** Omik-only `M1` ($k=0$) ve `M0(0)` referans tabanları olarak korunur.
2. **Seçilen İlaçların Değerlendirmeden Hariç Tutulması:**
   - Seçilen prob ilaçları her zaman $S_p \subset \mathcal{U}_{\text{tr}}$ (eğitim katında $\ge 100$ örnekte ölçülmüş ve $\mathcal{C}$ dışında kalan havuz) içinden seçilir.
   - Değerlendirme yalnızca 24 klinik AML ilacı olan hedef kümesi $\mathcal{C}$ üzerinden yapılır; $S_p \cap \mathcal{C} = \emptyset$ olduğundan seçilen hiçbir prob ilacı değerlendirme metriklerine girmez.
3. **OED Bilgi Kısıtı (Sıfır Test-AUC Sızıntısı):**
   - Panel seçicileri yalnızca **ilgili eğitim katından öğrenilmiş kovaryansları ($\hat{\Sigma}_{\text{res}}$)** ve test hastasının **seçim öncesi erişilebilir bilgilerini** (omik öncülü $\hat{y}^{\text{M1}}_p$ ve o hastanın retrospektif plakada hangi probların ölçülmüş olduğu ikili maskesi) kullanabilir. Test hastasının `auc` değerleri seçim sırasında **asla** okunmaz.
4. **Eşleştirilmiş Aynı-Hasta ve Aynı-Bütçe Filtresi:**
   - Tüm politikalar ve tüm $k \in \{1,2,3,5,8,12\}$ bütçeleri **birebir aynı değerlendirilebilir hastalar** üzerinde koşulur: test hastasında $\mathcal{C}$ içinde $\ge 5$ gözlenen hedef ve $\mathcal{U}_{\text{tr}}$ içinde $\ge 12$ gözlenen prob bulunması (`ID 5-Fold CV`: ~427 hasta; `Clean Waves1+2 → Waves3+4`: ~146 temiz hasta).
   - Sabit politikalar (`Static-Diversity`, `Static-D-Optimal`) eğitim katından tek bir sıralı öncelik listesi $[d_{(1)}, \dots, d_{(M)}]$ üretir; test hastasında bu sıradan ölçülmüş ilk $k$ prob alınarak her politikada tam bütçe $k$ eşitliği sağlanır. Ek olarak, geri-dolgu (fallback) olmadan doğrudan ilk-$k$ statik önekin hastada bulunma oranı (`top_k_direct_hit_rate`) ve **tam-panel (sıfır geri-dolgu) alt-grubu** raporlanır.

---

## 2. Karşılaştırılan 4 Toplu Panel Politikası (`M3` Tahmincisi Altında)

1. **`Random 50x`:** Her hasta $p$ için gözlenen $\mathcal{U}_{\text{tr}}$ havuzundan 50 sabit rastgele tohumla $k$ prob seçilir; eşleştirilmiş hasta karşılaştırmaları için 50 panel ortalaması, ayrıca tekil 50 rastgele panel dağılımı (medyan, P75, P90) raporlanır.
2. **`Static-Diversity`:** Eğitim katındaki `M3` OOF kalıntı korelasyon mesafesi $D_{ij} = 1 - |\rho_{\text{res}}(i,j)|$ üzerinde prob medoidinden başlayarak açgözlü Farthest-First (Max-Min) sabit öncelik sıralaması.
3. **`Static-D-Optimal`:** Eğitim katındaki $\hat{\Sigma}_{\text{res}}$ üzerinde hedef küme $\mathcal{C}$ ile prob seti $S$ arasındaki tam Gaussian Karşılıklı Bilgiyi $I(Y_{\mathcal{C}}; Y_S \mid X) = \frac{1}{2}\log\det(\Sigma_{\mathcal{C},\mathcal{C}}) - \frac{1}{2}\log\det(\Sigma_{\mathcal{C}\mid S})$ Schur tümleyeni ile açgözlü maksimize eden sabit öncelik sıralaması.
4. **`Patient-Adaptive Batch OED`:** Test hastasının yalnızca seçim öncesi omik öncülü $\mathcal{N}(\hat{y}^{\text{M1}}_{p,\mathcal{C}}, \hat{\Sigma}_{\text{res},\mathcal{C},\mathcal{C}})$ dağılımından her hedef ilacın hastanın en duyarlı ilk-3 adayı arasına girme olasılığı $w_{p,d}(x_p) = P(d \in \text{Top-3}(\mathcal{C}) \mid x_p)$ Monte Carlo ($M_{\text{mc}} = 500$, sabit seed) ile tahmin edilir; ardından hasta-özel karar ağırlıklı beklenen varyans azalışı
   $$\mathcal{A}(S; x_p) = \sum_{d \in \mathcal{C}} w_{p,d}(x_p) \left[\Sigma_{\text{res},d,d} - \Sigma_{\mathcal{C}\mid S, d, d}\right]$$
   hastanın mevcut probları içinden Schur tümleyeni ile açgözlü maksimize edilir.

---

## 3. Önceden Kayıtlı Metrikler (Düşük `AUC` = Yüksek Duyarlılık)

- `RMSE` ↓ ve `Residual R²` ↑
- **`Recall@3`** ↑: Gerçek en düşük `auc`'lu 3 ilaç ile tahmin edilen en düşük `auc`'lu 3 ilacın tie-duyarlı beklenen kesişimi $/ 3$.
- **`NDCG@3`** ↑ ve `Residual NDCG@3` ↑
- **`Regret@1`** ↓: $y_{p, \hat{d}_1(p)} - \min_{d \in \mathcal{T}_p} y_{p,d} \ge 0$ (tie-duyarlı beklenen AUC kaybı).
- **`Regret@3`** ↓: $\min_{d \in \text{Top3}_{\text{pred}}(p)} y_{p,d} - \min_{d \in \mathcal{T}_p} y_{p,d} \ge 0$ (tie-duyarlı beklenen Top-3 AUC kaybı).
- `Within-Patient Residual Spearman` ($\rho_{\text{pat}}^{\text{res}}$) ↑ ve kat-içi `Per-Drug Spearman` ($\rho_{\text{drug}}$) ↑
- Tüm eşleştirilmiş hasta karşılaştırmalarında **1.000 tekrarlı hasta-düzeyi bootstrap %95 GA** (seed `2026`).

---

## 4. Zorunlu `M3` Mekanizma Ablasyonları ($k \in \{3, 5, 8\}$)

`M3`'ün gerçekte ne öğrendiğini doğrulamak için aynı `Random 50x` (50 tekrar) ve `Static-D-Optimal` panelleri üzerinde 5 koşul karşılaştırılır:
1. **`M3_full`**: Tam model (omik öncül + `M1` OOF artık kovaryansı + test hastası prob artıkları).
2. **`M3_no_omics` (`M2a`)**: Omik öncülü tamamen kaldırılmış saf few-shot kovaryans modeli.
3. **`M3_perm_within_patient`**: Test hastası $p$'nin kendi $k$ prob artığı `r_p[S_p]` kendi içinde rastgele karıştırılır (skaler hasta ofseti korunur, ilaca özgü kovaryans bozulur).
4. **`M3_perm_across_patients`**: Test hastalarının prob artık vektörleri test hastaları arasında rastgele değiştirilir ($S_p$ maskesi korunur, başka hastanın değerleri verilir).
5. **`M3_perm_train_cov`**: Eğitimdeki `E_oof` sütunları her sütunda bağımsız permüte edilerek öğrenilen sahte kovaryansla eğitilmiş `M3`.

---

## 5. Önceden Kayıtlı Kapı 2 (Go / No-Go) Kriterleri

1. **Ablasyon Geçerliliği:** `M3_full`, $k \in \{5, 8\}$'de hem CV hem `Waves3+4` üzerinde `M3_no_omics`, `M3_perm_within_patient`, `M3_perm_across_patients` ve `M3_perm_train_cov`'dan daha düşük `RMSE` ve daha yüksek $\rho_{\text{pat}}^{\text{res}}$ vermelidir (%95 GA 0'ı dışlamalı).
2. **Kapı 2 `PASS`:** En az bir yapısal politika (`Static-D-Optimal`, `Static-Diversity` veya `Patient-Adaptive Batch OED`), eşit bütçede ($k \in \{3, 5, 8\}$) `Random 50x`'i **5-Fold CV**'de `RMSE` (%95 GA üst `< 0`) VE en az bir karar/sıralama metriğinde (`NDCG@3`, `Recall@3` veya `Regret@1`, %95 GA 0'ı dışlar) anlamlı geçer VE **Clean `Waves3+4`** OOD'de aynı `RMSE` iyileşme yönünü (`< 0`) korur.
3. **Sabit vs Adaptif Kararı:** `Patient-Adaptive Batch OED` vs `Static-D-Optimal` %95 GA'sına göre `ADAPTIVE_WINS`, `STATIC_EQUIVALENT` veya `STATIC_WINS` olarak raporlanır.
4. **Doygunluk Bütçesi ($k^*$):** Ardışık adım marjinal `RMSE` kazancının eklenen prob başına `< 1.0` AUC altına düştüğü veya %95 GA'nın 0'ı kestiği ilk bütçe $k^*$ olarak belirlenir.
