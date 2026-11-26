# OncoTwin v1 — Sprint 4 Protokol Eki (`PROTOCOL_SPRINT04.md`)

**Durum:** DONDURULMUŞ (Sprint 4 deneyleri çalıştırılmadan ve test sonuçlarına bakılmadan sabitlenmiştir; [PROTOCOL.md](PROTOCOL.md) ve [PROTOCOL_SPRINT03.md](PROTOCOL_SPRINT03.md) değiştirilmemiştir).  
**Referanslar:** [sol-architect-reviewer.md](agents/sol-architect-reviewer.md) §3.2 & §5 devir sözleşmesi; [DECISION_LOG.md](DECISION_LOG.md) `AD-005`, `AD-007`, `AD-008`.

---

## 1. Kohort, Analiz Birimi ve Karşılaştırılan Kollar

1. **Esas Hasta Birimi ve Kohort:**
   - Analiz birimi `dbgap_subject_id` (önceden deterministik olarak seçilmiş indeks örnekler: `471` benzersiz AML hastası; 8 yalnız-remisyon hastası dahil).
   - Değerlendirme hedef kümesi $\mathcal{C}$: aynı 24 klinik AML inhibitörü (`CLINICAL_CANDIDATE_DRUGS`).
   - Çıktı ve kısa liste yorumu yalnızca *ex vivo* araştırma AUC uç noktasına bağlıdır; klinik fayda veya prospektif aktarım iddiası içermez.
2. **Ana Kol ve Karşılaştırıcılar (`M3` Tahmincisi Altında):**
   - **Ana kol:** `StaticDOpt + M3, k=5`.
   - **Karşılaştırıcı bütçeler ve referans:** `k ∈ {2, 5, 8}` (`k=5` ana, `k=2` düşük bütçe, `k=8` duyarlılık) ve `k=0` omik-only `M1` referansı (ek taban olarak `M0(0)`). Tarihsel `k*=2` kaydı korunur.
   - **Saf Kişiselleştirme Kontrolü (Sol §6.1):** Aynı `M3` modeli ve aynı varyans-azaltma amaç fonksiyonu altında her budget $k \in \{2, 5, 8\}$ için üç seçici karşılaştırılır:
     1. `StaticDOpt` (eğitim kovaryansında log-det D-optimal statik sıralama — çalışma referansı),
     2. `EqualWeightVarRed` (eşit hedef ağırlıkları $w_c = 1/|\mathcal{C}|$ ile beklenen hedef varyans azaltma),
     3. `AdaptiveOED` (`compute_contender_weights` ile hastaya özgü Top-3 aday olasılık ağırlıklı beklenen varyans azaltma).

---

## 2. Hasta-Bazlı Bölme ve Dondurma (Sıfır İlaç Yanıtı Erişimi)

1. **ID Rejimi (`id_60_20_20`, Seed `2026`):**
   - `make_patient_disjoint_kfold(ds_idx.meta, n_splits=5, seed=2026, strata_col="cohort")` kullanılarak `471` indeks hasta deterministik ve ilaç yanıtlarına hiç bakılmadan 3 ayrık kümeye bölünür:
     - Kat `{0, 1, 2}` → `train_tune` (`284` hasta),
     - Kat `3` → `calibration` (`94` hasta),
     - Kat `4` → `test` (`93` hasta).
   - ID verileri Sprint 1–3'te toplu analize girdiği için bilimsel iddia *yeniden kullanılan kohort* kapsamıyla sınırlıdır.
2. **Temiz Zamansal Stres Testi (`wave_ood`: `Waves1+2 → Waves3+4`):**
   - `make_clean_wave_split` ile `Waves1+2` (`284` hasta) ve `Waves3+4` (`165` temiz hasta) ayrılır (`Both` ve çapraz-dalga hastaları dışlı).
   - `Waves1+2` içinde `make_patient_disjoint_kfold(..., n_splits=4, seed=2026, strata_col="cohort")` ile Kat `{0, 1, 2}` → `train_tune` (`213` hasta) ve Kat `3` → `calibration` (`71` hasta) bölünür; `test` = temiz `Waves3+4` (`165` hasta).
   - `Waves3+4` hiçbir eğitim, ölçek, hiperparametre, eşik veya kalibrasyon adımına girmez; tüm `Waves3+4` bulguları **keşifsel OOD stres testi** olarak etiketlenir.
3. **Dondurma Kuralları:**
   - Prob havuzu $\mathcal{U}_{\text{tr}}$ (`MIN_TRAIN_PROBE_OBS = 100` ve $\mathcal{U}_{\text{tr}} \cap \mathcal{C} = \emptyset$), filtreler, gen seçimi/z-skorlama, `M1`/`M3`, statik/kilitli paneller, PCA omik uzaklık skoru ve çekinme eşikleri **yalnızca `train_tune`** içinden belirlenir.
   - Kalibrasyon hasta etiketleri yalnızca `q_sim` / `q_per_drug` hesabına girer.
   - Tüm hasta atamaları koşu öncesi `data/processed/sprint04_split_manifest.json` dosyasına yazılır.

---

## 3. İki Panel Mevcudiyeti Senaryosu ve Payda Tanımları

1. **`archive_available` (Ana Arşiv Senaryosu — Mevcut Ölçümlerden Seçim):**
   - Politika, hastanın retrospektif plakada ölçülen prob maskesini görebilir (`auc` değerlerini görmez).
   - Değerlendirilebilir hasta filtresi: $\mathcal{C}$'de $\ge 5$ hedef ve $\mathcal{U}_{\text{tr}}$'de $\ge 8$ prob (`min_targets=5, min_probes=8`); `_require_full_budget` ile tüm değerlendirilebilir hastalarda tam `k` prob zorunludur.
2. **`locked_train_catalog` (Salt Eğitim Kataloğu Simülasyonu):**
   - Beat AML 2.0'da bağımsız prospektif plaka-kataloğu manifesti bulunmadığından *belgelenmiş plaka kataloğu* senaryosu raporlarda **`hesaplanamadı`** olarak kaydedilir; bunun yerine yalnızca `train_tune` kataloğuna dayalı kilitli seçim simülasyonu **ayrı etiketlenerek** raporlanır.
   - Seçici, test/kalibrasyon prob mevcudiyet maskesini **hiç açmadan** kilitli `k` prob seçer (`StaticDOpt` ve `EqualWeightVarRed` `train_tune` kovaryansından tek kilitli `k`-listesi; `AdaptiveOED` yalnızca `train_tune` prob havuzu ve test/cal hastasının omik verisinden kilitli `k`-listesi).
   - Sonradan arşiv maskesi açılır: eksik/ölçülmemiş prob varsa **geri doldurma yapılmaz**, tahmin üretilmez ve hasta **teknik çekinme (`technical_abstain`)** sayılır.
   - Panel bulunabilirliği tüm hedef-uygun hastalarda (`n_target_eligible`: $\mathcal{C}$'de $\ge 5$ hedef), performans farkları ise ortak tam-panel kesişiminde ayrı raporlanır; `n = 0` hücreler `hesaplanamadı` olarak yazılır. Arşiv mevcudiyetiyle seçilen ortak W3+4 havuzu ana uygulamaya aktarılmaz.

---

## 4. ID Split-Conformal Kalibrasyon ve Kısa Aday Listesi ($\alpha = 0.10$)

1. **Hasta-Düzeyi Eşzamanlı Skor (Nominal %90, $\alpha = 0.10$):**
   - Her kalibrasyon hastası $i$ için prob-dışı gözlenen uygun hedefler $O_i \subseteq \mathcal{C}$ üzerinde
     $$E_i = \max_{d \in O_i} |y_{i,d} - \hat{y}_{i,d}|$$
     hesaplanır.
   - $n$ geçerli kalibrasyon hastası için sonlu-örneklem sıra istatistiği $r = \lceil (n + 1)(1 - \alpha) \rceil$ alınır; $n = 0$ veya $r > n$ ise $\hat{q} = +\infty$ olur ve sonlu aralık yayımlanmaz (`finite_interval_published = False`).
2. **Ana ve Ek Kapsam Sonuçları:**
   - **Ana ID sonucu:** Eşzamanlı hasta kapsamı ($P(\max_{d \in O_i}|y_{i,d} - \hat{y}_{i,d}| \le \hat{q})$) ve aralık genişliği ($2\hat{q}$).
   - **Ek sonuçlar:** İlaç başına ampirik kapsam; ayrıca öncül literatür (Shokrzadeh & Shokrzadeh, bioRxiv Haziran 2026 `2026.06.07.730728v1`) ile konumlandırma için ayrı etiketli yardımcı *ilaç-bazlı marjinal split-conformal* kapsam.
   - **Kapsam sınırı:** Eksik arşiv hedefleri, yalnızca kabul edilen alt grup ve `Waves3+4` OOD için ayrı kapsam garantisi iddia edilmez (`Waves3+4` kapsamı yalnızca ampirik raporlanır).
3. **Konformal Kısa Aday Listesi:**
   - Düşük AUC daha duyarlı olduğundan, her test hastası için $[L_{i,d}, U_{i,d}] = [\hat{y}_{i,d} - \hat{q}, \hat{y}_{i,d} + \hat{q}]$ aralıklarıyla
     $$\mathcal{S}_i^{\text{short}} = \{ d \in O_i \mid L_{i,d} \le \min_{d' \in O_i} U_{i,d'} + 10^{-12} \}$$
     oluşturulur (eşzamanlı kapsam sağlandığında gerçek en düşük AUC'lu uygun hedef listede kalır).
   - Raporlananlar: `shortlist_contains_true_top1_rate`, `mean_shortlist_size_obs`, `definite_top1_rate_obs` (`|S_short| == 1`) ve bilgilendirme amaçlı tüm 24 hedef üzerinde `mean_shortlist_size_all24`.

---

## 5. OOD Çekinme (`ABSTAIN`) ve Risk–Kapsam (`AURC`)

1. **Skor Tanımları (Test AUC'si Sıfır Erişim):**
   - **Birincil (`omics_pca_dist`):** Yalnızca `train_tune` `_build_omics_features` (top-2000 yüksek varyanslı RNA + z-skorlu 17 mutasyon) üzerinde fit edilmiş merkezlenmiş en fazla `20` PCA bileşenindeki standartlaştırılmış RMS uzaklık.
   - **İkincil karşılaştırıcı (`m3_post_scale`):** `M3` koşullu kovaryansından 24 hedef üzerindeki RMS posterior standart sapma (kapsam garantisi yok).
2. **Sabit Eşik ve Eşleştirilmiş Risk–Kapsam Eğrisi:**
   - Çekinme eşiği $\tau_{0.80}$, yalnızca `train_tune` içi 5-kat hasta-ayrık doğrulamada **%80 hedef kabul oranına** göre (`quantile(..., 0.80, method="higher")`) kilitlenir (işletim noktası, performans iddiası değil).
   - Testte sabit eşikte **kabul oranı**, **kabul edilen hastalarda ortalama `Regret@1`** (ve çekinilenlerde tanısal `Regret@1`) raporlanır.
   - Risk–kapsam eğrisi ve `AURC` / `E-AURC` (`tie_tol = 1e-12` ile bağlı eşikler birlikte grup olarak eklenir; basamaklı integral; farklı kapsama tavanlarında ortak erişilebilir aralıkta ve eşleşmiş hastalarda kıyaslanır). Test kümesinde eşik seçimi yapılmaz.

---

## 6. Koşu Öncesi Zorunlu Doğrulama Sözleşmesi (Sol §5 #6)

Deney koşusundan önce ve test paketinde (`tests/test_sprint04.py`) şu 6 koşul doğrulanmalıdır:
- **(a)** `train_tune`, `calibration`, `test` hasta kümeleri her iki rejimde tamamen ayrıktır.
- **(b)** Gizli test `auc` değerleri değiştirilince model fit'i, sıralama, seçilen prob ve çekinme skoru değişmez.
- **(c)** Kalibrasyon yanıtları değiştirilince yalnızca conformal kalibrasyon ($\hat{q}$) ve kısa listeler değişir; model tahminleri ve çekinme skorları/eşikleri değişmez.
- **(d)** `locked_train_catalog` senaryosunda test mevcudiyet maskesi değişince ön-seçilen kilitli panel değişmez.
- **(e)** Hedef ve prob kümeleri ayrıktır ($\mathcal{S}_p \cap \mathcal{C} = \emptyset$).
- **(f)** Eksik bütçe sıfır bilgiyle geçilmiş gibi kaydedilmez; boş metrik dilimleri `RuntimeWarning` üretmeden `n=0` / `hesaplanamadı` ile görünür.

---

## 7. Çıktı Dosyaları

- Bölme manifesti: `data/processed/sprint04_split_manifest.json`
- Makine çıktısı: `data/processed/sprint04_results.json`
- Hasta düzeyi vektörler: `data/processed/sprint04_vectors.npz`
- Sprint 4 raporu: `docs/sprints/sprint-04-conformal-abstain.md`
