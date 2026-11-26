# OncoTwin v1 — Dondurulmuş Deney Protokolü (`PROTOCOL.md`)

**Durum:** DONDURULMUŞ (Sprint 2 deneyleri çalıştırılmadan önce sabitlenmiştir).  
**Kural:** Bu belgedeki kohort tanımı, örnek seçimi, bölme kuralları, `AUC` yönü, model formülasyonları, birincil metrikler ve Kapı 1 karar eşikleri test sonuçları görüldükten sonra **değiştirilemez**.

---

## 1. Veri Kaynağı ve Kohort Dışlama Kuralları

- **Veri Kaynağı:** `data/raw/beataml2/` (Beat AML 2.0, `biodev/beataml2.0_data`, CC BY 4.0, SHA-256 doğrulamalı).
- **Dışlanan Kayıtlar:**
  1. `beataml_waves1to4_sample_mapping.xlsx` içindeki **37 sağlıklı kontrol** (`Healthy pooled CD34+`, `Healthy Individual BM MNC`, `Healthy Individual CD34+`).
  2. `beataml_wv1to4_clinical.xlsx` içinde `ELN2017 == 'NonAML'` olarak etiketlenmiş satırlar (tüm tabloda 29 satır; `Drug ∩ RNA-seq` içinde 8 hasta / 8 örnek).
- **Eşleşmiş Saf AML Kohortu:** Hem `beataml_probit_curve_fits_v4_dbgap.txt` (`auc`) hem `beataml_waves1to4_norm_exp_dbgap.txt` (RNA-seq) verisine sahip **512 örnek / 471 benzersiz hasta (`dbgap_subject_id`)**.
- **Remisyon Notu:** Eşleşmiş kohortta `diseaseStageAtSpecimenCollection == 'Remission'` olan 13 örnek (11 hasta; bunların 8'inde başka evre örneği yok) protokol gereği kohortta tutulur, analizden keyfi olarak silinmez.

---

## 2. Hasta Bazlı Ayrım ve Deterministik İlk/İndeks Örnek Seçimi

- **Bölme Birimi:** Tüm CV ve dalga bölmelerinde bölme birimi **kesinlikle `dbgap_subject_id`**'dir. Aynı hastaya ait birden fazla örnek asla eğitim ve test katlarına dağıtılamaz.
- **Birincil Analiz Birimi (Hasta Başına İndeks Örnek):** Her hastadan hiçbir ilaç `auc` etiketine bakılmadan tamamen deterministik olarak tek bir indeks örnek seçilir (toplam **471 indeks örnek**). Sıralama önceliği (küçükten büyüğe):
  1. `timeOfSampleCollectionRelativeToInclusion` (eksik `NaN` değerler $+\infty$ kabul edilir),
  2. `diseaseStageAtSpecimenCollection` sırası: `Initial Diagnosis` (`0`) < `Relapse` (`1`) < `Residual` (`2`) < `Unknown` (`3`) < `Remission` (`4`),
  3. `dbgap_dnaseq_sample` mevcudiyeti: dolu (`0`) < eksik (`1`) *(aynı gün çekilen çoklu örneklerde tam çoklu-omik örneğini seçer; örn. hasta `2248`)*,
  4. `specimenType` sırası: `Bone Marrow Aspirate` (`0`) < `Peripheral Blood` (`1`) < diğer (`2`),
  5. `specimen_key` alfabetik artan.
- **İkincil Duyarlılık Analizi:** Temiz `Waves1+2 → Waves3+4` test kümesindeki hastaların **tüm** eşleşmiş örnekleri (`n = 173` örnek) hasta-küme ortalamalı olarak ikincil duyarlılık tablosunda raporlanır.

---

## 3. Sızıntısız Değerlendirme Rejimleri

1. **ID Rejimi (`Patient-Disjoint 5-Fold CV × 3 Seed`):**
   - Sabit rastgele tohumlar: `42, 137, 2026`.
   - 471 hasta `cohort` (`Waves1+2`, `Waves3+4`, `Both`) etiketine göre tabakalı (stratified) 5 kata bölünür.
   - Her katta `set(train_subjects).isdisjoint(set(test_subjects))` doğrulaması zorunludur.
2. **Zamansal OOD Rejimi (`Clean Waves1+2 → Waves3+4`):**
   - **Eğitim:** Yalnızca `cohort == 'Waves1+2'` indeks örnekleri (`n = 284` hasta).
   - **Test:** Yalnızca `cohort == 'Waves3+4'` olup **tüm 942 satırlık klinik tabloda hiçbir zaman `Waves1+2` veya `Both` içinde görülmemiş** temiz AML hastaları (`n = 165` hasta; duyarlılıkta tüm örnekleri `n = 173`).
   - `cohort == 'Both'` etiketli örnekler/hastalar zamansal eğitim ve testten çıkarılır.

---

## 4. Eğitim Katı Dışı Sıfır Bilgi Sızıntısı Kuralı

Bir değerlendirme katının test (veya dış) kümesinden **hiçbir** istatistik öğrenilemez. Aşağıdakilerin tamamı **yalnızca ilgili eğitim katı (`train_idx`)** içinde hesaplanır:
- Global ortalama ve ilaç ortalama etkileri ($\hat{\mu}_{\text{tr}}, \hat{b}_{d,\text{tr}}$),
- `residual_auc` referansı ($r_{p,d} = y_{p,d} - (\hat{\mu}_{\text{tr}} + \hat{b}_{d,\text{tr}})$),
- Prob havuzu kapsam filtresi ($\mathcal{U}_{\text{tr}}$),
- RNA-seq en yüksek varyanslı top-2000 gen seçimi ve gen-bazlı z-skorlama ($\hat{\mu}_g, \hat{\sigma}_g$),
- Sürücü mutasyon özelliklerinin eğitim ortalaması/std'si,
- Ampirik ilaç kovaryansı ($\hat{\Sigma}_{\text{emp}}$), düşük-rank SVD faktörleri ve OOF artık kovaryansı ($\hat{\Sigma}_{\text{res}}$),
- Tüm modellerin hiperparametre seçimi (yalnızca eğitim katı içinde hasta-ayrık 4-fold iç CV ile).

---

## 5. `AUC` Yönü, Hedef Kümesi ($\mathcal{C}$), Prob Havuzu ($\mathcal{U}_{\text{tr}}$) ve Few-Shot Bütçeleri

- **`AUC` Yönü (Ampirik Doğrulandı):** Beat AML 2.0'da **düşük `auc` = yüksek ilaç duyarlılığı** ($s_{p,d} = -\text{auc}_{p,d}$). Doğrulama (`FLT3-ITD+` vs `FLT3-ITD-`):
  - `Quizartinib (AC220)`: `114.9` vs `159.5` (fark `−44.5`)
  - `Gilteritinib`: `103.5` vs `148.3` (fark `−44.8`)
  - `Sorafenib`: `146.1` vs `201.5` (fark `−55.4`)
  - `Midostaurin`: `161.1` vs `208.2` (fark `−47.1`)
- **Hedef Aday Küme $\mathcal{C}$ (24 Klinik/Translasyonel AML İlacı):**
  `Venetoclax`, `Azacytidine`, `Cytarabine`, `Gilteritinib`, `Midostaurin`, `Quizartinib (AC220)`, `Sorafenib`, `Crenolanib`, `Trametinib (GSK1120212)`, `Selumetinib (AZD6244)`, `Dasatinib`, `Ruxolitinib (INCB018424)`, `Ibrutinib (PCI-32765)`, `Entospletinib (GS-9973)`, `AGI-5198`, `AGI-6780`, `Panobinostat`, `Selinexor`, `Rapamycin`, `Bortezomib (Velcade)`, `Palbociclib`, `Lenalidomide`, `Idelalisib`, `Volasertib (BI-6727)`.
- **Prob Havuzu $\mathcal{U}_{\text{tr}}$:** İlgili **eğitim katında** $\mathcal{C}$ dışında kalan ve o eğitim katında **$\ge 100$ örnekte ölçülmüş** tüm inhibitörler.
- **Bütçeler ve Eşleştirme:** $k \in \{0, 3, 5, 8\}$. Her $k > 0$ için test hastasının gözlenen $\mathcal{U}_{\text{tr}}$ ilaçlarından sabit `R = 10` rastgele tekrarla $k$ prob ilacı $S_p \subset \mathcal{U}_{\text{tr}}$ seçilir ($\mathcal{C}$ hiçbir zaman $S_p$'ye girmez, böylece değerlendirilen hedef çiftleri $k=0, 3, 5, 8$ arasında %100 aynı kalır).
- **Değerlendirilebilir Test Hastası Filtresi:** Test hastasında $\mathcal{C}$ içinde $\ge 5$ gözlenen hedef ve $\mathcal{U}_{\text{tr}}$ içinde $\ge 8$ gözlenen prob bulunması (her bölmede kesin `n_eval_patients` raporlanır).

---

## 6. Model Ailesi (`M0`, `M1`, `M2a`, `M2b`, `M3`)

1. **`M0(k)` — Adil Few-Shot Ortalama-Etki:**
   $$\hat{y}_{p,d}^{(0)} = \hat{\mu}_{\text{tr}} + \hat{b}_{d,\text{tr}} + \hat{b}_p^{(S_p)}, \qquad \hat{b}_p^{(S_p)} = \frac{1}{|S_p| + \lambda_0} \sum_{j \in S_p} \left(y_{p,j} - \hat{\mu}_{\text{tr}} - \hat{b}_{j,\text{tr}}\right)$$
   $k=0$ iken $\hat{b}_p = 0$; $k>0$ iken $\lambda_0 \in \{0.5, 1, 2, 5, 10, 20\}$ eğitim katında iç 4-fold CV ile seçilir.
2. **`M1` — Omik-Only Ridge ($k=0$):**
   Eğitimde seçilen top-2000 RNA geni (z-skorlu) + 17 ham mutasyon özelliği (`FLT3-ITD` ve `NPM1` klinik özet sütunlarından; `DNMT3A`, `TET2`, `IDH2`, `IDH1`, `TP53`, `NRAS`, `KRAS`, `RUNX1`, `ASXL1`, `SRSF2`, `WT1`, `PTPN11`, `CEBPA`, `KIT` WES çağrılarından + `has_wes` bayrağı; eğitimde z-skorlu) üzerinde dual Gram Ridge ($\lambda_1 \in \{30, 100, 300, 1000, 3000, 10000, 30000\}$). Ayrıca eğitim katı içi 4-fold out-of-fold (`Y_oof`) tahminlerini üretir.
3. **`M2a(k)` — Omiksiz Few-Shot Kovaryans Ridge (Koşullu Gaussian):**
   Eğitim kalıntılarından $\tilde{Y}_{\text{tr}} = Y_{\text{tr}} - \hat{\mu}_{\text{tr}} - \hat{b}_{d,\text{tr}}$ ikili-tamamlanmış PSD-projekte ve büzüştürülmüş kovaryans $\hat{\Sigma}_{\text{emp}}$ öğrenilir; $(\alpha_2 \in \{0.05, 0.15, 0.30, 0.50\}, \lambda_2 \in \{50, 200, 500, 1500, 5000\})$ eğitim içi 4-fold CV ile seçilir.
4. **`M2b(k)` — Omiksiz Düşük-Rank Maskeli SVD (Soft-Impute tarzı):**
   Eğitim kalıntılarından $\tilde{Y}_{\text{tr}}$ iteratif maskeli rank-$r$ SVD ile ayrıştırılır ($r \in \{3, 5, 8, 12\}$, $\lambda_{\text{svd}} \in \{0.5, 2.0, 10.0, 50.0\}$ eğitim içi CV ile seçilir); testte $S_p$'den Ridge ile hasta gizil vektörü kestirilir.
5. **`M3(k)` — Hibrit Omik Öncül + Koşullu Gaussian Few-Shot:**
   `M1`'in eğitim-içi OOF kalıntıları $E^{\text{oof}} = Y_{\text{tr}} - \hat{Y}^{\text{M1,oof}}$ üzerinden PSD-projekte büzüştürülmüş $\hat{\Sigma}_{\text{res}}$ öğrenilir; $(\alpha_3, \lambda_3)$ aynı ızgara üzerinde eğitim içi CV ile seçilir. Testte:
   $$\hat{y}_{p,\mathcal{C}}^{(3)} = \hat{y}_{p,\mathcal{C}}^{(1)} + \hat{\Sigma}_{\text{res},\mathcal{C},S_p}\left(\hat{\Sigma}_{\text{res},S_p,S_p} + \lambda_3 I\right)^{-1}\left(y_{p,S_p} - \hat{y}_{p,S_p}^{(1)}\right)$$
   $k=0$ iken `M3(0) == M1`.

---

## 7. Önceden Kayıtlı Metrikler, İstatistiksel Çıkarım ve Sızıntı Testi

- **`rmse`:** Gözlenen $(p, d \in \mathcal{C})$ çiftlerinde kök ortalama kare hata (ham `auc` ve `residual_auc` üzerinde özdeş).
- **`residual_r2`:** Eğitim ilaç ortalamaları çıkarıldıktan sonraki kalıntı varyansına kıyasla açıklanan varyans ($1 - \sum (y - \hat{y})^2 / \sum (y - \hat{\mu}_{\text{tr}} - \hat{b}_{d,\text{tr}})^2$).
- **`within_patient_residual_spearman` ($\rho_{\text{pat}}^{\text{res}}$):** Her test hastasında $|\mathcal{T}_p| \ge 5$ üzerinde tahmin kalıntısı $(\hat{y}_{p,d} - \hat{\mu}_{\text{tr}} - \hat{b}_{d,\text{tr}})$ ile gerçek kalıntı $(y_{p,d} - \hat{\mu}_{\text{tr}} - \hat{b}_{d,\text{tr}})$ arasındaki Spearman korelasyonunun hasta ortalaması (`M0(k)` için tanım gereği `0.0`).
- **`per_drug_spearman` ($\rho_{\text{drug}}$):** Her hedef ilaçta $\ge 10$ test hastası üzerinde tahmin ile gerçek `auc` arasındaki Spearman korelasyonunun ilaç ortalaması (`M0(0)` için tanım gereği `0.0`).
- **`ndcg_at_3` ve `ΔNDCG@3 (vs M0(k))`:** Hasta başına düşük `auc` = yüksek alaka ($\text{rel}_{p,d} = \max_{j \in \mathcal{T}_p} y_{p,j} - y_{p,d} \ge 0$), artan tahmin `auc` sırasına göre `NDCG@3`.
- **`residual_ndcg_at_3` (ikincil):** İlaç ortalama etkisi çıkarılmış kalıntı duyarlılık sıralaması ($-r_{p,d}$).
- **Güven Aralıkları:** Hasta düzeyinde **1.000 tekrarlı eşleştirilmiş bootstrap %95 GA** (sabit seed `2026`).
- **Zorunlu Permütasyon Sızıntı Testi:** Eğitim hasta etiketleri (`Y_train` ve `X` eşleşmesi) rastgele karıştırıldığında `M1` için `per_drug_spearman` %95 GA'sı 0'ı içermeli ve `residual_r2 <= 0.01` olmalıdır. Aksi hâlde sızıntı hatası kabul edilir ve tüm sonuçlar geçersiz sayılır.

---

## 8. Önceden Kayıtlı Kapı 1 (Go / No-Go) Kriterleri

1. **GEÇER (Sprint 3'e ilerle):**
   - Permütasyon sızıntı testi geçer; VE
   - $k=5$ veya $k=8$ bütçesinde en az bir few-shot model (`M2a`, `M2b` veya `M3`), hem adil `M0(k)` hem de omik-only `M1` karşısında **Patient-Disjoint 5-Fold CV**'de `rmse` farkının %95 GA üst sınırı `< 0` VE `within_patient_residual_spearman` farkının %95 GA alt sınırı `> 0` olur; VE
   - Aynı model **Clean `Waves1+2 → Waves3+4` OOD** bölmesinde `M0(k)` karşısında `rmse` (%95 GA üst `< 0`) ve `within_patient_residual_spearman` (%95 GA alt `> 0`) kazancını korur; VE
   - `ΔNDCG@3 (vs M0(k))` CV'de pozitif %95 GA'ya sahipse **TAM GEÇER**; değilse ama yukarıdaki iki metrik geçiyorsa **KOŞULLU GEÇER** (ham ilaç potens baskınlığı vurgulanarak).
2. **KALIR (Pivot / Negatif Sonuç):**
   - $k=8$ bütçesinde dahi hiçbir few-shot model `M0(k)` ve `M1` üzerinde anlamlı `rmse` ve `within_patient_residual_spearman` kazancı veremezse veya zamansal OOD (`Waves3+4`) bölmesinde tamamen çökerse Sprint 3'e (panel seçimine) geçmeden negatif sonuç raporlanır.
