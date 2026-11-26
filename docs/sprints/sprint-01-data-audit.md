# Sprint 1 — Beat AML 2.0 Ampirik Veri Denetimi ve Protokol Temeli

Bu rapor, `biodev/beataml2.0_data` (Waves 1–4, CC BY 4.0) resmi harmonize dosyalarının SHA-256 doğrulaması sonrası yapılan tam ampirik denetimini özetler. Tüm sayılar ham dosyalardan doğrudan ölçülmüştür.

---

## 1. Hasta ≠ Örnek Ayrımı ve Sızıntı Bulguları

- **Klinik özet (`beataml_wv1to4_clinical.xlsx`):** **805** benzersiz hasta (`dbgap_subject_id`), **942** benzersiz örnek (`specimen_key`).
- **Birden fazla örneği olan hastalar:** 108 hastanın 2+ örneği var (2 örnek: 84, ≥3 örnek: 24, maks: 5; toplam 245 örnek). **Örnek-bazlı rastgele bölme sızıntı yapar; bölme birimi kesinlikle `dbgap_subject_id` olmalıdır.**
- **Örnek haritası (`sample_map`):** 979 satır = 942 hasta örneği + **37 sağlıklı kontrol** (`Healthy pooled CD34+` / `Healthy Individual`). Analizde sağlıklı kontroller dışlanır.
- **Dalga dağılımı (örnek / hasta):** `Waves1+2`: 649 / 543, `Waves3+4`: 270 / 257, `Both`: 23 / 23.
- **Kritik dalga-arası sızıntı:** `Waves1+2` ile `Waves3+4` arasında **13 ortak hasta**, `Both` etiketi ile dalgalar arasında ek çakışmalarla toplam **18 ortak hasta** vardır. Zamansal dış bölmede (`Waves1+2 → Waves3+4`) bu hastalar ve `Both` örnekleri test kümesinden temizlenmelidir.

---

## 2. Modalite Kesişimi ve Alt Kohort Büyüklükleri

| Katman / Kesişim | Örnek | Hasta |
|---|---:|---:|
| Klinik özet (tüm AML/MDS-MPN örnekleri) | 942 | 805 |
| *Ex vivo* inhibitör verisi (`probit_curve_fits`) | 631 | 569 |
| RNA-seq (`norm_exp` içinde bulunan hasta sütunu) | 671 | — |
| **İnhibitör ∩ RNA-seq (`norm_exp`)** | **520** | **479** |
| İnhibitör ∩ DNA-seq (klinik kimlik) | 603 | — |
| İnhibitör ∩ RNA-seq ∩ DNA-seq | 494 | 458 |
| `Waves1+2` (İnhibitör ∩ RNA-seq) | 314 | 290 |
| `Waves3+4` (İnhibitör ∩ RNA-seq) | 183 | 174 |
| `Both` (İnhibitör ∩ RNA-seq) | 23 | 23 |

Not: `probit_curve_fits` içinde klinik tabloyla eşleşmeyen (orphan) örnek sayısı **0**, mükerrer `(örnek, inhibitör)` satır sayısı **0**.

---

## 3. İnhibitör Matrisi Kalitesi, Sansürleme ve Seyreklik

- **Neden birincil hedef `auc` (IC50 değil)?** `ic50` değerlerinin **%45.36**'si `max_conc` üst sınırında (`all_gt_50`), **%5.14**'si `min_conc` alt sınırında (`all_lt_50`) sansürlüdür (toplam **%50.50** sansürlü). `auc` kesintisizdir (min=0.0, medyan=207.5, std=64.28, maks=286.3); yakınsama oranı **%99.98**.
- **166 İnhibitörün `status` dağılımı:** `train/test` = **95** (51010 satır), `combined` = **28** (9363 satır), `exploratory` = **43** (3022 satır).
- **Her iki dalgada (`Waves1+2` ve `Waves3+4`) en az 100'er örnekte ölçülen ortak ilaç:** **114** ilaç ({'train/test': 95, 'combined': 19}); her iki dalgada RNA-seq'li en az 80'er örnekte ortak ölçülen: **114** ilaç.

| Alt Küme | Örnek | Hasta | İlaç | Gözlem | Doluluk (%) | Eksiklik (%) | Örnek Başına İlaç (Medyan [P10–P90]) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tüm panel (`all_166`) | 631 | 569 | 166 | 63395 | 60.5 | 39.5 | 114 [18–121] |
| `train/test` (95 ilaç) | 631 | 569 | 95 | 51010 | 85.1 | 14.9 | 93 [17–95] |
| Çift-dalga ≥100 (`cross_wave`) | 631 | 569 | 114 | 57978 | 80.6 | 19.4 | 105 [17–114] |
| Aday Küme $\mathcal{C}$ (24 ilaç) | 631 | 569 | 24 | 10890 | 71.9 | 28.1 | 20 [7–24] |

### Örnek Başına İnhibitör Kapsam Eşikleri (`cross_wave` ortak ilaç seti)

| Eşik | Toplam Örnek | Toplam Hasta | RNA-seq'li Örnek | RNA-seq'li Hasta | W1+2 RNA Hasta | W3+4 RNA Hasta |
|---|---:|---:|---:|---:|---:|---:|
| ≥30 ilaç | 551 | 507 | 467 | 434 | 264 | 156 |
| ≥50 ilaç | 549 | 505 | 466 | 433 | 264 | 155 |
| ≥80 ilaç | 529 | 486 | 449 | 417 | 253 | 150 |
| ≥100 ilaç | 376 | 342 | 330 | 305 | 165 | 137 |

---

## 4. Çift-Yönlü `M0` (Ortalama-Etki) Varyans Ayrıştırması — Gerçek Tahmin Tavanı

*DrEval* (2026) bulgusuna paralel olarak, herhangi bir gelişmiş model kurmadan önce ham `auc` varyansının ne kadarının yalnızca ortalama etkilerle açıklandığı ölçülmüştür ($y_{p,d} = \mu + b_d + b_p + r_{p,d}$):

| Alt Küme | Toplam Std | İlaç Ort. $R^2$ | Hasta Ort. $R^2$ | Çift-Yön `M0` $R^2$ | `M0` RMSE | Artık ($r_{p,d}$) Std | Hasta-İçi Sıralama $R^2$ (`M0`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tüm 166 ilaç | 64.28 | 0.416 | 0.134 | 0.549 | 43.16 | 43.16 | 0.479 |
| `train/test` (95 ilaç) | 63.31 | 0.407 | 0.153 | 0.557 | 42.12 | 42.12 | 0.477 |
| Aday Küme $\mathcal{C}$ (24 ilaç) | 66.62 | 0.306 | 0.168 | 0.469 | 48.54 | 48.54 | 0.362 |

**Kritik metodolojik sonuç:** Ham `auc` üzerinde `M0` zaten tek başına yüksek $R^2$ açıklamaktadır (bunun ana nedeni ilaçların kendi ortalama duyarlılık farklarıdır). Bu yüzden modellerin başarısını yalnızca ham `auc` korelasyonu ile ölçmek **yanıltıcıdır**; Sprint 2 protokolünde mutlaka **ilaç-ortalamasından arındırılmış kalıntı ($r_{p,d}$) korelasyonu / RMSE'si** ile **hasta-içi sıralama kazancı (`ΔNDCG@3` vs `M0`)** esas alınmalıdır.

---

## 5. Klinik/Translasyonel Aday Küme ($\mathcal{C}$) Taslağı ve İlaç Aileleri

- `beataml_drug_families.xlsx` dosyası 166 inhibitörün **117** tanesini kapsıyor (**49** kinaz-dışı/ek ajan burada yok; `drug_gene` sekmesi **117** ilacı hedeflerine bağlıyor).
- Hem `Waves1+2` hem `Waves3+4`'te iyi kapsanan, AML'de klinik olarak anlamlı **24 aday ilaç ($\mathcal{C}$)** aşağıdadır (`Venetoclax`, `Azacytidine`, `Cytarabine`, `Gilteritinib`, `Midostaurin`, `Quizartinib`, `Sorafenib` dahil):

| İnhibitör | Status | Toplam Örnek | W1+2 | W3+4 | W1+2 (RNA) | W3+4 (RNA) | Ort. AUC | Std AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dasatinib | train/test | 598 | 369 | 210 | 298 | 176 | 167.6 | 59.9 |
| Sorafenib | train/test | 598 | 373 | 203 | 301 | 171 | 188.0 | 49.5 |
| Trametinib (GSK1120212) | train/test | 581 | 347 | 212 | 286 | 176 | 130.8 | 67.3 |
| Quizartinib (AC220) | train/test | 556 | 351 | 184 | 281 | 159 | 148.8 | 51.2 |
| Rapamycin | train/test | 553 | 354 | 177 | 291 | 153 | 164.3 | 56.1 |
| Bortezomib (Velcade) | train/test | 550 | 339 | 189 | 281 | 160 | 156.4 | 58.0 |
| Midostaurin | train/test | 550 | 326 | 205 | 271 | 172 | 197.1 | 58.8 |
| Ruxolitinib (INCB018424) | train/test | 550 | 344 | 188 | 278 | 153 | 225.8 | 35.7 |
| Selumetinib (AZD6244) | train/test | 538 | 330 | 187 | 275 | 160 | 189.9 | 70.9 |
| Idelalisib | train/test | 535 | 331 | 182 | 270 | 152 | 190.3 | 47.2 |
| Crenolanib | train/test | 531 | 337 | 176 | 278 | 150 | 208.0 | 53.2 |
| Ibrutinib (PCI-32765) | train/test | 525 | 314 | 198 | 254 | 171 | 198.1 | 53.7 |
| Venetoclax | combined | 430 | 221 | 197 | 183 | 172 | 149.4 | 79.0 |
| Azacytidine | combined | 422 | 238 | 173 | 199 | 157 | 231.6 | 46.1 |
| Volasertib (BI-6727) | combined | 400 | 216 | 173 | 181 | 157 | 157.9 | 42.9 |
| Cytarabine | combined | 375 | 194 | 170 | 163 | 155 | 188.8 | 52.9 |
| Entospletinib (GS-9973) | combined | 357 | 179 | 168 | 149 | 154 | 178.7 | 51.8 |
| AGI-5198 | combined | 341 | 213 | 117 | 178 | 110 | 241.0 | 38.8 |
| AGI-6780 | combined | 339 | 213 | 115 | 179 | 107 | 233.6 | 44.1 |
| Palbociclib | combined | 338 | 154 | 173 | 128 | 156 | 227.9 | 44.7 |
| Panobinostat | combined | 328 | 151 | 166 | 125 | 150 | 93.9 | 74.8 |
| Gilteritinib | combined | 317 | 136 | 170 | 112 | 155 | 136.9 | 61.4 |
| Lenalidomide | combined | 304 | 121 | 172 | 97 | 157 | 235.4 | 55.0 |
| Selinexor | combined | 274 | 94 | 172 | 76 | 157 | 104.4 | 50.0 |

---

## 6. Eşleşmiş Kohortta (`Drug ∩ RNA-seq`) Başlıca Sürücü Mutasyonlar ve Tanı Kapsamı

- **Kritik biyoinformatik tespit (`FLT3-ITD`):** WES (`beataml_wes_wv1to4_mutations_dbgap.txt`) yalnızca nokta/küçük indel çağrılarını (`FLT3` %10.3) içerir; `FLT3-ITD` ise klinik özet sütunundadır ve tüm kohortta **219** örnekte (**185** hasta), `Drug ∩ RNA-seq` eşleşmiş kohortta ise **129** örnekte (**123** hasta) pozitiftir. `NPM1` pozitif hasta sayısı eşleşmiş kohortta **132**'tir.
- **Tanı kapsamı (`ELN2017 == NonAML`):** Tüm tabloda **29** örnek, `Drug ∩ RNA-seq` eşleşmiş kohortta ise **8** örnek (**8** hasta) `NonAML` (MDS/MPN veya diğer) olarak etiketlidir.

| Gen (WES çağrısı) | Mutasyonlu Hasta | WES Hastası | Frekans (%) |
|---|---:|---:|---:|
| FLT3 | 47 | 458 | 10.3 |
| NPM1 | 108 | 458 | 23.6 |
| DNMT3A | 98 | 458 | 21.4 |
| TET2 | 69 | 458 | 15.1 |
| IDH2 | 55 | 458 | 12.0 |
| IDH1 | 34 | 458 | 7.4 |
| TP53 | 41 | 458 | 9.0 |
| NRAS | 75 | 458 | 16.4 |
| KRAS | 26 | 458 | 5.7 |
| RUNX1 | 49 | 458 | 10.7 |
| ASXL1 | 48 | 458 | 10.5 |
| SRSF2 | 60 | 458 | 13.1 |

---

## 7. Kapı 0 Değerlendirmesi ve Sprint 2 Öncesi Karar Soruları

**Kapı 0 Sonucu: GEÇTİ (Veri yeterli, ama sıkı sızıntı temizliği zorunlu).**
- Hem ilaç hem RNA-seq verisine sahip **479** benzersiz hasta (**520** örnek) mevcut.
- Çift-dalgada ortak ölçülen **114** inhibitör içinde `≥80 ilaç` ölçülmüş **RNA-seq'li 417 hasta** (W1+2: 253, W3+4: 150) var — güçlü `5-Fold CV` ve `Waves1+2 → Waves3+4` doğrulaması için yeterli.

### Protokolü (`docs/PROTOCOL.md`) Dondurmadan Önce Netleştirilecek 4 Karar
1. **Hedef Aday Küme ($\mathcal{C}$) ve Prob Havuzu ($\mathcal{U}$):** Yukarıdaki 24 AML klinik adayı $\mathcal{C}$ olarak dondurulsun mu, yoksa yalnızca `train/test` (95) içindeki klinik ilaçlar mı yoksa `combined` (`Venetoclax`, `Gilteritinib`, `Azacytidine`, `Cytarabine`, `Panobinostat`) mutlaka dahil edilsin mi? *(Öneri: Hepsi çift-dalga kapsamlı olduğundan bu 24 klinik ilacın tamamını $\mathcal{C}$ olarak almak; prob havuzu $\mathcal{U}$ ise `cross_wave_ge100` havuzu — 114 ilaç.)*
2. **Çoklu Örnek Kuralı:** Birden fazla örneği olan hastalarda ana bölme `dbgap_subject_id` üzerinden yapılacak; test kümesinde hasta başına yalnızca **ilk/tanı örneği** mi puanlansın yoksa tüm zaman noktaları mı (küme ağırlıklı)? *(Öneri: Testte hasta başına ilk örnek, duyarlılık analizinde tüm örnekler.)*
3. **`NonAML` Örnekleri:** `ELN2017 == NonAML` olan 8 eşleşmiş örnek (8 hasta) ana AML çalışmasından çıkarılsın mı? *(Öneri: Evet, birincil analiz saf AML'de yapılıp `NonAML` yalnızca duyarlılıkta incelenmeli.)*
4. **Birincil Başarı Ölçütü:** İlaçların ortalama AUC farkının yarattığı sahte başarıyı önlemek için birincil metrik **`residual_auc` (ilaç ortalamasından arındırılmış RMSE / hasta-içi Spearman)** ve **`ΔNDCG@3 (vs M0)`** olarak dondurulsun mu?
