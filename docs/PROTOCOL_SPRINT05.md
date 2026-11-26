# OncoTwin v1 — Sprint 5 Protokolü (`PROTOCOL_SPRINT05.md`)

**Durum:** DONDURULMUŞ (Sprint 5 deneyleri çalıştırılmadan ve Sprint 5 sonuçlarına bakılmadan sabitlenmiştir; [PROTOCOL.md](PROTOCOL.md), [PROTOCOL_SPRINT03.md](PROTOCOL_SPRINT03.md), [PROTOCOL_SPRINT04.md](PROTOCOL_SPRINT04.md), [PROTOCOL_SPRINT04_ADDENDUM_19.md](PROTOCOL_SPRINT04_ADDENDUM_19.md) ve tüm `sprint01..04` çıktıları değiştirilmemiştir).  
**Referanslar:** [sol-architect-reviewer.md](agents/sol-architect-reviewer.md) §7 & §8.3; [opus-optimization-qa.md](agents/opus-optimization-qa.md) §3.8–§3.9; [DECISION_LOG.md](DECISION_LOG.md) `AD-001`, `AD-007`, `AD-008`; ön-modelleme katalog denetimi `data/processed/sprint05_fimm_catalog_audit.json`.  
**Güncel Literatür Konumlandırması (Kasım 2026):** Tercan (*PLOS ONE*, 2026, `doi:10.1371/journal.pone.0343422`, Beat AML → FPMTB sıra-tabanlı batch-dirençli aktarım); Traub ve ark. (*NeurIPS* 2024, `arXiv:2407.01032`, `AUGRC`); Liao ve ark. (`arXiv:2601.23128`, 2026, DCR konformal sıralama); Sale & Hüllermeier (*COPA* 2026, PMLR 329, konformal kısmi sıralama).

---

## 1. Kulvar A — `M4` Yolak / Hedef-Ailesi Yapılı Kovaryans Büzüştürme ve Kapı 3 (`AD-001`)

### 1.1 Model Tanımı (`M4`) ve İç İçe Geçmişlik
- **Referans:** `StaticDOpt + M3, k=5` korunur (Sol §8.3 #1).
- **Biyolojik Öncül:** Yalnızca Beat AML 2.0 ile birlikte yayımlanan `beataml_drug_families.xlsx` (`drug_family` sekmesi; elle sonradan hedef ekleme yok) ikili `(ilaç × aile)` üyelik matrisi $B \in \{0,1\}^{p \times m}$ kullanılır (`id_60_20_20` aktif `120` ilaçta `89` anotasyonlu [`15/24` hedef, `74/96` prob], `327` üyelik kenarı, `843` paylaşımlı-aile çifti).
- Satırları birim $\ell_2$ normuna çekilmiş $\tilde{B}$ matrisinden kesin yarı-pozitif tanımlı (PSD) birim-diyagonalli kosinüs Gram matrisi kurulur:
  $$G_{\text{unit}} = \tilde{B}\tilde{B}^\top + \operatorname{diag}\!\left(\max(1 - \operatorname{diag}(\tilde{B}\tilde{B}^\top), 0)\right) \succeq 0.$$
- Marjinal eğitim kalıntı standart sapmaları $s_i = \sqrt{\max(\hat{\Sigma}_{\text{psd}, ii}, 10^{-4})}$ ile kesin pozitif tanımlı biyolojik büzüştürme hedefi ve `M4` kovaryansı tanımlanır:
  $$T_{\text{bio}}(\rho) = \operatorname{diag}(s)\bigl[(1-\rho)I + \rho G_{\text{unit}}\bigr]\operatorname{diag}(s), \qquad \Sigma_{M4}(\alpha, \rho) = (1-\alpha)\hat{\Sigma}_{\text{psd}} + \alpha T_{\text{bio}}(\rho).$$
- **İç içe özellik:** $\rho = 0$ olduğunda $\Sigma_{M4}(\alpha, 0) \equiv \Sigma_{M3}(\alpha)$ (birebir `M3` özdeşliği).
- **Hiperparametre Ayarı:** $(\alpha, \rho, \lambda)$ üçlüsü yalnız eğitim katı içi 4-kat hasta-ayrık CV ile seçilir: $\alpha \in \{0.05, 0.15, 0.30, 0.50\}$, $\rho \in \{0.0, 0.15, 0.30, 0.50, 0.70\}$, $\lambda \in \{50, 200, 500, 1500, 5000\}$. Ayrıca `M4` iç-CV'de $\rho = 0$ seçerse biyolojik öncülün yönünü izlemek için zorunlu pozitif $\rho \in \{0.15, 0.30, 0.50, 0.70\}$ tanısal kolu (`M4_forced_pos`) ayrı raporlanır.

### 1.2 Aile-Boyutu ve İlaç-Derecesi Korumalı Karıştırılmış Öncül Kontrolü (`M4_perm_null`)
- Bipartite çift-kenar takası (`permute_bipartite_incidence_preserving_degrees`, `30 × |E|` takas girişimi) ile **her bireysel ilacın aile sayısı (`row_sum`)** ve **her ailenin büyüklüğü (`col_sum`)** birebir korunurken ilaç–aile biyolojik eşleşmesi karıştırılır.
- Tek-çekiliş gürültüsünü önlemek için **`M = 20`** deterministik karıştırma çekilişi (`seed = 5000 + m`) koşulur; hasta-düzeyi kayıplar 20 çekiliş üzerinden ortalanarak `M4` ile eşleştirilir.

### 1.3 İki Ayrı Deney ve Bölmeler (Sol §8.3 #1)
- **Deney A1 (Birincil — Sabit Panel):** Yalnızca kovaryans öncülü değiştirilir; **aynı hastalar, aynı hedefler ve aynı `StaticDOpt(M3)` paneli** üzerinde `M3`, `M4`, `M4_forced_pos` ve `M4_perm_null` kıyaslanır (`k = 5` ana; `k ∈ {2, 8}` duyarlılık).
  1. **Birincil ID:** Sprint 3 hasta-ayrık 5-Kat CV × 3 tohum (`n_paired = 427` indeks hasta).
  2. **İkincil ID Holdout:** Sprint 4 `id_60_20_20` ayrımı (`n_paired = 91` test hastası).
  3. **Keşifsel Zamansal Stres Testi:** Temiz `Waves1+2 → Waves3+4` (`n_paired = 146` Sprint 3 bölmesi ve `n_paired = 148` Sprint 4 bölmesi; hiçbir seçim/ayar için kullanılmaz).
- **Deney A2 (İkincil — Yeniden Seçilmiş Panel):** `M4` kovaryansı kullanılarak yeniden seçilen `StaticDOpt(M4) + M4, k=5` ile `StaticDOpt(M3) + M3, k=5` aynı hastalar ve hedefler üzerinde ayrı deney olarak kıyaslanır.

### 1.4 Ön Kayıtlı Kapı 3 (`Gate 3`, `AD-001`) Karar Kuralı
- **Ana metrik ve payda:** Deney A1 (`StaticDOpt, k=5`) üzerinde hasta-kümeli eşleştirilmiş `ΔRMSE` (negatif = iyi), `n_paired = 427` (ID 5-Fold CV × 3 tohum); eş-metrikler `ΔRegret@1` ve `ΔNDCG@3`.
- **Anlamlı etki eşiği:** `0.50 AUC` (`ΔRMSE ≤ −0.50`).
- **PASS (`v2` 15–25 düğümlü / 3–5 serbest parametreli daraltılmış ODE değerlendirmesine geçiş)** yalnız aşağıdaki 3 koşulun **tamamı** sağlandığında verilir:
  1. ID CV (`n=427`): `ΔRMSE(M4 − M3) ≤ −0.50` ve %95 hasta-bootstrap GA üst sınırı `< 0` (`p < 0.05`), `Regret@1` ve `NDCG@3`'te istatistiksel bozulma yok.
  2. ID CV (`n=427`): `ΔRMSE(M4 − M4_perm_null) ≤ −0.50` ve %95 GA üst sınırı `< 0` (`p < 0.05`).
  3. Keşifsel `Waves3+4` (`n=146`): nokta kestirimi `ΔRMSE(M4 − M3) < 0` ve `ΔRMSE(M4 − M4_perm_null) < 0`.
- Herhangi biri sağlanmazsa: **Kapı 3 = `NO-GO (v2 ODE ERTELENİR)`**.

---

## 2. Kulvar B — Hasta-Düzeyi Sıra-Kalibreli Konformal Kısa Aday Listesi & `AUGRC` (Sol §8.3 #2)

- **Girdi:** Sprint 4 `v2` dondurulmuş bölmeleri ve tahminleri (`id_60_20_20` `n_cal=90, n_test=91` birincil; `wave_ood` `n_cal=71, n_test=148` keşifsel stres testi). `Waves3+4` hiçbir yöntem veya eşik seçiminde kullanılmaz.
- **Kıyaslanan 4 Konformal Kısa Liste Yöntemi ($\alpha = 0.10$, nominal %90, `SOL-11` sonlu-eşik korumalı):**
  1. `EqualWidthAUC` (Sprint 4 v2 tabanı): $\max_{O_i}|y - \hat{y}|$ ile $\hat{q}$; $[L, U]$ kesişimi $\iff \hat{y}_{j,d} \le \min_{O_j}\hat{y} + 2\hat{q}$.
  2. `RankConformal` (Sol §8.3 #2 birincil): Her kalibrasyon hastasında gerçek en iyi hedefin bağ-koruyucu 1-tabanlı tahmin sırası $R_i = \max_{d^* \in \arg\min y} (1 + \#\{d \in O_i : \hat{y}_{i,d} < \hat{y}_{i,d^*}\})$ üzerinden sonlu-örneklem konformal eşik $\hat{k}_\alpha = R_{(\lceil (n+1)(1-\alpha) \rceil)}$; testte ilk $\hat{k}_\alpha$ sıradaki (eşik bağı dahil) hedefler seçilir.
  3. `RelRankConformal`: Göreli sıra $R_i / |O_i|$ üzerinden $\hat{q}_{\text{rel}}$; testte ilk $\lceil \hat{q}_{\text{rel}} |O_j| \rceil$ hedef seçilir.
  4. `Top1GapConformal`: Gerçek en iyi hedefin tahmin edilen #1'e göre marjı $M_i = \max_{d^*}(\hat{y}_{i,d^*} - \min_{O_i}\hat{y}) \ge 0$ üzerinden $\hat{m}_\alpha$; testte $\hat{y}_{j,d} \le \min_{O_j}\hat{y} + \hat{m}_\alpha$ (hasta-uyarlanabilir liste boyutu ve tekil aday olanağı).
- **Ana Kol ve Karşılaştırıcılar:** `StaticDOpt_k5` ana; `M0_k0`, `M1_k0`, `StaticDOpt_k2`, `StaticDOpt_k8`, `AdaptiveOED_k5` karşılaştırıcı.
- **Birlikte Raporlanan Metrikler:** `shortlist_contains_true_top1_rate`, `shortlist_mean_size_obs`, `shortlist_mean_fraction_obs`, `definite_top1_rate_obs` ve **AUC kaybı** (`shortlist_best_regret1` = $\min_{S} y - \min_{O} y$, `shortlist_mean_regret1` = $\operatorname{mean}_{S} y - \min_{O} y$, `shortlist_worst_regret1` = $\max_{S} y - \min_{O} y$) + eşleştirilmiş bootstrap farkları.
- **Ek Seçici Risk Duyarlılığı (`AUGRC`, Opus §3.9 / Traub et al., NeurIPS 2024):** Sprint 4 `v2` `omics_pca_dist` ve `m3_post_scale` çekinme skorları için `AUGRC` ($G(c) = c \cdot R(c)$ yamuk integrali), `E-AUGRC` ve `Random AUGRC` ($0.5\,\bar{\ell}$) ek duyarlılık tablosu olarak raporlanır.

---

## 3. Kulvar C — FIMM / Malani Dış Sıralama Tekrarı (`Zenodo 7370747`, Sol §8.3 #3)

### 3.1 Ön-Modelleme Katalog, Kimyasal Kimlik, Doz ve Assay QC Özeti (`sprint05_fimm_catalog_audit.json`)
- **Doğrulanan Paket:** `Functional_Precision_Medicine_Tumor_Board_AML.zip` (`38,810,683 B`, MD5 `3db5280e20f315d56d6bf07dd6a9e241`).
- **Örnek ve Hasta Sayıları:** `File_0`: `252` AML örnek / `186` hasta; `File_3.1` sDSS: `515` bileşik × `181` örnek (`164 AML + 17 sağlıklı`); `File_7` RNA log2-CPM: `167` örnek (`163 AML + 4 sağlıklı CD34+`, Opus §3.9 düzeltmesi doğrulandı). Eşleşmiş AML RNA + sDSS = **`132` örnek / `109` benzersiz `Patient_ID`** (`92` tek örnekli, `13` iki, `2` üç, `2` dört örnekli hasta; `125` örnekte `File_6` ikili mutasyon paneli mevcut).
- **Hasta Gruplaması:** Birincil analiz deterministik indeks örnek seçimiyle **`109` benzersiz hasta** (`Diagnosis < Relapse < Refractory`, `Bone marrow < Blood`, en çok gözlenen ortak ilaç, leksikografik `Sample_ID`) üzerinde yürütülür; ek olarak tüm **`132` örnek** üzerinde hasta-kümeli (`Patient_ID`) bootstrap duyarlılığı verilir.
- **Kimyasal Kimlik, Doz Aralığı ve Assay Farkı:**
  - Beat AML `166` bileşiğin **`102`** tanesi FIMM `Drug_ID` ile tekil ve belirsizsiz eşleşti (`0` belirsiz; `Azacytidine → FIMM023825`, `Rapamycin → FIMM023795`).
  - Hedef kümesi: **`23/24`** hedef ortak (`Gilteritinib` FIMM kütüphanesinde yok). Beat AML tüm-471 havuzundan (`min_obs=100`) **`64/101`** prob ortak.
  - Maksimum konsantrasyon uyumu (`23` ortak hedef): `12` birebir `1×`, `10` FIMM daha düşük maks. doz (`0.5×`–`0.01×`, örn. `Trametinib` `250 nM` vs `10,000 nM`, `Rapamycin` `100 nM` vs `10,000 nM`), `1` FIMM daha yüksek (`Lenalidomide` `100,000 nM` vs `10,000 nM`).
  - Assay farkı: Beat AML `72h MTS` (7-nokta, probit AUC, düşük = duyarlı) vs FIMM `72h CellTiter-Glo` (`CM: 71`, `MCM: 61`, 5-nokta 10×, sağlıklı-kontrollü `sDSS`, yüksek = duyarlı); RNA kütüphanesi `Scriptseq: 102`, `Nextera: 30`; ortak Ensembl gen sayısı **`14,688`**, ortak sürücü mutasyon **`15`**.
- **Yön Uyumu ve Konformal Sınır:** `y_fimm_aligned = -sDSS` (düşük = daha duyarlı). Beat AML konformal aralığı FIMM'e **kesinlikle taşınmaz**.

### 3.2 İki Ayrı FIMM Deneyi (Donmuş Aktarım ≠ Yöntem Tekrarı — Sol §8.3 #3)
1. **Deney C1 — Donmuş Beat AML Model Aktarımı (Sıfır FIMM Yeniden Fit, Sıfır FIMM Kohort İstatistiği):**
   - Beat AML (`471` indeks hasta) içinde yalnız `Beat AML ∩ FIMM` ortak kataloğunda (`23` hedef, `64` ortak prob, `14,688` ortak gen) modeller ve `StaticDOpt` paneli eğitilip **dondurulur**; FIMM verisi panele veya ölçeklere asla girmez.
   - Birim/batch farkını test kohortuna bakmadan aşmak için iki tek-hasta normalizasyonu uygulanır: (i) RNA için hasta-içi yüzdelik sıra `within_sample_rank_normalize` (Tercan 2026 *PLOS ONE* yaklaşımı), (ii) yanıt için hasta-içi z-skoru $z_{p,d} = (y_{p,d} - \bar{y}_p) / s_p$ (Beat AML eğitiminde tüm ölçülen ortak ilaçlar üzerinden; FIMM test hastasında few-shot güncellemesi için **yalnızca o hastanın gözlenen $k$ probu** üzerinden $z_{p,S}^{\text{obs}} = (y_{p,S} - \bar{y}_{p,S}) / \max(\text{sd}(y_{p,S}), \epsilon)$). Birim-varyanslı z-skor ölçeğine uygun ridge ızgarası $\lambda \in \{0.05, 0.20, 0.50, 1.50, 5.00\}$.
   - Kollar (`k ∈ {0, 2, 5, 8}`): `C1_M0_transfer` (`k=0`), `C1_M1_rank_transfer` (`k=0`), `C1_M2a_ztransfer` (`StaticDOpt` & `Random 20x`), `C1_M3_rank_ztransfer` (`StaticDOpt` & `Random 20x`).
   - Değerlendirme (yalnız sıralama ve sDSS regret): `NDCG@3`, `Recall@3`, `Within-Patient Spearman`, `Regret@1 (sDSS)` (eşleştirilmiş hasta-kümeli bootstrap GA ile).
2. **Deney C2 — FIMM İçi Yöntem Tekrarı (Donmuş Aktarımdan Ayrı Etiketli):**
   - Yalnızca FIMM `109` benzersiz hasta üzerinde hasta-ayrık **5-Kat CV × 3 sabit tohum (`2026, 2027, 2028`)** ile OncoTwin boru hattı (`M0`, `M1`, `M2a`, `M3`, `M4` [Beat AML `drug_family` ortak alt kümesi], `StaticDOpt` vs `Random 20x`, `k ∈ {0, 2, 5, 8}`) doğrudan FIMM `-sDSS` ölçeğinde sınanır (`-sDSS` varyansına ölçek-uyumlu $\lambda \in \{1.0, 5.0, 20.0, 75.0, 250.0\}$).
   - Her katta prob seçimi, gen filtresi, z-skorlama ve kovaryans **yalnız o katın eğitim hastalarından** öğrenilir.
   - Bu deney açıkça **"yöntem tekrarı (FIMM içi yeniden fit)"** olarak etiketlenir; donmuş aktarım (C1) ile karıştırılmaz.

---

## 4. Koşu Öncesi Zorunlu Doğrulama Sözleşmesi (`tests/test_sprint05_*.py`)
1. `M4`: $\rho = 0$ iken `M3` ile bit-düzeyinde özdeşlik; `G_unit` ve $\Sigma_{M4}$ simetrik/PD; `permute_bipartite_incidence_preserving_degrees` satır/sütun toplamlarını birebir korur; gizli test AUC'si değişince tahmin/panel değişmez.
2. Kısa liste & `AUGRC`: Sentetik değiştirilebilir Monte Carlo'da `RankConformal` ve `Top1GapConformal` kapsamasının $[1-\alpha, 1-\alpha+1/(n+1)]$ bandında olması; $q=\infty$ iken `hesaplanamadı`/`NaN`; sabit skorda `AUGRC == 0.5 * mean(loss)` (`Random AUGRC`).
3. FIMM: MD5 doğrulaması, hasta-ayrık CV sızıntısızlığı, `C1` aktarımında FIMM test etiketlerinin ve diğer test hastalarının sıfır etkisi (tek-hasta değişmezliği), `-sDSS` yön tutarlılığı ve `0` `RuntimeWarning`.

---

## 5. Çıktı Dosyaları
- FIMM katalog denetimi: `data/processed/sprint05_fimm_catalog_audit.json`
- Makine çıktısı: `data/processed/sprint05_results.json`
- Hasta düzeyi vektörler: `data/processed/sprint05_vectors.npz`
- Sprint 5 raporu: `docs/sprints/sprint-05-m4-shortlist-fimm.md`
