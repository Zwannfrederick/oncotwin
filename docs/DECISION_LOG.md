# Mimari Karar Kayıtları (Architecture Decision Log)

Bu belge, OncoTwin projesinin kapsam daraltma ve bilimsel yöntem kararlarını gerekçeleriyle kaydeder.

---

### AD-001: Tam Dijital İkiz ve Hill-ODE'nin v1'den Çıkarılması
- **Karar:** v1 sürümünde tam hasta dijital ikizi iddiası ve sürekli Hill-ODE simülatörü yer almayacaktır.
- **Gerekçe:**
  1. Bazal RNA-seq ve mutasyon verilerinden büyük kinetik ODE ağlarının hız sabitleri pratik olarak belirlenemez (Raue et al., 2009, [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)).
  2. ChEMBL $K_i$/$\text{IC}_{50}$ değerleri doğrudan hücresel etkili konsantrasyon değildir; hücre geçirgenliği, protein bağlanması, ATP rekabeti, hedef işgali ve efflux gibi ara basamaklar eksiktir.
  3. Az sayıda (3–5) skaler canlılık ölçümüyle 8–15 parametre serbest bırakmak öncül-baskın (prior-dominated) sonuç üretir.
- **Revizyon Koşulu:** Yalnızca Sprint 5'te (`M4` yolak-yapılı basit model) istatistiksel baseline'lara karşı tekrarlanabilir üstünlük ve derece-koruyucu kontrolü yenme sağlanırsa (Kapı 3), v2'de sabit 15–25 düğüm ve en fazla 3–5 serbest hasta parametresiyle (sentetik kurtarma testi dahil) değerlendirilir.

---

### AD-002: HR+/HER2− Meme Kanseri Yerine Tek Çekirdek Olarak Beat AML 2.0 Seçimi
- **Karar:** İlk prototip yalnızca **Beat AML 2.0** (Waves 1–4) primer hasta verisi üzerine kurulacaktır; HR+/HER2− meme kanseri, NSCLC, mCRC ve PDO doğrulaması v1'den çıkarılmıştır.
- **Gerekçe:**
  1. Guillen et al. (*Nature Cancer*, 2022, [doi:10.1038/s43018-022-00337-6](https://doi.org/10.1038/s43018-022-00337-6)) çalışmasında ER+ PDX oluşturma başarısı primer tümörlerde %9, metastazlarda %16 kalmıştır; fulvestrant ve CDK4/6 inhibitörlerinin sitostatik etkisi kısa süreli canlılık testlerinde güvenilir biçimde ayrışmamaktadır.
  2. *DrEval* (*Nature Communications*, 2026, [doi:10.1038/s41467-026-72903-w](https://doi.org/10.1038/s41467-026-72903-w)) hücre hatlarından primer/PDX verisine aktarımın zayıf olduğunu göstermiştir.
  3. Beat AML 2.0 primer insan lösemi hücreleri, hızlı 72 saatlik *ex vivo* inhibitör yanıtı ve çoklu-omik profillemesiyle tek veri kümesinde en temiz yöntem testini sunar (ancak hasta ≠ örnek ve matris seyrekliği denetimi zorunludur).

---

### AD-003: Ardışık Aktif Öğrenme Yerine Tek-Tur Toplu Panel Seçimi (Batch OED)
- **Karar:** Hastadan doku alınıp tek seferde laboratuvara gönderilen gerçekçi klinik/deneysel iş akışına uygun olarak ardışık döngü yerine **tek-tur toplu panel seçimi (Batch OED)** kullanılacak; $k \in \{1, 2, 3, 5, 8, 12\}$ panel boyutu sabit bir varsayım olarak değil öğrenme eğrisiyle test edilecektir.
- **Gerekçe:** Birincil hücre canlılığı kısa sürede düşer; laboratuvarda tek turda $k$ adet önceden seçilmiş inhibitörün plakalanması biyolojik ve lojistik olarak tek geçerli modeldir.

---

### AD-004: Toksisite Motoru (openFDA/CPIC/PDS) ve Tahoe-100M'nin v1'den Çıkarılması
- **Karar:** Toksisite risk modülü ve Tahoe-100M transkripsiyonel pertürbasyon atlası v1 kapsamı dışındadır.
- **Gerekçe:**
  1. Beat AML açık klinik özetinde boylamsal kan/biyokimya ve CTCAE toksisite dereceleri yer almaz; etkinlik ve toksisite aynı açık veri kümesinde birlikte doğrulanamaz.
  2. `openFDA` kendi belgelerinde verilerin doğrulanmamış olabileceğini ve klinik kararlarda kullanılmaması gerektiğini belirtir; `CPIC` *UGT1A1*–irinotekan kılavuzu 2024 durum belgesinde beklemededir; `Project Data Sphere` erişimi kayıt ve DUA sözleşmesine tabidir.
  3. Tahoe-100M canlılık/klinik yanıt veri kümesi değildir ve ilk hipotezi test etmek için gereksiz veri/depolama maliyeti yaratır.

---

### AD-005: ID Konformal Kalibrasyon ile OOD Çekinme (`ABSTAIN`) Ayrımı
- **Karar:** Değiştirilebilir (In-Distribution / ID) bölmelerde ayrı kalibrasyon kümesiyle marjinal split-conformal aralıklar raporlanacak; dalga kayması (`Waves 3+4`) veya özellik uzayında uzaklık (OOD) durumlarında kapsam garantisi iddia edilmeyip doğrudan `ABSTAIN` çekinmesi uygulanacaktır.
- **Gerekçe:** Değiştirilebilirlik bozulduğunda klasik konformal kapsam garantisi geçerliliğini yitirir; OOD ve aktif seçim altında sahte güven aralığı üretmek yerine çekinmek ve Selective Risk–Coverage (`AURC`) raporlamak bilimsel olarak dürüst yaklaşımdır.

---

### AD-006: Veri-Güdümlü Go/No-Go Eşikleri
- **Karar:** Önceden keyfi "+0.10 Spearman" veya "%30 az deney" gibi eşikler tanımlanmayacak; tüm Go/No-Go eşikleri Sprint 1 (Kapı 0) veri denetiminden çıkacak gerçek örneklem büyüklüğü, ilaç kapsaması, `M0` varyans ayrıştırması ve bootstrap güven aralıklarına göre `docs/PROTOCOL.md` içinde dondurulacaktır.
- **Gerekçe:** Beklenen varyans ve efektif örneklem büyüklüğü bilinmeden yazılan sayısal eşikler güç hesabına dayanmaz.

---

### AD-007: Kolektif 3-Ajan Yönetişimi ve Tarihsel Çıktı Değişmezliği (Immutability)
- **Karar:** Proje, kullanıcı denetiminde üç imzalı ajan panosuyla yönetilir (`Gemini 4 Neon` — Ana Geliştirici, `GPT-6.2 Sol` — Araştırma Mimarı ve Hakem, `Claude Opus 5.5` — Optimizasyon ve Son Kontrol). Dondurulmuş `docs/PROTOCOL*.md` dosyaları sonradan değiştirilmez; ilk koşu çıktıları (`data/processed/sprint0{1,2,3}_*.json`, `docs/sprints/sprint-0{1,2,3}-*.md`) ezilmez. Düzeltici koşular ayrı sürüm adıyla (`*_v2_*`) kaydedilir ve yeniden bootstrap/yeniden analiz için hasta düzeyindeki vektörler `.npz` dosyasında kalıcılaştırılır.
- **Gerekçe:** Sol §5–§7 ve Opus `OPUS-001`/`OPUS-010` denetimlerinde, hasta vektörleri silindiği için yeniden bootstrap yapılamaması (`NEEDS_EVIDENCE`) ve tarihsel ilk kaydın keşifsel yeniden analizden ayrılması ihtiyacı doğrulandı.

---

### AD-008: Sprint 4 Hasta-Birimli Konformal Kalibrasyon, Bütçe/Politika Referansı ve `Waves 3+4` Keşifsel Stres Testi Etiketi
- **Karar:** `AD-005` ve Sol §3.2 hükümleri somutlaştırılarak:
  1. **Panel politikası referansı:** Sprint 4'te çalışma referansı **`StaticDOpt`**, karşılaştırıcı **`AdaptiveOED`** olacaktır (`ADAPTIVE_WINS` etiketi, `Waves 3+4` `k=8`'de `AdaptiveOED NDCG@3 = 0.868` vs `StaticDOpt NDCG@3 = 0.879` tersliği ve asimetrik karar kuralı nedeniyle Sprint 4 seçimine temel yapılmaz).
  2. **Bütçe kolları:** Ön-kayıtlı `k*=2` tarihsel kaydı korunarak Sprint 4 bütçeleri **`k = 5` (ana)**, **`k = 2` (düşük bütçe)** ve **`k = 8` (duyarlılık)** olarak dondurulur.
  3. **Kalibrasyon izolasyonu:** Hasta düzeyinde (`dbgap_subject_id`) ayrı eğitim/ayar, kalibrasyon ve test bölmesi uygulanır; bu ayrım sağlandığında nested-OOF split-conformal kapsam için önkoşul değildir (Sol §3.2 #1). OOF $\Sigma_{\text{res}}$ posterior varyansı ham kapsam aralığı olarak kullanılmaz (`OPUS-005`).
  4. **OOD çekinme (`ABSTAIN`) ve alt gruplar:** Birincil OOD skoru yalnızca eğitimde öğrenilen omik uzaklık, ikincil karşılaştırıcı ise `M3` posterior ölçeğidir (kapsam garantisiz); çekinme eşiği eğitim/ayar doğrulamasında önceden belirlenmiş kabul oranıyla kilitlenir. `Waves 3+4` kümesi Sprint 2–3'te incelendiği için *keşifsel OOD stres testi* olarak etiketlenir. `Remission` alt grubundaki 8 yalnız-remisyon hastası ana kohortta kalır.
- **Gerekçe:** Opus `OPUS-005` (`audits/opus-2026-11-21/realdata_probe_v1.json`), test kalıntı varyansı / OOF $\Sigma_{\text{res}}$ diyagonal medyan oranının CV'de `0.75–1.23`, `Waves 3+4`'te `1.41` olduğunu gösterdi (ham aralık OOD'de ~%40 dar kalıyor). Sol §3.2 (#1–#7) ve §6.2–§6.3, çoklu ilaç korelasyonu altında değiştirilebilirlik biriminin hasta olması gerektiğini, `ADAPTIVE_WINS` bulgusunun karışık kaldığını ve bütçe/OOD kurallarının Sprint 4 verisi görülmeden kilitlenmesi gerektiğini kesinleştirdi.

---

### AD-009: Sprint 5 Kapı 3 (`NO-GO — v2 ODE ERTELENİR`), Sıra-Kalibreli Konformal Kısa Liste ve FIMM Dış Tekrar Sınırları
- **Karar:**
  1. **Kapı 3 (`AD-001` revizyon koşulu): `NO-GO (v2 ODE ERTELENİR)`.** `beataml_drug_families.xlsx` yolak/hedef-ailesi yapılı kovaryans büzüştürme modeli (`M4`), sabit `StaticDOpt(M3)` paneli (`k=5`, Deney A1) üzerinde ne yolaksız `M3`'ü ne de 20 çekilişli derece- ve aile-boyutu korumalı permüte öncülü (`M4_perm_null`) geçebilmiştir (`ID 5-Fold CV n=427`: `ΔRMSE(M4 − M3) = +0.01 [-0.01, +0.03], p=0.294`; `ΔRMSE(M4 − M4_perm_null) = +0.02 [-0.00, +0.04], p=0.128`; FIMM `C2 n=109`: `ΔRMSE = -0.00 [-0.01, +0.01], p=0.444`). Bu nedenle v2 kinetik Hill-ODE geliştirimi ertelenir; ana yöntem çekirdeği `M3 + StaticDOpt` olarak kalır.
  2. **Kısa liste kalibrasyonu & `AUGRC`:** Eşit-genişlikli AUC aralıkları (`EqualWidthAUC`, `StaticDOpt_k5` ID'de `18.70` ilaç / `%98.7` — klinik olarak ayırt edici değil) yerine hasta-düzeyi sıra-kalibreli konformal kısa listeler (`RankConformal` `k̂=6`, `%93.4` gerçek #1 kapsamı, `6.00` ilaç [`-12.70`, `p=0.002`]; `Top1GapConformal` `m̂=74.2`, `%92.3` kapsam, `5.45` ilaç, `%9.9` tekil karar ve `Mean Regret@1` `116.77 → 53.83`, `p=0.002`) benimsenir. OOD (`Waves3+4`) altında `RankConformal` ölçek-değişmezliği sayesinde `%92.6` kapsamını korurken `Top1GapConformal` `%86.5`'e kayar; bu nedenle OOD riski varken ölçek-değişmez `RankConformal` birincil güvenli kısa liste, `Top1GapConformal` ID tekil-karar tamamlayıcısı olarak raporlanır. `AUGRC` (Traub et al., NeurIPS 2024, `arXiv:2407.01032`) Sprint 4 `v2` `AURC` yönünü teyit eder (`W3+4`: `omics 14.27 > Rand 12.30`, `m3_post 10.57 < Rand 12.30`).
  3. **FIMM/Malani (`Zenodo 7370747`) dış tekrarı:** Donmuş Beat AML aktarımı (**Deney C1**, sıfır FIMM yeniden fit / sıfır FIMM kohort istatistiği, tek-hasta sıra-normalize RNA [Tercan 2026] + tek-hasta prob z-skoru, `23/24` hedef, `64` ortak prob, `109` benzersiz hasta) ile FIMM içi 5-kat CV **yöntem tekrarı (Deney C2)** kesin biçimde ayrı raporlanır; Beat AML konformal aralıkları FIMM'e taşınmaz (`conformal_intervals_transferred = False`).
- **Gerekçe:** Ön kayıtlı [PROTOCOL_SPRINT05.md](PROTOCOL_SPRINT05.md) kuralları altında tüm kanıtlar [sprint-05-m4-shortlist-fimm.md](sprints/sprint-05-m4-shortlist-fimm.md) ve `data/processed/sprint05_{results.json,vectors.npz}` dosyalarında kalıcılaştırılmıştır. Ampirik artık kovaryansı (`M3`) iki bağımsız kohortta da (Beat AML `n=427` ve FIMM `n=109`) el ile küratörlüğü yapılmış ilaç-aile etiketlerinin ötesindeki çapraz-hedef/polifarmakoloji korelasyonlarını zaten yakalamaktadır.
