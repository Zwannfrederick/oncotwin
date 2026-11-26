# OncoTwin v1 — Yöntem ve Tekrarlanabilirlik Raporu

**Sprint 6 sürümü · 26 Kasım 2026 · GPT-6.2 Sol (bilimsel çekirdek, `2026-11-25`) & Gemini 4 Neon (tekrarlanabilir araştırma paketi & izlenebilirlik, `2026-11-26`).** Bu rapor Sprint 1–5'in bilimsel bulgularını, negatif sonuçları, kayıtlı protokol ile gerçekleşen koşular arasındaki tarihli sapmaları, `OPUS-028` duyarlılık deneyini ve uçtan uca CPU-only yeniden çalıştırma paketini tek belgede birleştirir.

> **Sorumluluk Reddi:** Bu çalışma ve yazılım yalnızca araştırma amaçlıdır (`Research Use Only`). Klinik tanı, tedavi seçimi veya hasta yönetimi için kullanılamaz.

## 1. Araştırma sorusu ve ana yöntem

Primer AML örneklerinde önceden ayrılmış hedeflerin *ex vivo* duyarlılık sırasını, aynı hastada az sayıda ek prob ölçümüyle tahmin ediyoruz. Araştırma referansı eğitimde öğrenilen `StaticDOpt` prob sırası, beş ölçüm ve `M3` koşullu kovaryans güncellemesi. `M1` omik tabanını ve eşit bütçeli rastgele paneli karşılaştırıcı olarak kullanıyoruz. Prob ve değerlendirilen hedef kümeleri ayrık. Beat AML’de düşük AUC, FIMM’de yüksek sDSS daha yüksek duyarlılık. Ölçekler arasındaki aktarımda FIMM yanıtının yönü ters çevrildi. Sonuçlar laboratuvarda ölçülen duyarlılık ve arşivdeki hedeflerle sınırlı. [Beat AML 2.0](https://doi.org/10.1016/j.ccell.2022.07.002) · [FIMM veri sürümü](https://zenodo.org/records/7370747) · [Sprint 5 protokolü](PROTOCOL_SPRINT05.md)

## 2. Sprint 1–5 uçtan uca kanıt ve karar kapıları özeti

Tüm mimari kapsam daraltmaları [DECISION_LOG.md](DECISION_LOG.md) (`AD-001`–`AD-009`) içinde, dondurulmuş protokoller ise `docs/PROTOCOL*.md` dosyalarında tarihli olarak saklanmıştır. Tarihsel `v1` çıktıları ezilmeden korunmuştur (`AD-007`).

| Aşama / Kapı | Birincil Rejim & Payda | Temel Gözlenen Kanıt (`%95 GA`) | Karar / Durum |
|---|---|---|---|
| **Sprint 1 — Kapı 0** ([rapor](sprints/sprint-01-data-audit.md)) | Beat AML 2.0: `942` örnek / `805` hasta (`108` çoklu-örnekli hasta, `18` örtüşen hasta, `Both` dahil) | Eşleşmiş ilaç + RNA-seq: `520` örnek / `479` hasta (`471` indeks AML hastası, `24` klinik hedef ilaç, `166` toplam inhibitör); sonraki bölmelerde hasta kimliği ayrım birimi oldu | **PASS** ([PROTOCOL.md](PROTOCOL.md) donduruldu) |
| **Sprint 2 — Kapı 1** ([rapor](sprints/sprint-02-baselines.md)) | Hasta-ayrık 5-Kat CV × 3 (`n=448`) & temiz `W1+2→W3+4` (`n=148`) | Permütasyon sızıntı testi geçti. `M3(k=5)` vs `M1(k=0)`: ID `ΔRMSE = −4,29 [−4,98, −3,60]` (`44,78` vs `49,07`, `Res R² = 0,362`, `NDCG@3 = 0,869`); W3+4 `ΔRMSE = −2,49 [−3,69, −1,35]` | **PASS** (`M3(k=5,8)`) |
| **Sprint 3 (`v2`) — Kapı 2** ([rapor](sprints/sprint-03-v2-panel-selection.md), [Holm](sprints/sprint-03-v2-reanalysis.md)) | ID CV × 3 (`n=427`) & temiz `W3+4` (`n=146`); `k ∈ {1,2,3,5,8,12}` | `StaticDOpt(k=5)` vs `Random50(k=5)`: ID `ΔRMSE = −2,37 [−2,84, −1,89]` (`42,74` vs `45,11`); W3+4 `ΔRMSE = −1,81 [−2,94, −0,65]` (`50,81` vs `52,62`). Simetrik 9-test Holm: `AdaptiveOED` vs `StaticDOpt` iki rejimde de **`INSUFFICIENT_EVIDENCE`** → `StaticDOpt` referans (`AD-008`) | **PASS** (Sol & Opus `QA_PASS`) |
| **Sprint 4 (`v2` & Ek 19)** ([v2](sprints/sprint-04-v2-conformal-abstain.md), [Ek 19](sprints/sprint-04-v2-dev-opus19-sensitivity.md)) | ID 60/20/20 (`n_cal=90`, `n_test=91`) & `W3+4` (`n_cal=70`, `n_test=148`) | `StaticDOpt_k5` vs `M1_k0` ID `ΔRMSE = −7,29 [−9,45, −5,42]`. Eşzamanlı %90 split-conformal kapsam: ID `%86,8 (79/91)`, W3+4 `%74,3 (110/148)`, fakat `EqualWidthAUC` kısa listesi `%98,7` geniş kaldı. W3+4 omik çekinme taşınmadı (`AURC 32,44 > Rand 24,60`). Ek 19 keşifsel `m3_post` eşiği: ID `84/91`, W3+4 `65/148` | **`QA_PASS`** (`SOL-09..12`, `DEV-OPUS-13..20` kapandı) |
| **Sprint 5 — Kulvar A (`M4`, Kapı 3)** ([rapor](sprints/sprint-05-m4-shortlist-fimm.md)) | Sabit `StaticDOpt(M3)` paneli (`A1`, `k=5`), ID CV (`n=427`), `20×` derece-korumalı permütasyon | `M4 − M3`: `ΔRMSE = +0,011 [−0,010, +0,033]`; `M4 − perm`: `+0,017 [−0,005, +0,039]` (eşik `≤ −0,50`). FIMM `C2 (n=109)` kohort içi yöntem tekrarı: `≈0,00 [−0,01, +0,01]` | **`NO-GO (v2 ODE ERTELENİR)`** (`AD-009`) |
| **Sprint 5 — Kulvar B & C** ([rapor](sprints/sprint-05-m4-shortlist-fimm.md)) | Beat AML `v2` tahminleri (`91`/`148`) & FIMM `Zenodo 7370747` (`109` hasta, `23/24` hedef, `64` prob) | `RankConformal`: ID `85/91 (%93,4)` & `6,00` ilaç (`−12,70` vs `EqualWidthAUC`), W3+4 `137/148 (%92,6)` & `6,00` ilaç. FIMM `C1` donmuş aktarım (`k=5`): `ΔNDCG@3 +0,032` vs `M1`, `+0,033` vs `Random20x` (üst-3 zenginleşmesi); `C2` FIMM içi CV: `ΔRMSE −1,00 sDSS` vs `M1` | **`QA_PASS (teknik)`** (Opus §3.10) ve **Sol v1 bilimsel onayı** (§9) |

Paydalar ilgili deney ve uygunluk kuralına aittir. Sprint 2–5’te yeniden kullanılan Beat AML bölmeleri bağımsız tekrar olarak sayılmaz. W3+4 panel sonuçları arşiv mevcudiyeti koşuluyla yorumlanır.

## 3. Kısa liste ve çekinme — SOL-13 ve SOL-14

`RankConformal` sabit tahmin politikasından sonra, her kalibrasyon hastasında ölçülen gerçek en iyi hedefin tahmin sırasını kullanır. Birden fazla eşit gerçek birinci varsa en kötü bağ sırası kalibre edilir. Deney tablosunda **en az bir gerçek birincinin içerilmesi** sayıldı. Ölçüm mevcudiyeti maskesi listeyi oluştururken açıktır. Bu nedenle ampirik kapsam arşivdeki ölçülen hedeflere aittir. Nominal split-conformal açıklaması uygun yeni hastalar ve aynı seçim/ölçüm politikası altında değiştirilebilirlik gerektirir. Tüm kataloğa, alt gruplara ve tek aday verilen hastalara otomatik olarak aktarılmaz. W3+4, keşifsel zamansal stres testidir. [Regresyonda split-conformal](https://doi.org/10.1080/01621459.2017.1307116) · [Liste kurucu](../src/oncotwin/eval/rank_conformal.py)

| Beş-prob statik kol | İç test (uygun hasta 91) | W3+4 (uygun hasta 148) |
|---|---:|---:|
| Eşit genişlikli AUC, gerçek birinci listede / ort. aday sayısı | 91/91 · 18,70 | 148/148 · 21,41 |
| `RankConformal`, aynı ölçüler | 85/91 · 6,00 | 137/148 · 6,00 |
| `Top1Gap`, aynı ölçüler | 84/91 · 5,45 | 128/148 · 4,76 |
| `Top1Gap`, tek aday verilenlerde gerçek birinci | 5/9 (Wilson %95 GA %26,7–81,1) | 19/25 (%56,6–88,5) |

`RankConformal` ana kısa liste, `Top1Gap` karşılaştırıcı. Eşit genişlik ve `Top1Gap` aynı tahmin farkı liste ailesinde farklı kalibrasyon ölçütlerine dayanır. `RelRank` üst tamsayıya yuvarlayarak doğrudan göreli sıra kuralına göre kapsayıcı bir liste üretir. W3+4 için dondurulmuş protokoldeki **71** kişi kalibrasyon bölmesini, uygulama raporundaki **70** ise arşiv uygunluğundan sonra kalibrasyona giren hastaları gösterir. İç testte 90 kalibrasyon ve 91 test hastası kullanıldı. [Sprint 5 kısa liste tabloları](sprints/sprint-05-m4-shortlist-fimm.md) · [Bağımsız Opus kontrolü §3.10](agents/opus-optimization-qa.md)

W3+4 ana statik kolda omik çekinme skoru rastgele kabul sırasından daha kötü çıktı (AURC 32,44 / 24,60, AUGRC 14,27 / 12,30). `m3_post_scale` sıralaması daha düşük alan verdi. Ek 19, aynı skorları kullanarak kabulü 146/148’den 65/148’e indiren keşifsel işletim noktasıdır. Bu eşik analizi bağımsız risk kontrolü sağlamaz. [Sprint 4 Ek 19](sprints/sprint-04-v2-dev-opus19-sensitivity.md) · [AUGRC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract-Conference.html)

## 4. FIMM — SOL-15 ve SOL-16

- **C1, donmuş aktarım.** Beat AML’de ortak katalog için öğrenilen model ve prob sırası FIMM’de yeniden fit edilmedi. 109 indeks hastada beş-prob M3 eksi omik M1 için **ΔNDCG@3 +0,032 [+0,007, +0,058]** ve **ΔRegret@1 −0,23 [−1,26, +0,76] sDSS**. Aynı bütçeli rastgele M3’e karşı ΔNDCG@3 **+0,033 [+0,013, +0,054]** ve ΔRegret@1 **−0,26 [−0,93, +0,41] sDSS**. Bulguyu **ölçülen hedeflerin üst-3 sırasındaki sinyal** olarak yazıyoruz. Hasta içi Spearman ve Regret@1 güçlü bir genel sıralama veya birinci aday üstünlüğü göstermedi. Kayıp ve aralıklar sabit tahminlere koşullu ve çoklu karşılaştırma düzeltmesiz. [Sprint 5 C1 tabloları](sprints/sprint-05-m4-shortlist-fimm.md)
- **Mevcudiyet ve assay.** FIMM’de 23 ortak hedef ve 64 ortak prob var. 98/109 indeks hastada sabit beş-prob öneki tam gözleniyor. Raporlanan C1 performansı 109 hastanın tamamında arşiv maskesine göre geri doldurmayla elde edildi. Beat AML’nin MTS/AUC ölçümüyle FIMM’in CTG/sDSS ölçümü ve doz serileri farklı. 23 ortak hedeften 11’inde üst konsantrasyon farklı. NDCG’nin gerçek yanıt ağırlığı `max(y)−y` olduğundan hasta içi pozitif afin dönüşüm korunur. Doz ve assay kayması genel olarak sıralamayı veya ağırlığı değiştirebilir. [FIMM katalog denetimi](../data/processed/sprint05_fimm_catalog_audit.json) · [Çok merkezli AML assay çalışması](https://pmc.ncbi.nlm.nih.gov/articles/PMC13408654/)
- **C2, FIMM içi yöntem tekrarı.** Beş hasta-ayrık kat × üç tohumla FIMM üzerinde yeniden fit. Bu koşuda M3 beş-prob eksi M1 RMSE **−1,00 [−1,26, −0,76] sDSS**. FIMM `File_7` RNA matrisi kaynakta batch-düzeltilmiş. Yerel model ve gen seçimi eğitim katına ait. Ham sayımdan uçtan uca tek örnekli doğrulama ayrı analiz gerektirir. Beat AML konformal aralıkları FIMM’e taşınmadı. [FIMM veri tanımı](https://zenodo.org/records/7370747)
- **Gerçekleşen deney kolları.** C1’de `Random20x` yalnız **M3, k=5** için koşuldu. `M2a` rastgele kolları ve `M3` rastgele `k=2,8` hücreleri koşulmadı. C2’de raporlanan rastgele karşılaştırıcı da yalnız **M3, k=5**. Bu farklar tarihli rapor notu olarak kaydedildi. [Dondurulmuş protokol §3.2](PROTOCOL_SPRINT05.md) · [C1/C2 koşucusu](../src/oncotwin/experiments/run_sprint05_fimm.py)

## 5. OPUS-028 — Beat AML içinde prob ölçeği duyarlılığı

**Tasarım.** Eski Sprint 4 v2 iç bölmesi yeniden kullanıldı. Eğitim 284 indeks hasta, kalibrasyondaki 94 kişi kullanılmadı, test 93 hasta. 93 hastanın tamamında en az beş ölçülmüş hedef, 91’inde ayrıca en az sekiz aday prob mevcut. M1 ve M3, C1’deki hasta içi sıra temsili ve **102 ortak ilaç içindeki** eğitim hastası satır z-skoruyla yeniden fit edildi. 23 hedef, en az 100 eğitim ölçümü olan 62 ortak prob, beş-prob D-opt sırası ve her test hastasının seçili probları kollar arasında sabit. Test prob seçiminde mevcut arşiv maskesi kullanıldı. Beş prob ana kıyas, iki prob mekanizma duyarlılığı. Ölçek kolları test etiketleriyle yeniden fit edilmedi.

| Test girdisi | Nasıl hesaplandı? | Beş prob NDCG@3 | Beş prob Regret@1 (AUC) |
|---|---|---:|---:|
| M1 omik taban | Prob kullanılmadı | 0,854 | 25,32 |
| M3, eğitim referansı | Eğitim hasta ortalamalarının medyanı ve hasta SD’lerinin medyanı | 0,898 | 16,79 |
| M3, prob ortalaması | Kendi prob ortalaması ve eğitimden sabit SD | 0,895 | 14,59 |
| M3, C1 prob z-skoru | Kendi prob ortalaması ve prob SD’si | 0,888 | 18,41 |
| M3, tam satır tanısı | Diğer test ölçümleri dahil hasta ortalaması ve SD. Uygulanabilir kıyasın dışında | 0,894 | 16,18 |

**Eşleştirilmiş hasta farkları.** Sabit tahminler üzerinde 2.000 bootstrap çekilişi, 91 hasta, noktasal %95 GA.

| Fark | ΔNDCG@3 | ΔRegret@1 (AUC) |
|---|---:|---:|
| Beş prob, C1 ölçeği eksi eğitim referansı | −0,010 [−0,021, +0,001] | +1,61 [−3,62, +7,73] |
| Beş prob, C1 ölçeği eksi yalnız prob merkezleme | −0,007 [−0,017, +0,002] | +3,82 [−1,39, +10,35] |
| Beş prob, C1 ölçeği eksi M1 | +0,034 [+0,014, +0,055] | −6,91 [−14,88, +0,21]* |
| İki prob, C1 ölçeği eksi eğitim referansı | −0,016 [−0,031, −0,002] | +4,79 [−1,19, +11,57] |

*\*Not (`OPUS-035`): Beş prob `C1 ölçeği − M1` `Regret@1` aralığı sınırda ve bootstrap tohumuna bağlıdır (Sol 2.000 tekrar `[−14,88, +0,21]`, Opus §3.11 4.000 tekrar `[−14,34, −0,05]`); bu nedenle ne kesin kazanç ne de kesin sıfır olarak raporlanır.*

**Yorum.** Aynı kohort içinde beş probla C1 ölçeği altında üst-3 sinyali görüldü. Eğitim referansına karşı farkın aralığı sıfırı içeriyor. İki probda tüm 91 hasta için z-skorları birer −1 ve +1 oldu. Bu dönüşüm iki ölçümün fark büyüklüğünü siliyor. Beş probda tek başına merkezleme ile ayrıca hasta SD’si kullanmanın tahmin üzerindeki etkileri ayrık olarak raporlandı. Her hedefe eşit bir tahmin ofseti eklenseydi NDCG ve Regret@1 değişmezdi. Prob girdisindeki merkez değişikliği M3’te hedeflere eşit yansımak zorunda değildir. Dolayısıyla bu deney ölçek duyarlılığını ölçer, FIMM’deki Regret@1 sonucuna tek bir neden atfetmeye yetmez.

**Denetim.** Maskeyi sabit tutarak seçilmeyen tüm sonlu test yanıtları değiştirildi. Her bütçede eğitim referansı, prob merkezleme ve C1 ölçeğinin girdi ve tahmin dizileri birebir eşleşti. Tam satır tanısı bu değişiklikten etkilendi. Eğitimde kilitlenen önek arşiv değerlendirme kümesinde k=2 için 83/91, k=5 için 51/91 tam ölçülmüş. Tablodaki başarı geri doldurmalı ortak değerlendirme kümesine ait. [Çalıştırılabilir denetim](../audits/sol-sprint06/opus028_beataml_probe_scale.py) · [Tam sonuç JSON’u](../audits/sol-sprint06/opus028_beataml_probe_scale.json)

## 6. Negatif bulgular ve literatürdeki yer

- **M4 ve Kapı 3.** Ön kayıtlı beş-prob A1 kıyasında ID CV’de M4 eksi M3 RMSE **+0,011 [−0,010, +0,033] AUC**. M4 eksi derece-korumalı karıştırılmış aile öncülü **+0,017 [−0,005, +0,039] AUC**. Her ikisi de −0,50 eşiğinin üzerinde. Sonuç `NO-GO`. ODE ertelenir. Sonuç sınanan aile öncülüne aittir. [Sprint 5 Kapı 3](sprints/sprint-05-m4-shortlist-fimm.md)
- **Yakın öncül.** Shokrzadeh ve Shokrzadeh’in Haziran 2026 ön baskısı Beat AML’de 318 hasta ve 122 bileşikte split-conformal regresyon aralıkları raporladı. %90 nominal düzeyde %90,7 ampirik kapsam, ilaç bazında toplulaştırılan ölçüdür. OncoTwin’in odak noktası, ölçüm bütçesi altında hasta düzeyinde prob seçimi, üst sıralama ve ölçülen birinciyi içeren kısa listedir. Tercan ve arkadaşları aynı FIMM kaynağına aktarımda göreli gen ifadesi kullandı. OncoTwin’in probla yanıt güncellemesi ve kalibrasyon hedefi ayrı yöntem seçimleridir. [Haziran 2026 ön baskısı](https://www.biorxiv.org/content/10.64898/2026.06.07.730728v1) · [Tercan ve ark.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0343422)
- **Genişletilmiş literatür konumlandırması (Sol §10):** Çok merkezli AML assay uyumsuzluğu ([Willoch-Allen ve ark., *npj Precis Oncol* 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13408654/)), DSS eğri/doz duyarlılığı ([Yadav ve ark., *Sci Rep* 2014](https://doi.org/10.1038/srep05193)), konformal risk kontrolü ve seçim ([Angelopoulos ve ark., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html); [Jin & Candès, *JMLR* 2023](https://jmlr.org/papers/v24/22-1176.html)), konformal kısmi/tam sıralama ([Sale & Hüllermeier, COPA 2026](https://proceedings.mlr.press/v329/sale26a.html); [Liao ve ark., 2026](https://arxiv.org/abs/2601.23128)), yapılandırılmamış kovaryans büzüştürme ([Ledoit & Wolf, *JMVA* 2004](https://doi.org/10.1016/S0047-259X%2803%2900096-4)) ve prospektif tek-ilaç klinik tasarım ayrımı ([VenEx, *Blood* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13122316/)).

## 7. Tarihli protokol sapmaları ve iddia sınırları tablosu

| Kimlik | Tarih | Konu / Gözlenen Sapma | Durum ve İddia Sınırı |
|---|---|---|---|
| **`SOL-08`** | `2026-11-22` | Sprint 3 `v2` 20 çekilişli ofset-sabitli permütasyon ölçek geri dönüşü sonrası yeniden merkezlenmedi (W3+4 ham ortalama kayması ≤ `27,22` AUC) | Açık, Sprint 4–5'te kullanılmıyor; ofset üstüne ek tahmin kanıtı olarak yalnız `M3` vs `M3_offset_only` kullanılır |
| **`SOL-12`** | `2026-11-22` | Sprint 4 `v1` koşusunda `PROTOCOL_SPRINT04.md §1.2`'deki `M0(0)` eksikti | Sprint 4 `v2`'de (`DEV-OPUS-16`) aynı hasta/hedef maskesinde eklendi; `v1` tarihsel korundu |
| **`SOL-13` / `OPUS-027`** | `2026-11-25` | `Top1GapConformal` tek aday doğruluğu ID `5/9` (`%26,7–81,1`), W3+4 `19/25` (`%56,6–88,5`); liste kurucu arşiv ölçüm maskesi `O_i` açıkken çalışıyor ve bağda kalibrasyon tüm gerçek birincileri, tablo en az birini sayıyor | Genel `%90` marjinal kapsam tek adaylı hastalara veya ölçülmemiş kataloğa aktarılmaz; `RankConformal` ana kısa liste, `Top1Gap` karşılaştırıcı ve tek aday paydası her zaman ayrı raporlanır |
| **`SOL-14`** | `2026-11-25` | `PROTOCOL_SPRINT05.md §2` W3+4 kalibrasyon bölmesini `n_cal=71` yazıyor; arşiv uygunluğu (`min_targets=5, min_probes=8`) sonrası kalibrasyona giren hasta `70` (`test=148`) | Bölme (`71`) ile arşiv-uygun kalibrasyon (`70`) ayrımı bu raporda (§3) ve `verify_package.py`'de açıkça kilitlendi |
| **`SOL-15` / `OPUS-028`** | `2026-11-25` | FIMM `C1` arşiv geri doldurması (`k=5,8` tam-panel `98/109`), `11/23` hedefte doz üst sınırı farkı ve eğitim (tüm ortak ilaçlar) vs test (`k` prob) z-skor ölçeği farkı (`k=2` `91/91` `±1`) | Beat AML iç kontrolü [opus028_beataml_probe_scale.py](../audits/sol-sprint06/opus028_beataml_probe_scale.py) ile ölçüldü (§5); FIMM `C1` sonuçları aynı kohortta eşik/ölçek seçimi için kullanılmaz |
| **`SOL-16`** | `2026-11-25` | `PROTOCOL_SPRINT05.md §3.2` `Random20x`'i geniş listelerken uygulamada `C1` ve `C2`'de yalnız `M3, k=5` için koşuldu; FIMM `File_7` RNA matrisi kaynakta batch-düzeltilmiş | Koşulmayan `M2a` ve `k=2,8` rastgele hücreleri açıkça "koşulmadı" olarak belgelenir (§4); uçtan uca ham-sayım doğrulaması için `File_8` ayrı analiz gerektirir |
| **`OPUS-029`** | `2026-11-25` | `within_sample_rank_normalize` bağları ortalama sıra yerine sütun sırasıyla (ordinal) kırıyor (Beat AML'de satır başına `%19,5` bağ, FIMM'de ≈`0`; ortalama fark `2e-5`, en çok `0,0016`) | Ölçülen fark RNA sıra girdisinde küçük. Tahmin metrikleri için ayrı eşdeğerlik analizi yapılmadı. v1 sıra kuralı tarihsel `sprint05_*` bit-eşitliğini koruyor |
| **`OPUS-030`** | `2026-11-25` | FIMM `C1` `M3(k=5) − M1`: Opus’un bağımsız bootstrap koşusunda `NDCG@3 +0,032 [+0,007, +0,055]`, `Recall@3 +0,037 [+0,006, +0,067]`; ancak `Spearman +0,022 [−0,001, +0,045]` ve `Regret@1 −0,23 [−1,22, +0,77]` sıfırı kesiyor | İddia "genel sıralama" veya "birinci aday kaybı üstünlüğü" değil, **ölçülen hedeflerde üst-3 zenginleşmesi** olarak sınırlandırıldı |
| **`OPUS-032`** | `2026-11-26` | `rank_conformal.py`, `uncertainty._finite_sample_conformal_quantile` private yardımcısını dışarıdan import ediyordu | Sprint 6’da public `finite_sample_conformal_quantile` olarak yeniden adlandırıldı; sayısal fark `0,0`, Opus §3.11 ile **KAPANDI** |

## 8. Veri kaynakları, donanım/ortam ve tekrarlanabilir araştırma paketi

### 8.1 Veri kaynakları ve bütünlük özetleri
- **Beat AML 2.0** ([biodev/beataml2.0_data](https://github.com/biodev/beataml2.0_data), CC BY 4.0): 6 veri dosyası ve lisans metni (toplam yaklaşık 291 MiB, Git LFS medya uçlarından [fetch_beataml.py](../src/oncotwin/data/fetch_beataml.py) ile SHA-256 doğrulamalı indirilir).
- **FIMM / Malani AML** ([Zenodo 7370747](https://zenodo.org/records/7370747), CC BY 4.0): `Functional_Precision_Medicine_Tumor_Board_AML.zip` (`38,810,683 B`, exact MD5 `3db5280e20f315d56d6bf07dd6a9e241`; [fetch_fimm.py](../src/oncotwin/data/fetch_fimm.py) ile indirilir ve [fimm_dataset.py](../src/oncotwin/data/fimm_dataset.py) ile ön-modelleme QC denetimi `sprint05_fimm_catalog_audit.json` üretilir). `File_7` RNA tablosu `167` örnek (`163` AML + `4` sağlıklı CD34+), eşleşmiş AML analitik kümesi `132` örnek / `109` benzersiz hastadır.

### 8.2 Doğrulanmış ortam ve performans profili (CPU-only)
- **Ortam dosyaları:** [pyproject.toml](../pyproject.toml), [requirements.txt](../requirements.txt), [pytest.ini](../pytest.ini) (`filterwarnings = error::RuntimeWarning`), [CITATION.cff](../CITATION.cff).
- **Doğrulanmış sürüm kilidi (`2026-11-26`):** `Python 3.14.10` (Linux x86_64), yalnız **3 çalışma paketi** (`numpy==2.5.5`, `pandas==3.0.3`, `openpyxl==3.1.5`) + test (`pytest==9.2.0`). Çalıştırma CPU üzerinde doğrulandı, GPU/VRAM gerekmiyor.
- **Ölçülen çalışma süreleri ve tepe bellek (Opus & Neon CPU ölçümleri):**
  - Sprint 2: `301,0 s` · Sprint 3 `v2`: `440,7 s` · Sprint 4 `v2`: `48,6–54,9 s` (peak RSS `389 MB`) · Sprint 5 (Kulvar A/B/C): `244,1–251,0 s` (peak RSS `599 MB`) · `verify_package` (ham veri + 32 dosya manifest + 16 bilimsel değişmez): `~1 s` · Tam `pytest` paketi (`88` test): `~15 s`.
- **Dondurulmuş artefakt manifesti:** [release_manifest.sha256](../data/processed/release_manifest.sha256) (`32` dosya: 5 dondurulmuş protokol, 16 işlenmiş JSON/NPZ, 9 sprint raporu ve `OPUS-028` betik/JSON).

### 8.3 Uçtan uca yeniden çalıştırma ve tek-komut doğrulama

Paket doğrulama komutlarını OncoTwin kök dizininden çalıştırın. Doğrulama, arşivlenmiş dosyaların bütünlüğünü ve seçili bilimsel değişmezleri kontrol eder. Model deneylerini baştan çalıştırma komutları ayrı verilmiştir.

**Temiz ortam kurulumu (`OPUS-035`, doğrulandı).** `python3 -m venv` → `pip install -r requirements.txt` → tek-komut `verify_package --run-tests` zinciri Opus §3.11 tarafından temiz sanal ortamda doğrulandı (**`PASS`, çıkış kodu `0`, `88 passed`, `15,6 s`**). Sınır: aynı makinede Linux x86_64 ve `Python 3.14.10` üzerinde doğrulandı; farklı işletim sistemi veya daha eski Python sürümleri (`requires-python >= 3.11`) denenmedi. v1 için desteklenen kurulum yolu repo kökünden `pip install -r requirements.txt` ve `PYTHONPATH=. python3 -m src...` kullanımıdır (`pip install .` üst düzey `src` ad alanı kurduğu için önerilmez, `OPUS-033`).

```bash
# 0. Kurulum (yalnız 3 çalışma bağımlılığı + pytest)
python3 -m pip install -r requirements.txt

# 1. Ham verileri bütünlük doğrulamasıyla indir (Beat AML SHA-256, FIMM MD5)
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_beataml
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_fimm

# 2. Tek-komut paket bütünlüğü, ham veri, manifest, değişmezler ve test doğrulaması
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python3 -W error::RuntimeWarning -m src.oncotwin.experiments.verify_package --run-tests
```

**İsteğe bağlı deney tekrarı.** Aşağıdaki komutları, arşiv dosyalarıyla yazma bağlantısı paylaşmayan ayrı bir proje kopyasında çalıştırın. Varsayılan çıktı yolları dondurulmuş JSON/NPZ ve sprint raporlarıyla çakışır. Özgün paket manifestini arşiv kopyasında doğrulayın. Yeniden üretilen sonuçları ayrı karşılaştırın ve süre alanlarını sayısal kıyastan çıkarın. Buradaki koşucular güncel analiz sürümlerini üretir.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.data.fimm_dataset
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.data.audit_beataml
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.run_sprint02
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.run_sprint03
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.sprint03_reanalysis
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.run_sprint04
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.sprint04_dev_opus19
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.run_sprint05
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 -W error::RuntimeWarning audits/sol-sprint06/opus028_beataml_probe_scale.py
```

## 9. Açık sınırlılıklar, gelecek çalışma sınırı ve Sprint 6 kapanış kararı

- **Açık sınırlılıklar (v1 kapsamı dışında bırakılanlar):**
  1. Kilitli eğitim kataloğundan tek sabit beş-problu plakanın geri-doldurmasız, prospektif bağımsız hasta kohortunda denenmesi ve teknik çekinme oranının ölçülmesi.
  2. FIMM `File_8` ham RNA sayımlarından uçtan uca tek-örnekli ön işleme (kaynak `File_7` batch-düzeltmesini kaldırarak) ve baştan tanımlanmış assay/doz uyumlu alt küme duyarlılığı.
  3. Klinik yanıt, toksisite, kombine rejimler ve kinetik Hill-ODE (`AD-001` / `AD-009` gereği ertelendi).
- **Kapanış kararı (v1 mühürlü — `2026-11-26`):** Beş-prob `StaticDOpt + M3` ve `RankConformal` araştırma referansı olarak dondurulur (`Top1Gap` karşılaştırıcı ve tek aday paydası açıkça raporlanır). `OPUS-028`, C1 prob-ölçeği kuralının keşifsel duyarlılığı olarak belgelenir; farklı bir ölçek seçimi bağımsız hastalarla önceden tanımlanmış karşılaştırmaya bırakılır. **Sol’un v1 bilimsel kapanış hükmü (`2026-11-26`, §12): `ONAY`.** **Opus’un Sprint 6 sürüm paketi teknik QA hükmü (`2026-11-26`, §3.11): `QA_PASS (teknik)`** (`OPUS-028` birebir yeniden üretim, `OPUS-029`/`032` kapalı, temiz sanal ortam `88 passed`). Kapı 3 `NO-GO` ve ODE ertelemesi geçerli. Proje sahibi onayıyla (`2026-11-26`) **OncoTwin v1 resmen kapatılıp mühürlenmiştir.** [Sol denetimi](agents/sol-architect-reviewer.md) · [Opus teknik QA](agents/opus-optimization-qa.md) · [Neon pano](agents/neon-lead-dev.md)
