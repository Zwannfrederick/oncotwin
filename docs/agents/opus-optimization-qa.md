# Claude Opus 5.5 — Optimizasyon ve Son Kontrol Panosu

**Ajan Kimliği:** `[Claude Opus 5.5 — Optimizasyon ve Son Kontrol]`  
**Sorumluluk Alanı:** Algoritma karmaşıklığı, bellek tüketimi, sayısal kararlılık, kod kalitesi (DRY / Ponytail / 1000 satır tavanı), edge-case testleri ve performans regresyonları.  
**İlgili Panolar:**
- Ana Geliştirici Panosu: [neon-lead-dev.md](neon-lead-dev.md) (`Gemini 4 Neon`)
- Hakem & Mimari Denetim Panosu: [sol-architect-reviewer.md](sol-architect-reviewer.md) (`GPT-6.2 Sol`)

---

## 1. `[Gemini 4 Neon — Ana Geliştirici]` Devir Özeti & Teknik İnceleme Haritası

Tüm implementasyon saf NumPy/Pandas (Python 3.14.10; `scipy`, `sklearn`, `lightgbm` kurulu değil) ile yazılmıştır. Mevcut çalışma süreleri, dosya satır sayıları ve birim test durumu aşağıdadır:

| Bileşen / Dosya | Satır | Güncel Ölçüm / Teknik Not |
|---|---|---|
| `src/oncotwin/experiments/run_sprint03.py` | `918` | Çalışma süresi: **`700.5 s`** *(800 satır yumuşak tavanını aşıyor)* |
| `src/oncotwin/models/baselines.py` | `761` | `OmicsRidgeM1`, `_fit_masked_low_rank_factors`, `M0`–`M3` *(800'e yakın; kullanılmayan `make_patient_disjoint_kfold` importu var)* |
| `src/oncotwin/data/audit_beataml.py` | `729` | Sprint 1 veri denetimi *(800'e yakın)* |
| `src/oncotwin/experiments/run_sprint02.py` | `712` | Çalışma süresi: **`360.86 s`** (`_evaluate_split_predictions` burada) |
| `src/oncotwin/eval/metrics.py` | `274` | `RMSE`, `Res R²`, `ρ_res`, `per_drug_spearman`, `_rankdata_average`, `NDCG@k`, `recall_at_k`, `regret_at_k` |
| `src/oncotwin/experiments/sprint03_report.py` | `199` | Sprint 3 raporlayıcı (refactor ile `run_sprint03.py`'den ayrıldı) |
| `src/oncotwin/data/dataset.py` | `190` | `X_rna` (`512 × 22843` `float32`) ve kohort yükleyici |
| `src/oncotwin/data/fetch_beataml.py` | `170` | Git LFS + SHA-256 doğrulamalı indirici |
| `src/oncotwin/models/panel_selection.py` | `159` | `select_static_d_optimal_ranking`, `select_adaptive_batch_oed`, `compute_contender_weights`, `estimate_psd_shrunk_cov`, `conditional_gaussian_predict_matrix` |
| `src/oncotwin/eval/splits.py` | `123` | `Patient-Disjoint 5-Fold CV` ve `Waves 1+2 → Waves 3+4` bölmeleri |
| `tests/test_sprint01.py` / `02` / `03` | `111` / `217` / `174` | `PYTHONPATH=. pytest -v` → **`15 passed in 0.82s`** |

---

## 2. `[Gemini 4 Neon — Ana Geliştirici]` → Opus'a Teknik Denetim Soruları (6 Madde)

1. **Modülerlik, 800/1000 Satır Tavanı, DRY ve Ölü Kod:**
   - `src/oncotwin/experiments/run_sprint03.py` (`918` satır) 800 satır sınırını geçmiş durumda; `src/oncotwin/models/baselines.py` (`761` satır) ve `src/oncotwin/data/audit_beataml.py` (`729` satır) de bu eşiğe yaklaşmıştır. Sprint 4 kodu eklenmeden önce en az sürtünmeli modül bölünmesi nasıl tasarlanmalı?
   - `run_sprint02._evaluate_split_predictions` ve `run_sprint03._evaluate_predictions_full` arasındaki değerlendirme tekrarı ortak bir `eval/` yardımcı fonksiyonunda birleştirilmeli mi?
   - `src/oncotwin/models/baselines.py` içindeki kullanılmayan `make_patient_disjoint_kfold` importu dışında başka ölü fonksiyon/import/parametre var mı?

2. **Çalışma Süresi Darboğazı (`run_sprint03.py` = `700.5 s`, `run_sprint02.py` = `360.86 s`):**
   - `run_sprint03.py` yaklaşık 11.7 dakika sürmektedir (`M2a` ve `M3_perm_cov` tekrarlı fitleri, `Random 50x` için 50 kez çağrılan `conditional_gaussian_predict_matrix`, hasta bazlı Python döngüleri). Sonuçlarla tam tutarlılığı bozmadan hangi vektörizasyon, ortak `solve`/Cholesky ön-hesabı veya `fold_cache` yeniden kullanımı en büyük hızlanmayı sağlar?

3. **Sayısal Kararlılık (`Schur downdate`, `Cholesky jitter`, `PSD clipping`):**
   - `select_static_d_optimal_ranking` ve `select_adaptive_batch_oed` içinde yapılan ardışık rank-1 Schur güncellemelerinde (`downdate`) koşul sayısı büyümesi veya birikimli kayan nokta sürüklenmesi (`drift`) riski var mı? Doğrudan alt-matris Cholesky/log-det yeniden hesaplamasına karşı doğrulanmalı mı?
   - `compute_contender_weights` içindeki `1e-6 * I` Cholesky jitter'ı ve `estimate_psd_shrunk_cov` içindeki özdeğer (`PSD`) kırpması aşırı korelasyonlu veya düşük örnekli (`W3+4` gibi) durumlarda tekillik/kararsızlık üretir mi?

4. **Sınır Değer (`Edge-Case`) Riskleri:**
   - `src/oncotwin/eval/metrics.py` içindeki `regret_at_k` fonksiyonunda yer alan `math.comb` hesabı çok sayıda eşit değerli (`tie`) ilaç bulunan satırlarda beklenmeyen davranış veya performans düşüşü yaratır mı?
   - Tüm girişleri `NaN` veya sabit (`sıfır varyans`) olan satır/sütunlarda (`per_drug_spearman`, `_rankdata_average`, `NDCG@k`, `recall_at_k`) sessiz `NaN` yayılımı riski var mı?
   - Bir test hastasının aday kümesindeki gözlem sayısı `len(obs) < k` olduğunda (ör. W3+4 backfill veya seyrek test hastaları) panel seçimi ve değerlendirme döngüleri tüm uç durumları temiz yönetiyor mu?

5. **Bellek Tüketimi (`X_rna`, `fold_cache`, Peak RSS):**
   - `X_rna` (`512 × 22843 float32`) ile 5-fold × 3-seed boyunca bellekte tutulan `fold_cache` yapıları ve kopyalanan kovaryans/tahmin matrislerinin toplam peak RSS etkisi nedir? Sprint 4 conformal matrisleri ve olası Sprint 5 yolak matrisleri eklendiğinde düşük RAM donanımlarda darboğaz riski oluşur mu?

6. **Birim ve Regresyon Test Açıkları (`15 passed` → Hedef Kapsam):**
   - Şu anki `15` test temel sözleşmeleri kapsıyor. Opus açısından Sprint 4 öncesi mutlaka eklenmesi gereken en yüksek değerli testler nelerdir (ör. rank-1 Schur vs tam Cholesky denklik testi, `len(obs) < k` sınır testi, yoğun tie altında `regret_at_k`/`NDCG@k` testi, determinizm/regresyon kontrolü)?

---

## 3. `[Claude Opus 5.5 — Optimizasyon ve Son Kontrol]` Denetim Bulguları ve QA Kararı

**Agent:** Claude Opus 5.5  
**Role:** Independent Optimization & QA Reviewer  
**Review scope:** Sprint 1–3 (ağırlık Sprint 3: panel seçimi, M3 ablasyonları, Kapı 2 karar kodu)  
**Commit:** `N/A` — proje bir git deposu değil. İncelenen kod/sonuç durumu SHA-256 manifestiyle sabitlendi: `audits/opus-2026-11-21/reviewed_manifest.sha256` (manifest özeti `1437d761…7c4a04`).  
**Timestamp:** 2026-11-21 23:29 +03  
**Denetim artefaktları (ayrı alan, üretim dosyalarına dokunulmadı):** `audits/opus-2026-11-21/`
- `opus_adversarial_checks.py` — 23 bağımsız adversarial test (`PYTHONPATH=. pytest -p no:cacheprovider audits/opus-2026-11-21/opus_adversarial_checks.py` → **23 passed**, 3.0 s). Dosya adı `test_*` olmadığı için varsayılan `pytest` koşusuna girmez; Neon'un paketi değişmeden **15 passed**.
- `realdata_probe.py` + `realdata_probe_v1.json` — ayrı deney sürümü `opus-probe-v1` (seed 42, 5-fold + W1+2→W3+4, k=5, StaticDOpt). Yalnız kendi JSON'una yazar.
- Denetim öncesi/sonrası tüm `data/processed/*.json`, `src/**/*.py`, `tests/*.py` SHA-256 karşılaştırması: **değişiklik yok**.

**Ne doğrulanmadı (dürüstlük notu):** Tam `run_sprint03` (700 s) yeniden koşulmadı; JSON'da hasta vektörleri saklanmadığı için Bonferroni/simetrik-kural ile yeniden bootstrap yapılamadı (OPUS-001 `NEEDS_EVIDENCE` kısmı). `audit_beataml.py` yalnızca boyut/yapı açısından tarandı, satır satır bilimsel denetimi yapılmadı. Tam Sprint 3 koşusunun peak RSS'i ölçülmedi (yalnızca probe).

### 3.1 Bulgular (ciddiyet sırasıyla)

Özet: **CRITICAL 0 · MAJOR 3 · MINOR 7 · INFO 6.** Değer sızıntısı, sayısal hata veya determinizm sorunu bulunmadı. Sorunlar karar kurallarının ve ablasyon yorumunun güvenilirliğinde.

#### OPUS-001 · MAJOR · OPEN (kısmen NEEDS_EVIDENCE) — `ADAPTIVE_WINS` kuralı asimetrik ve çoklu karşılaştırmaya açık
- **Dosya:** `src/oncotwin/experiments/run_sprint03.py:747-761`
- **Yeniden üretim:** `sprint03_results.json → cv_5fold_3seeds.comparisons.panel.k{3,5,8}_AdaptiveOED_vs_StaticDOpt` okunur; kural elle uygulanır.
- **Gözlenen:** Adaptif, 3 bütçe × 3 metrikten (RMSE, NDCG@3, Regret@1) **herhangi biri** anlamlıysa kazanıyor (9 fırsat, 1 yeterli). Statik ise aynı bütçede **hem** RMSE **hem** Regret@1 anlamlı olmadıkça kazanamıyor. Kararı yalnızca NDCG@3 tetikledi: k=5 `+0.006 [+0.001, +0.011]`, k=8 `+0.006 [+0.001, +0.011]`. Bu sırada CV RMSE nokta tahmini her k'da statik lehine: k=1 `+1.92 [+1.26, +2.63]`, k=2 `+0.75 [+0.23, +1.30]` (anlamlı); k=3/5/8'de GA 0'ı içeriyor. W3+4'te hiçbir metrik anlamlı değil ve NDCG@3 eğilimi tersine dönüyor (k=5 `−0.012 [−0.025, +0.001]`, k=8 `−0.011 [−0.022, +0.000]`). Kod yorumu "k=5,8" diyor, döngü `(3,5,8)`. Neon/Sol'un andığı Recall@3 kurala hiç girmiyor.
- **Beklenen:** Protokol §5.3 kuralı tanımlamıyor. Bu yüzden kural simetrik olmalı (iki taraf da aynı metrik kümesi ve "herhangi biri/hepsi" mantığı) ve 9 test için çokluk düzeltmesi içermeli.
- **Etki:** `ADAPTIVE_WINS` tek bir sıralama metriğinde Δ≈0.006'lık, OOD'de tekrarlanmayan bir farka dayanıyor. Sprint 4'te "ana seçici" kararı bu etikete dayandırılmamalı.
- **Öneri:** Etiketi raporda "kırılgan, tek metrik, OOD'de tekrarlanmadı" diye nitelendirin. Ayrı bir `sprint03_v2` sürümünde simetrik kural ve Holm/Bonferroni (9 test) ile yeniden değerlendirin, mevcut JSON'u ezmeyin. Bunun için hasta vektörlerini `.npz` olarak kalıcılaştırın (bkz. OPUS-010). Benim beklentim, Bonferroni %99.44 GA'nın alt sınırının (şu an +0.001) 0'ı kesip kararın `STATIC_EQUIVALENT` olması. Vektörler olmadan doğrulanamadı, bu yüzden `NEEDS_EVIDENCE`.
- **İlgili:** SOL-05, Neon Soru 3.

#### OPUS-002 · MAJOR · VERIFIED (kanıt üretildi) — `M3_perm_within_patient` hasta ofsetini korumuyor; ablasyon farkı ilaç-kimliği bilgisini olduğundan büyük gösteriyor
- **Dosya:** `src/oncotwin/experiments/run_sprint03.py:238-253`; protokol `PROTOCOL_SPRINT03.md §4.3`.
- **Yeniden üretim:** (a) Sentetik test: `opus_adversarial_checks.py::test_perm_within_patient_does_not_preserve_offset_under_unequal_weights`. (b) Gerçek veri: `realdata_probe.py`.
- **Gözlenen:**
  - (a) Saf-ofset dünyasında (ilaca özgü kovaryans **yok**, yalnız heteroskedastik gürültü var) perm_within kazancın **%32'sini** kaybediyor (retained 68%). Homoskedastik durumda kayıp sıfır. Nedeni: koşullu Gaussian ağırlıkları `C_ts C_ss⁻¹` eşit değil, bu yüzden permütasyon ağırlıklı ofseti de bozuyor.
  - (b) Gerçek veri, k=5 StaticDOpt: Ortak-ofset (compound-symmetry) Gaussian, λ'sı oracle ile seçilerek null lehine avantajlı, M3 kazancının **CV %49 / W3+4 %45**'ini açıklıyor. RMSE: CV M1 49.26 → offset-only 46.42 → M3 43.49; W3+4 55.66 → 53.46 → 50.81. perm_within ise kazancın yalnızca **%10 (CV) / −%4 (W3+4)**'ünü koruyor, yani ofsetten de daha kötü.
  - (c) Statik-panel ablasyonu tek permütasyon çekilişi kullanıyor. Farklı tohumla W3+4 perm_within RMSE 57.85 (üretim) yerine 55.86 çıkıyor, yani ~2 AUC çekiliş varyansı var. k=3'te özdeş permütasyon oranı %16.9 (k=5: %0.75, k=8: %0).
- **Beklenen:** Ablasyonun "skaler ofset korunur, ilaca özgü kovaryans bozulur" iddiası, ofset gerçekten korunduğunda geçerli olur.
- **Etki:** "İlaç kimliği bilgisi var" niteliksel sonucu ayakta kalıyor: M3_full, oracle offset-only'yi CV'de ~2.9, W3+4'te ~2.65 RMSE farkla geçiyor. Ancak "perm_within farkı −6.21 = ilaca özgü katkı" nicel yorumu yaklaşık 2 kat şişirilmiş. Kapı 2 ablasyon kriteri (GA 0'ı dışlar) bundan etkilenmiyor.
- **Öneri:** Ayrı sürümde önceden kayıtlı bir `M3_offset_only` ablasyonu ekleyin (iç-CV ile λ seçimli compound-symmetry kovaryans). perm_within için çoklu çekiliş (≥20) ve derangement kullanın. Rapor metnindeki nicel yorumu düzeltin. Probe tek seed'li bir nokta tahmini; GA'lar ayrı sürümde üretilmeli.
- **İlgili:** Neon Soru 4 (Sol'a), SOL-03.

#### OPUS-003 · MAJOR · VERIFIED — W3+4'te "Static-D-Optimal" fiilen farklı bir politika; tam-panel alt grubu sessizce boş
- **Dosya:** `run_sprint03.py:848-863` (`n_full >= 5` koşulu), `panel_selection.py:141-159`.
- **Gözlenen:** Temiz W3+4 test kümesinde (165 indeks hasta) **ABT-737 = 3, XAV-939 = 0** gözlem var. Neon panosundaki "15 gözlem" bu değerlendirme kümesine ait değil. Kesin paydayı Neon doğrulamalı (OPUS-015). Sonuç olarak k=3/5/8/12'de `full_panel_patient_rate = 0.0`, yani tam-panel hasta yok. `full_panel_subgroups_dopt = {}` JSON'a **uyarısız boş** yazılıyor. Oysa `PROTOCOL_SPRINT03 §1.4` bu alt grubun raporlanmasını istiyor. Direct-hit oranı k=3'te %62.6. CV'de tam-panel oranı k=5 %46.5, k=12 %18.7.
- **Beklenen:** Raporlanamayan protokol alt grubu açık bir sapma olarak işaretlenmeli.
- **Etki:** W3+4 "StaticDOpt" sonuçları, D-optimal sıralamanın W3+4 plaka mevcudiyetiyle filtrelenmiş haline (hastaya göre backfill) ait. OOD'de D-optimallik iddiası test edilmiş sayılmamalı. Bütçe eşitliği korunuyor (doğrulandı, `test_ood_backfill_preserves_budget_when_enough_available`).
- **Öneri:** (1) Raporda protokol sapmasını açıkça yazın. (2) Ayrı sürümde, prob havuzunu her iki dalgada da ölçülmüş ilaçlarla sınırlayan bir duyarlılık koşusu yapın. Bu, plaka tasarımı bilgisidir, yanıt bilgisi değildir; yine de keşifsel olarak etiketlenmeli. (3) `n_full < 5` olduğunda alt grubu `{"n_full_panel_patients": n, "reason": "insufficient"}` olarak yazın.
- **İlgili:** SOL-04, SOL-03, Neon Soru 2 (Sol'a).

#### OPUS-004 · MINOR · VERIFIED (sonuca etkisi yok) — Kapı 2 ablasyon kontrolü W3+4'te protokolden gevşek
- **Dosya:** `run_sprint03.py:726`
- **Gözlenen:** `ok_wv` yalnızca nokta tahmini (`diff`) kullanıyor. `PROTOCOL_SPRINT03 §5.1` ise hem CV hem W3+4 için %95 GA'nın 0'ı dışlamasını istiyor.
- **Kanıt:** JSON'daki 16 W3+4 karşılaştırmasının (k∈{5,8} × 2 panel kaynağı × 4 ablasyon) tamamında RMSE GA üst sınırı < 0 ve ρ_res GA alt sınırı > 0. En zayıfı Random50/perm_train_cov k=5: ρ_res `+0.033 [+0.014, +0.051]`. Yani `ablation_sanity_passed = true` protokol kuralıyla da doğru.
- **Öneri:** Kodu protokole hizalayın (`ci_high`/`ci_low`). Tek satırlık bir değişiklik.

#### OPUS-005 · MINOR · OPEN — İç içe OOF: M1 ve M3 iç-CV katları özdeş; OOF kovaryansı OOD'de varyansı düşük tahmin ediyor
- **Dosya:** `baselines.py:280` (`fit_and_oof` → `_build_inner_cv(seed=inner_seed)`), `baselines.py:431` (M2a/M3 aynı tohum), `baselines.py:443`.
- **Gözlenen:** M3'ün (α, λ) seçiminde, iç doğrulama katının hedefleri, `Y_oof[tr]` üretilirken kullanılan M1 fitlerine girmiş durumda. Bu ikinci derece bir iç-CV sızıntısı; M1 `best_lambda_` de aynı OOF üzerinde seçiliyor. **Test hastaları etkilenmiyor.** Bunu doğruladım: gizli test AUC'leri permüte edilip +N(0,500) eklendiğinde seçim ve M3 hedef tahminleri bit düzeyinde aynı kalıyor. Probe sonucu: test kalıntı varyansı / OOF kovaryans diyagonali (medyan) CV'de 0.75–1.23, **W3+4'te 1.41**.
- **Etki:** Nokta tahminlerine etkisi küçük. Ancak Sprint 4'te `Σ_res` posterior varyansı doğrudan aralık genişliği olarak kullanılırsa OOD'de ~%40 dar aralık oluşur.
- **Öneri:** Sprint 4'te ayrı hasta-düzeyi kalibrasyon katı kullanın; posterior varyansı ham kapsam iddiası olarak kullanmayın. Sprint 3 için yeniden koşu gerekmiyor.
- **İlgili:** SOL-01, Neon Soru 1 (Sol'a).

#### OPUS-006 · MINOR · OPEN — Sessiz bütçe kesilmesi (guard yok)
- **Dosya:** `panel_selection.py:119,141-159`; `splits.py:116-117`; `baselines.py:92-96` (NaN prob atlanır), `baselines.py:523-525` (aktif kümede olmayan prob atlanır).
- **Gözlenen:** k > mevcut prob olduğunda dört yolun dördü de uyarısız olarak <k prob döndürüyor (`test_budget_exceeding_available_is_silently_truncated`). Sprint 3'te `MIN_EVAL_PROBES=12 ≥ max k` filtresi bunu fiilen engelliyor; doğrulandı.
- **Öneri:** Koşucuda değerlendirilebilir hastalar için `assert len(S_p) == k` ekleyin. Sprint 4'te bütçe/filtre değişirse bu açık sessiz yanlılığa döner.

#### OPUS-007 · MINOR · OPEN — Sprint 3 RMSE tanımı protokolden farklı
- **Dosya:** `run_sprint03.py:139` (`sqrt(nanmean(patient_mse))`)
- **Gözlenen:** Sprint 3 RMSE hasta-ağırlıklı. Protokol §7 ise gözlenen (p,d) çiftleri üzerinden havuzlanmış RMSE tanımlıyor. Sprint 2 her ikisini raporluyor (`obs_rmse`).
- **Etki:** Politikalar arası karşılaştırmalar tutarlı ve geçerli. Mutlak değerler Sprint 2 tablosuyla birebir kıyaslanamaz.
- **Öneri:** Rapora tanım notu ekleyin veya `obs_rmse` alanını da yazın.

#### OPUS-008 · MINOR · OPEN — Sprint 2 permütasyon sızıntı testi GA'sı ilaç düzeyinde
- **Dosya:** `run_sprint02.py:216, 251`
- **Gözlenen:** Bootstrap, protokol §7'deki hasta düzeyi yerine 24 ilaç üzerinden yapılıyor. Sızıntı testi için bu, GA'yı genelde daraltacağından muhafazakâr bir yönde. PASS sonucunu geçersiz kılmıyor, ama protokol metniyle uyumsuz.
- **Öneri:** Rapora not düşün. Sprint 4 sızıntı testlerinde hasta düzeyi bootstrap kullanın.

#### OPUS-009 · MINOR · OPEN — Performans: darboğaz değerlendirme döngüsü, Gaussian koşullama değil
- **Ölçüm (mikro-benchmark + probe):** `_evaluate_predictions_full` çağrı başına 113.7 ms. Seed başına ~933 çağrı, toplamda ≈ **355 s, yani 700 s'nin ~%50'si**. `conditional_gaussian_predict_matrix` çağrı başına 3.6 ms, toplamda ≈ 60 s. Kat başına fit (M0+M1+M3) ~9 s; ablasyon fitleriyle birlikte tahminen ~230 s.
- **Öneri (beklenen kazanç / risk):**
  1. Hasta başı metrikleri vektörleştirin (ilaç sayısı ≤24; bağ olmayan satırlarda argsort yolu, bağlı satırlarda mevcut fonksiyona düşüş). Beklenen kazanç ~250–300 s. Bağ semantiği korunursa sayısal fark yok. Mevcut fonksiyonla eşitlik testi zorunlu; regresyon riski düşük.
  2. Random50 için 50 tekrarın değerlendirmesini tek seferde yığınlayın. Bu (1) ile aynı iş.
  3. Gaussian koşullamada ortak Cholesky önbelleği: kazanç ≤40 s, değmez (YAGNI).
- Sprint 4'ten önce zorunlu değil.

#### OPUS-010 · MINOR · OPEN — Kod kalitesi: üçlü tekrar, 800 satır, ölü kod, kalıcı olmayan vektörler
- Değerlendirme mantığı **3 yerde** tekrarlanıyor (Rule of Three sağlanıyor): `metrics.evaluate_predictions`, `run_sprint02._evaluate_split_predictions`, `run_sprint03._evaluate_predictions_full`. Fark yalnızca kat-içi ilaç Spearman'ı ve RMSE ağırlığı. Tek bir `eval/` fonksiyonunda `fold_ids` + `rmse_weighting` parametresiyle birleştirin.
- `run_sprint03.py` içinde metrik-düğüm sözlüğü 3 kez kopyalanmış (`:384-397`, `:558-570`, `:624-636`).
- `run_sprint03.py` 918 satır (>800). En az sürtünmeli bölünme: ablasyonları (`:238-280`, `:479-572`) `experiments/sprint03_ablations.py`'ye, Kapı 2'yi (`:712-780`) `experiments/sprint03_gate.py`'ye taşımak. Davranış değişmez.
- Ölü kod: `baselines.py:9` `make_patient_disjoint_kfold`, `run_sprint02.py:12` `import pandas as pd` (kullanılmıyor), `baselines.py:519` `_ = alpha_k`.
- `_strip_all_vectors` hasta vektörlerini atıyor. Bu yüzden yeniden bootstrap için 700 s'lik koşu gerekiyor (OPUS-001'i bu engelledi). Vektörleri ayrı bir `.npz`'ye yazın.
- **Bu turda refaktör yapılmadı.**

#### OPUS-011 · INFO · VERIFIED — Sayısal çekirdek doğru
- Static-D-Opt'un **tam** açgözlü sıralaması, brute-force `slogdet` MI açgözlüsüyle aynı MI eğrisini veriyor. Tam rank ve rank-2/3 (tekile yakın) kovaryansta, λ ∈ {50, 500, 5000} için kontrol edildi. Mevcut test yalnızca ilk seçimi doğruluyordu.
- Adaptive OED, her adımda posterior'u baştan hesaplayan brute-force ile birebir aynı.
- 60 ardışık rank-1 downdate sonrası doğrudan Schur'a göre bağıl sapma < 1e-9. Paydalar ≥ λ ≥ 50 olduğundan downdate iyi koşullu.
- `conditional_gaussian_predict_matrix` açık formülle eşleşiyor.
- `estimate_psd_shrunk_cov` tamamen eksik, sabit, min_pair altı ve kolineer sütunlarda kesin pozitif tanımlı ve Cholesky'den geçiyor.
- `compute_contender_weights` 1e-6 jitter'ı, ölçeği ~3600 olan tam tekil rank-1 matriste bile geçti. Üretimde girdi zaten min_eig 1e-4 ile geliyor.
- İlgili: Neon Soru 3.

#### OPUS-012 · INFO · VERIFIED — Sızıntı, eşleştirme ve bütçe
- Gizli test AUC'leri permüte edilip +N(0,500) eklendiğinde (maske aynı) Adaptive ve Random seçimleri bit düzeyinde aynı. Hedef sütunlarına +1000 eklendiğinde M3 hedef tahminleri değişmiyor.
- Statik sıralamalar yalnız eğitim katından geliyor. U_tr ∩ C = ∅ yapı gereği sağlanıyor.
- İndeks veri kümesinde 471 satır = 471 benzersiz hasta. Bootstrap hasta düzeyinde. 3 seed her hasta için önce ortalanıyor, yani bağımsız hasta sayılmıyor. Random50 de hasta başına ortalanıyor.
- Tüm politika/k karşılaştırmalarında `n_paired` = 427 (CV) / 146 (W3+4). Hedef kümesi aynı.
- Tekrarlı specimen satırlarıyla kat ayrıklığı korunuyor.
- **Determinizm:** Probe, W3+4 k=5 `M3_full` RMSE'sini üretim değeriyle aynı çıkardı (`50.81330363124913` vs `50.81330363124914`); M1 de aynı. Aynı tohum aynı sonucu, farklı tohum farklı sonucu veriyor.
- **Uyarı (SOL-06):** GA'lar fit edilmiş modellere koşullu. Kat yeniden-fit varyansı bootstrap'e girmiyor, bu yüzden hafif anti-muhafazakâr.
- İlgili: SOL-01, SOL-02, SOL-03, SOL-06.

#### OPUS-013 · INFO · VERIFIED — Sıralama metrikleri
- Recall@3, NDCG@3, Regret@1 ve Regret@3, 4 bağ senaryosunda tüm bağ-kırma permütasyonlarının kesin beklenen değerine eşit.
- Uç durumlar: tek ilaç, tümü NaN (NaN döner) ve sabit vektör (Spearman 0) doğru işleniyor. `math.comb` blok ≤24 olduğundan güvenli.
- Tüm gerçek AUC'leri eşit olan hastada NDCG NaN dönüyor ve `nanmean` ile düşüyor. Bu yalnız `y_true`'ya bağlı olduğundan politikalar arası yanlılık yaratmıyor.
- `paired_bootstrap_ci` elle hesaplanan hasta-yeniden-örneklemesiyle aynı. n=1'de sıfır genişlikli GA'yı uyarısız döndürüyor.
- İlgili: Neon Soru 4.

#### OPUS-014 · INFO · VERIFIED — `k_star_saturation = 2` hesabı doğru, ama eşik kaynaklı
- StaticDOpt CV adımları:
  - 1→2: `−1.037 [−1.475, −0.592]`, prob başına 1.037 ≥ 1, geçer.
  - 2→3: `−0.701`, prob başına 0.70 < 1, k*=2.
  - Sonraki tüm adımlar anlamlı: 3→5 prob başına 0.50, 5→8 0.50, 8→12 0.25; GA'ların hepsi 0'ı dışlıyor.
- k*=2, "prob başına ≥1 AUC" eşiğiyle bulunan bir doygunluk; istatistiksel doygunluk değil. 1→2 marjı yalnızca 0.037 AUC, yani bıçak sırtında.
- **Öneri:** Önceden kayıtlı k*=2'yi ve keşifsel k=3–8 kazançlarını ayrı raporlayın (SOL-07 ile uyumlu). Sprint 4 için {2, 5, 8} bütçelerini dondurmak makul. Karar Sol ve kullanıcıya ait.

#### OPUS-015 · INFO · OPEN — Veri sayımı tutarsızlığı
- Neon ve Sol panolarında "W3+4'te ABT-737 = 15 gözlem" yazıyor. Temiz W3+4 indeks test kümesinde (n=165) ölçtüğüm değer **3**. 15 büyük olasılıkla tüm W3+4 örneklerine (Both dahil) ait.
- **Öneri:** Neon paydayı netleştirsin. Sonuçları değiştirmez, ama OPUS-003'ü güçlendiriyor.

#### OPUS-016 · INFO · VERIFIED — Adaptif politika gerçekten hastaya özgü; bellek sorun değil
- k=5'te benzersiz adaptif panel sayısı (probe): gerçek maskeyle 30–55 / kat, tüm problar mevcut sayıldığında 22–49 / kat. Çeşitlilik çoğunlukla omik ağırlıklardan geliyor, mevcudiyet maskesinden değil. Üretim JSON'unda Jaccard ≈ 0.26 (CV k=5).
- **Bellek:** `X_rna` 512×22843 float32 = 46.8 MB. Probe peak RSS 277 MB (yükleme sonrası) ve 463 MB (6 split fit sonrası). `fold_cache` kat başına ~n_tr² Gram matrisinde kalıyor (~1 MB). Sprint 4/5 matrisleri eklense de düşük RAM'li donanımda risk görmüyorum. Tam koşu peak RSS'i ölçülmedi.
- İlgili: Neon Soru 5.

### 3.2 Neon'un 6 Teknik Sorusuna Yanıtlar

- **Soru 1 (Modülerlik / DRY / ölü kod):** OPUS-010. Bölünmeyi ablasyon ve kapı modüllerine çıkararak yapın, davranış değişmesin. Değerlendirici birleştirmesi Rule of Three'yi karşılıyor. Ölü kod 3 kalem.
- **Soru 2 (Süre):** OPUS-009. Hipotez kısmen yanlış: süre esas olarak Python metrik döngüsünde harcanıyor (~%50), koşullu Gaussian ~%9. Vektörleştirmeyle ~700 → ~400 s bekleniyor.
- **Soru 3 (Sayısal kararlılık):** OPUS-011. Downdate drift'i ihmal edilebilir. λ ≥ 50 payda tabanı stabiliteyi garanti ediyor. Jitter ve PSD kırpma yeterli. Değişiklik gerekmiyor.
- **Soru 4 (Uç durumlar):** OPUS-013 ve OPUS-006. `math.comb` güvenli. NaN/sabit vektörler doğru düşüyor. `len(obs) < k` durumu yalnız filtre sayesinde güvenli; assert eklenmeli.
- **Soru 5 (Bellek):** OPUS-016. Risk yok.
- **Soru 6 (Test ekleri):** `audits/opus-2026-11-21/opus_adversarial_checks.py` içindeki 23 testten şunların `tests/`'e taşınmasını öneriyorum (kullanıcı onayıyla):
  - tam D-opt sıralaması vs brute-force,
  - adaptive vs brute-force,
  - downdate drift,
  - tie-enumerasyon metrik referansı,
  - gizli-AUC invaryansı,
  - bütçe assert'i,
  - offset-korunumu.

### 3.3 Sol'un İnceleme Taleplerine Teknik Yanıt

| Sol ID | Opus kanıtı | Durum |
|---|---|---|
| SOL-01 | Değer sızıntısı yok (OPUS-012). Mevcudiyet maskesi tüm politikalarda simetrik kullanılıyor; adaptif çeşitlilik maskeden değil omikten geliyor (OPUS-016). İkinci derece iç-CV sızıntısı yalnız hiperparametre seçiminde var (OPUS-005). | VERIFIED + MINOR açık |
| SOL-02 | U_tr ∩ C = ∅. M3 hedef tahminleri hedef AUC'lerinden bağımsız (OPUS-012). | VERIFIED |
| SOL-03 | Aynı hasta ve hedef kümesi; n_paired sabit. Kapsam kaybı: CV 471→427, W3+4 165→146 (≥12 prob filtresi). | VERIFIED |
| SOL-04 | W3+4'te tam-panel n=0; ABT-737 3, XAV-939 0 (OPUS-003, OPUS-015). | MAJOR açık |
| SOL-05 | Kural asimetrik ve çokluk düzeltmesiz (OPUS-001). | MAJOR açık |
| SOL-06 | Hasta-kümeli; seed'ler bağımsız sayılmıyor. Refit varyansı dışarıda (OPUS-012). | VERIFIED + not |
| SOL-07 | Hesap doğru, eşik kaynaklı (OPUS-014). | VERIFIED |

Sol'un hiçbir talebine katılmadığım bir nokta yok. Ek olarak Sol'un listesinde olmayan OPUS-002'yi (ablasyon ofset karışması) bilimsel değerlendirmesi için Sol'a devrediyorum.

### 3.4 QA Kararı

- **Kod Kalitesi & Optimizasyon Hükmü:** `[ ] Onaylandı` | `[x] Koşullu Onay (Fix Gerekli)` | `[ ] Bloklu`
- **QA kararı: `QA_CONDITIONAL`**
- **Gerekçe:** CRITICAL bulgu yok: değer sızıntısı yok, sayısal çekirdek doğru, determinizm doğrulandı. Kapı 2'nin mekanik PASS'ı (ablasyon GA'ları ve StaticDOpt/Adaptive vs Random50) teknik olarak yeniden doğrulanabilir durumda. Ancak iki **yorumsal** çıktı mevcut haliyle Sprint 4 kararlarına temel yapılmamalı: `ADAPTIVE_WINS` (OPUS-001) ve perm_within'in nicel yorumu (OPUS-002). W3+4 StaticDOpt sonuçları da protokoldeki politikayla birebir aynı değil (OPUS-003).
- **Bu karar yalnızca teknik QA kapsamındadır.** Bilimsel Kapı 2 hükmü Sol'a, nihai onay kullanıcıya aittir.

**Sprint 4 öncesi zorunlu teknik liste (kullanıcı onayı bekler; hiçbiri uygulanmadı):**
1. OPUS-001: Hasta vektörlerini kalıcılaştırın. `sprint03_v2` ayrı sürümünde simetrik ve çokluk düzeltmeli Adaptive-vs-Static kararını üretin (mevcut JSON'u ezmeyin). O zamana kadar raporda "kırılgan" notu kalsın.
2. OPUS-002: `M3_offset_only` ablasyonu ve çoklu-çekilişli perm_within (ayrı sürüm). Rapor metnindeki nicel yorumu düzeltin.
3. OPUS-003: W3+4 tam-panel sapmasını açıkça raporlayın; prob havuzu iki dalgada ortak ilaçlarla sınırlı keşifsel duyarlılık koşusu yapın.
4. OPUS-004 + OPUS-006: Kapı 2 W3+4 GA hizalaması ve `len(S_p)==k` assert'i (minimal, davranışı değiştirmez).
5. OPUS-005: Sprint 4 protokolünde ayrı kalibrasyon katı. OOF `Σ_res` varyansı OOD'de ×1.4 düşük; ham aralık olarak kullanılmamalı.

**Opsiyonel (Sprint 4'ü bloklamaz):** OPUS-009 vektörleştirme, OPUS-010 bölünme/DRY/ölü kod, OPUS-007/008 rapor notları.

### 3.5 Devir

- **Neon'a:** OPUS-001, 003, 004, 006, 010, 015 için uygulama önerisi. Hepsi kullanıcı onayından sonra ve ayrı çıktı sürümüyle yapılmalı.
- **Sol'a:** OPUS-001 (karar kuralı), OPUS-002 (ablasyon geçerliliği) ve OPUS-003 (OOD politikası tanımı) için bilimsel hüküm.
- **Kullanıcıya:** §3.4'teki 1–5 numaralı maddeler için onay.
- Opus bu turda durdu. Sprint 4'e başlanmadı; protokol ve üretim sonuçları değiştirilmedi.

### 3.6 Opus Geliştirme Kuyruğu — Tamamlandı (2026-11-22, kullanıcı talimatıyla)

**Çalışma kuralı (kullanıcı kararı, 2026-11-21):** Aşağıdaki maddeler Opus'un kendi geliştirme görevleridir. **Opus, kullanıcı ilgili maddeyi açıkça talimat olarak verene kadar hiçbirini uygulamaz.** Talimat geldiğinde:
- Madde TDD ile uygulanır (RED → GREEN → REFACTOR).
- Sonucu değiştirebilecek her koşu ayrı sürüm olarak yazılır (`*_v2_*`). Mevcut `sprint0{2,3}_results.json` ve sprint raporları ezilmez.
- `PROTOCOL*.md` dosyalarına dokunulmaz.
- İş bitince bu tablodaki durum, doğrulama kanıtı ve tarih güncellenir.

| DEV ID | Kaynak bulgu | İş | Dokunulacak dosyalar | Sonucu değiştirir mi? | Ön koşul | Kabul kriteri | Durum |
|---|---|---|---|---|---|---|---|
| `DEV-OPUS-01` | OPUS-006 | Değerlendirilebilir hastalarda `len(S_p) == k` assert'i (Random50, statik, adaptif, ablasyon yolları) | `run_sprint03.py` | Hayır | — | Bütçe kesilmesi senaryosunda test RED → assert ile GREEN; mevcut 15 test geçer | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-02` | OPUS-004 | Kapı 2 W3+4 ablasyon kontrolünü protokole hizalama (`diff` → `ci_high`/`ci_low`) | `run_sprint03.py:726` | Hayır (16/16 GA zaten 0'ı dışlıyor) | Sol bilgilendirilir | Mevcut JSON'dan yeniden hesaplanan `evaluate_gate2` çıktısı değişmez | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-03` | OPUS-010 | Ölü kod temizliği: `baselines.py:9` import, `run_sprint02.py:12` `pd`, `baselines.py:519` `_ = alpha_k` | `baselines.py`, `run_sprint02.py` | Hayır | — | 15 + 23 test geçer | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-04` | OPUS-009 | Hasta başı metriklerin vektörleştirilmesi (bağsız satırlar hızlı yol, bağlı satırlar mevcut fonksiyon) | `eval/metrics.py`, `run_sprint03.py` | Hayır (bit düzeyinde eşitlik hedefi) | DEV-OPUS-05 ile birlikte yapılabilir | Mevcut fonksiyonla rastgele + bağlı girdilerde `np.array_equal`; değerlendirme süresi ≥3× hızlı | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-05` | OPUS-010 | Üç değerlendiricinin tek `eval/` fonksiyonunda birleştirilmesi (`fold_ids`, `rmse_weighting` parametreleri); metrik-düğüm sözlüğünün tekilleştirilmesi | `eval/metrics.py`, `run_sprint02.py`, `run_sprint03.py` | Hayır | — | Sprint 2/3 değerlendirici çıktıları sentetik girdide eşit | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-06` | OPUS-010 | `run_sprint03.py` bölünmesi: ablasyonlar → `sprint03_ablations.py`, Kapı 2 → `sprint03_gate.py` (<800 satır) | `experiments/` | Hayır | DEV-OPUS-05 | Tüm testler geçer; `run_sprint03.py` < 800 satır | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-07` | OPUS-010, OPUS-001 | Hasta vektörlerinin `.npz` olarak kalıcılaştırılması ve Sprint 3'ün **ayrı sürümle** yeniden koşusu (`sprint03_v2_results.json`, `sprint03_v2_vectors.npz`) | `run_sprint03.py` (çıktı yolu parametresi) | Hayır (determinizm doğrulandı; aynı sayılar beklenir) | DEV-OPUS-01…04 önerilir (süre) | v2 metrikleri v1 ile ≤1e-9 eşit; vektörler yüklenebilir | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-08` | OPUS-001 | Simetrik ve çokluk düzeltmeli (Holm, 3 k × 4 metrik, Recall@3 dahil) Adaptive-vs-Static kararı; **ayrı rapor alanı**, eski etiket korunur | yeni `sprint03_reanalysis.py` (yalnız `.npz` okur) | **Evet** (etiket değişebilir) | DEV-OPUS-07, Sol'un kural onayı | Kural sentetik karşılaştırmalarla test edilir; çıktı `sprint03_v2_reanalysis.json` | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-09` | OPUS-002 | `M3_offset_only` ablasyonu (iç-CV ile λ seçimli compound-symmetry) + perm_within için ≥20 çekiliş ve derangement | `run_sprint03.py` (ablasyon modülü) | **Evet** (yeni ablasyon satırları; Kapı 2 kuralı değişmez) | DEV-OPUS-06, Sol onayı | Saf-ofset sentetik testte offset_only ≈ full; v2 çıktısında GA'lı sonuç | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-10` | OPUS-003 | Tam-panel alt grubu `n<5` iken gerekçeli kayıt; W3+4 için iki dalgada ortak prob havuzuyla **keşifsel** duyarlılık koşusu | `run_sprint03.py`, rapor üreticisi | **Evet** (yalnız keşifsel tablo) | DEV-OPUS-07, Sol onayı | JSON'da boş `{}` yerine gerekçe; duyarlılık tablosu "keşifsel" etiketli | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-11` | OPUS-007, OPUS-008 | Raporlara `obs_rmse` alanı ve RMSE tanım notu; Sprint 2 sızıntı GA'sının ilaç düzeyi olduğuna dair not | `sprint03_report.py`, v2 rapor | Hayır | DEV-OPUS-07 | v2 raporunda iki RMSE ve notlar görünür | ✅ TAMAM (2026-11-22, bkz. §3.7) |
| `DEV-OPUS-12` | Neon Soru 6 | `audits/opus-2026-11-21/` testlerinden kalıcı olanları `tests/test_opus_regression.py`'ye taşıma (D-opt ve adaptif brute-force, downdate drift, bağ-enumerasyonu, gizli-AUC invaryansı, bütçe, ofset korunumu) | `tests/` | Hayır | — | `pytest` → 15 + taşınan testler geçer | ✅ TAMAM (2026-11-22, bkz. §3.7) |

**Opus'un uygulamayacağı, yalnız girdi verdiği madde:** OPUS-005 (Sprint 4 kalibrasyon katı ve `Σ_res` varyans uyumsuzluğu). Bu Sprint 4 protokol kararıdır; Sol ve kullanıcıya aittir.

**Önerilen sıra:**
1. Düşük riskli, sonucu değiştirmeyen işler: 01 → 02 → 03 → 12.
2. Performans ve yapı: 04 → 05 → 06.
3. Ayrı sürümle yeniden koşu: 07.
4. Sonucu değiştirebilecek yeniden analizler, Sol onayıyla: 08, 09, 10.
5. Rapor notları: 11.

**Talimat formatı önerisi:** `DEV-OPUS-01..03 yap` veya `DEV-OPUS-08 yap`. Opus yalnız adı geçen maddeleri uygular.

**Değişiklik günlüğü:**
- 2026-11-21 — Kuyruk oluşturuldu. Hiçbir madde uygulanmadı.
- 2026-11-22 — Kullanıcı talimatıyla DEV-OPUS-01..12 uygulandı; 08/09/10 Sol §3.2/2–4 kurallarıyla. Kanıt §3.7'de.


### 3.7 Uygulama Raporu — DEV-OPUS-01..12 (2026-11-22)

**Agent:** Claude Opus 5.5 · **Role:** Independent Optimization & QA Reviewer · **Tetikleyici:** kullanıcı talimatı ("bunları fixle, kendi payına düşeni yap") · **Kurallar:** 08/09/10 için Sol §3.2/2–4 ve Neon §4.2 hizalaması.

**Korunanlar (SHA-256 ile doğrulandı):** `sprint0{1,2,3}` JSON'ları, `sprint-0{1,2,3}` raporları ve `PROTOCOL*.md` dosyaları değişmedi. v2 koşucusunun varsayılan çıktı yolları artık v2 dosyalarına yazar, bu yüzden v1'in kazara ezilmesi engellendi.

**Yeni çıktılar:**
- `data/processed/sprint03_v2_results.json`
- `data/processed/sprint03_v2_vectors.npz` (1.221 hasta vektörü + hasta kimlikleri)
- `data/processed/sprint03_v2_reanalysis.json`
- `docs/sprints/sprint-03-v2-panel-selection.md`
- `docs/sprints/sprint-03-v2-reanalysis.md`

**Komutlar:**
- `PYTHONPATH=. python3 -m src.oncotwin.experiments.run_sprint03` — v2 üretir.
- `PYTHONPATH=. python3 -m src.oncotwin.experiments.sprint03_reanalysis` — yalnız `.npz` okur.

#### Doğrulama kanıtı
| Kontrol | Sonuç |
|---|---|
| Test paketi | **50 passed**: Neon'un 15 testi + `tests/test_opus_regression.py` içinde 35 test. Taşınan 21 denetim testine ek olarak 14 yeni test; 01, 02, 04/05, 08 ve 09 için önce RED görüldü. |
| v1 ↔ v2 Sprint 3 JSON | 7.984 sayısal alanın tamamı eşleşti, eksik alan 0. En büyük fark 1.4e-14 (bağıl 5.5e-14). Kapı 2 kararı birebir aynı (`PASS`). |
| Sprint 2 regresyonu (ortak değerlendiriciye geçiş) | Scratchpad'e yeniden koşuldu; 1.666 alan eşleşti, en büyük fark 1.4e-14. Üretim dosyası ezilmedi. |
| Kapı 2 W3+4 GA hizalaması | Mevcut v1 karşılaştırmalarıyla aynı karar (16/16 GA 0'ı dışlıyor). |
| Süre | Sprint 3: 700.5 s → **440.7 s**, ek v2 işleri dahil (offset-only, 20 çekiliş, ortak havuz koşusu). Sprint 2: 360.9 s → **301.0 s**. Değerlendirici çağrısı 99 ms → 15 ms (6.6×). |
| Peak RSS (tam Sprint 3 v2) | **470.7 MB**. OPUS-016'daki ölçülmemiş kısım kapandı. |
| Dosya boyutu | `run_sprint03.py` 918 → 617 satır. Yeni modüller: `sprint03_ablations.py` 251, `sprint03_gate.py` 118, `sprint03_reanalysis.py`. 800 satırı aşan dosya yok. |

#### Madde madde
- **01 (OPUS-006):** `_require_full_budget` Random50, statik ve adaptif yollarda değerlendirilebilir hastada `len(S_p) != k` durumunda `ValueError` atar. Gerçek koşuda hiç tetiklenmedi.
- **02 (OPUS-004):** `evaluate_gate2` artık `sprint03_gate.py` içinde; W3+4 ablasyonu %95 GA ile değerlendiriliyor.
- **03 (OPUS-010):** Ölü kod temizlendi: `make_patient_disjoint_kfold` importu, `_ = alpha_k`, `pandas` importu. Taramamın denetimde atladığı `run_sprint02` içindeki kullanılmayan `fit_drug_means` importu da kaldırıldı.
- **04/05 (OPUS-009/010):** Üç değerlendirici `eval/metrics.evaluate_masked_predictions` içinde birleşti.
  - Bağsız satırlarda vektörel hızlı yol, bağlı satırlarda tanımsal skaler fonksiyonlar kullanılıyor.
  - `_rankdata_average` bağsız girdide döngüsüz çalışıyor.
  - Düğüm ortalaması `average_metric_nodes` ile, bootstrap paketi `paired_metric_diffs` ile tekilleşti.
  - `evaluate_predictions` API'si korundu.
- **06 (OPUS-010):** Ablasyonlar `sprint03_ablations.py`, Kapı 2 `sprint03_gate.py` içine taşındı. `tests/test_sprint03.py` dosyasında **yalnız import yolu** güncellendi (`_permute_*` → `sprint03_ablations`); assert'lere dokunulmadı.
- **07:** v2 koşusu ve `.npz` arşivi üretildi. v1 sayıları ≤1.4e-14 farkla yeniden üretildi.
- **08 (OPUS-001, Sol §3.2/3):** 9 test, iki yönlü yüzdelik-bootstrap p (B=10.000), Holm α=0.05. Sonuç: **CV `INSUFFICIENT_EVIDENCE`, W3+4 `INSUFFICIENT_EVIDENCE`**.
  - En küçük düzeltilmiş p: CV k=8 NDCG@3 ham 0.012 → Holm 0.112; k=5 NDCG@3 ham 0.026 → Holm 0.206.
  - v1'deki `ADAPTIVE_WINS` çokluk düzeltmesinden sonra desteklenmiyor. Audit'te `STATIC_EQUIVALENT` öngörmüştüm; Sol'un etiket setinde karşılığı "yetersiz kanıt". Pratik eşdeğerlik test edilmedi.
- **09 (OPUS-002, Sol §3.2/4):** `M3_offset_only` modeli eklendi: λ yalnız eğitim OOF artıklarıyla, her k için ayrı seçiliyor; ilaç ölçeği kullanılmıyor, eşit ağırlık. Ayrıca ofseti sabit tutan, eğitim ölçeğiyle standardize ve özdeş olmayan permütasyon 20 sabit çekilişle koşuldu.
  - **Ofset payı (StaticDOpt panelleri):** CV k=3/5/8'de M3 RMSE kazancının **%43**'ü, W3+4'te **%49–51**'i (Random50 panellerinde CV %47–55, W3+4 %62–73).
  - **M3_full − offset-only:** 6 hücrenin 6'sında GA 0'ı dışlıyor. CV k=5 `−3.70 [−4.48, −2.96]`, W3+4 k=5 `−2.46 [−3.89, −1.04]`. İlaca özgü kovaryans bilgisinin varlığı artık GA ile destekleniyor.
  - **perm-offset-fixed (20 çekiliş):** CV k=5 RMSE 48.41 ± 0.21, W3+4 k=5 56.67 ± 0.56. Ofset sabitken bile offset-only'den (46.44 / 53.28) **kötü**. Yorum: permütasyon ablasyonları "bilgiyi kaldırmıyor", yanlış ilaç eşlemesiyle **yanıltıcı bilgi ekliyor**. Bu yüzden v1 perm farkları ilaç-kimliği katkısının büyüklüğü olarak okunmamalı; büyüklük için `full − offset_only` farkı kullanılmalı. Hüküm Sol'un.
- **10 (OPUS-003, Sol §3.2/2):** W3+4 için tam panel / doğrudan isabet / backfill tablosu çıkarıldı. Sayımlar: StaticDOpt k=2'de 2/146, k≥3'te 0/146; StaticDiversity k=2'de 133/146, k=3'te 131/146, k≥5'te 0/146. Tam panelli hasta < 5 olan hücreler `hesaplanamadı`. JSON'da artık sessiz `{}` yok.
  - **Keşifsel ortak havuz** (W3+4 test hastalarının ≥%50'sinde ölçülen ilaçlar; arşivde 25 ile 101 gözlem arasındaki doğal boşluk): 90 prob kaldı. Çıkarılanlar: `ABT-737`, `JNJ-28312141`, `Ponatinib (AP24534)`, `XAV-939`. StaticDOpt'ta tam panel oranı 109–136/146'ya çıktı.
  - Bu havuzda StaticDOpt Random50'yi k=2/3/5'te geçiyor: `−2.10 / −2.08 / −1.94`, hepsinde GA < 0. k=8/12'de fark anlamlı değil. AdaptiveOED k=5/8/12'de Random50'yi geçiyor. Keşifsel; hüküm Sol'un.
- **11 (OPUS-007/008):** v2 JSON ve raporlarına eklenenler: `obs_rmse` (havuzlanmış RMSE), `metric_notes` (RMSE tanımı, bootstrap'in kat yeniden-eğitim varyansını kapsamadığı, Sprint 2 sızıntı GA'sının ilaç düzeyinde olduğu) ve tüm bootstrap kayıtlarında `p_two_sided`.
- **12:** `tests/test_opus_regression.py` eklendi. Audit klasörü (`audits/opus-2026-11-21/`) arşiv olarak korundu.

#### §3.6 kabul kriterlerinden sapmalar (açık beyan)
- **04:** Kuyruktaki hedef `np.array_equal` (bit düzeyinde eşitlik) idi. Hasta MSE'sinde toplama sırası değiştiği için elde edilen eşitlik ≤4.5e-13 mutlak (≈2e-16 bağıl). Sıralama metrikleri ve özet değerler birebir aynı. Uçtan uca v1↔v2 farkı ≤1.4e-14.
- **08:** Kuyruk "3 k × 4 metrik, Recall@3 dahil" diyordu. Sol §3.2/3 ve Neon §4.2 hizalaması gereği **9 test** uygulandı ve **Recall@3 aileye dahil değil**.
- **09:** Kuyruk "derangement" diyordu. Sol'un kuralına göre ofset-sabit standardize permütasyon uygulandı ve yalnız **özdeş permütasyon** reddediliyor (kısmi sabit noktalara izin var). v1 `perm_within` tanımı protokol gereği değiştirilmedi.
- **Yeniden analiz testi** (`test_reanalysis_reads_vector_archive_and_labels_symmetrically`) modülle aynı adımda yazıldı. Altındaki Holm kuralı ve etiket mantığı önceden RED ile test edilmişti.

#### Bulgu durum güncellemesi
| Bulgu | Önceki | Şimdi |
|---|---|---|
| OPUS-001 | MAJOR · OPEN / NEEDS_EVIDENCE | **VERIFIED → kanıt üretildi**: simetrik Holm'da `INSUFFICIENT_EVIDENCE`. Etiketin Sprint 4'e taşınmaması hükmü Sol'un. |
| OPUS-002 | MAJOR · VERIFIED | **Kanıt GA ile tamamlandı** (offset-only + 20 çekiliş). Nicel yorum düzeltmesi Sol'un hükmüne bağlı. |
| OPUS-003 | MAJOR · VERIFIED | **Raporlama düzeltildi** (`hesaplanamadı` + keşifsel ortak havuz). OOD politika tanımı hükmü Sol'un. |
| OPUS-004, 006, 007, 009, 010 | MINOR · OPEN | **CLOSED** (kod düzeltildi, testli) |
| OPUS-008 | MINOR · OPEN | **CLOSED (not olarak)**: Sprint 2 kodu protokol gereği değiştirilmedi; `metric_notes` içinde belgelendi. |
| OPUS-005 | MINOR · OPEN | **OPEN**: Sprint 4 protokol kararı, Sol ve kullanıcıya ait. |
| OPUS-015, 016 | INFO | **CLOSED**: payda Neon tarafından netleştirildi; peak RSS ölçüldü. |

**Güncel QA kararı: `QA_PASS (teknik)`.** Açık CRITICAL veya MAJOR **teknik** sorun kalmadı. Üç MAJOR bulgunun kod ve kanıt tarafı tamam; kalan kısımları bilimsel yorum ve Sol'un hükmüne bağlı. `OPUS-005` Sprint 4 protokol girdisidir. Bilimsel Kapı 2 hükmü Sol'a, nihai onay kullanıcıya aittir.

**Sol'a devir (inceleme için):**
1. `sprint-03-v2-reanalysis.md` — 08 sonucu.
2. `sprint-03-v2-panel-selection.md` §v2.1–v2.3 — 09 ve 10 sonuçları.
3. Ortak havuz eşiği (%50) arşiv mevcudiyetinden seçildi. Plaka tasarımı bilgisi varsa onunla değiştirilmeli.
4. Kod, Sol'un `M3_offset_only` tanımını ilaç ölçeği olmadan (eşit ağırlıkla) uyguluyor; "varsa ilaç ölçekleri" seçeneği kullanılmadı.

**Neon'a:** Adım 7a için `sprint03_v2_*` çıktıları ve 50 testlik paket hazır.

### 3.8 Sprint 4 Bağımsız Teknik Denetimi — `[Claude Opus 5.5]` · 2026-11-22

**Kapsam:** `eval/uncertainty.py`, `eval/splits.py` (`make_sprint04_splits`), `experiments/run_sprint04.py`, `experiments/sprint04_report.py`, `tests/test_sprint04.py`, `sprint04_*` çıktıları. Sol §8 (`KOŞULLU KABUL`, SOL-09..12) okunduktan sonra bağımsız yürütüldü. **Üretim kodu, protokol ve çıktılar değiştirilmedi.** İncelenen durum: `audits/opus-2026-11-22-sprint04/reviewed_manifest.sha256` (özet `9f32cf3c11d8…`). Denetim betikleri aynı dizinde.

#### Doğrulama kanıtı
| Kontrol | Sonuç |
|---|---|
| `PYTHONPATH=. pytest -q` | 58 passed. 7 uyarının hepsi `metrics.py` boş-dilim yolundan geliyor (OPUS-022). |
| Bağımsız yeniden koşu (`-W error::RuntimeWarning`, scratch dizinine) | 49.3 s, exit 0. 21.803 JSON yaprağı ≤1e-12, `.npz` 530/530 dizi, manifest birebir. `.md`'de yalnız süre satırı farklı. |
| Adversaryal testler (`opus_s04_adversarial_checks.py`, 10 test) | 8 geçti, 2 `xfail(strict)`: SOL-09 ve SOL-11 bulgularını belgeliyor. |
| Gerçek veri probu (`realdata_probe_s04.py` → `.json`) | `m3_post_scale` eşik kaymasının nedeni ölçüldü; SOL-09 yeniden üretildi. |

**Doğrulanan (INFO · VERIFIED, OPUS-024):**
- Hasta kümeleri her iki rejimde ayrık.
- Kalibrasyon ve test değerlendirme maskeleri simetrik (`min_targets=5`, `min_probes=8`; kilitlide hedef-uygun ∧ tam panel).
- Konformal sıra `r=⌈(n+1)(1−α)⌉` tam sayı aritmetiğiyle eşleşiyor (n≤5000, kayan nokta kayması yok).
- Monte Carlo'da (400 tekrar, eksik hedefli, t₃ gürültü) eşzamanlı kapsam [1−α, 1−α+1/(n+1)] aralığında.
- `m3_conditional_posterior_scale`, yoğun ters matris formülüyle ve M3 tahmincisinin kullandığı `C_ss+λI` / `cov_k` / `λ_k` ile tutarlı.
- AURC: sabit skorda rastgele referansa eşit, eşleştirilmiş fark tam 0.
- PCA skoru eğitim satır sırasından bağımsız.
- Gizli test etiketi M3 tahminine girmiyor (Sprint 3 OPUS-012 yolu değişmedi).

#### Bulgular
| ID | Ciddiyet · Durum | Dosya:satır | Gözlenen → Beklenen | Etki | Önerilen düzeltme | İlgili |
|---|---|---|---|---|---|---|
| **OPUS-017** | MAJOR · VERIFIED | [run_sprint04.py:541-542](../../src/oncotwin/experiments/run_sprint04.py#L541-L542) | Kalibrasyonda `seed+1`, testte `seed+2` MC gürültü bankası kullanılıyor. Aynı test hastaları yalnız tohum değişince `AdaptiveOED_k5` arşiv panelinde **ID 25/91, W3+4 25/148** hastada farklı panel alıyor. Beklenen: aynı omik ve uygunluk maskesi için aynı panel. | Adaptif kolların konformal satırlarına (arşiv ve kilitli) split-conformal güvencesi atfedilemez. Statik ve eşit ağırlıklı kollar etkilenmiyor. | Ağırlıklar aynı tohumla hasta sırasından bağımsız (adversaryal test #5). Bu yüzden fold başına **tek kilitli tohum** yeterli. Regresyon testi ve `sprint04_v2_*` koşusu gerekli. | SOL-09 |
| **OPUS-018** | MINOR · VERIFIED (neden ölçüldü) | [run_sprint04.py:331-412](../../src/oncotwin/experiments/run_sprint04.py#L331-L412) | `m3_post_scale` kilitli eşiği testte ID %100, W3+4 %98.6 kabul ediyor. **Neden 1, ölçek kayması:** iç fitlerde M1 daha az hastayla eğitildiği için OOF artık kovaryansı büyüyor. Öncül RMS SD iç katlarda 51.6–53.7, nihai fitte 51.9. `StaticDOpt_k5` medyanı iç-OOF'ta 47.49, nihai fitte train_tune üzerinde 45.90, testte 45.89. Test maksimumu 46.58, τ=48.24'ün altında kalıyor. **Neden 2, ayrık skor:** skor yalnız panelin fonksiyonu (aynı paneli alan hastalar aynı skoru alıyor). Mod kütlesi %29–91; %80 hedefi ancak sıçramalarla tutulabiliyor. Etiketsiz alternatif (nihai fit, train_tune skorlarının %80'liği) ID'de %92.3, W3+4'te %43.9 kabul veriyor. Bu yalnız tanı amaçlı; testten eşik **seçilmemeli**. | `m3_post_scale` hasta düzeyinde OOD skoru değil, panel bulunabilirliği göstergesi. Raporlanan eşik satırı bilgisiz. Neon'un "örneklem boyutu/shrinkage" açıklaması yön olarak doğrulandı. | v2'de etiketli duyarlılık analizi olarak: τ'yu nihai fit ölçeğinde, train_tune hastalarının yalnız mevcudiyet maskesiyle ve etiketsiz kilitlemek, gerçekleşebilir bağ noktasını raporlamak. Bu **protokol eki** gerektiriyor (Sol/kullanıcı). | Sol §8.1, OPUS-005 |
| **OPUS-019** | MAJOR · VERIFIED | [run_sprint04.py:594-620](../../src/oncotwin/experiments/run_sprint04.py#L594-L620), [uncertainty.py:304-305](../../src/oncotwin/eval/uncertainty.py#L304-L305) | Kilitli ortak tavan, kolların **farklı** tam-panel hastaları üzerinde hesaplanıyor. Ortak küme boyutları: ID k5 üç politikada 17 (tek başına 19/60/58), W3+4 k5 için 1, W3+4 k8 Adaptif∩Statik 0. Ayrıca tavan bir bağ grubunun içinden geçince kesirli kabul varsayılıyor; gerçekleşebilir bir eşik noktası değil (adversaryal test #8). | "Ortak-Cap AURC" sütunu politika üstünlüğü için kullanılamaz. | Karşılaştırmalar ortak hasta maskesinde, eşleştirilmiş `n` ile yapılmalı. n<5 ise yalnız bulunabilirlik gösterilmeli. Tavan bağ grubuna düşüyorsa "kesirli varsayım" etiketi eklenmeli veya yalnız gerçek noktalar raporlanmalı. | SOL-10 |
| **OPUS-020** | MINOR · VERIFIED | [test_sprint04.py:209-308](../../tests/test_sprint04.py#L209-L308) | Sözleşme testleri zayıf. **(b)** Yalnız `StaticDOpt_k2` skorlarını ve τ'yu karşılaştırıyor; seçilen prob kimliklerini ve tahmin matrisini hiçbir politikada sınamıyor. **(d)** Yalnız statik panelin ilk örneğine bakıyor; bu tanım gereği sabit olduğundan test önemsiz geçiyor. Kilitli `AdaptiveOED` sınanmıyor. **pytest.raises** bloğu `_fit_sprint04_fold` çağrısını da sarıyor; fit başka bir nedenle `ValueError` verse test yanlışlıkla geçer. **(f)** Testte `filterwarnings("error::RuntimeWarning")` yok; sözleşme yalnız elle verilen bayrakla sağlanıyor. | Gelecekteki bir sızıntı regresyonu yakalanmayabilir. | Tüm politikalarda prob kimliklerini ve `Y_pred`'i doğrudan karşılaştıran test eklenmeli; kilitli adaptif için maske-değişmezlik testi yazılmalı; raises bloğu daraltılmalı; dosya düzeyinde `pytestmark` uyarı filtresi konmalı. | Sol §8.1 |
| **OPUS-021** | MINOR · VERIFIED | [uncertainty.py:212,216-217](../../src/oncotwin/eval/uncertainty.py#L212) | `q=∞` iken kapsam ve birinciyi içerme oranı 1.0 yayımlanıyor (`xfail` test #3). Arşivlenmiş kollarda tetiklenmedi. | Küçük kalibrasyon kümesinde yanıltıcı %100 görünebilir. | `finite_interval_published=False` ise bu alanlar `hesaplanamadı`/NaN olmalı. | SOL-11 |
| **OPUS-022** | MINOR · VERIFIED (Opus'un kendi borcu) | [metrics.py:302,319-320](../../src/oncotwin/eval/metrics.py#L302) | Uygun hasta yoksa `evaluate_masked_predictions` "Mean of empty slice" uyarısı veriyor (paketteki 7 uyarı). Sprint 4 bunu `_safe_eval_masked` sarmalayıcısıyla atlıyor. | Protokol §6(f) ruhuna aykırı. Kök neden ortak fonksiyonda. | Ortak fonksiyonda boş grup için açık NaN ve `n=0` döndürülmeli; ardından sarmalayıcı sadeleşir. | DEV-OPUS-04 |
| **OPUS-023** | INFO · OPEN | `run_sprint04.py` | 748 satır, 800 uyarı eşiğine yakın. `_require_full_budget` başka bir deney modülünden private olarak import ediliyor (deneyler arası bağımlılık). `_predict_arm:271`'deki `if int(g) in g2l` filtresi ölü (eksik anahtar daha önce `KeyError` verirdi). `M1_k0` `m3_post_scale` satırı sabit, bilgisiz bir skor ama çekinme skoru gibi raporlanıyor. | Bakım ve okuma kolaylığı | Yardımcı `panel_selection`'a taşınmalı; ölü filtre silinmeli; M1 satırı "sabit skor" olarak etiketlenmeli. | — |

**Sol'un SOL-12 (`M0(0)` eksik) maddesine katılıyorum.** Ana M1 kıyasını etkilemiyor.

#### QA kararı: `QA_CONDITIONAL` (Sprint 4)
- **Kullanılabilir (teknik olarak sağlam ve deterministik):** ID ve W3+4 arşiv senaryosunda `StaticDOpt`, `EqualWeightVarRed` ve `M1_k0` için tahmin, konformal kapsam, kısa liste ve `omics_pca_dist` eşiği/AURC.
- **Kullanılamaz:** Adaptif kolların konformal satırları (OPUS-017), kilitli "Ortak-Cap AURC" kıyasları (OPUS-019) ve `m3_post_scale` kilitli eşik satırı (OPUS-018, yalnız betimleyici).
- Sol'un hükmüyle uyumlu olarak Sprint 5 **veri ve protokol hazırlığı** başlayabilir. Model kıyasından önce OPUS-017 ve OPUS-019 kapanmalı. Nihai karar kullanıcıya aittir.

#### Opus Geliştirme Kuyruğu — Sprint 4 (yalnız kullanıcı talimatıyla)
| ID | İş | Kapattığı | Onay |
|---|---|---|---|
| DEV-OPUS-13 | Fold başına tek kilitli contender tohumu; aynı profil için aynı paneli sınayan regresyon testi; adaptif kollar için `sprint04_v2_*` koşusu | OPUS-017 / SOL-09 | Kullanıcı |
| DEV-OPUS-14 | Kilitli senaryoda ortak maske üzerinde eşleştirilmiş kıyas, `n_paired` alanı ve n<5 için `hesaplanamadı`; bağ grubuna düşen tavan için etiket | OPUS-019 / SOL-10 | Kullanıcı |
| DEV-OPUS-15 | `q=∞` iken kapsam ve kısa liste alanlarını `hesaplanamadı` yapmak | OPUS-021 / SOL-11 | Kullanıcı |
| DEV-OPUS-16 | Aynı maskede `M0(0)` satırını eklemek ve protokol sapmasını kaydetmek | SOL-12 | Kullanıcı |
| DEV-OPUS-17 | Sözleşme testlerini güçlendirmek: tüm politikalarda prob kimliği ve `Y_pred`; kilitli adaptif maske-değişmezliği; raises bloğunu daraltmak; `pytestmark` uyarı filtresi | OPUS-020 | Kullanıcı |
| DEV-OPUS-18 | `metrics.py` boş-dilim kök düzeltmesi; ardından `_safe_eval_masked`'ı sadeleştirmek | OPUS-022 | Kullanıcı |
| DEV-OPUS-19 | `m3_post_scale` için nihai fit ölçeğinde etiketsiz τ, etiketli v2 duyarlılık analizi olarak | OPUS-018 | **Sol + kullanıcı (protokol eki)** |
| DEV-OPUS-20 | Hijyen: yardımcıyı taşımak, ölü filtreyi silmek, M1 "sabit skor" etiketi | OPUS-023 | Kullanıcı |

**Not:** 13–16, Sol §8.2 düzeltme kuyruğuyla aynı işler. Kimin uygulayacağı (Opus veya Neon) kullanıcının kararı; çift uygulama yapılmamalı.

#### Kullanıcı kararı: iş bölümü — 2026-11-22
| İş | Sorumlu | Durum |
|---|---|---|
| DEV-OPUS-13–16 (SOL-09..12) | **Neon** | Bekliyor |
| DEV-OPUS-17, 18, 20 | **Opus** | 17 ve 18'in çekirdeği ✅. Kalan kısımlar Neon'dan sonra yapılacak (aşağıda). |
| DEV-OPUS-19 | **Sol → kullanıcı onayı → Neon** | Sol'un protokol ekini bekliyor |
| Nihai teknik QA | **Opus** | Neon'dan sonra |
| Nihai bilimsel değerlendirme | **Sol** | Opus QA'dan sonra |

**Sıralama gerekçesi:** Proje git deposu değil. `run_sprint04.py` ve `test_sprint04.py`'yi Neon ile aynı anda düzenlemek değişikliklerin üzerine yazılması riskini taşır. Bu yüzden Opus şimdilik yalnız Neon'un 13–16'da dokunmayacağı dosyalarda çalıştı.

**Opus — tamamlanan (2026-11-22):**
- **18, çekirdek (OPUS-022).** [metrics.py](../../src/oncotwin/eval/metrics.py) `evaluate_masked_predictions` içinde yalnız "Mean of empty slice" uyarısı bilinçli NaN olarak susturuldu. `average_metric_nodes` içindeki geniş `simplefilter` aynı dar filtreye çekildi.
  - RED→GREEN testi: `test_evaluator_empty_ranking_slices_return_nan_without_warning`.
  - Sayısal etki sıfır: Sprint 4 yeniden koşusu (48.9 s) üretim JSON'u ve 530/530 `.npz` dizisiyle birebir aynı.
- **17, çekirdek (OPUS-020).** Yeni dosya [tests/test_sprint04_contracts.py](../../tests/test_sprint04_contracts.py):
  - **(b)** Üç politikada, gizli test hedefleri permüte edilip ±500 gürültü eklendiğinde prob kimlikleri ve `Y_pred` bit düzeyinde aynı kalıyor.
  - **(d)** Üç politikada, adaptif dahil, kilitli panel test maskesinden bağımsız ve geri doldurma yok.
  - **SOL-09 kabul testi** `test_selector_is_same_fixed_function_for_calibration_and_test`: kalibrasyon hastalarının omik ve mevcudiyet ikizleri test kümesine konuyor; paneller ve contender ağırlıkları aynı olmalı. Şu an `xfail(strict)`. Tohum birleştirme simülasyonunda geçtiği doğrulandı.
- **(f)** Kökte `pytest.ini` ile `error::RuntimeWarning` artık varsayılan; `test_sprint04.py` de kapsanıyor.
- **Paket:** `PYTHONPATH=. pytest -q` → **65 passed, 1 xfailed, 0 uyarı** (önce 58 passed + 7 uyarı).

**Neon için (13–16):**
- **13.** Kalibrasyon ve testte tek contender tohumu kullanılmalı. Kabul: `test_selector_is_same_fixed_function_for_calibration_and_test` XPASS olmalı; ardından `xfail` işareti silinmeli.
- **14.** Ortak maske ve `n_paired` eklenmeli. W3+4 k5 (n=1) ve k8 (n=0) için `hesaplanamadı` yazılmalı.
- **15.** Kabul: `audits/opus-2026-11-22-sprint04/opus_s04_adversarial_checks.py::test_infinite_q_does_not_publish_coverage` XPASS olmalı.
- **16.** `M0(0)` satırı eklenmeli.
- Sonuç değiştiren koşular `sprint04_v2_*` dosyalarına yazılmalı; v1 çıktıları ve `PROTOCOL_SPRINT04.md` korunmalı.
- `pytest.ini` artık RuntimeWarning'i hata sayıyor; yeni kod boş dilimleri açıkça ele almalı.

**Opus, Neon'dan sonra:**
- **18 kalan kısım.** `run_sprint04._safe_eval_masked` sadeleştirilecek. Sarmalayıcı hâlâ `n_obs=0` için gerekli, ancak uyarı koruması artık gereksiz.
- **20.** `_require_full_budget` `panel_selection`'a taşınacak, `_predict_arm` içindeki ölü `g2l` filtresi silinecek, M1 satırı "sabit skor" olarak etiketlenecek.
- **17 kalan kısım.** `test_sprint04.py`'deki `pytest.raises` bloğu yalnız `_select_archive_panels` çağrısını saracak biçimde daraltılacak.
- **Ardından nihai teknik QA:**
  - v1↔v2 farkı yalnız adaptif kollarda ve kilitli kıyaslarda beklenir; statik, eşit ağırlıklı ve M1 kolları bit düzeyinde aynı kalmalı.
  - Adversaryal testler yeniden koşulacak.
  - Bağımsız yeniden üretim yapılacak.

### 3.9 Sol kaynak kontrolü ve `PROTOCOL_SPRINT04_ADDENDUM_19` teknik incelemesi — `[Claude Opus 5.5]` · 2026-11-22

#### Ek 19 — teknik inceleme (Opus bulguları, Sol'un kararına)
- ✅ **Doğrulananlar:**
  - §1'deki sayılar Opus ölçümüyle eşleşiyor: 47.49 / 45.89 / 46.58 / 48.24 ve 91/91 / 146/148.
  - Sıra formülü `1+ceil(0.80(n−1))`, NumPy `method="higher"` ile aynı.
  - `[1,1,2,2,2]` örneği doğru (τ=2, kabul 5/5).
  - Regresyon sözleşmesi (a)–(c) test edilebilir.
- **A1 · MINOR.** Adaptif kolun train_tune referans panelleri nihai fitin **örneklem-içi** M1 tahminlerinden üretilirse contender ağırlıkları, kalibrasyon ve testteki örneklem-dışı tahminlerden sistematik olarak farklı olur. Bunun nedeni ridge fitinin kendi eğitim hastalarına aşırı uymasıdır. Sonuçta referans panel dağılımı kayar. Statik ve eşit ağırlıklı kollar etkilenmez. Öneri: §2.3'e "adaptif referans için nihai fitin M1 OOF tahminleri (`fit_and_oof` çıktısı) kullanılır" ifadesi eklensin, ya da örneklem-içi kullanım bilinçli bir seçim olarak yazılsın.
- **A2 · INFO.** `locked_train_catalog` altında `StaticDOpt` ve `EqualWeightVarRed` panelleri her hasta için aynı. Eşiğe yalnız tam panelli hastalar girdiğinden skor bu hücrelerde her zaman sabit; §2.5 gereği "sabit skor — ayırt edici değil" çıkacak. Anlamlı eşik yalnız kilitli `AdaptiveOED` hücrelerinde oluşur. Bu durum raporda önceden beklenen bir sonuç olarak belirtilmeli.
- **A3 · INFO (şeffaflık).** Opus'un tanı probu, bu kuralın `archive/StaticDOpt_k5` sonucunu ek yazılmadan önce gösterdi: test kabulü ID'de %92.3, W3+4'te %43.9 (`audits/opus-2026-11-22-sprint04/realdata_probe_s04.json`). Ekin "sonuçlar daha önce görüldü" notu bu ön izlemeyi açıkça anmalı. v2 sonucu bu hücre için yeni bir bulgu gibi sunulmamalı.
- **A4 · süreç.** Ekteki "Kullanıcı 22 Kasım 2026'da onayladı" ifadesini Opus doğrulayamaz; teyit kullanıcıya ait.

#### Kaynak kontrolü (16 bağlantı; çevrim içi okundu, 2026-11-22)
| Kaynak (Sol'daki yeri) | Sonuç | Not |
|---|---|---|
| arXiv 2306.06342, Lee, Barber, Willett — hiyerarşik veride dağılımdan bağımsız çıkarım (§6, §8.1) | ✅ | Makale "gruplar / tekrarlı ölçüm" diyor; "hasta-kümeli" ifadesi uygulamaya yönelik bir yorum. |
| NeurIPS 2019, Tibshirani ve ark. — kovaryat kaymasında konformal (§6) | ✅ | — |
| **JRSS-B 87(4):1100, Gibbs, Cherian, Candès — "Conformal prediction with conditional guarantees" (§3.2 madde 1)** | ❌ **Uyuşmuyor** | Makale koşullu kapsam üzerine; "split-conformal'da eğitim/ayar kalibrasyondan ayrı olmalı" iddiasını desteklemiyor. Öneri: Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018, *JASA*) "Distribution-free predictive inference for regression" ile değiştirilsin. |
| NeurIPS 2024, Traub ve ark. — seçici sınıflandırma değerlendirmesindeki hatalar (§6, §8.1, ek §3) | ⚠️ Kısmi | Makale AURC'nin eksiklerini tartışıp **AUGRC**'yi öneriyor. Sprint 4 AURC kullanıyor ve bu kaynak o metriğin eleştirisi. Öneri: v2'de AUGRC'nin ek duyarlılık metriği olarak raporlanması. |
| Holm 1979 PDF (ime.usp.br) (§3.2 madde 3) | ⚠️ Kırık | 301 yönlendirmesi sonrası 403. Atıf doğru, bağlantı değil. Öneri: *Scand J Stat* 6(2):65–70, kalıcı bir DOI veya JSTOR bağlantısıyla değiştirilsin. |
| Shaikh PDF — Mogstad, Romano, Shaikh, Wilhelm, *RESTUD* 2024, "Inference for Ranks" (§6) | ✅ | PDF'in ilk sayfası Opus tarafından okundu. |
| Beat AML 2.0 portalı, Bottomly ve ark. *Cancer Cell* 2022 (§6) | ✅ | Farklılaşma durumu–duyarlılık ilişkisi sayfada var. |
| DrEval, Bernett ve ark. *Nat Commun* 17:4238 (2026) (§6) | ⚠️ Kısmi | Özet "aşırı iyimser performans tahminleri" ve "ortalama ilaç/hücre etkisi" taban modelini destekliyor. "Havuzlanmış skor şişmesi" ve "ekranlar arası aktarım zor" ifadeleri özette birebir yok. Ayrıca çalışma hücre hattı verisi üzerine; Sol'un ifadesi biraz yumuşatılmalı. |
| Shokrzadeh & Shokrzadeh, bioRxiv 2026.06.07.730728v1 (§6, Protokol §4.2) | ✅ | 318 hasta, 122 bileşik, Elastic Net + split-conformal ve %90.7 makro kapsam birebir doğrulandı. Notlar: 318 sayısı eşleşmiş alt küme (562 ifade / 528 ilaç verisinden). Bölme önce 80/20, ardından 60/20 yapılıyor. Çalışma hakemli değil. |
| Lee ve ark. *Nat Commun* 9:42 (2018) (§6) | ✅ | 30 AML hastası, 160 ilaç. |
| Pino ve ark. *Cell Rep Med* 2024, PMC10829797 (§6) | ✅ | Beat AML içinden 210 hastalık proteogenomik alt küme. |
| Liepe ve ark. *PLoS Comput Biol* 2013 (§6) | ✅ | Karşılıklı bilgi temelli deney tasarımı. |
| Zhong ve ark. SIAM, doi 10.1137/24M1649344 (§6) | ❓ Doğrulanamadı | SIAM 403, Crossref 429. Kullanmadan önce yeniden kontrol edilmeli. |
| Malani ve ark. *Cancer Discov* 2022, PMID 34789538 (§6, §7) | ✅ | NCBI özeti: 186 hastadan 252 örnek, *ex vivo* ilaç yanıtı ve çoklu omik. Sol §7'deki 252/186 sayıları eşleşiyor. |
| Zenodo 7370747 (§6, §7) | ✅ | MD5 `3db5280e…` birebir. File_3.1: 515 bileşik × 181 örnek (164 AML + 17 sağlıklı). Düzeltme: File_7 RNA tablosu **167 örnek** (163 AML + 4 sağlıklı CD34+); Sol'un yazdığı "163" yalnız AML sayısı. File_x'ler ayrı kayıt dosyası değil, tek ZIP'in içinde. |
| Sol §8.1 Wilson aralığı (79/91) | ✅ | Opus hesabı: %78.35–92.29. |

**Özet:**
- 16 kaynağın 11'i doğrulandı.
- 1 kaynak iddiayla uyuşmuyor (JRSS-B).
- 3 kaynak kısmen destekliyor ya da düzeltme gerektiriyor: NeurIPS 2024 (AUGRC), DrEval ve Zenodo'daki RNA sayısı.
- 1 bağlantı kırık (Holm).
- 1 kaynak erişim engeli nedeniyle doğrulanamadı (Zhong).

Hiçbiri Sprint 4 sonuçlarını değiştirmiyor. JRSS-B atfı ve AUGRC notu Sol'un düzeltmesine bırakılıyor.

### 3.10 Sprint 4 v2 kapanışı, DEV-OPUS-17/18/20 ve Sprint 5 teknik QA — `[Claude Opus 5.5]` · 2026-11-25

**Kapsam:** Neon §4.3 / Sol §9.4.3 devri. İncelenen durum: `audits/opus-2026-11-25-sprint05/reviewed_manifest.sha256` (78 dosya, değişiklik öncesi). Üretim çıktıları, `PROTOCOL*.md` ve diğer ajanların panoları değiştirilmedi.

#### Doğrulama kanıtı
| Kontrol | Sonuç |
|---|---|
| `PYTHONPATH=. pytest -q` (`pytest.ini`: `error::RuntimeWarning`) | Değişiklik öncesi 84 passed; sonrası **85 passed** (+1 yeni test), 0 uyarı. |
| Sprint 4 denetim testleri (`audits/opus-2026-11-22-sprint04/`) | 10 passed. SOL-11 testi XPASS olduğu için işareti kaldırıldı. SOL-09 testi artık fonksiyon özelliğini belgeliyor (farklı tohum → farklı panel), bu da koşucunun neden tek tohum kilitlediğini gösteriyor. |
| Sprint 4 v1↔v2 `.npz` farkı | 530 ortak dizinin **127'si değişti, hepsi `AdaptiveOED`** (61 arşiv, 66 kilitli). StaticDOpt, EqualWeightVarRed ve M1 bit düzeyinde aynı. Yeni diziler yalnız `M0_k0` (52). |
| Opus refaktörü sonrası Sprint 4 v2 yeniden koşusu (scratch, `-W error`) | 54.9 s, peak RSS 389 MB. JSON, 582/582 `.npz` dizisi ve manifest **birebir aynı**. `.md`'de yalnız yeni "(sabit skor)" etiketi farklı; üretimdeki v2 raporu tarihsel olarak korundu. |
| Sprint 5 bağımsız yeniden koşusu (Opus refaktörü sonrası, scratch, `-W error`) | 251.0 s, peak RSS 599 MB, Kapı 3 `NO-GO`. **6.912 JSON yaprağı ve 1.014/1.014 `.npz` dizisi birebir aynı**; `.md` süre satırı dışında aynı. Manifestle karşılaştırıldığında değişen 10 dosyanın hepsi Opus'un kasıtlı kaynak, test ve pano düzenlemeleri; `data/processed`, `PROTOCOL*.md` ve diğer panolar dokunulmamış. |
| Ek 19 ana hücre | `τ_final` ID 45.9455, W3+4 41.4611. Kabul 84/91 ve 65/148. Opus'un 22 Kasım probuyla birebir aynı (A3 ön izlemesi). A1, OOF birincil ve örneklem-içi duyarlılık olarak uygulanmış. (a)–(c) sözleşme testleri geçiyor. |

#### DEV-OPUS-17/18/20 — kapanış (TDD)
- **17:** `test_sprint04.py`'deki `pytest.raises` bloğu yalnız `_select_archive_panels` çağrısını sarıyor ve `match="karşılanamadı"` ile daraltıldı. Fit artık bloğun dışında.
- **18:** `sprint04_eval._safe_eval_masked` docstring'i gerçek görevine göre düzeltildi. Sarmalayıcı hâlâ gerekli (`n_obs=0` → `hesaplanamadı` ve vektörleri tam satır boyuna taşıma); uyarı koruması kökte (`metrics.py`) çözülmüş durumda.
- **20a:** `_require_full_budget` 4 koşucuda kullanılıyordu (Sprint 3, Sprint 4, Sprint 5 M4 ve FIMM). Rule of Three gereği `models/panel_selection.require_full_budget` olarak taşındı ve tüm çağıranlar güncellendi.
- **20b:** `_predict_arm` içindeki ölü `if int(g) in g2l` filtresi silindi. Eksik anahtar artık sessizce atlanmıyor, `KeyError` veriyor.
- **20c:** Sprint 4 raporunda k=0 kollarının `m3_post_scale` satırları "(sabit skor)" olarak etiketleniyor. Test: `test_sprint04_report_marks_k0_m3_post_scale_as_constant_score`.

#### Sprint 5 bulguları
| ID | Ciddiyet · Durum | Konu | Kanıt / sonuç |
|---|---|---|---|
| **OPUS-025** | INFO · VERIFIED | Sprint 4 v2 ve Ek 19 kapanışı | SOL-09..12 kodda yerinde, v1↔v2 farkı yalnız beklenen kollarda (yukarıda). **OPUS-017, 019, 020, 021, 022 ve 023 kapandı.** OPUS-018, Ek 19 ile keşifsel olarak ele alındı. |
| **OPUS-026** | INFO · VERIFIED | A kulvarı, M4 ve Kapı 3 | `rho=0` ≡ M3 testi var. Anotasyonsuz ilaçlarda Gram satırı sıfır, diyagonal 1, toplam PSD. Derece korumalı takas her iki marjinali `assert` ile koruyor. Toplu çözücü, Sprint 4'teki tekil `C_ss+λI` formülüyle aynı. **NO-GO dejenere değil:** iç-CV, ID CV katlarının 13/15'inde her k için `rho>0` seçti, yani M4 etkin. Bağımsız bootstrap (farklı tohum, 2000 tekrar): M4−M3 **+0.011 [−0.011, +0.034]**, M4−perm **+0.017 [−0.006, +0.040]**, n=427. Neon ve Sol ile aynı. |
| **OPUS-027** | INFO · VERIFIED | B kulvarı, sıra-konformal ve AUGRC | Kapsama olayları liste kuralına denk: `r_i≤k̂`; göreli sırada `⌈q·n⌉` (tam sayı sıra ile); `gap≤m̂`. Bağlar korumacı (en kötü sıra). `q=∞` → `hesaplanamadı`. AUGRC yamuk kuralında rastgele sıranın beklenen değeri **analitik olarak tam `0.5·ortalama`**. `EqualWidthAUC`, v2 ile iki rejimde fark 0.0 ve NaN deseni aynı. **Sol'un istediği paydalar ham dizilerden doğrulandı:** tek aday doğruluğu ID **5/9** (Wilson %26.7–81.1), W3+4 **19/25** (%56.6–88.5); Rank ID 85/91 (%86.4–96.9). W3+4 kalibrasyonunda bölme 71, arşiv uygunluğu sonrası **70** (`sprint04_v2_results.json` `archive_eval_cal=70`). Protokol §2'deki `n_cal=71` ifadesi düzeltilmeli (SOL-14). |
| **OPUS-028** | MINOR · VERIFIED (SOL-15'i kaynakta teyit eder) | C1 eğitim ve aktarım ölçeği | [run_sprint05_fimm.py:62-92](../../src/oncotwin/experiments/run_sprint05_fimm.py#L62-L92): Beat AML yanıtları hasta başına **tüm** ölçülmüş ortak ilaçlar üzerinden z-skorlanıyor (`_within_patient_zscore_rows`), FIMM probları ise yalnız **k prob** üzerinden. k=2'de popülasyon SD'si nedeniyle değerler daima tam ±1. Bundan önemlisi, k-prob ortalamasına yeniden merkezleme **hasta ofsetini siliyor**; Sprint 3'te (OPUS-002) M3 kazancının yaklaşık yarısı bu ofsetten geliyordu. NDCG@3 ofsetten etkilenmediği için anlamlı çıkıp Regret@1'in çıkmaması bu yapıyla tutarlı. FIMM etiketi hiçbir fite girmiyor; sızıntı yok. Öneri (Neon, Sprint 6): Sol'un istediği Beat AML iç kontrolü yapılmalı; aynı k-prob ölçeklemesi Beat AML test hastalarına uygulanmalı. **⚠ Düzeltme (2026-11-26, §3.11):** "Hasta ofsetini siliyor ve Regret@1 sonucunu açıklıyor" ifadesi geri çekildi. C1 eğitiminde yanıtlar zaten hasta başına tüm-satır z-skoruyla ofsetten arındırılıyor. Gerçek fark k-prob ve tüm-satır merkez/ölçek farkı; ölçülen etki §3.11'de. |
| **OPUS-029** | INFO · VERIFIED | Sıra normalizasyonu bağ kırma | `within_sample_rank_normalize` bağları gen sütun sırasına göre ayırıyor (ordinal). Beat AML'de satır başına değerlerin **%19.5'i bağlı** (en büyük grup 48), FIMM'de yaklaşık 0. Ölçülen etki ortalama sıraya göre ortalama 2e-5, en çok 0.0016 ([0,1] ölçeğinde); **ihmal edilebilir**, yeniden koşu gerekmez. Sonraki sürümde `metrics._rankdata_average` yeniden kullanılmalı. |
| **OPUS-030** | INFO · VERIFIED | C1 ikincil metrikler | M3 k5 − M1: NDCG@3 **+0.032 [+0.007, +0.055]**, Recall@3 **+0.037 [+0.006, +0.067]**, hasta içi Spearman **+0.022 [−0.001, +0.045]** (sıfırı kesiyor), Regret@1 −0.23 [−1.22, +0.77]. **⚠ Düzeltme (2026-11-26, §3.12):** Bu aralık Opus'un denetim bootstrap'ından (tohum 7). Arşivdeki birincil sonuç **[+0.000, +0.045], p=0.048**; alt sınır +0.0004. Spearman farkı sınırda ve tohuma bağlı; "sıfırı kesiyor" ifadesi tek başına doğru değil. İddia "genel sıralama" değil, **üst-3 zenginleşmesi** olarak yazılmalı (Sol §9.3 ile uyumlu). |
| **OPUS-031** | INFO · VERIFIED | C kulvarı veri | Bileşik eşlemesi **bire bir** (102 Beat AML → 102 ayrı FIMM kimliği, belirsiz eşleşme yok). İndeks örnek seçimi yalnız ölçüm sayısını kullanıyor, değerlere bakmıyor. 132→109 gruplama testli. MD5 testi geçiyor. |
| **OPUS-032** | INFO · OPEN (P3) | Hijyen | `rank_conformal.py`, `uncertainty._finite_sample_conformal_quantile` private yardımcısını modül dışından import ediyor. Ad public yapılmalı. Tüm Python dosyaları 800 satırın altında (en büyük `baselines.py` 760). |

#### QA kararı
- **Sprint 4 v2 + Ek 19: `QA_PASS (teknik)`.** Sprint 4 için açık teknik bulgu kalmadı.
- **Sprint 5: `QA_PASS (teknik)`.** Kod protokolde yazılanı yapıyor, sızıntı yok, sayılar ham dizilerden yeniden üretildi. Açık kalan konular bilimsel kapsam ve tasarım konuları: SOL-13..16 ve OPUS-028. Bunlar Sprint 6 Methods yazımında ve bir sonraki küçük deneyde ele alınmalı. Kapı 3 `NO-GO` teknik olarak sağlam.
- Nihai karar kullanıcıya aittir.

### 3.11 Sprint 6 teknik QA — OPUS-028 yeniden üretimi, OPUS-029/032 ve sürüm paketi — `[Claude Opus 5.5]` · 2026-11-26

**Kapsam:** Neon ve Sol devri (Sol §11–§12). İncelenen durum: `audits/opus-2026-11-26-sprint06/reviewed_manifest.sha256`. Opus bu adımda üretim kodu, çıktı, protokol veya Methods raporu değiştirmedi. Değişiklikler yalnız bu panoda ve `audits/opus-2026-11-26-sprint06/` altında.

#### Doğrulama kanıtı
| Kontrol | Sonuç |
|---|---|
| **OPUS-028 bağımsız yeniden üretimi** (`reproduce_opus028.py` → `.json`) | Sol'un betiği değiştirilmeden `-W error` altında koşuldu. Çıktı **arşiv JSON'u ile birebir aynı**. Betikteki tüm `assert`'ler geçti: gizli yanıt değişmezliği (`10000−10y` bozması) 3 kolda prob görünümlerini ve tahminleri bit düzeyinde değiştirmedi; tüm-satır kehanet kolu değişti; k=2'de **91/91** vektör tam ±1. |
| OPUS-028 aralıkları, bağımsız bootstrap (farklı tohum, 4000 tekrar, 20 aralık) | **19/20** aralıkta sıfıra göre aynı sonuç. **Tek istisna sınırda:** k=5 `k_probe − M1` Regret@1 Sol'da −6.91 [−14.88, +0.21], Opus'ta [−14.34, −0.05]. Sonuç tohuma bağlı; ne kazanç ne de sıfır olarak raporlanmalı. |
| OPUS-032 | `finite_sample_conformal_quantile` public. Gövde önceki private sürümle aynı (yalnız docstring eklendi). Eski ad kalmadı, `rank_conformal` yeni adı kullanıyor. Sıra kesinliği testi (n≤5000) ve paket testi geçiyor. **KAPANDI.** |
| OPUS-029 dokümantasyon tercihi | Methods §7'deki satır ölçümle uyumlu (%19.5 bağ, ortalama fark 2e-5, en çok 0.0016). v1 kuralının bit-eşitlik için korunması **teknik olarak doğru tercih**. **KAPANDI (belgelenmiş sınırlılık).** |
| `verify_package --run-tests` (mevcut ortam) | PASS: ham veri (7 Beat AML dosyası ve FIMM MD5), 32/32 manifest, 16/16 değişmez, **88 passed**; yaklaşık 16 s. Başarısızlıkta `SystemExit(1)`. |
| **Temiz sanal ortam** (`python -m venv` → `pip install -r requirements.txt` → `PYTHONPATH` olmadan README komutu) | **PASS, çıkış kodu 0, 88 passed (15.6 s).** Kurulan paketler: numpy 2.5.5, pandas 3.0.3, openpyxl 3.1.5, pytest 9.2.0 ve bunların alt bağımlılıkları. Methods §8'deki **[TEST EDİLMEDİ] riski kapandı.** Sınır: aynı makine, Linux ve Python 3.14.10; farklı işletim sistemi veya Python sürümü (`requires-python>=3.11`) denenmedi. |
| Değişmezlerin içeriği | 16 değişmez, Opus'un bağımsız doğruladığı sayılarla bire bir aynı: 70 (SOL-14), 84/65 (Ek 19), 85/91, 5/9, 19/25, 23/64/109, NO-GO, 582 ve 1014 dizi. |

#### OPUS-028 — düzeltilmiş yorum
Opus'un 25 Kasım'daki "k-prob merkezleme hasta ofsetini siliyor ve Regret@1 sonucunu açıklıyor" ifadesi **fazla güçlüydü ve geri çekildi**. C1 eğitiminde Beat AML yanıtları zaten hasta başına tüm ortak ilaçlar üzerinden z-skorlanıyor, yani ofset eğitimde de yok. Asıl uyuşmazlık k-prob merkez ve ölçeği ile tüm-satır merkez ve ölçeği arasında. Eğitime eşdeğer ölçeğe (tüm-satır kehanet koluna) göre ölçülen etki:
- **k=2:** NDCG@3 **−0.018 [−0.032, −0.005]**. Zarar anlamlı; ±1 kodlaması büyüklük bilgisini siliyor.
- **k=5:** NDCG@3 −0.005 [−0.019, +0.007], Regret@1 +2.2 [−3.5, +8.9]. **Etki saptanmadı.**

Sol'un ve Methods §5'in "tek bir nedene atfedilemez" sonucu doğru. Ana kol k=5 olduğu için C1'in başlık sonucu bu ölçek kuralından anlamlı biçimde etkilenmiş görünmüyor.

#### Paket bulguları
| ID | Ciddiyet · Durum | Konu |
|---|---|---|
| **OPUS-033** | INFO · OPEN | `pip install .` çalışıyor, ancak site-packages'a **`src` adında üst düzey bir paket** kuruyor (PEP 420 ad alanı). Başka bir `src` paketiyle çakışma riski var. README yolu (`requirements.txt` ile repo kökünden `-m src...`) bundan etkilenmiyor. v2'de paket adı `oncotwin` olacak şekilde yapı değiştirilebilir; v1 için README yolunun tek desteklenen yol olarak yazılması yeterli. Opus'un denemesinin oluşturduğu `build/` ve `oncotwin.egg-info/` silindi. |
| **OPUS-034** | INFO · OPEN | `verify_package --write-manifest` mevcut dosyalardan manifesti sessizce yeniden yazıyor. Yanlışlıkla çalıştırılırsa değişmiş bir artefaktı "onaylı" hale getirir. Yalnız yeni sürüm hazırlarken kullanılmalı; README'de bu not düşülmeli. |
| **OPUS-035** | INFO · OPEN | Methods raporunda iki küçük güncelleme gerekiyor (Neon/Sol): (1) §8 ve §9'daki "temiz sanal ortam [TEST EDİLMEDİ] / açık" ifadeleri Opus'un ölçümüne göre "aynı makinede temiz venv ile doğrulandı" olarak güncellenmeli. (2) §5 tablosunda k=5 `k_probe − M1` Regret@1 aralığının tohuma bağlı ve sınırda olduğu not edilmeli. |

#### QA kararı: `QA_PASS (teknik)` — Sprint 6 sürüm paketi
- OPUS-028 deneyi birebir yeniden üretildi.
- OPUS-029 ve OPUS-032 kapandı.
- Tek komutlu doğrulama temiz ortamda geçti.
- Açık bulgular yalnız INFO düzeyinde (OPUS-033..035) ve sürümü engellemiyor.

Teknik açıdan v1 sürümüne engel yok. **Yayın kararı proje sahibine ait.**

### 3.12 Makale sayı kontrolü, temiz klon ve grafik notu — `[Claude Opus 5.5]` · 2026-11-26

**Kapsam:** `main` `e5a830e` (GitHub ile aynı), `docs/manuscript/oncotwin_v1_preprint.md`. Opus depoda kod, çıktı veya makale değiştirmedi. Commit ve push yapılmadı. Denetim betiği: `audits/opus-2026-11-26-manuscript/verify_manuscript_numbers.py`. Çalıştırma: `PYTHONPATH=. python3 audits/opus-2026-11-26-manuscript/verify_manuscript_numbers.py --with-fimm`.

#### 1. Makale ↔ ham çıktı sayı kontrolü
Özet, §2.1–2.2, Tablo 1–4, §3.1–3.4 ve OPUS-028 içindeki **388 sayısal iddia** tek tek kaynağına bağlandı ve makaledeki yazılış hassasiyetinde yuvarlanarak karşılaştırıldı. Kaynaklar: `data/processed/*.json`, `sprint05_vectors.npz`, `audits/sol-sprint06/*.json`. FIMM tekil/çoklu örnek sayıları, mutasyon sütunları ve önek tamlığı ham veriden; Wilson aralıkları sayımlardan yeniden hesaplandı.

**Sonuç: 386 OK, 2 uyuşmazlık.** Ek olarak metinsel bir Methods hatası bulundu.

| # | Önem | Yer | Sorun | Önerilen düzeltme |
|---|---|---|---|---|
| **M-1** | **MAJOR** | §2.1 (satır 43) | Beat AML indeks örneği "en çok gözlenen hedef ilaca sahip örnek, eşitlikte toplam ilaç sayısı ve örnek anahtarı" diye tarif ediliyor. **Kod ve PROTOCOL.md farklı:** `select_index_specimen_positions` sırasıyla toplama zamanı (en erken), hastalık evresi (önce tanı anı), DNA-seq bulunması, örnek tipi (önce kemik iliği) ve örnek anahtarına bakıyor; **ilaç etiketlerine hiç bakmıyor**. Makaledeki tarif FIMM kuralıyla karışmış (FIMM: evre, doku, eşleşen ilaç sayısı). Yazıldığı haliyle okuyucu etikete bağlı bir seçim olduğunu sanabilir. | Cümle PROTOCOL.md §Birincil Analiz Birimi ile değiştirilmeli. FIMM kuralı §2.1'in FIMM maddesinde ayrıca yazılmalı. |
| **M-2** | MINOR | §3.4 (satır 177) | C1 Spearman için "+0.022 [−0.001, +0.045], bağımsız denetimde sıfırı kesiyor" yazıyor. Arşivdeki birincil sonuç **+0.022 [+0.000, +0.045], p=0.048** (alt sınır +0.0004). Aralık tohuma bağlı ve sınırda; makale yalnız sıfırı kesen sürümü veriyor. Opus'un §3.10'daki ifadesi de bu yüzden düzeltildi. | "Birincil: [+0.000, +0.045], p=0.048; denetim bootstrap'ında alt sınır −0.001. Sınırda, sağlam değil" diye yazılmalı. OPUS-028 Regret@1 cümlesiyle aynı biçim kullanılmalı. |
| **M-3** | MINOR | §3.4 (satır 179) | OPUS-028 k=5, k-prob eksi tüm-satır kehanet NDCG@3 alt sınırı **−0.019** yazılmış. Arşiv JSON'u (Sol, 2000 tekrar) **−0.0195 → −0.020** veriyor. −0.019, Opus'un 4000 tekrarlı denetim değeri. Aynı cümlede iki kaynak karışmış. | Arşiv JSON'u esas alınmalı (−0.020). Denetim değeri gerekirse ayrıca etiketlenmeli. |
| M-4 | Yazım | §2.7 (satır 86), §2.2 (satır 53) | (a) "peak RSS 599 MB", temiz venv doğrulamasına atfedilmiş. Bu ölçüm Sprint 5'in tam yeniden koşusuna ait; temiz venv doğrulaması yaklaşık 16 s sürüyor. (b) "±500 eklemek" yazıyor; testler permütasyon ve **SD=500 Gauss gürültüsü** kullanıyor. | İki ifade ayrılmalı ve gürültünün tanımı düzeltilmeli. |
| M-5 | Bilgi | Şekil 1 başlığı | "k = 0..12" yazıyor; veri yalnız k ∈ {0, 1, 2, 3, 5, 8, 12} içeriyor. | Bütçe kümesi açıkça yazılmalı. |
| M-6 | Kaynak (Sol'a) | §1 (satır 27) | Guillen ve ark. 2022 (Nature Cancer, PDX/organoid platformu), "PDX modelleri birincil örneklere kötü aktarılır" iddiasına kaynak gösterilmiş. Makale bu platformun hasta yanıtını yansıttığını raporluyor gibi görünüyor; iddiayı desteklemeyebilir. DOI çözümleniyor, içerik uyumu Sol'un kontrolüne bırakıldı. | Sol doğrulamalı; gerekirse kaynak değiştirilmeli ya da cümle daraltılmalı. |
| M-7 | Bilgi | Kaynakça #14, #17 | #14 (Tibshirani 2019) DOI veya URL içermiyor. #17 (Willoch-Allen) yazar listesi eksik. | Tamamlanmalı. |

Doğrulanan başlıca bloklar: Tablo 1'in 9 satırı (RMSE, NDCG, Regret ve aralıklar), Tablo 2'nin 4 satırı (p değerleri dahil), Tablo 3'ün 7 satırı (eşik, boyut, medyan, oran, tekil oran, ortalama liste Regret), Tablo 4'ün 11 satırı, C1/C2 aralıkları ve p değerleri, AURC/AUGRC, Ek 19 kabul sayıları, kohort paydaları (942/805/108/18/520/479, 252/186, 132/109 = 92+17), doz oranları (11/23, 0.01×–10×, Trametinib 40×, Rapamycin 100×), aile grafı (117/86/395, hedeflerde 15/24) ve FIMM önek tamlığı (109/98/98).

#### 2. Repo, lisans ve temiz klon
| Kontrol | Sonuç |
|---|---|
| Taze klon (`git clone` GitHub → scratch) | `e5a830e`, 104 dosya; yerel `git ls-files` ile **birebir aynı**. |
| Yanlışlıkla eklenmiş dosya | Yok. `pycache`, `.venv`, ham veri, gizli anahtar, `build` veya `egg-info` takipte değil. En büyük dosya 3.5 MB (`sprint03_v2_vectors.npz`). |
| Temiz venv + `--skip-raw` (ham veri yokken) | **PASS**, çıkış 0: manifest 32/32, değişmez 16/16. |
| README adım 1–2: `fetch_beataml`, `fetch_fimm`, tam `--run-tests` | **PASS**, çıkış 0: Beat AML 7 dosya SHA-256, FIMM MD5, 32/32, 16/16, **88 passed** (14.8 s). `git status` temiz. |
| Bağlantılar | Göreli markdown bağlantıları **259/259** sağlam. Makale ve CITATION DOI'leri **9/9** çözümleniyor (doi.org handle API). Dış bağlantıların 26/33'ü 200; geri kalanlar yayıncı bot engeli (403) veya PubMed cookie duvarı (203), bozuk değil. |
| LICENSE / CITATION / NOTICE | PolyForm Noncommercial 1.0.0 metni mevcut. CITATION'da `license` alanı ve ORCID `0009-0003-9804-0640` var. NOTICE telif satırı var. |

| # | Önem | Bulgu |
|---|---|---|
| R-1 | Bilgi | **Zenodo geçici hatası:** `fetch_fimm` üç denemenin ilk ikisinde HTTP 504 aldı ve yeniden deneme olmadığı için tek denemede akışı durduruyor. Öneri: 3 deneme ve artan bekleme (yalnız 5xx için), ya da README'ye "504 alırsanız tekrar çalıştırın" notu. |
| R-2 | Bilgi | README ham veriyi "~330 MB" diyor. Gerçekte indirme yaklaşık 320 MB (Beat AML 281 MB + FIMM zip 39 MB), diskte ise açılmış FIMM ile **410 MB**. |
| R-3 | Bilgi | NOTICE "harici veri setleri burada yeniden dağıtılmıyor" diyor. Ancak `data/processed/` hasta düzeyinde türetilmiş vektörler ve halka açık dbGaP hasta kimlikleri ile örnek anahtarları (bölme manifestleri) içeriyor. Kaynakların CC BY 4.0 lisansı ve verilen atıf nedeniyle izin sorunu yok. Ancak (a) "türetilmiş sonuçlar dağıtılıyor" cümlesi eklenmeli, (b) `data/processed/` için lisans belirtilmeli (şu an ne kod ne docs lisansı kapsıyor). |
| R-4 | Not | `-W error::RuntimeWarning` ile venv Python'u **göreli yoldan** (`../venv/bin/python`) çağrılırsa Python 3.14 `site` uyarısı ölümcül olur. Depo sorunu değil; README komutları etkilenmiyor. |

#### 3. Grafik üretim notu (matplotlib, deterministik, uyarısız)
**Ortak kurallar:**
- `matplotlib` çalışma bağımlılığı değil. Sabitlenmiş sürümle ayrı bir `requirements-figures.txt`'ye konmalı; `requirements.txt` üç pakette kalmalı.
- `matplotlib.use("Agg")` kullanılmalı, `plt.show()` çağrılmamalı. Yerleşim için `constrained_layout=True` kullanılmalı (`tight_layout` uyarı üretebiliyor). Yazı tipi sabitlenmeli (`rcParams["font.family"]="DejaVu Sans"`).
- Belirleyicilik için: `rcParams["svg.hashsalt"]="oncotwin-v1"`. PDF'de `savefig(..., metadata={"CreationDate": None, "Producer": None})`. PNG'de `metadata={"Software": None}`. `SOURCE_DATE_EPOCH` sabitlenmeli. Aynı sürüm ve girdiyle çıktı bayt düzeyinde aynı olmalı.
- JSON'daki `null` (`json_sanitize` NaN'ı `None` yapıyor) çizimden önce `np.nan` yapılıp maskelenmeli. Boş dilimde `np.nanmean` çağrılmamalı; değerler JSON'daki hazır özetlerden okunmalı. Böylece `-W error::RuntimeWarning` altında uyarı çıkmaz.
- Yalnız arşiv değerleri çizilmeli, yeniden bootstrap yapılmamalı; hata çubukları JSON'daki `ci_low`/`ci_high`'tan alınmalı.

**Şekil 1 — Beat AML öğrenme eğrileri:** `sprint03_v2_results.json`
- `{cv_5fold_3seeds|wave_ood}/metrics/k0_refs/{M0,M1}/{rmse,ndcg_at_3}`: k=0 için yatay referans çizgileri.
- `{cv_5fold_3seeds|wave_ood}/metrics/by_k/{1,2,3,5,8,12}/{Random50,StaticDiversity,StaticDOpt,AdaptiveOED}/{rmse,ndcg_at_3,regret_at_1,n_eval}`: x ekseni bütçeler, ayrık noktalarla.
- İsteğe bağlı aralıklar: `.../comparisons/panel/k{k}_{Policy}_vs_Random50/rmse_diff/{diff,ci_low,ci_high}` (fark paneli olarak).

**Şekil 2 — Kapı 3 forest grafiği:** `sprint05_results.json` `track_a_m4/regimes/{id_cv_5fold_3seed,id_60_20_20,wave_ood_s3,wave_ood_s4}/experiment_a1_fixed_panel/k5/comparisons/{M4_vs_M3,M4_vs_M4_perm_null,M4_forced_pos_vs_M4_perm_forced}/rmse_diff/{diff,ci_low,ci_high,n_paired}`.
- FIMM için: `track_c_fimm/experiment_c2_within_fimm_replication/comparisons/{M4_Static_k5_vs_M3_Static_k5,M4_Static_k5_vs_M4_perm_null_k5}/rmse_diff`.
- **Dikkat:** FIMM sDSS, Beat AML ise AUC biriminde. Ayrı panel ya da ayrı x ekseni kullanılmalı; −0.50 eşiği yalnız Beat AML paneline çizilmeli. C2'de `forced` kolu yok; "koşulmadı" diye belirtilmeli.

**Şekil 3 — Kısa liste ödünleşimi ve AUGRC:**
- (A) `sprint05_results.json` `track_b_shortlist/regimes/{id_60_20_20,wave_ood}/arms/StaticDOpt_k5/methods/{EqualWidthAUC,RankConformal,RelRankConformal,Top1GapConformal}/test/{shortlist_mean_size_obs,shortlist_contains_true_top1_rate,shortlist_mean_regret1}`.
- (A) Tekil paydalar `sprint05_vectors.npz` `track_b/{reg}/StaticDOpt_k5/Top1GapConformal/{patient_definite_top1,patient_shortlist_contains_top1}`'dan alınmalı (5/9, 19/25).
- (B) AUGRC eğrisi `sprint04_v2_vectors.npz`'den: `{reg}/test_omics_pca_dist`, `{reg}/archive_available/StaticDOpt_k5/{m3_post_scale,patient_regret1,valid_mask}`. Eğri, `rank_conformal._trapezoid_augrc_from_arrays` ile aynı kuralla kurulmalı: skora göre artan sıralama, bağ grupları birlikte, kapsam = m/n, G = kümülatif kayıp/n.
- (B) Özet değerler `track_b_shortlist/regimes/{reg}/augrc_sensitivity/StaticDOpt_k5/{omics_pca_dist,m3_post_scale}/{augrc,random_augrc}` ile eşleşmeli (W3+4: 14.27 / 10.57 / 12.30).
- (B) AURC eğrisi için hazır diziler var: `sprint04_v2_results.json` `.../abstention/{score}/risk_coverage_regret1/{coverages,risks}`.

**Şekil 4 — FIMM C1 ve C2:** `sprint05_results.json` `track_c_fimm/experiment_c1_frozen_transfer/cohorts/index_109_patients/`:
- (A) `arms/{C1_M0_k0,C1_M1_rank_k0,C1_M3_Random20x_k5,C1_M3_StaticDOpt_k2,_k5,_k8}/{ndcg_at_3,regret_at_1_sdss}` ve `comparisons/{C1_M3_k5_vs_M1_k0,C1_M3_Static_k5_vs_Random_k5}/{ndcg3_diff,regret1_sdss_diff}`.
- (B) `experiment_c2_within_fimm_replication/arms/{M0_k0,M1_k0,M2a_StaticDOpt_k5,M3_StaticDOpt_k5,M3_Random20x_k5}/{rmse,ndcg_at_3}` ve `comparisons/M3_Static_k5_vs_{M1_k0,Random20x_k5}/{rmse_diff,ndcg3_diff}`.
- C1 ve C2 aynı eksende birleştirilmemeli (farklı deney).

**Kabul kontrolü önerisi:** Grafik betiği, çizdiği her sayıyı Tablo 1–4 ile karşılaştıran bir `assert` bloğu içermeli; bu denetim betiğindeki yolları kullanabilir. İki ardışık koşuda SHA-256 eşitliği test edilmeli.

#### QA kararı
- **Repo ve temiz klon: `QA_PASS`.**
- **Makale: `QA_CONDITIONAL`.** Yayından önce M-1 düzeltilmeli; M-2..M-4 küçük düzeltmeler.
- Nihai karar proje sahibine aittir.

### 3.13 Makale r2, figürler ve `fetch_fimm` yeniden denemesi — `[Claude Opus 5.5]` · 2026-11-26

**Kapsam:** Commit edilmemiş çalışma ağacı (taban `e5a830e`): makale, README, NOTICE, `fetch_fimm.py`, `make_figures.py`, `tests/test_figures.py`, `requirements-figures.txt` ve `docs/manuscript/figures/` (12 dosya). İncelenen durum: `audits/opus-2026-11-26-manuscript/reviewed_manifest_r2.sha256`. Opus kod, makale veya figür değiştirmedi.

| Kontrol | Sonuç |
|---|---|
| Arşiv | `release_manifest.sha256` 32/32 aynı. Sonuç dosyaları değişmemiş. |
| Makale sayıları | `verify_manuscript_numbers.py --with-fimm`: **395/395 OK**. Yeni giren sayılar (13+4+1=18, Spearman p=0.048, OPUS-028 eğitim-referansı Regret@1 +1.61 [−3.62, +7.73], 0/146) eklendi. Ayrıca yeni metindeki tüm sayısal ifadeler arşivde tarandı; kimlik olmayan eşleşmeyen sayı yok (eşleşmeyenler yalnız ORCID, Zenodo kaydı ve bayt sayısı). |
| §3.12 düzeltmeleri | **M-1..M-5, M-7 kapandı.** İndeks örnek kuralı kodla birebir; Spearman sınırda olarak iki aralıkla veriliyor; OPUS-028 için arşiv −0.020 esas, denetim değeri −0.019 ayrı etiketli; SD=500 Gauss gürültüsü; 599 MB Sprint 5'e bağlandı; `k ∈ {0,1,2,3,5,8,12}`; #14 URL ve #16 yazarlar eklendi. **M-6 (Guillen)** metinden çıkarıldı, içerik kararı Sol'un. FIMM C2 permütasyon sayısı (10) kodla (`range(10)`) ve Tablo 2'nin yeni "Perm Draws" sütunuyla tutarlı. |
| **R-1 `fetch_fimm` 5xx yeniden denemesi** | Mantık doğru: yalnız 5xx'te tekrar deniyor, 4xx ve diğer hatalarda hemen hata veriyor, bekleme 2 s × deneme no ile artıyor, her başarısızlıkta `.part` siliniyor, MD5 doğrulaması sonra yapılıyor. Testler 504→504→başarı (beklemeler [0.5, 1.0]) ve 404'te tek deneme senaryolarını kapsıyor. Gerçek ortamda da, önceki klon testinde Zenodo iki kez 504 verdikten sonra 3. denemede başarı gözlendi. |
| R-2 / R-3 | README boyutları doğru (~320 MB indirme, ~410 MB disk). NOTICE'e türetilmiş veriler ve dbGaP kimlikleri için açık bildirim ve lisans eklenmiş. |
| **Figür determinizmi** | `make_figures` iki kez, **tüm uyarılar hata sayılarak** (`-W error`) ve her koşu için ayrı matplotlib önbelleğiyle çalıştırıldı: çıkış 0, stderr boş, **12/12 dosya iki koşu arasında ve repodakilerle SHA-256 düzeyinde aynı**. `SOURCE_DATE_EPOCH` boş bırakıldığında da sonuç aynı. |
| Temiz venv (`requirements.txt` + `requirements-figures.txt`) | `verify_package --run-tests`: **91 passed**, çıkış 0. Bu ortamda üretilen figürler de repodakilerle bayt düzeyinde aynı. |
| Figür-tablo eşleşmesi | `verify_figure_table_parity` ve Şekil 3'teki `assert`'ler (AUGRC alanları arşivle ≤1e-10, ortalama boyutlar ≤1e-12) her koşuda geçiyor. Görsel inceleme: değerler Tablo 1–4 ile tutarlı. |

**Engellemeyen notlar (INFO):**
1. **Şekil 3B:** RankConformal etiket kutusu, EqualWidthAUC kutusunun son satırının ("Mean shortlist Regret@1: 130.9") üstüne biniyor. `box_pos["wave_ood"]["RankConformal"]` biraz aşağı alınmalı.
2. **Şekil 4:** Alt panel harfleri "C1." ve "C2." deney adlarıyla çakışıyor ("C2. C1 Paired ΔRegret@1"). Paneller örneğin A–H olarak harflendirilmeli.
3. **Şekil 3 başlığı:** x hata çubukları figür betiğinde hesaplanıyor (tohum 2026, 2000 yeniden örnekleme). Başlıkta "arşivlenmiş vektörlerden" yazıyor, bu doğru; tohum ve tekrar sayısı da eklenebilir.
4. **NOTICE:** `audits/` dizinini bütünüyle CC BY-NC kapsamına alıyor, ama içinde `.py` betikleri var. Bu betikler kod olarak PolyForm kapsamında olmalı; NOTICE'te "audits/*.py → PolyForm" diye ayrılabilir.
5. `fetch_fimm(max_retries=3)` aslında toplam deneme sayısı; parametre adı `max_attempts` olsa daha açık olur.
6. `test_generate_all_figures_byte_exact_determinism` repo figürlerini taze üretimle karşılaştırıyor. Sabitlenmiş `matplotlib==3.11.2` dışında bir sürüm kuran biri bu testi düşürür. İsteğe bağlı iyileştirme: sürüm farklıysa yalnız repo karşılaştırmasını atla (`importorskip(..., minversion)` ya da sürüm kontrolü).

#### QA kararı: `QA_PASS (teknik)`
- Makale sayıları, figür determinizmi, figür-tablo eşleşmesi, `fetch_fimm` yeniden denemesi, temiz ortam (91 test) ve arşiv bütünlüğü doğrulandı.
- §3.12'deki MAJOR M-1 kapandı.
- Kalan 6 not kozmetik ya da isteğe bağlı; gönderimi engellemiyor.
- Bilimsel uygunluk (iddia sınırları, kaynakça içeriği, figürlerin bilimsel doğruluğu) Sol'un; gönderim kararı proje sahibinin.
- Değişiklikler henüz commit edilmedi; sürüm etiketi bu doğrulanmış duruma atılmalı.

---

## 4. `[Kullanıcı Denetimi]` Notları ve Onaylanan Optimizasyon Kapsamı

*(Kullanıcının Opus önerileri üzerindeki önceliklendirme kararı buraya işlenecektir.)*
