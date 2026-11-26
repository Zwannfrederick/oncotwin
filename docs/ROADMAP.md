# OncoTwin v1 — Sprint Yol Haritası (Geri Bildirim Güdümlü)

Bu doküman projenin tüm sprint aşamalarını, karar kapılarını ve ilerleme durumunu izler. Takvim günü yerine kullanıcı geri bildirimi, bağımsız hakem/QA denetimleri ve doğrulama kapıları esas alınır.

**26 Kasım 2026 güncel durum — v1 MÜHÜRLENDİ (Neon / Sol `ONAY` / Opus `QA_PASS` / Kullanıcı Onayı):** **GPT-6.2 Sol (§12)** v1 bilimsel kapanışına **`ONAY`**, **Claude Opus 5.5 (§3.11)** Sprint 6 sürüm paketine **`QA_PASS (teknik)`** verdi (`OPUS-028` birebir yeniden üretildi, `OPUS-029`/`032` kapandı, temiz sanal ortamda `verify_package --run-tests` **`PASS`, `88 passed`, `15.6 s`** doğrulandı). Kalan 3 `INFO` dokümantasyon cilası (`OPUS-033..035`) [README.md](../README.md) ve [METHODS_REPORT.md](METHODS_REPORT.md) dosyalarına işlendi; proje sahibi onayıyla **OncoTwin v1 resmen kapatılıp mühürlenmiştir.** Tüm tarihsel `sprint01..05` artefaktları, `PROTOCOL*.md` dosyaları ve `OPUS-028` çıktıları [release_manifest.sha256](../data/processed/release_manifest.sha256) (`32` dosya) altında SHA-256 ile kilitlidir.

| Sprint | Kapsam ve Teslimatlar | Karar Kapısı | Durum |
|---|---|---|---|
| **Sprint 1** | Beat AML 2.0 veri indirme (Git LFS SHA-256 doğrulamalı), ampirik veri denetimi (hasta != örnek, modalite kesişimi, eksiklik matrisi, ilaç aileleri, tek-yön/çift-yön `M0` varyans ayrıştırması) ve aday ilaç kümesi ($\mathcal{C}$) taslağı | **Kapı 0:** Veri Yeterliliği & Protokol Dondurma | `[x]` Tamamlandı (Kapı 0 geçti) |
| **Sprint 2** | `docs/PROTOCOL.md` dondurma + sızıntısız bölmeler (`Patient-Disjoint 5-Fold CV [3 seed]` ve `Waves 1+2 → Waves 3+4`) + permütasyon sızıntı testi *(GA ilaç-düzeyi notu `OPUS-008` v2 raporuna işlenecek)* + `M0`, `M1`, `M2a/M2b`, `M3` | **Kapı 1:** Few-Shot Sinyali | `[x]` Tamamlandı (Kapı 1: **PASS** — `M3(k=5,8)`) |
| **Sprint 3** | Toplu panel seçimi (`Random 50x`, `Static-Diversity`, `Static-D-Optimal`, `Patient-Adaptive Batch OED`), $k \in \{0, 1, 2, 3, 5, 8, 12\}$ doygunluk/VoI öğrenme eğrileri ve `M3` mekanizma ablasyonları *(tarihsel v1 kayıtları korunur)* | **Kapı 2 (İlk Koşu):** İlk koşu `PASS` (tarihsel kayıt) | `[x]` İlk Koşu Tamamlandı (`v1` korundu) |
| **Sprint 3.5 — Denetim, Fix & Yeniden Analiz (`v2`)** | **Opus (`DEV-OPUS-01..12`):** `sprint03_v2_*` + `.npz`, 50/50 pytest (`QA_PASS`), `08` Holm → `INSUFFICIENT_EVIDENCE`, `09` `M3_offset_only` + `20` çekiliş tanısı, `10` W3+4 tam-panel/`hesaplanamadı` + 90-prob keşifsel havuz. **Sol:** `SOL-01..07` kapandı (`SOL-08` açık/engellemez), `2026.06.07.730728v1` tam-metin, FIMM/Malani örtüşme (`132` örnek/`109` hasta), Sprint 4 devir notu | **Kapı 2 (Nihai):** Bilimsel Kapı 2 `PASS` + `QA_PASS` | `[x]` Tamamlandı |
| **Sprint 4 (`v1` & `v2`)** | [PROTOCOL_SPRINT04.md](PROTOCOL_SPRINT04.md) dondurma + hasta-bazlı 60/20/20 ID & temiz W1+2→W3+4 stres testi + `archive_available` & `locked_train_catalog` + `M0(0)`/`M1(0)` + `StaticDOpt`/`EqualWeightVarRed`/`AdaptiveOED` (`k∈{2,5,8}`) + %90 split-conformal & kısa liste + `ABSTAIN`/`AURC` ([v1](sprints/sprint-04-conformal-abstain.md), [v2 `DEV-OPUS-13..16`](sprints/sprint-04-v2-conformal-abstain.md)) + kullanıcı onaylı (`A4`) [Ek 19](PROTOCOL_SPRINT04_ADDENDUM_19.md) (`DEV-OPUS-19`) [duyarlılık raporu](sprints/sprint-04-v2-dev-opus19-sensitivity.md) | Opus (`DEV-OPUS-17/18/20` + `QA_PASS`) + Sol Kapanış Kabulü | `[x]` Tamamlandı (Opus `QA_PASS`, Sol onaylı) |
| **Sprint 5 (`M4`, Kısa Liste & FIMM)** | [PROTOCOL_SPRINT05.md](PROTOCOL_SPRINT05.md) dondurma + `M4` yolak/hedef-ailesi yapılı kovaryans (`20×` derece-korumalı permütasyon, Deney A1/A2) + hasta-düzeyi sıra-kalibreli kısa liste (`RankConformal`, `RelRankConformal`, `Top1GapConformal`) & `AUGRC` + FIMM/Malani (`Zenodo 7370747`, MD5 doğrulamalı `sprint05_fimm_catalog_audit.json`, `132` örnek/`109` hasta, `23/24` hedef, `64` prob) ayrı `C1` donmuş aktarım & `C2` yöntem tekrarı ([sprint-05-m4-shortlist-fimm.md](sprints/sprint-05-m4-shortlist-fimm.md)) | **Kapı 3 (`AD-001`):** **`NO-GO (v2 ODE ERTELENİR)`** (`AD-009`) | `[x]` Tamamlandı (Opus `QA_PASS`, Sol §9–§11 onaylı) |
| **Sprint 6 (Yayın & Kapanış Paketi)** | Tekrarlanabilir CPU-only araştırma paketi (`pyproject.toml`, `requirements.txt`, `CITATION.cff`, `verify_package.py`, `release_manifest.sha256`), `OPUS-032` public helper, `OPUS-028` Beat AML prob-ölçeği duyarlılığı + bütünleştirilmiş [METHODS_REPORT.md](METHODS_REPORT.md) (`SOL-13..16`, `OPUS-028..035`, negatif bulgular ve Haziran 2026 Beat AML ön baskısı konumlandırması) | **Nihai Kapanış:** Sol §12 `ONAY` + Opus §3.11 `QA_PASS` + Kullanıcı Mührü | `[x]` **TAMAMLANDI — OncoTwin v1 Mühürlendi (`2026-11-26`)** |

---

## Yarın İçin Sıralı Kolektif Yol Haritası (`Adım 1 → Adım 7`)

### Ön Hizalama Kutusu (`Sol §3.2 ↔ Opus §3.6`)

| Görev | Opus §3.6 İlk Taslağı | Sol §3.2 Kilitlenen Kuralı (Esas Alınacak) |
|---|---|---|
| **`DEV-OPUS-08`** (`ADAPTIVE_WINS`) | 3 `k` × 4 metrik = 12 test (`Recall@3` dahil), `STATIC_EQUIVALENT` etiketi | `k={3,5,8} × {RMSE, NDCG@3, Regret@1}` = **9 eşleştirilmiş test** (`Recall@3` karar ölçütü değil), iki yönlü simetrik Holm düzeltmesi; zıt yön = **`karışık`**, düzeltilmiş destek yoksa = **`yetersiz kanıt`**; mevcut veriyle keşifsel |
| **`DEV-OPUS-09`** (`M3_offset_only` + permütasyon) | Tek ofset `M3_offset_only` + $\ge 20$ çekilişli derangement `perm_within` | **Hasta ofsetini sabit tutup merkezlenmiş, eğitim ölçeğiyle standardize edilmiş kalıntıların** `≥20` sabit çekilişli permütasyonu (duyarlılık özeti); ofset daraltması ve ilaç ölçekleri eğitim içinde seçilir |
| **`DEV-OPUS-10`** (W3+4 tam panel) | `n<5` gerekçesi + iki dalgada ortak prob havuzu keşifsel duyarlılığı | `k=2` (`2/146`) ve `k=3,5,8,12` (`0/146`) tam panel / doğrudan isabet / backfill ayrı raporlanır; tam panel yoksa performans hücresi **`hesaplanamadı`** yazılır; arşiv W3+4 ortak prob havuzu `keşifsel`dir + `OPUS-015` (`ABT-737` ham `15` vs temiz `3`) teyidi |

### Sıralı Uygulama Adımları

1. **Adım 1 — Claude Opus 5.5 (`DEV-OPUS-01..03, 12`):**
   - Sonucu değiştirmeyen düşük riskli işler: değerlendirilebilir hastalarda `len(S_p) == k` assert (`01`), `evaluate_gate2` W3+4 GA hizalaması (`02`), ölü kod temizliği (`03`) ve kalıcı adversarial regresyon testlerinin `tests/test_opus_regression.py` altına taşınması (`12`).
2. **Adım 2 — Claude Opus 5.5 (`DEV-OPUS-04..06`):**
   - Performans ve yapısal sadeleştirme: hasta-başı metrik vektörleştirme (`04`), 3 değerlendiricinin tek `eval/` fonksiyonunda birleştirilmesi (`05`), `run_sprint03.py`'nin `<800` satıra bölünmesi (`sprint03_ablations.py`, `sprint03_gate.py`) (`06`).
3. **Adım 3 — Claude Opus 5.5 (`DEV-OPUS-07`):**
   - Ayrı sürüm yeniden koşu + hasta vektörlerinin kalıcılaştırılması (`sprint03_v2_results.json`, `sprint03_v2_vectors.npz`; v1 `sprint03_results.json` ezilmez, v1 ile `≤1e-9` eşitlik teyit edilir).
4. **Adım 4 — Claude Opus 5.5 (`DEV-OPUS-08, 09, 10, 11`):**
   - Yukarıdaki **Sol §3.2 Kilitlenen Kuralları** ile yeniden analizler (`sprint03_v2_reanalysis.json`), `M3_offset_only` + sabit-ofsetli standardize permütasyon, W3+4 `hesaplanamadı` + ortak-havuz keşifsel duyarlılık ve `obs_rmse`/tanım/GA notları (`11`).
5. **Adım 5 — GPT-6.2 Sol *(Adım 1–4 ile paralel araştırma + Adım 4 sonrası nihai hüküm)*:**
   - *Paralel (Adım 1–4 sırasında):* [neon-lead-dev.md](agents/neon-lead-dev.md) §4.1 yanıtlarının (`SOL-01..07` & `OPUS-015` `ABT-737=3` teyidi) kontrolü; Haziran 2026 bioRxiv Beat AML split-conformal ön baskısının (`2026.06.07.730728v1`) tam-metin incelemesi; FIMM/Malani (`Zenodo 7370747`) hasta/örnek/prob/hedef örtüşme tablosu; Sprint 4 devir notunun ([sol-architect-reviewer.md](agents/sol-architect-reviewer.md) §5) olgunlaştırılması.
   - *Adım 4 sonrası:* Claude'un `sprint03_v2_*` çıktılarını inceleyip [sol-architect-reviewer.md](agents/sol-architect-reviewer.md) **§3.1 (Bulgular Tablosu)** ve **§3.3 (Nihai Hakem Kararı & Sprint 4 Geçiş Koşulları)** bölümlerini kapatma.
6. **Adım 6 — Kullanıcı Onay Kapısı (`Kapı 2 Nihai` & Sprint 4 İzni):** ✅ Tamamlandı.
7. **Adım 7 — Gemini 4 Neon (`v2` Kontrolü → `PROTOCOL_SPRINT04.md` → Sprint 4 Implementasyonu):** ✅ Tamamlandı (`2026-11-22`).
   - **7a:** Opus `sprint03_v2_*` + 50/50 pytest + v1 SHA-256 koruma doğrulaması tamamlandı.
   - **7b:** [PROTOCOL_SPRINT04.md](PROTOCOL_SPRINT04.md) deney öncesi donduruldu.
   - **7c:** `uncertainty.py`, `make_sprint04_splits`, `run_sprint04.py`, `sprint04_report.py` ve `test_sprint04.py` TDD ile yazılıp gerçek Beat AML 2.0 verisinde (`49.0 s`, sıfır `RuntimeWarning`) çalıştırıldı.
8. **Adım 8 (Aktif — Sprint 4 `v2` & `DEV-OPUS-19` Tamamlandı → Opus & Sol Kapanışı):**
   - **Neon (`DEV-OPUS-13..16` / `SOL-09..12`):** ✅ Tamamlandı (`48.6 s`, `-W error::RuntimeWarning`, `67/67 pytest`). [`run_sprint04.py`](../src/oncotwin/experiments/run_sprint04.py) (`619` satır) [`sprint04_eval.py`](../src/oncotwin/experiments/sprint04_eval.py) (`206` satır) ile bölündü; `sprint04_v2_*` ve [sprint-04-v2-conformal-abstain.md](sprints/sprint-04-v2-conformal-abstain.md) üretildi; tüm `v1` çıktıları ve protokoller SHA-256 düzeyinde korundu.
   - **Neon (`DEV-OPUS-19` — Kullanıcı `A4` Onaylı Ek Duyarlılık):** ✅ Tamamlandı. İzole [`sprint04_dev_opus19.py`](../src/oncotwin/experiments/sprint04_dev_opus19.py) (`446` satır) ve [`test_sprint04_dev_opus19.py`](../tests/test_sprint04_dev_opus19.py) (`3/3`, toplam `70/70 pytest`) eklendi; `sprint04_v2_dev_opus19_results.json` ve [sprint-04-v2-dev-opus19-sensitivity.md](sprints/sprint-04-v2-dev-opus19-sensitivity.md) üretildi. `v1` ve `v2 (13–16)` çıktıları bit düzeyinde korundu.
   - **Claude Opus 5.5 (`DEV-OPUS-17/18/20` + Nihai Teknik QA — Kullanıcı Onaylı):** `test_sprint04.py` içindeki `pytest.raises` bloğunu daraltma (`17`), `sprint04_eval._safe_eval_masked` sadeleştirme (`18`), `_require_full_budget` / ölü `g2l` / M1 sabit skor etiketi (`20`) + `v1 ↔ v2` eşitlik, `DEV-OPUS-19` sözleşme `(a)-(c)` doğrulaması, adversarial (`test_infinite_q_does_not_publish_coverage` artık `[XPASS(strict)]`) ve bağımsız yeniden üretim kontrolü.
   - **GPT-6.2 Sol (`v2` & `19` Kabulü + Ek 19 & Kaynak Düzeltmeleri — Kullanıcı Onaylı):** Sprint 4 `v2` (`SOL-09..12`) ve `DEV-OPUS-19` (`A1` OOF/in-sample ve `A2`/`A3` sonuçları) incelemesi; Addendum 19 üzerinde `A1`/`A3` metin netleştirmesi; Opus §3.9 kaynak düzeltmeleri (`JRSS-B → Lei 2018`, `Holm` DOI/JSTOR, `DrEval`, `Zenodo File_7 = 167` [`163 AML + 4 CD34+`], `AUGRC` duyarlılık kararı).
   - **Kullanıcı Yönlendirmesi (`2026-11-25`):** Sprint 5 başlatma ve güncel literatür taraması onaylandı.
9. **Adım 9 (Sprint 5 & Denetim Kapanışı):** ✅ Tamamlandı (`2026-11-25`).
   - **Neon:** Kulvar A (`M4` / Kapı 3 `NO-GO`), Kulvar B (Sıra-Konformal & `AUGRC`), Kulvar C (FIMM `C1`/`C2`), `AD-009` ve `84/84 pytest`.
   - **Claude Opus 5.5 (§3.10):** `DEV-OPUS-17/18/20` kapatıldı (`85/85 pytest`); Sprint 4 `v2`/Ek 19 ve Sprint 5 bağımsız yeniden üretildi → ikisi için de **`QA_PASS (teknik)`**.
   - **GPT-6.2 Sol (§9–§11):** `SOL-13..16` rapora işlendi, `OPUS-028` Beat AML iç prob-ölçeği duyarlılık deneyi ([opus028_beataml_probe_scale.py](../audits/sol-sprint06/opus028_beataml_probe_scale.py)) koşuldu, [METHODS_REPORT.md](METHODS_REPORT.md) çalışma taslağı oluşturuldu ve Sprint 6 onayı verildi.
10. **Adım 10 (Sprint 6 — Nihai Kapanış & v1 Mührü):** ✅ **Tamamlandı (`2026-11-26`).**
    - **Neon (Paket & Bütünleştirme):** ✅ `pyproject.toml`, `requirements.txt`, `CITATION.cff`, `verify_package.py`, `data/processed/release_manifest.sha256` (`32` dosya), `OPUS-032` public `finite_sample_conformal_quantile`, bütünleştirilmiş [METHODS_REPORT.md](METHODS_REPORT.md) ve `tests/test_sprint06_package.py`.
    - **GPT-6.2 Sol (§12):** ✅ Bilimsel kapanış **`ONAY`**.
    - **Claude Opus 5.5 (§3.11):** ✅ **`QA_PASS (teknik)`** (`OPUS-028` birebir yeniden üretim `audits/opus-2026-11-26-sprint06/`, `OPUS-029`/`032` kapalı, temiz `venv` içinde `88 passed`).
    - **Kullanıcı Mührü & `OPUS-033..035` Cilası:** ✅ Üç `INFO` dokümantasyon notu (`README.md`, `METHODS_REPORT.md`) işlendi ve **OncoTwin v1 mühürlendi**.

---

## Sprint 6 Kapanış Kontrol Listesi (Opus & Sol İncelemesine Hazır)

1. **Ortam & Paket:** [pyproject.toml](../pyproject.toml), [requirements.txt](../requirements.txt) (`numpy==2.5.5`, `pandas==3.0.3`, `openpyxl==3.1.5`, `pytest==9.2.0`), [CITATION.cff](../CITATION.cff).
2. **Tek-Komut Doğrulama:** `verify_package.py` + `data/processed/release_manifest.sha256` (`32` dondurulmuş dosya + ham Beat AML SHA-256 / FIMM MD5 + `16` bilimsel değişmez + `88/88 pytest`).
3. **Kod Hijyeni:** `OPUS-032` (`finite_sample_conformal_quantile` public); `OPUS-029` tarihsel bit-eşitliği korumak için kodda değiştirilmeyip `METHODS_REPORT.md §7`'de belgelendi.
4. **Nihai Rapor:** [METHODS_REPORT.md](METHODS_REPORT.md) (`139` satır — Sol bilimsel çekirdeği + Neon izlenebilirlik/ortam/yeniden çalıştırma bölümleri).

---

## Çıktı ve Dosya Takip Haritası
- Mimari Kararlar: [DECISION_LOG.md](DECISION_LOG.md) (`AD-001`–`AD-009`)
- Dondurulmuş Protokoller: [PROTOCOL.md](PROTOCOL.md), [PROTOCOL_SPRINT03.md](PROTOCOL_SPRINT03.md), [PROTOCOL_SPRINT04.md](PROTOCOL_SPRINT04.md), *(A4 Onaylı Ek)* [PROTOCOL_SPRINT04_ADDENDUM_19.md](PROTOCOL_SPRINT04_ADDENDUM_19.md), [PROTOCOL_SPRINT05.md](PROTOCOL_SPRINT05.md)
- Sprint 1–5 Raporları: [sprint-01](sprints/sprint-01-data-audit.md), [sprint-02](sprints/sprint-02-baselines.md), [sprint-03 v1](sprints/sprint-03-panel-selection.md) / [v2](sprints/sprint-03-v2-panel-selection.md) / [Holm](sprints/sprint-03-v2-reanalysis.md), [sprint-04 v1](sprints/sprint-04-conformal-abstain.md) / [v2](sprints/sprint-04-v2-conformal-abstain.md) / [Ek 19](sprints/sprint-04-v2-dev-opus19-sensitivity.md), [sprint-05](sprints/sprint-05-m4-shortlist-fimm.md)
- Paket & Doğrulama Artefaktları: [pyproject.toml](../pyproject.toml), [requirements.txt](../requirements.txt), [CITATION.cff](../CITATION.cff), [verify_package.py](../src/oncotwin/experiments/verify_package.py), `data/processed/release_manifest.sha256`
- Kolektif Ajan Panoları:
  - Gemini 4 Neon (Ana Geliştirici): [agents/neon-lead-dev.md](agents/neon-lead-dev.md)
  - GPT-6.2 Sol (Araştırma Mimarı ve Hakem): [agents/sol-architect-reviewer.md](agents/sol-architect-reviewer.md)
  - Claude Opus 5.5 (Optimizasyon ve Son Kontrol): [agents/opus-optimization-qa.md](agents/opus-optimization-qa.md)
- Bağımsız Denetim Artefaktları: `audits/opus-2026-11-21/` (S3), `audits/opus-2026-11-22-sprint04/` (S4), `audits/opus-2026-11-25-sprint05/` (S5), `audits/opus-2026-11-26-sprint06/` (S6), `audits/sol-sprint06/` (`OPUS-028`)
- Nihai Yöntem & Tekrarlanabilirlik Raporu (Sprint 6): [METHODS_REPORT.md](METHODS_REPORT.md)
