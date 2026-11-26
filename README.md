> **Free for science. Not for commercial exploitation.**
>
> OncoTwin is a research-only computational framework made available to advance scientific understanding and reproducible cancer research. Commercial exploitation and clinical decision-making are prohibited under the research license.

# OncoTwin (v1)

**Yazar / Proje Sahibi:** Muhammed Emin Çelik  
**Depo:** https://github.com/Zwannfrederick/oncotwin  
**Lisans:** Kod için [PolyForm Noncommercial 1.0.0](LICENSE) · Rapor, protokol ve grafikler için [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (bkz. [NOTICE](NOTICE))

OncoTwin v1, birincil akut miyeloid lösemi (AML) örneklerinde tüm ilaç panelini taramak yerine **az sayıda (örneğin 5) *ex vivo* ilaç ölçümü** yaparak geri kalan klinik aday ilaçları hasta içinde sıralamanın ne kadar işe yaradığını ve nerede tıkandığını test eden açık, tekrarlanabilir bir araştırma çalışmasıdır.

Bu çalışma "yeni bir mucize algoritma" veya "tam hasta dijital ikizi" iddiası taşımaz; iki açık AML veri seti (**Beat AML 2.0** ve **FIMM / Malani**) üzerinde ön-kayıtlı deney sonuçlarını, çalışan kısımları ve çalışmayan (negatif) bulguları birlikte raporlar.

---

## Ne Sorduk, Ne Bulduk?

- **Veri:** [Beat AML 2.0](https://github.com/biodev/beataml2.0_data) (`471` benzersiz indeks AML hastası, `24` hedef ilaç, `166` toplam inhibitör; Waves 1–4) ve dış tekrar için [FIMM / Malani AML](https://zenodo.org/records/7370747) (`132` eşleşmiş örnek / `109` benzersiz hasta, `23/24` ortak hedef, `64` ortak prob).
- **Çalışan kısım (5 prob + `M3`):** Eğitimde seçilen sabit 5 ilaçlık `StaticDOpt` paneli ve koşullu kovaryans güncellemesi (`M3`), yalnızca omik veriye (`M1`) ve aynı bütçeli rastgele panele karşı Beat AML'de (`ID 60/20/20`: `ΔRMSE −7.29 [−9.45, −5.42]` vs `M1`; `ID CV`: `ΔRMSE −2.37 [−2.84, −1.89]` vs `Random50`) ve FIMM içi 5-kat CV tekrarında (`C2`: `ΔRMSE −1.00 [−1.26, −0.76] sDSS` vs `M1`) tekrarlanabilir kazanç sağladı. Beat AML'de dondurulmuş model sıfır yeniden-eğitimle FIMM'e taşındığında (`C1`, `109` hasta) üst-3 zenginleşmesi (`ΔNDCG@3 +0.032` vs `M1`, `+0.033` vs `Random`) korundu, ancak `Regret@1` farkı anlamlı çıkmadı.
- **Kısa aday listesi (`RankConformal`):** Sprint 4'teki eşit genişlikli AUC aralıkları neredeyse tüm hedefleri (`%98.7`, `18.70` ilaç) listede tutuyordu. Hasta düzeyinde sıra kalibrasyonu (`RankConformal`, nominal `%90`), listeyi **`6.00` ilaca** indirirken gerçek en iyi ilacı iç testte **`85/91 (%93.4)`**, zamansal `Waves 3+4` testinde **`137/148 (%92.6)`** oranında korudu.
- **Çalışmayan kısımlar (negatif bulgular):**
  1. **Yolak / ilaç-ailesi kovaryansı (`M4`) ek kazanç vermedi (`Kapı 3: NO-GO`):** Sabit 5-prob panelinde `M4` ne yolaksız `M3`'ü (`+0.011 AUC`) ne de 20 çekilişli derece-korumalı rastgele aile grafını (`+0.017 AUC`) geçebildi; FIMM'de de fark `0` çıktı. Bu nedenle kinetik Hill-ODE (v2) ertelendi ([AD-001](docs/DECISION_LOG.md), [AD-009](docs/DECISION_LOG.md)).
  2. **Omik uzaklık tabanlı OOD çekinmesi yeni dalgaya taşınmadı:** `Waves 3+4` testinde omik uzaklık skoru rastgele seçimden daha kötü performans verdi (`AURC 32.44 vs 24.60`, `AUGRC 14.27 vs 12.30`).
  3. **Tek aday tahmini (`Top1Gap`, `|S|=1`) güvenilir değil:** Genel kapsam `%92.3` görünse de yalnız tek ilaç önerilen alt grupta gerçek birinci oranı iç testte `5/9`, `Waves 3+4`'te `19/25` kaldı.
  4. **Kapsam sınırı:** Tüm bulgular geriye dönük arşivlerdeki *ex vivo* 72 saatlik canlılık ölçümleriyle sınırlıdır; klinik yanıt veya tedavi seçimi kanıtı değildir.

---

## Araştırma Hiyerarşisi ve Yapay Zeka Kullanım Şeffaflığı

Bu çalışma **insan yönetimi ve sorumluluğu** altında, kodlama ve iç hata kontrolü için üç farklı yapay zeka modeline ayrık görevler verilerek yürütüldü:

- **Proje Sahibi ve Tek Yazar — Muhammed Emin Çelik:** Araştırma sorusunu belirledi; başlangıçtaki "tam dijital ikiz / kinetik ODE" fikrini bilimsel kanıtlar ışığında gerçekçi *ex vivo* panel seçimi çekirdeğine daralttı; veri sızıntısı yasaklarını, hasta bazlı bölmeleri, ablasyon ve kontrol koşullarını koydu; her sprint geçişinde (`Kapı 0–3`, `Ek 19`, `v1 mührü`) devam/durdurma kararlarını bizzat verdi. Yazar olarak tüm çalışmanın bilimsel sorumluluğu kendisine aittir. *(Not: Sayısal hesaplamalar elle tek tek yeniden hesaplanmamış; kod, birim/sözleşme testleri ve farklı ortam/araçlarla yapılan bağımsız yeniden çalıştırmalar üzerinden doğrulanmıştır.)*
- **Araç olarak kullanılan yapay zeka modelleri (yazar değildir):**
  - **Gemini 4 Neon:** Veri indirme/doğrulama boru hatları, model/seçici/konformal kodları, deney koşucuları ve paketleme.
  - **GPT-6.2 Sol:** Protokol ve veri sızıntısı denetimi (`SOL-01..16`), iddia sınırlarının daraltılması, kaynak kontrolü, `OPUS-028` ölçek duyarlılığı betiği ve [METHODS_REPORT.md](docs/METHODS_REPORT.md) bilimsel çekirdeği.
  - **Claude Opus 5.5:** Çekişmeli (adversarial) köşe-durum testleri (`OPUS-001..035`), bellek/hız profili, temiz sanal ortamda bağımsız yeniden üretim ve Sprint 3/4 `v2` refaktörleri (`DEV-OPUS-01..12, 17, 18, 20`).

> **Önemli Şeffaflık Notu:** Yapay zeka ajanlarının bu depodaki denetimleri (`docs/agents/*.md`, `audits/**`) **bağımsız akademik hakemlik (peer review) değildir**; bunlar proje içi otomatik hata yakalama, sızıntı tarama ve tekrarlanabilirlik kontrolleridir.

---

## Temel Dokümanlar

- **bioRxiv Ön Baskı Taslağı (İngilizce):** [docs/manuscript/oncotwin_v1_preprint.md](docs/manuscript/oncotwin_v1_preprint.md)
- **Yöntem ve Tekrarlanabilirlik Raporu (Türkçe):** [docs/METHODS_REPORT.md](docs/METHODS_REPORT.md)
- **Yol Haritası & Mimari Kararlar:** [docs/ROADMAP.md](docs/ROADMAP.md), [docs/DECISION_LOG.md](docs/DECISION_LOG.md) (`AD-001`–`AD-009`)
- **Dondurulmuş Protokoller:** [PROTOCOL.md](docs/PROTOCOL.md), [PROTOCOL_SPRINT03.md](docs/PROTOCOL_SPRINT03.md), [PROTOCOL_SPRINT04.md](docs/PROTOCOL_SPRINT04.md), [PROTOCOL_SPRINT04_ADDENDUM_19.md](docs/PROTOCOL_SPRINT04_ADDENDUM_19.md), [PROTOCOL_SPRINT05.md](docs/PROTOCOL_SPRINT05.md)
- **Sprint 1–5 Raporları:** [sprint-01](docs/sprints/sprint-01-data-audit.md) · [sprint-02](docs/sprints/sprint-02-baselines.md) · [sprint-03 v1](docs/sprints/sprint-03-panel-selection.md) / [v2](docs/sprints/sprint-03-v2-panel-selection.md) / [Holm](docs/sprints/sprint-03-v2-reanalysis.md) · [sprint-04 v1](docs/sprints/sprint-04-conformal-abstain.md) / [v2](docs/sprints/sprint-04-v2-conformal-abstain.md) / [Ek 19](docs/sprints/sprint-04-v2-dev-opus19-sensitivity.md) · [sprint-05](docs/sprints/sprint-05-m4-shortlist-fimm.md)
- **İç Denetim Panoları ve Artefaktları:** [neon-lead-dev.md](docs/agents/neon-lead-dev.md), [sol-architect-reviewer.md](docs/agents/sol-architect-reviewer.md), [opus-optimization-qa.md](docs/agents/opus-optimization-qa.md), `audits/`
- **Paket ve Ortam Dosyaları:** [requirements.txt](requirements.txt) (yalnız 3 çalışma paketi: `numpy`, `pandas`, `openpyxl` + `pytest`, CPU-only), [pyproject.toml](pyproject.toml), [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), [NOTICE](NOTICE), [release_manifest.sha256](data/processed/release_manifest.sha256)

---

## Çalıştırma ve Doğrulama Komutları

> **Kurulum & Manifest Notu (`OPUS-033`, `OPUS-034`):** v1 için desteklenen kurulum ve çalıştırma yolu repo kökünden `pip install -r requirements.txt` ve `PYTHONPATH=. python3 -m src...` kullanımıdır (`pip install .` üst düzey `src` ad alanı kurduğu için önerilmez). `verify_package --write-manifest` bayrağı mevcut `release_manifest.sha256` dosyasının üzerine yazar; yalnızca yeni sürüm keserken kullanın.

Ham veri dosyaları (`data/raw/`, ~`330 MB`) depoda yeniden dağıtılmaz; resmi kaynaklardan SHA-256/MD5 doğrulamasıyla indirilir. İşlenmiş özet çıktılar ve 32 dosyalı SHA-256 manifesti (`data/processed/`, ~`9 MB`) hemen denetlenebilmesi için depoda yer alır.

```bash
# 0. Ortam kurulumu (CPU-only, yalnız 3 çalışma paketi + pytest)
python3 -m pip install -r requirements.txt

# 1. Ham verileri bütünlük doğrulamasıyla indir (Beat AML SHA-256, FIMM MD5)
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_beataml
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_fimm

# 2. Tek-komut paket bütünlüğü (ham veri + 32 dosya manifest + 16 değişmez + 88 pytest)
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python3 -W error::RuntimeWarning -m src.oncotwin.experiments.verify_package --run-tests

# 3. Ham veri indirmeden yalnız arşiv manifesti ve bilimsel değişmezleri kontrol etmek için
PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.verify_package --skip-raw

# 4. İsteğe bağlı: Sprint 1–5 deneylerini ve OPUS-028 duyarlılığını ayrı bir kopyada baştan üret
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

---

> **Sorumluluk Reddi:** Bu yazılım ve rapor yalnızca bilimsel araştırma amaçlıdır (`Research Use Only`). Ticari kullanım, klinik tanı, tedavi seçimi veya hasta yönetimi kesinlikle kapsam dışıdır.
