> **Free for science. Not for commercial exploitation.**
>
> OncoTwin is a research-only computational framework made available to advance scientific understanding and reproducible cancer research. Commercial exploitation and clinical decision-making are prohibited under the research license.

# OncoTwin (v1)

**Yazar / Proje Sahibi:** Muhammed Emin Çelik ([ORCID](https://orcid.org/0009-0003-9804-0640))  
**Depo:** https://github.com/Zwannfrederick/oncotwin  
**Lisans:** Kod için [PolyForm Noncommercial 1.0.0](LICENSE) · Rapor, protokol, grafik ve `data/processed/` türetilmiş çıktıları için [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (bkz. [NOTICE](NOTICE))

OncoTwin v1, akut miyeloid lösemide (AML) arşiv temelli *ex vivo* panel seçimi ve hasta düzeyinde kısa aday listesi araştırmasıdır. Tüm ilaç panelini taramak yerine **az sayıda (örneğin 5) *ex vivo* prob ölçümü** kullanarak önceden belirlenen AML hedeflerindeki ölçülmüş duyarlılıkları tahmin etmenin ve hasta içinde sıralamanın ne kadar işe yaradığını ve nerede sınırlandığını iki açık veri seti (**Beat AML 2.0** ve **FIMM / Malani**) üzerinde, belirlenen koşulardan önce dondurulmuş yazılı iç protokollerle test eder.

---

## Ne Sorduk, Ne Bulduk?

- **Veri:** [Beat AML 2.0](https://github.com/biodev/beataml2.0_data) (`471` benzersiz indeks AML hastası, `24` hedef ilaç, `166` toplam inhibitör; Waves 1–4) ve dış tekrar için [FIMM / Malani AML](https://zenodo.org/records/7370747) (`132` eşleşmiş örnek / `109` benzersiz hasta, `23/24` ortak hedef, `64` ortak Beat AML probu).
- **Çalışan kısım (5 arşiv-mevcut prob + `M3`):** Eğitimde öğrenilen `StaticDOpt` prob sırasıyla hastada ölçülmüş ilk 5 probu (eksik öneklerde geri doldurmayla) kullanan `M3` koşullu kovaryans güncellemesi, belirlenen kıyaslarda omik tabanını (`M1`) ve eşit bütçeli rastgele paneli geçti: Beat AML `ID 60/20/20` (`n=91`) `ΔRMSE −7.29 [−9.45, −5.42]` AUC vs `M1`; Beat AML `ID CV` (`n=427`) `ΔRMSE −2.37 [−2.84, −1.89]` AUC vs `Random50`; FIMM içi 5-kat CV (`C2`, `n=109`, kat içi yeniden kurulan prob havuzu) `ΔRMSE −1.00 [−1.26, −0.76]` sDSS vs `M1` ve `−0.46 [−0.61, −0.32]` vs `Random20x`. Kilitli beş-prob önek desteği (geri doldurmasız): Sprint 4 ID `19/93`, Sprint 4 `Waves 3+4` `1/163`, Sprint 3 `Waves 3+4` `0/146`, FIMM `C1` `98/109`. Beat AML'de dondurulmuş model yeniden eğitilmeden FIMM'e taşındığında (`C1`, `109` hasta, kaynakta batch-düzeltilmiş FIMM RNA) üst-3 sıralama kalitesi iyileşti (`ΔNDCG@3 +0.032 [+0.007, +0.058]` vs `M1`, `+0.033 [+0.013, +0.054]` vs `Random20x`), ancak `Regret@1` farkının %95 GA'sı sıfırı içerdi (`−0.23 [−1.26, +0.76]` sDSS).
- **Kısa aday listesi (`RankConformal`):** Sprint 4'teki eşit genişlikli AUC aralıkları (`EqualWidthAUC`) ölçülen en iyi hedefi `91/91` (ID) ve `148/148` (`Waves 3+4`) oranında kapsasa da hedeflerin `%98.7`'sini (`18.70` ve `21.41` ilaç) listede tutuyordu. Hasta düzeyinde sıra kalibrasyonu (`RankConformal`, nominal `%90`), listeyi **`6.00` ölçülmüş hedefe** indirirken en iyi **ölçülmüş** hedefi iç testte **`85/91 (%93.4)`**, keşifsel zamansal `Waves 3+4` testinde **`137/148 (%92.6)`** ampirik oranla içerdi.
- **Çalışmayan kısımlar (negatif bulgular):**
  1. **Yolak / ilaç-ailesi kovaryansı (`M4`) Kapı 3 eşiğini karşılamadı (`NO-GO`):** Sabit 5-prob panelinde `M4` ne yolaksız `M3`'e (`+0.011 [−0.010, +0.033]` AUC) ne de 20 çekilişli derece-korumalı rastgele aile grafına (`+0.017 [−0.005, +0.039]` AUC) karşı `−0.50` AUC eşiğini karşıladı; FIMM `C2`'de de (10 çekiliş) fark `−0.00 [−0.01, +0.01]` sDSS çıktı. Bu nedenle kinetik Hill-ODE (v2) ertelendi ([AD-001](docs/DECISION_LOG.md), [AD-009](docs/DECISION_LOG.md)).
  2. **Uyarlanabilir seçim (`AdaptiveOED`):** Simetrik 9-test Holm düzeltmesi sonrasında `StaticDOpt`'a karşı `INSUFFICIENT_EVIDENCE` verdi ([AD-008](docs/DECISION_LOG.md)).
  3. **Omik uzaklık tabanlı OOD çekinmesi yeni dalgaya taşınmadı:** `Waves 3+4` testinde omik uzaklık skoru rastgele seçimden daha kötü performans verdi (`AURC 32.44 vs 24.60`, `AUGRC 14.27 vs 12.30`).
  4. **Tek aday tahmini (`Top1Gap`, `size = 1`) güvenilir değil:** Marjinal kapsam ID'de `%92.3` görünse de yalnız tek ilaç önerilen alt grupta en iyi ölçülmüş hedef oranı iç testte `5/9` (Wilson `%26.7–81.1`), `Waves 3+4`'te `19/25` (`%56.6–88.5`) kaldı.
  5. **Kapsam sınırı:** Tüm bulgular geriye dönük arşivlerdeki *ex vivo* 72 saatlik canlılık ölçümleriyle sınırlıdır; klinik tanı veya tedavi seçimi kanıtı değildir.

---

## Yazarlık, Katkılar ve Yapay Zekâ Kullanımı

- **Tek Yazar ve Bilimsel Sorumlu — Muhammed Emin Çelik:** Araştırma sorusunu belirledi; kapsamı kinetik ODE modellerinden *ex vivo* panel seçimine daralttı; veri sızıntısı yasaklarını, hasta bazlı bölmeleri, kontrol koşullarını ve her sprint karar kapısını (`Kapı 0–3`, `Ek 19`) onayladı; analizleri, kaynakları ve metinleri inceleyip onayladı. Çalışmanın tüm bilimsel ve etik sorumluluğu kendisine aittir.
- **Araç olarak kullanılan yapay zekâ modelleri (yazar değildir):**
  - **Gemini 4 Neon:** Boru hatları, model/konformal kodu, deterministik grafik üretimi, paketleme ve taslak metin düzenleme desteği.
  - **GPT-6.2 Sol:** Protokol ve veri sızıntısı denetimi (`SOL-01..16`), iddia/kaynak denetimi, `OPUS-028` analiz kodu ve bilimsel metin taslağı ([METHODS_REPORT.md](docs/METHODS_REPORT.md)).
  - **Claude Opus 5.5:** Çekişmeli (adversarial) köşe-durum testleri (`OPUS-001..035`), bellek/hız profili, refaktör (`DEV-OPUS-01..12, 17, 18, 20`), makale sayı denetimi ve ayrı proje içi tekrar koşuları.

> **Şeffaflık Notu:** Bu depodaki yapay zekâ destekli denetimler (`docs/agents/*.md`, `audits/**`) **bağımsız akademik hakemlik (peer review) değildir**; bunlar proje içi otomatik hata yakalama, sızıntı tarama ve tekrarlanabilirlik kontrolleridir.

---

## Temel Dokümanlar

- **Ön Baskı Taslağı (İngilizce) ve Grafikler:** [docs/manuscript/oncotwin_v1_preprint.md](docs/manuscript/oncotwin_v1_preprint.md) · `docs/manuscript/figures/`
- **Yöntem ve Tekrarlanabilirlik Raporu (Türkçe):** [docs/METHODS_REPORT.md](docs/METHODS_REPORT.md)
- **Yol Haritası & Mimari Kararlar:** [docs/ROADMAP.md](docs/ROADMAP.md), [docs/DECISION_LOG.md](docs/DECISION_LOG.md) (`AD-001`–`AD-009`)
- **Dondurulmuş İç Protokoller:** [PROTOCOL.md](docs/PROTOCOL.md), [PROTOCOL_SPRINT03.md](docs/PROTOCOL_SPRINT03.md), [PROTOCOL_SPRINT04.md](docs/PROTOCOL_SPRINT04.md), [PROTOCOL_SPRINT04_ADDENDUM_19.md](docs/PROTOCOL_SPRINT04_ADDENDUM_19.md), [PROTOCOL_SPRINT05.md](docs/PROTOCOL_SPRINT05.md)
- **Sprint 1–5 Raporları:** [sprint-01](docs/sprints/sprint-01-data-audit.md) · [sprint-02](docs/sprints/sprint-02-baselines.md) · [sprint-03 v1](docs/sprints/sprint-03-panel-selection.md) / [v2](docs/sprints/sprint-03-v2-panel-selection.md) / [Holm](docs/sprints/sprint-03-v2-reanalysis.md) · [sprint-04 v1](docs/sprints/sprint-04-conformal-abstain.md) / [v2](docs/sprints/sprint-04-v2-conformal-abstain.md) / [Ek 19](docs/sprints/sprint-04-v2-dev-opus19-sensitivity.md) · [sprint-05](docs/sprints/sprint-05-m4-shortlist-fimm.md)
- **İç Denetim Panoları ve Artefaktları:** [neon-lead-dev.md](docs/agents/neon-lead-dev.md), [sol-architect-reviewer.md](docs/agents/sol-architect-reviewer.md), [opus-optimization-qa.md](docs/agents/opus-optimization-qa.md), `audits/`
- **Paket ve Ortam Dosyaları:** [requirements.txt](requirements.txt) (yalnız 3 çalışma paketi: `numpy`, `pandas`, `openpyxl` + `pytest`, CPU-only), [requirements-figures.txt](requirements-figures.txt) (isteğe bağlı `matplotlib`), [pyproject.toml](pyproject.toml), [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), [NOTICE](NOTICE), [release_manifest.sha256](data/processed/release_manifest.sha256)

---

## Çalıştırma ve Doğrulama Komutları

> **Kurulum & Manifest Notu (`OPUS-033`, `OPUS-034`):** v1 için desteklenen kurulum ve çalıştırma yolu repo kökünden `pip install -r requirements.txt` ve `PYTHONPATH=. python3 -m src...` kullanımıdır (`pip install .` üst düzey `src` ad alanı kurduğu için önerilmez). `verify_package --write-manifest` bayrağı mevcut `release_manifest.sha256` dosyasının üzerine yazar; yalnızca yeni sürüm keserken kullanın.

Ham veri dosyaları (`data/raw/`, indirme ~`320 MB` [`281 MB` Beat AML + `39 MB` FIMM zip], FIMM açıldığında diskte ~`410 MB`) depoda yeniden dağıtılmaz; resmi kaynaklardan SHA-256/MD5 doğrulamasıyla indirilir (`fetch_fimm` geçici Zenodo 5xx hatalarında artan beklemeyle 3 kez dener). Türetilmiş özet çıktılar, hasta vektörleri ve 32 dosyalı SHA-256 manifesti (`data/processed/`, ~`9 MB`, CC BY-NC 4.0 + kaynak atıflarıyla) hemen denetlenebilmesi için depoda yer alır.

```bash
# 0. Ortam kurulumu (CPU-only, yalnız 3 çalışma paketi + pytest)
python3 -m pip install -r requirements.txt

# 1. Ham verileri bütünlük doğrulamasıyla indir (Beat AML SHA-256, FIMM MD5)
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_beataml
PYTHONPATH=. python3 -m src.oncotwin.data.fetch_fimm

# 2. Tek-komut paket bütünlüğü (ham veri + 32 dosya manifest + 16 değişmez + pytest)
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python3 -W error::RuntimeWarning -m src.oncotwin.experiments.verify_package --run-tests

# 3. Ham veri indirmeden yalnız arşiv manifesti ve bilimsel değişmezleri kontrol etmek için
PYTHONPATH=. python3 -W error::RuntimeWarning -m src.oncotwin.experiments.verify_package --skip-raw

# 4. İsteğe bağlı: 4 makale grafiğini (PNG/PDF/SVG) deterministik olarak yeniden üret
python3 -m pip install -r requirements-figures.txt
SOURCE_DATE_EPOCH=1700000000 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python3 -W error::RuntimeWarning -m src.oncotwin.experiments.make_figures

# 5. İsteğe bağlı: Sprint 1–5 deneylerini ve OPUS-028 duyarlılığını ayrı bir kopyada baştan üret
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
