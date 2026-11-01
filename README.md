# VoxRecte — Fonem Tanıma Temel Testi

Önceden eğitilmiş bir wav2vec 2.0 fonem tanıma modelini örnek İngilizce kayıtlar üzerinde çalıştırır ve her kayıt için IPA fonem dizisini, çıkarım süresini ve çalışma ortamı bilgisini kaydeder.

- **Model:** [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) (CTC, eSpeak IPA fonemleri)
- **Veri:** [`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy) (LibriSpeech clean/validation, 16 kHz)

## Gereksinimler

- Python 3.13 (test edilen sürüm)
- İnternet bağlantısı (model, veri seti ve revizyon bilgisi Hugging Face'ten alınır)
- İsteğe bağlı: CUDA 13.0 destekleyen sürücüye sahip NVIDIA GPU; yoksa CPU'da çalışır

## Kurulum

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`, torch ve torchaudio'nun CUDA 13.0 sürümlerini PyTorch paket deposundan kurar ve projenin `voxrecte` paketini düzenlenebilir modda (`pip install -e .`) yükler; script'ler ve testler bu paketi kullanır. GPU yoksa önce CPU sürümlerini kurun, ardından kalan paketleri yükleyin:

```bash
pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

eSpeak veya phonemizer kurulumu gerekmez; script yalnızca model çıktısını çözer.

**Windows notu:** torch paketinin dosya yolları uzundur. Depo derin bir klasöre klonlanırsa kurulum `OSError: No such file or directory` hatasıyla durabilir. Depoyu kısa bir yola (ör. `C:\voxrecte`) klonlayın veya Windows'ta [uzun yol desteğini](https://pip.pypa.io/warnings/enable-long-paths) açın.

## Çalıştırma

```bash
python scripts/run_phoneme_baseline.py          # ilk 5 kayıt, varsa GPU
python scripts/run_phoneme_baseline.py -n 10    # ilk 10 kayıt
python scripts/run_phoneme_baseline.py --cpu    # GPU olsa da CPU kullan
```

İlk çalıştırmada model (~1,2 GB) ve veri seti Hugging Face'ten indirilir.

## Çıktılar

Sonuçlar `outputs/` klasörüne yazılır:

| Dosya | İçerik |
| --- | --- |
| `phoneme_baseline.json` | Kayıt bazında tahminler, ses ve çıkarım süreleri |
| `environment.json` | Python, kütüphane, CUDA, GPU sürümleri; model ve veri seti revizyonları |

Depodaki çıktılar 14 Ekim 2026'da NVIDIA GeForce RTX 4060 Laptop GPU üzerinde üretildi. Örnek:

| ID | Süre (s) | Çıkarım (s) | Referans metin | Tahmin (IPA) |
| --- | --- | --- | --- | --- |
| 1272-128104-0001 | 4,82 | 0,053 | NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER | n ɔːɹ ɹ ɪ z m ɪ s t ɚ k w ɪ l t ɚ z m æ n ɚ l ɛ s ɪ n t ɚ ɹ ɛ s t ɪ ŋ ð æ n h ɪ z m æ ɾ ɚ |

## speechocean762 veri hazırlığı

[speechocean762](https://www.openslr.org/101/) (OpenSLR SLR101, CC BY 4.0), anadili Mandarin olan 250 konuşmacının 5000 İngilizce cümlesini ve beş uzmanın telaffuz puanlarını içerir.

```bash
python scripts/prepare_speechocean762.py
```

Script arşivi (~520 MB) OpenSLR'dan `data/` klasörüne indirir, SHA-256 özetini doğrular ve açar. Ardından ses dosyalarını, konuşmacı bilgilerini, cümleleri ve uzman puanlarını kayıt kimliğiyle eşleştirir; eksik dosya, tekrar ve tutarlılık kontrollerini yapar. Orijinal eğitim/test ayrımı korunur. `data/` klasörü depoya eklenmez.

| Dosya | İçerik |
| --- | --- |
| `speechocean762_manifest.csv` | Kayıt başına bir satır: bölüm, konuşmacı, cümle, ses bilgisi, cümle düzeyi puanlar |
| `speechocean762_checks.json` | Bölüm özetleri ve eksik dosya, tekrar, tutarlılık kontrollerinin sonuçları |

## Okutulacak cümlelerin seçimi

```bash
python scripts/select_sentences.py
```

speechocean762'deki cümleler tekrarsız hale getirilir; her cümlenin fonemleri veri setinin kendi ARPAbet transkripsiyonlarından (vurgu rakamları atılarak, 39 fonem) alınır. Açgözlü bir algoritma önce tüm fonemleri kapsayan, ardından her fonemin birden fazla cümlede geçmesini ve nadir fonemleri öne çıkaran 20 temel ve 5 yedek cümle seçer. Temel cümlelerden pilot kayıtlar için 5 cümlelik bir alt küme de işaretlenir. Cümleler katılımcılara okutulacağı için dilbilgisi bozuk veya hassas içerikli cümleler [`resources/excluded_sentences.csv`](resources/excluded_sentences.csv) listesiyle gerekçeleriyle birlikte havuzdan çıkarılır.

| Dosya | İçerik |
| --- | --- |
| `sentence_selection.csv` | Seçilen cümleler (temel/yedek, pilot sırası), fonemleri ve kaynak kayıt kimlikleri |
| `phoneme_coverage.csv` | Fonem başına veri setindeki ve seçimdeki sayılar |

## Referans ve model fonemlerinin ortak biçimi

```bash
python scripts/transcribe_speechocean762.py   # tüm kayıtlar için model çıktısı (~3 dk, GPU) → data/predictions/
python scripts/check_phoneme_map.py           # eşleştirme tablosu ve train kayıtlarında kontrol
python -m unittest discover tests             # birim testleri
```

Veri setinin ARPAbet referansları ve modelin eSpeak IPA çıktısı, CMU sözlüğünün 39 fonemine (vurgusuz) ve modelin ürettiği üç ek sese (DX çırpma, Q gırtlak durması, IX indirgenmiş ünlü) çevrilir. Dönüşüm tablosu [`voxrecte/phonemes.py`](voxrecte/phonemes.py), vurgu ve kabul edilebilir telaffuz farkları (ör. r'li/r'siz aksan, çırpma) [`voxrecte/rules.py`](voxrecte/rules.py) içinde açık kurallar olarak tanımlıdır. Kontrol yalnızca train kayıtlarında yapılır.

| Dosya | İçerik |
| --- | --- |
| `phoneme_mapping.csv` | Model sembolü → ortak fonem tablosu |
| `phoneme_mapping_check.json` | Sembol eşleşme oranları, kurallarla/kurallarsız hata oranı, kural kullanımı, kalan farklar, kelime örnekleri |

## Telaffuz karşılaştırma

```bash
python scripts/compare_pronunciation.py kayit.wav "DO YOU HAVE ANY QUESTIONS FOR HER"
python scripts/compare_pronunciation.py kayit.wav "..." --json sonuc.json   # sonucu JSON olarak da yaz
```

Hedef cümle [`resources/lexicon.tsv`](resources/lexicon.tsv) sözlüğüyle fonemlere çevrilir. Sözlük speechocean762 referans transkripsiyonlarından `scripts/build_lexicon.py` ile üretilmiştir ve 2604 kelime içerir; seçilen cümlelerin tüm kelimeleri sözlükte vardır. Kayıt modelden geçirilir, iki fonem dizisi ortak biçime çevrilip hizalanır. Her kelime için hedef ve duyulan fonemler, değiştirme/silme/ekleme farkları ve kabul edilebilir farklar (kural numarasıyla) gösterilir. Kurala uymayan farklar kesin hata değil, **olası hata** olarak raporlanır.

## Değerlendirme (speechocean762 test kayıtları)

```bash
python scripts/evaluate_speechocean762.py     # transcribe_speechocean762.py çıktısını kullanır
```

Karşılaştırma 2500 test kaydında çalıştırılır ve beş uzmanın fonem etiketleriyle (`scores-detail.json`) karşılaştırılır. Uzman puanlarının ortalaması yerine her fonemdeki oylar kullanılır: işaretleme oranı "yanlış" diyen uzman sayısına göre verilir; precision/recall/F1 için "yanlış fonem" uzman çoğunluğuyla (≥3/5) tanımlanır ve alternatif tanımlar da raporlanır. Tek bir uzmanın diğer dört uzmana karşı başarısı insan referansı olarak verilir.

| Dosya | İçerik |
| --- | --- |
| `evaluation_metrics.json` | Tüm ölçümler: oy gruplarına göre işaretleme oranı, precision/recall/F1, insan referansı, korelasyonlar, yaş grubu ve fonem kırılımları, hata örüntüleri |
| `evaluation_utterances.csv` | Her test kaydı için uzman puanları, sistem sayımları ve model çıktısı |

## Ek çalışma: cümle düzeyi puan tahmini (plan dışı)

Proje planının adımlarından biri değildir; plan çıktılarından ayrı tutulmak için `extras/sentence_scores/` klasöründedir. speechocean762 hazırlandıktan sonra:

```bash
python extras/sentence_scores/extract_features.py        # her kayıt için wav2vec 2.0 katman özellikleri (~3 dk, GPU)
python extras/sentence_scores/train_sentence_scores.py   # Ridge ile uzman puanlarının tahmini
```

Model değiştirilmez; her katmanın zaman ortalaması alınır ve üzerine Ridge regresyonu eğitilir. Katman ve düzenleme katsayısı, eğitim kümesinden ayrılan 15 konuşmacılık doğrulama kümesinde seçilir; test kümesi yalnızca son ölçümde kullanılır. Özellikler `data/features/` klasörüne, sonuçlar `extras/sentence_scores/outputs/` klasörüne yazılır.

| Dosya | İçerik |
| --- | --- |
| `sentence_scores_baseline.json` | Seçilen ayarlar, PCC/MSE sonuçları, katmana göre doğrulama sonuçları |
| `sentence_scores_split.json` | Doğrulama konuşmacıları |
| `sentence_scores_test_predictions.csv` | Test kayıtları için tahminler |

## Proje yapısı

Yeniden kullanılan kod `voxrecte/` paketinde, komut satırı araçları `scripts/` klasöründedir. Script'ler yalnızca girdileri okur, paketi çağırır ve çıktıları yazar.

```
voxrecte/                         Python paketi
  paths.py                        dosya yolları
  text.py                         cümle normalleştirme
  speechocean.py                  speechocean762 okuma
  phonemes.py                     fonem envanteri, model IPA → ortak fonem dönüşümü
  rules.py                        referans fonemler, kabul edilebilir fark kuralları
  alignment.py                    referans ve model fonem dizilerinin hizalanması
  lexicon.py                      telaffuz sözlüğü
  comparison.py                   kayıt ↔ hedef cümle karşılaştırması
  evaluation.py                   uzman fonem etiketleriyle değerlendirme
  recognizer.py                   ses okuma ve wav2vec 2.0 fonem modeli
  selection.py                    fonem kapsamaya göre cümle seçimi
  files.py                        CSV / JSON yazma
scripts/                          komut satırı araçları
  run_phoneme_baseline.py         Adım 1: fonem tanıma temel testi
  prepare_speechocean762.py       Adım 2: indirme, eşleştirme ve kontrol
  select_sentences.py             Adım 3: okutulacak cümlelerin seçimi
  transcribe_speechocean762.py    Adım 4–6: tüm kayıtlar için model çıktısı
  check_phoneme_map.py            Adım 4: eşleştirmenin gerçek kayıtlarda kontrolü
  build_lexicon.py                Adım 5: speechocean762'den telaffuz sözlüğü
  compare_pronunciation.py        Adım 5: ses + hedef metin → olası fonem farkları
  evaluate_speechocean762.py      Adım 6: test kayıtlarında değerlendirme
extras/sentence_scores/           plan dışı ek çalışma: cümle düzeyi puan tahmini (script'ler ve çıktıları)
tests/                            birim testleri
resources/                        telaffuz sözlüğü, elle çıkarılan cümleler
outputs/                          üretilen veri çıktıları (CSV / JSON)
reports/                          danışmana teslim edilen raporlar (PDF)
data/                             veri seti, model çıktıları, özellikler (git'e eklenmez)
```

## Lisans

Bu depodaki kod [MIT lisansı](LICENSE) ile sunulur. Kullanılan model ([Apache 2.0](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)) LibriSpeech kayıtları ([CC BY 4.0](https://www.openslr.org/12)) ve speechocean762 ([CC BY 4.0](https://www.openslr.org/101/)) kendi lisanslarına tabidir.
