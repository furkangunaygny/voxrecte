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

`requirements.txt`, torch ve torchaudio'nun CUDA 13.0 sürümlerini PyTorch paket deposundan kurar. GPU yoksa önce CPU sürümlerini kurun, ardından kalan paketleri yükleyin:

```bash
pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

eSpeak veya phonemizer kurulumu gerekmez; script yalnızca model çıktısını çözer.

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
| `phoneme_baseline.md` | Referans metin ve tahmin edilen fonemlerin tablosu |
| `phoneme_baseline.json` | Kayıt bazında tahminler, ses ve çıkarım süreleri |
| `environment.json` | Python, kütüphane, CUDA, GPU sürümleri; model ve veri seti revizyonları |

Depodaki çıktılar 14 Ekim 2026'da NVIDIA GeForce RTX 4060 Laptop GPU üzerinde üretildi. Örnek:

| ID | Süre (s) | Çıkarım (s) | Referans metin | Tahmin (IPA) |
| --- | --- | --- | --- | --- |
| 1272-128104-0001 | 4,82 | 0,053 | NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER | n ɔːɹ ɹ ɪ z m ɪ s t ɚ k w ɪ l t ɚ z m æ n ɚ l ɛ s ɪ n t ɚ ɹ ɛ s t ɪ ŋ ð æ n h ɪ z m æ ɾ ɚ |

## Proje yapısı

```
scripts/run_phoneme_baseline.py   ana script
requirements.txt                  sabitlenmiş bağımlılıklar
outputs/                          örnek çalıştırma çıktıları
```

## Lisans

Bu depodaki kod [MIT lisansı](LICENSE) ile sunulur. Kullanılan model ([Apache 2.0](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)) ve LibriSpeech kayıtları ([CC BY 4.0](https://www.openslr.org/12)) kendi lisanslarına tabidir.
