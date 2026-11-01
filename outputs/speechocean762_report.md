# speechocean762 — Veri kontrol raporu

Kaynak: [OpenSLR SLR101](https://www.openslr.org/101/) (`https://openslr.elda.org/resources/101/speechocean762.tar.gz`, CC BY 4.0)  
Arşiv SHA-256: `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`  
Üretim: 2026-10-26T23:22:48+00:00 · `scripts/prepare_speechocean762.py`

## Özet

| Bölüm | Kayıt | Konuşmacı | Süre (saat) | Kadın / Erkek | Çocuk / Yetişkin |
| --- | --- | --- | --- | --- | --- |
| train | 2500 | 125 | 2.876 | 67 / 58 | 58 / 67 |
| test | 2500 | 125 | 2.691 | 58 / 67 | 64 / 61 |

Orijinal eğitim/test ayrımı (`train/`, `test/` Kaldi dizinleri) aynen korundu; tabloda `split` sütunu bu ayrımı gösterir. Her konuşmacının kayıt sayısı: 20 kayıt → 250 konuşmacı.

## Kontroller

| Kontrol | Sonuç |
| --- | --- |
| `wav.scp`'de olup diskte olmayan ses dosyası | ✅ Sorun yok |
| Kayıtların `text`, `utt2spk`, `spk2utt`, `scores.json`, `scores-detail.json` dosyalarında eksiksiz bulunması | ✅ Sorun yok |
| Konuşmacıların yaş ve cinsiyet bilgisi | ✅ Sorun yok |
| Konuşmacı kimliği tutarlılığı (`utt2spk`, kayıt kimliği, `WAVE/SPEAKERxxxx` klasörü) | ✅ Sorun yok |
| Kaldi dosyalarında tekrarlanan anahtar | ✅ Sorun yok |
| Hem eğitimde hem testte bulunan kayıt | ✅ Sorun yok |
| Hem eğitimde hem testte bulunan konuşmacı | ✅ Sorun yok |
| Birebir aynı ses içeriğine sahip kayıt (SHA-1) | ✅ Sorun yok |
| Aynı konuşmacının aynı cümleyi tekrar okuması | ✅ Sorun yok |
| Okunamayan veya 0,5 sn'den kısa ses | ✅ Sorun yok |
| Ses biçimi | 16000 Hz / 1 ch / PCM_16: 5000 |
| Uzman sayısı (`scores-detail.json`) | 5 uzman: 5000 kayıt |
| Cümle puanlarının aralık dışında olması | ✅ Sorun yok |
| Kelime/fonem puanlarının aralık dışında olması | ✅ Sorun yok |
| Puanlanan kelimelerin cümle metniyle eşleşmesi | ✅ Sorun yok |
| Kaldi metni ile `scores.json` metni arasında noktalama dışı fark | ✅ Sorun yok |

## Dikkat edilmesi gerekenler

1. **Listede olmayan 245 WAV dosyası.** `WAVE/` altında `wav.scp`'de geçmeyen 245 dosya var (toplam 0.411 saat, 36 konuşmacı, hepsi `SPEAKER96xx`). Bunların puanı yoktur ve hiçbir bölüme ait değildir; tabloya alınmadı. Hiçbiri listedeki bir kaydın birebir kopyası değil. Tam liste `speechocean762_checks.json` içindedir.
2. **Noktalama farkı.** 720 kayıtta `scores.json` metni noktalama içerir (ör. `RIGHT!`), Kaldi `text` dosyası içermez. Kelimeler birebir aynı; tabloda Kaldi metni kullanıldı.
3. **Tamlık (completeness) ölçeği.** README 0–1 aralığını tarif eder, ancak `scores.json` değerleri 0–10 ölçeğindedir (4975 kayıt 10,0). Değerler olduğu gibi bırakıldı.
4. **Ortak cümleler.** 52 cümle metni hem eğitimde hem testte geçer (farklı konuşmacılar). Konuşmacı düzeyinde sızıntı yoktur.

## Tablo sütunları (`speechocean762_manifest.csv`)

| Sütun | Açıklama |
| --- | --- |
| `utt_id` | Kayıt kimliği (`0` + 4 haneli konuşmacı + 4 haneli sıra) |
| `split` | Orijinal bölüm: `train` veya `test` |
| `speaker_id`, `gender`, `age` | `utt2spk`, `spk2gender`, `spk2age` dosyalarından |
| `age_group` | Türetilmiş: yaş < 18 → `child`, aksi halde `adult` |
| `text`, `num_words` | Kaldi `text` dosyasındaki cümle ve kelime sayısı |
| `wav_path`, `wav_exists` | Veri seti köküne göre ses yolu ve dosyanın varlığı |
| `duration_s`, `sample_rate`, `channels`, `subtype` | Ses dosyası başlığından |
| `audio_sha1` | Ses dosyasının SHA-1 özeti (tekrar kontrolü için) |
| `accuracy`, `completeness`, `fluency`, `prosodic`, `total` | `scores.json` cümle düzeyi uzman puanları (5 uzmanın ortalaması/medyanı) |
| `n_experts`, `n_scored_words` | `scores-detail.json`'daki uzman sayısı ve puanlanan kelime sayısı |
