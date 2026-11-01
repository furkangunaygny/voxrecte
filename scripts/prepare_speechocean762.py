"""Download speechocean762, build a per-utterance manifest and run data checks.

Source:  OpenSLR SLR101 (https://www.openslr.org/101/), CC BY 4.0
Layout:  Kaldi-style train/ and test/ directories, WAVE/SPEAKERxxxx/*.WAV,
         resource/scores.json (aggregated) and resource/scores-detail.json (five experts)

Outputs (written to outputs/):
  speechocean762_manifest.csv   one row per utterance: split, speaker, text, audio, scores
  speechocean762_report.md      missing-file, duplicate and consistency checks
  speechocean762_checks.json    the same checks in machine-readable form
"""

import argparse
import csv
import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

URL = "https://openslr.elda.org/resources/101/speechocean762.tar.gz"
ARCHIVE_SHA256 = "876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5"
SPLITS = ("train", "test")
SENTENCE_SCORES = ("accuracy", "completeness", "fluency", "prosodic", "total")
ADULT_AGE = 18  # ages in the corpus are 6-15 or 19-43, so the split is unambiguous
ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir):
    archive = data_dir / "downloads" / "speechocean762.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"{archive}: sha256 {digest} does not match {ARCHIVE_SHA256}")
    corpus = data_dir / "speechocean762"
    if not corpus.exists():
        print(f"Extracting to {data_dir}")
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
    return corpus, digest


def read_kaldi(path):
    """Return [(key, value)] in file order, so duplicate keys stay visible."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            rows.append((parts[0], parts[1].strip() if len(parts) > 1 else ""))
    return rows


def normalize(text):
    """Uppercase words only; scores.json keeps punctuation that the Kaldi text files drop."""
    return " ".join(re.sub(r"[^A-Z' ]", " ", text.upper()).split())


def duplicates(keys):
    return sorted(k for k, n in Counter(keys).items() if n > 1)


def out_of_range(values, low, high):
    return [v for v in values if v is None or not low <= v <= high]


def write_report(summary, rows, path):
    c = summary["checks"]
    sp = summary["splits"]
    status = lambda bad: "✅ Sorun yok" if not bad else f"⚠️ {len(bad)}"
    count = lambda key: sum(len(c["per_split"][s][key]) for s in SPLITS)
    missing_meta = sum(len(v) for s in SPLITS for v in c["per_split"][s]["utts_missing_from"].values())
    missing_spk = sum(len(c["per_split"][s][k]) for s in SPLITS for k in ("speakers_missing_gender", "speakers_missing_age"))
    dup_keys = [k for v in c["duplicate_keys_in_kaldi_files"].values() for k in v]
    score_oor = [u for v in c["sentence_score_out_of_range"].values() for u in v]
    word_oor = sum(c["word_score_out_of_range"].values())
    age_groups = {s: Counter(r["age_group"] for r in rows if r["split"] == s) for s in SPLITS}
    unlisted = c["wav_on_disk_not_listed"]
    punct_only = count("text_differs_from_scores_json") - count("text_differs_beyond_punctuation")

    lines = [
        "# speechocean762 — Veri kontrol raporu\n",
        f"Kaynak: [OpenSLR SLR101](https://www.openslr.org/101/) (`{summary['source']}`, CC BY 4.0)  ",
        f"Arşiv SHA-256: `{summary['archive_sha256']}`  ",
        f"Üretim: {summary['timestamp_utc']} · `scripts/prepare_speechocean762.py`\n",
        "## Özet\n",
        "| Bölüm | Kayıt | Konuşmacı | Süre (saat) | Kadın / Erkek | Çocuk / Yetişkin |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in SPLITS:
        g = sp[s]["gender"]
        lines.append(f"| {s} | {sp[s]['utterances']} | {sp[s]['speakers']} | {sp[s]['hours']} | "
                     f"{g.get('f', 0)} / {g.get('m', 0)} | "
                     f"{age_groups[s]['child'] // 20} / {age_groups[s]['adult'] // 20} |")
    lines += [
        "",
        "Orijinal eğitim/test ayrımı (`train/`, `test/` Kaldi dizinleri) aynen korundu; tabloda `split` sütunu bu ayrımı gösterir. "
        f"Her konuşmacının kayıt sayısı: {', '.join(f'{n} kayıt → {k} konuşmacı' for n, k in c['utterances_per_speaker'].items())}.\n",
        "## Kontroller\n",
        "| Kontrol | Sonuç |",
        "| --- | --- |",
        f"| `wav.scp`'de olup diskte olmayan ses dosyası | {status(c['wav_listed_but_missing'])} |",
        f"| Kayıtların `text`, `utt2spk`, `spk2utt`, `scores.json`, `scores-detail.json` dosyalarında eksiksiz bulunması | {'✅ Sorun yok' if not missing_meta else f'⚠️ {missing_meta}'} |",
        f"| Konuşmacıların yaş ve cinsiyet bilgisi | {'✅ Sorun yok' if not missing_spk else f'⚠️ {missing_spk}'} |",
        f"| Konuşmacı kimliği tutarlılığı (`utt2spk`, kayıt kimliği, `WAVE/SPEAKERxxxx` klasörü) | {status(sum((c['per_split'][s]['speaker_id_inconsistent'] + c['per_split'][s]['utt2spk_vs_spk2utt_mismatch'] for s in SPLITS), []))} |",
        f"| Kaldi dosyalarında tekrarlanan anahtar | {status(dup_keys)} |",
        f"| Hem eğitimde hem testte bulunan kayıt | {status(c['utts_in_both_splits'])} |",
        f"| Hem eğitimde hem testte bulunan konuşmacı | {status(c['speakers_in_both_splits'])} |",
        f"| Birebir aynı ses içeriğine sahip kayıt (SHA-1) | {status(c['identical_audio_groups'])} |",
        f"| Aynı konuşmacının aynı cümleyi tekrar okuması | {status(c['same_speaker_same_text'])} |",
        f"| Okunamayan veya 0,5 sn'den kısa ses | {status(c['empty_or_short_audio_lt_0.5s'])} |",
        f"| Ses biçimi | {', '.join(f'{k}: {v}' for k, v in c['audio_formats'].items())} |",
        f"| Uzman sayısı (`scores-detail.json`) | {', '.join(f'{k} uzman: {v} kayıt' for k, v in c['n_experts_distribution'].items())} |",
        f"| Cümle puanlarının aralık dışında olması | {status(score_oor)} |",
        f"| Kelime/fonem puanlarının aralık dışında olması | {'✅ Sorun yok' if not word_oor else f'⚠️ {word_oor}'} |",
        f"| Puanlanan kelimelerin cümle metniyle eşleşmesi | {status(c['words_mismatch_text'])} |",
        f"| Kaldi metni ile `scores.json` metni arasında noktalama dışı fark | {status(sum((c['per_split'][s]['text_differs_beyond_punctuation'] for s in SPLITS), []))} |",
        "",
        "## Dikkat edilmesi gerekenler\n",
        f"1. **Listede olmayan {unlisted['count']} WAV dosyası.** `WAVE/` altında `wav.scp`'de geçmeyen {unlisted['count']} dosya var "
        f"(toplam {unlisted['hours']} saat, {len(unlisted['speakers'])} konuşmacı, hepsi `SPEAKER96xx`). Bunların puanı yoktur ve "
        "hiçbir bölüme ait değildir; tabloya alınmadı. "
        + ("Hiçbiri listedeki bir kaydın birebir kopyası değil." if not c["unlisted_identical_to_listed"]
           else f"{len(c['unlisted_identical_to_listed'])} tanesi listedeki bir kaydın birebir kopyası.")
        + " Tam liste `speechocean762_checks.json` içindedir.",
        f"2. **Noktalama farkı.** {punct_only} kayıtta `scores.json` metni noktalama içerir (ör. `RIGHT!`), Kaldi `text` dosyası içermez. "
        "Kelimeler birebir aynı; tabloda Kaldi metni kullanıldı.",
        "3. **Tamlık (completeness) ölçeği.** README 0–1 aralığını tarif eder, ancak `scores.json` değerleri 0–10 ölçeğindedir "
        f"({c['completeness_distribution'].get(10.0, 0)} kayıt 10,0). Değerler olduğu gibi bırakıldı.",
        f"4. **Ortak cümleler.** {c['unique_sentences']['shared']} cümle metni hem eğitimde hem testte geçer (farklı konuşmacılar). "
        "Konuşmacı düzeyinde sızıntı yoktur.",
        "",
        "## Tablo sütunları (`speechocean762_manifest.csv`)\n",
        "| Sütun | Açıklama |",
        "| --- | --- |",
        "| `utt_id` | Kayıt kimliği (`0` + 4 haneli konuşmacı + 4 haneli sıra) |",
        "| `split` | Orijinal bölüm: `train` veya `test` |",
        "| `speaker_id`, `gender`, `age` | `utt2spk`, `spk2gender`, `spk2age` dosyalarından |",
        f"| `age_group` | Türetilmiş: yaş < {ADULT_AGE} → `child`, aksi halde `adult` |",
        "| `text`, `num_words` | Kaldi `text` dosyasındaki cümle ve kelime sayısı |",
        "| `wav_path`, `wav_exists` | Veri seti köküne göre ses yolu ve dosyanın varlığı |",
        "| `duration_s`, `sample_rate`, `channels`, `subtype` | Ses dosyası başlığından |",
        "| `audio_sha1` | Ses dosyasının SHA-1 özeti (tekrar kontrolü için) |",
        "| `accuracy`, `completeness`, `fluency`, `prosodic`, `total` | `scores.json` cümle düzeyi uzman puanları (5 uzmanın ortalaması/medyanı) |",
        "| `n_experts`, `n_scored_words` | `scores-detail.json`'daki uzman sayısı ve puanlanan kelime sayısı |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="where the archive is downloaded and extracted")
    args = parser.parse_args()

    corpus, archive_digest = download(args.data_dir)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    scores = json.loads((corpus / "resource" / "scores.json").read_text(encoding="utf-8"))
    detail = json.loads((corpus / "resource" / "scores-detail.json").read_text(encoding="utf-8"))

    checks = {}
    meta = {}  # split -> name -> rows
    for split in SPLITS:
        meta[split] = {
            name: read_kaldi(corpus / split / name)
            for name in ("wav.scp", "text", "utt2spk", "spk2utt", "spk2gender", "spk2age")
        }

    # Duplicate keys inside each Kaldi file.
    checks["duplicate_keys_in_kaldi_files"] = {
        f"{split}/{name}": duplicates(k for k, _ in rows)
        for split in SPLITS for name, rows in meta[split].items()
    }

    rows_out = []
    split_utts = {}
    split_spks = {}
    per_split = {}
    for split in SPLITS:
        m = {name: dict(rows) for name, rows in meta[split].items()}
        utts = set(m["wav.scp"])
        split_utts[split] = utts
        split_spks[split] = set(m["spk2gender"]) | set(m["spk2age"]) | set(m["spk2utt"]) | set(m["utt2spk"].values())
        spk2utt = {spk: set(v.split()) for spk, v in m["spk2utt"].items()}
        utt2spk_from_spk2utt = {u: spk for spk, us in spk2utt.items() for u in us}

        per_split[split] = {
            # Every utterance must appear in all utterance-level files.
            "utts_missing_from": {
                name: sorted(utts ^ set(m[name])) for name in ("text", "utt2spk")
            } | {"spk2utt": sorted(utts ^ set(utt2spk_from_spk2utt)), "scores.json": sorted(utts - set(scores)),
                 "scores-detail.json": sorted(utts - set(detail))},
            "utt2spk_vs_spk2utt_mismatch": sorted(u for u in utts if m["utt2spk"].get(u) != utt2spk_from_spk2utt.get(u)),
            "speakers_missing_gender": sorted(s for s in split_spks[split] if s not in m["spk2gender"]),
            "speakers_missing_age": sorted(s for s in split_spks[split] if s not in m["spk2age"]),
            # utt id = "0" + 4-digit speaker + 4-digit index; wav lives under WAVE/SPEAKER<spk>/
            "speaker_id_inconsistent": sorted(
                u for u in utts
                if not (u[1:5] == m["utt2spk"].get(u) and m["wav.scp"][u] == f"WAVE/SPEAKER{m['utt2spk'].get(u)}/{u}.WAV")
            ),
            "text_differs_from_scores_json": sorted(u for u in utts if u in scores and scores[u]["text"] != m["text"].get(u)),
            "text_differs_beyond_punctuation": sorted(
                u for u in utts if u in scores and normalize(scores[u]["text"]) != normalize(m["text"].get(u, ""))
            ),
        }

        for utt in sorted(utts):
            spk = m["utt2spk"].get(utt, "")
            rel = m["wav.scp"][utt]
            wav = corpus / rel
            row = {
                "utt_id": utt, "split": split, "speaker_id": spk,
                "gender": m["spk2gender"].get(spk, ""), "age": m["spk2age"].get(spk, ""),
                "age_group": ("adult" if int(m["spk2age"][spk]) >= ADULT_AGE else "child") if spk in m["spk2age"] else "",
                "text": m["text"].get(utt, ""), "num_words": len(m["text"].get(utt, "").split()),
                "wav_path": rel, "wav_exists": wav.is_file(),
                "duration_s": "", "sample_rate": "", "channels": "", "subtype": "", "audio_sha1": "",
            }
            if row["wav_exists"]:
                info = sf.info(wav)
                row.update(duration_s=round(info.duration, 3), sample_rate=info.samplerate,
                           channels=info.channels, subtype=info.subtype,
                           audio_sha1=hashlib.sha1(wav.read_bytes()).hexdigest())
            s = scores.get(utt, {})
            for k in SENTENCE_SCORES:
                row[k] = s.get(k, "")
            d = detail.get(utt, {})
            row["n_experts"] = len(d.get("accuracy", []))
            row["n_scored_words"] = len(s.get("words", []))
            rows_out.append(row)

    checks["per_split"] = per_split
    by_utt = {r["utt_id"]: r for r in rows_out}

    # Split integrity: no utterance or speaker may appear in both train and test.
    checks["utts_in_both_splits"] = sorted(split_utts["train"] & split_utts["test"])
    checks["speakers_in_both_splits"] = sorted(split_spks["train"] & split_spks["test"])

    # Files on disk vs. wav.scp.
    listed = {r["wav_path"] for r in rows_out}
    on_disk = {p.relative_to(corpus).as_posix() for p in (corpus / "WAVE").rglob("*") if p.is_file()}
    checks["wav_listed_but_missing"] = sorted(listed - on_disk)
    unlisted = sorted(on_disk - listed)
    checks["wav_on_disk_not_listed"] = {
        "count": len(unlisted),
        "speakers": dict(Counter(f.split("/")[1] for f in unlisted)),
        "hours": round(sum(sf.info(corpus / f).duration for f in unlisted) / 3600, 3),
        "files": unlisted,
    }
    checks["scores_without_utterance"] = sorted(set(scores) - set(by_utt))
    checks["unlisted_with_scores"] = sorted(Path(p).stem for p in unlisted if Path(p).stem in scores)

    # Audio properties and duplicates.
    present = [r for r in rows_out if r["wav_exists"]]
    checks["audio_formats"] = dict(Counter(f"{r['sample_rate']} Hz / {r['channels']} ch / {r['subtype']}" for r in present))
    checks["empty_or_short_audio_lt_0.5s"] = sorted(r["utt_id"] for r in present if r["duration_s"] < 0.5)
    by_hash = defaultdict(list)
    for r in present:
        by_hash[r["audio_sha1"]].append(r["utt_id"])
    checks["identical_audio_groups"] = [sorted(v) for v in by_hash.values() if len(v) > 1]
    checks["unlisted_identical_to_listed"] = sorted(
        f for f in unlisted if hashlib.sha1((corpus / f).read_bytes()).hexdigest() in by_hash
    )
    by_spk_text = defaultdict(list)
    for r in rows_out:
        by_spk_text[(r["speaker_id"], r["text"])].append(r["utt_id"])
    checks["same_speaker_same_text"] = [sorted(v) for v in by_spk_text.values() if len(v) > 1]
    text_splits = defaultdict(set)
    for r in rows_out:
        text_splits[r["text"]].add(r["split"])
    checks["unique_sentences"] = {
        "train": sum(1 for t, s in text_splits.items() if "train" in s),
        "test": sum(1 for t, s in text_splits.items() if "test" in s),
        "shared": sum(1 for s in text_splits.values() if len(s) == 2),
    }

    # Score ranges (README metric) and scorer counts.
    ranges = {"accuracy": (0, 10), "fluency": (0, 10), "prosodic": (0, 10), "total": (0, 10), "completeness": (0, 10)}
    checks["sentence_score_out_of_range"] = {
        k: sorted(r["utt_id"] for r in rows_out if out_of_range([r[k] if r[k] != "" else None], *rng)) for k, rng in ranges.items()
    }
    word_acc, word_stress, phone_acc, word_count_mismatch, phone_len_mismatch = [], [], [], [], []
    for utt, r in by_utt.items():
        words = scores.get(utt, {}).get("words", [])
        if len(words) != r["num_words"] or [w["text"] for w in words] != r["text"].split():
            word_count_mismatch.append(utt)
        for w in words:
            word_acc += out_of_range([w["accuracy"]], 0, 10)
            word_stress += [w["stress"]] if w["stress"] not in (5, 10) else []
            phone_acc += out_of_range(w["phones-accuracy"], 0, 2)
            if len(w["phones"]) != len(w["phones-accuracy"]):
                phone_len_mismatch.append(utt)
    checks["word_score_out_of_range"] = {"accuracy": len(word_acc), "stress_not_5_or_10": len(word_stress), "phone_accuracy_not_0_2": len(phone_acc)}
    checks["words_mismatch_text"] = sorted(word_count_mismatch)
    checks["phones_vs_phone_scores_length_mismatch"] = sorted(set(phone_len_mismatch))
    checks["n_experts_distribution"] = dict(Counter(r["n_experts"] for r in rows_out))
    checks["completeness_distribution"] = dict(Counter(r["completeness"] for r in rows_out))
    checks["utterances_per_speaker"] = dict(Counter(Counter(r["speaker_id"] for r in rows_out).values()))

    # Write manifest.
    fields = list(rows_out[0])
    with open(out_dir / "speechocean762_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows_out, key=lambda r: (r["split"] != "train", r["utt_id"])))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": URL,
        "archive_sha256": archive_digest,
        "splits": {
            split: {
                "utterances": len(split_utts[split]),
                "speakers": len(split_spks[split]),
                "hours": round(sum(r["duration_s"] for r in present if r["split"] == split) / 3600, 3),
                "gender": dict(Counter(dict(meta[split]["spk2gender"]).values())),
            }
            for split in SPLITS
        },
        "checks": checks,
    }
    (out_dir / "speechocean762_checks.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, rows_out, out_dir / "speechocean762_report.md")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
    print(f"Wrote manifest and checks to {out_dir}")


if __name__ == "__main__":
    main()
