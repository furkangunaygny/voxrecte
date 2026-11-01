"""Step 2: download speechocean762, build a per-utterance manifest and run data checks.

Source:  OpenSLR SLR101 (https://www.openslr.org/101/), CC BY 4.0
Layout:  Kaldi-style train/ and test/ directories, WAVE/SPEAKERxxxx/*.WAV,
         resource/scores.json (aggregated) and resource/scores-detail.json (five experts)

Outputs (written to outputs/):
  speechocean762_manifest.csv   one row per utterance: split, speaker, text, audio, scores
  speechocean762_checks.json    split summary and missing-file, duplicate and consistency checks
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from voxrecte.files import write_dict_csv, write_json
from voxrecte.paths import DATA_DIR, MANIFEST_PATH, OUTPUTS_DIR, corpus_dir
from voxrecte.speechocean import SENTENCE_SCORES, SPLITS
from voxrecte.text import normalize_sentence

URL = "https://openslr.elda.org/resources/101/speechocean762.tar.gz"
ARCHIVE_SHA256 = "876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5"
KALDI_FILES = ("wav.scp", "text", "utt2spk", "spk2utt", "spk2gender", "spk2age")
ADULT_AGE = 18  # ages in the corpus are 6-15 or 19-43, so the split is unambiguous
MIN_DURATION_S = 0.5
SENTENCE_SCORE_RANGE = (0, 10)  # completeness too, although the corpus README describes it as 0-1
WORD_SCORE_RANGE = (0, 10)
PHONEME_SCORE_RANGE = (0, 2)
WORD_STRESS_VALUES = (5, 10)


@dataclass
class SplitFiles:
    """The Kaldi files of one split, as raw (key, value) rows and as dictionaries."""
    name: str
    raw: dict[str, list[tuple[str, str]]]

    def __post_init__(self):
        """Derive lookup maps, utterance ids and speaker ids from the raw rows."""
        self.maps = {file: dict(rows) for file, rows in self.raw.items()}
        self.utterances = set(self.maps["wav.scp"])
        self.speakers = (set(self.maps["spk2gender"]) | set(self.maps["spk2age"]) | set(self.maps["spk2utt"])
                         | set(self.maps["utt2spk"].values()))


# --- download and reading -------------------------------------------------------------------------

def sha256(path: Path) -> str:
    """SHA-256 of a (large) file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1(path: Path) -> str:
    """SHA-1 of a file's bytes, used to find identical recordings."""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def download_corpus(data_dir: Path) -> tuple[Path, str]:
    """Download (once), verify and extract the archive; returns (corpus directory, archive SHA-256)."""
    archive = data_dir / "downloads" / "speechocean762.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    digest = sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"{archive}: sha256 {digest} does not match {ARCHIVE_SHA256}")
    corpus = corpus_dir(data_dir)
    if not corpus.exists():
        print(f"Extracting to {data_dir}")
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
    return corpus, digest


def read_kaldi(path: Path) -> list[tuple[str, str]]:
    """[(key, value)] in file order, so duplicate keys stay visible."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if parts:
            rows.append((parts[0], parts[1].strip() if len(parts) > 1 else ""))
    return rows


def read_split(corpus: Path, split: str) -> SplitFiles:
    """Read the Kaldi files of one split."""
    return SplitFiles(split, {file: read_kaldi(corpus / split / file) for file in KALDI_FILES})


# --- manifest -------------------------------------------------------------------------------------

def manifest_rows(split: SplitFiles, corpus: Path, scores: dict, detail: dict) -> list[dict]:
    """One row per utterance listed in wav.scp, with speaker, audio and sentence-score columns."""
    m = split.maps
    rows = []
    for utt in sorted(split.utterances):
        speaker = m["utt2spk"].get(utt, "")
        wav_path = m["wav.scp"][utt]
        wav = corpus / wav_path
        text = m["text"].get(utt, "")
        row = {
            "utt_id": utt, "split": split.name, "speaker_id": speaker,
            "gender": m["spk2gender"].get(speaker, ""), "age": m["spk2age"].get(speaker, ""),
            "age_group": age_group(m["spk2age"].get(speaker)),
            "text": text, "num_words": len(text.split()),
            "wav_path": wav_path, "wav_exists": wav.is_file(),
            "duration_s": "", "sample_rate": "", "channels": "", "subtype": "", "audio_sha1": "",
        }
        if row["wav_exists"]:
            info = sf.info(wav)
            row.update(duration_s=round(info.duration, 3), sample_rate=info.samplerate,
                       channels=info.channels, subtype=info.subtype, audio_sha1=sha1(wav))
        utt_scores = scores.get(utt, {})
        row.update({k: utt_scores.get(k, "") for k in SENTENCE_SCORES})
        row["n_experts"] = len(detail.get(utt, {}).get("accuracy", []))
        row["n_scored_words"] = len(utt_scores.get("words", []))
        rows.append(row)
    return rows


def age_group(age: str | None) -> str:
    """'adult' or 'child' from the age; '' if the age is unknown."""
    if age is None:
        return ""
    return "adult" if int(age) >= ADULT_AGE else "child"


# --- checks ---------------------------------------------------------------------------------------

def duplicates(keys) -> list[str]:
    """Keys that occur more than once."""
    return sorted(k for k, n in Counter(keys).items() if n > 1)


def out_of_range(values, low, high) -> list:
    """Values that are missing or outside [low, high]."""
    return [v for v in values if v is None or not low <= v <= high]


def check_split_files(split: SplitFiles, scores: dict, detail: dict) -> dict:
    """Consistency of the Kaldi files of one split with each other and with the score files."""
    m, utts = split.maps, split.utterances
    spk2utt = {spk: set(v.split()) for spk, v in m["spk2utt"].items()}
    utt2spk_from_spk2utt = {u: spk for spk, us in spk2utt.items() for u in us}
    return {
        # every utterance must appear in all utterance-level files
        "utts_missing_from": {name: sorted(utts ^ set(m[name])) for name in ("text", "utt2spk")}
                             | {"spk2utt": sorted(utts ^ set(utt2spk_from_spk2utt)),
                                "scores.json": sorted(utts - set(scores)),
                                "scores-detail.json": sorted(utts - set(detail))},
        "utt2spk_vs_spk2utt_mismatch": sorted(u for u in utts if m["utt2spk"].get(u) != utt2spk_from_spk2utt.get(u)),
        "speakers_missing_gender": sorted(s for s in split.speakers if s not in m["spk2gender"]),
        "speakers_missing_age": sorted(s for s in split.speakers if s not in m["spk2age"]),
        # utt id = "0" + 4-digit speaker + 4-digit index; the wav lives under WAVE/SPEAKER<spk>/
        "speaker_id_inconsistent": sorted(
            u for u in utts
            if not (u[1:5] == m["utt2spk"].get(u) and m["wav.scp"][u] == f"WAVE/SPEAKER{m['utt2spk'].get(u)}/{u}.WAV")
        ),
        "text_differs_from_scores_json": sorted(
            u for u in utts if u in scores and scores[u]["text"] != m["text"].get(u)
        ),
        "text_differs_beyond_punctuation": sorted(
            u for u in utts
            if u in scores and normalize_sentence(scores[u]["text"]) != normalize_sentence(m["text"].get(u, ""))
        ),
    }


def check_disk_files(rows: list[dict], corpus: Path, scores: dict) -> tuple[dict, list[str]]:
    """Files listed in wav.scp vs. files on disk; returns (checks, unlisted files)."""
    listed = {r["wav_path"] for r in rows}
    on_disk = {p.relative_to(corpus).as_posix() for p in (corpus / "WAVE").rglob("*") if p.is_file()}
    unlisted = sorted(on_disk - listed)
    utt_ids = {r["utt_id"] for r in rows}
    checks = {
        "wav_listed_but_missing": sorted(listed - on_disk),
        "wav_on_disk_not_listed": {
            "count": len(unlisted),
            "speakers": dict(Counter(f.split("/")[1] for f in unlisted)),
            "hours": round(sum(sf.info(corpus / f).duration for f in unlisted) / 3600, 3),
            "files": unlisted,
        },
        "scores_without_utterance": sorted(set(scores) - utt_ids),
        "unlisted_with_scores": sorted(Path(p).stem for p in unlisted if Path(p).stem in scores),
    }
    return checks, unlisted


def check_audio(rows: list[dict], corpus: Path, unlisted: list[str]) -> dict:
    """Audio formats, too-short files and byte-identical recordings."""
    present = [r for r in rows if r["wav_exists"]]
    by_hash = defaultdict(list)
    for r in present:
        by_hash[r["audio_sha1"]].append(r["utt_id"])
    return {
        "audio_formats": dict(Counter(f"{r['sample_rate']} Hz / {r['channels']} ch / {r['subtype']}" for r in present)),
        "empty_or_short_audio_lt_0.5s": sorted(r["utt_id"] for r in present if r["duration_s"] < MIN_DURATION_S),
        "identical_audio_groups": [sorted(v) for v in by_hash.values() if len(v) > 1],
        "unlisted_identical_to_listed": sorted(f for f in unlisted if sha1(corpus / f) in by_hash),
    }


def check_texts(rows: list[dict]) -> dict:
    """Repeated readings by one speaker and sentences shared between the splits."""
    by_speaker_text = defaultdict(list)
    text_splits = defaultdict(set)
    for r in rows:
        by_speaker_text[(r["speaker_id"], r["text"])].append(r["utt_id"])
        text_splits[r["text"]].add(r["split"])
    return {
        "same_speaker_same_text": [sorted(v) for v in by_speaker_text.values() if len(v) > 1],
        "unique_sentences": {
            "train": sum("train" in s for s in text_splits.values()),
            "test": sum("test" in s for s in text_splits.values()),
            "shared": sum(len(s) == 2 for s in text_splits.values()),
        },
    }


def check_scores(rows: list[dict], scores: dict) -> dict:
    """Score ranges (README metric), word entries vs. sentence text, scorer counts."""
    word_acc, word_stress, phone_acc, words_mismatch, phone_len_mismatch = [], [], [], [], set()
    for r in rows:
        words = scores.get(r["utt_id"], {}).get("words", [])
        if len(words) != r["num_words"] or [w["text"] for w in words] != r["text"].split():
            words_mismatch.append(r["utt_id"])
        for w in words:
            word_acc += out_of_range([w["accuracy"]], *WORD_SCORE_RANGE)
            word_stress += [w["stress"]] if w["stress"] not in WORD_STRESS_VALUES else []
            phone_acc += out_of_range(w["phones-accuracy"], *PHONEME_SCORE_RANGE)
            if len(w["phones"]) != len(w["phones-accuracy"]):
                phone_len_mismatch.add(r["utt_id"])
    return {
        "sentence_score_out_of_range": {
            k: sorted(r["utt_id"] for r in rows if out_of_range([r[k] if r[k] != "" else None], *SENTENCE_SCORE_RANGE))
            for k in ("accuracy", "fluency", "prosodic", "total", "completeness")
        },
        "word_score_out_of_range": {"accuracy": len(word_acc), "stress_not_5_or_10": len(word_stress),
                                    "phone_accuracy_not_0_2": len(phone_acc)},
        "words_mismatch_text": sorted(words_mismatch),
        "phones_vs_phone_scores_length_mismatch": sorted(phone_len_mismatch),
        "n_experts_distribution": dict(Counter(r["n_experts"] for r in rows)),
        "completeness_distribution": dict(Counter(r["completeness"] for r in rows)),
        "utterances_per_speaker": dict(Counter(Counter(r["speaker_id"] for r in rows).values())),
    }


def run_checks(splits: list[SplitFiles], rows: list[dict], corpus: Path, scores: dict, detail: dict) -> dict:
    """All corpus checks, in the order they appear in the JSON output."""
    train, test = splits
    disk, unlisted = check_disk_files(rows, corpus, scores)
    return {
        "duplicate_keys_in_kaldi_files": {f"{s.name}/{file}": duplicates(k for k, _ in raw)
                                          for s in splits for file, raw in s.raw.items()},
        "per_split": {s.name: check_split_files(s, scores, detail) for s in splits},
        # split integrity: no utterance or speaker may appear in both train and test
        "utts_in_both_splits": sorted(train.utterances & test.utterances),
        "speakers_in_both_splits": sorted(train.speakers & test.speakers),
        **disk,
        **check_audio(rows, corpus, unlisted),
        **check_texts(rows),
        **check_scores(rows, scores),
    }


def split_summary(split: SplitFiles, rows: list[dict]) -> dict:
    """Recordings, speakers, hours and gender counts of one split."""
    present = [r for r in rows if r["wav_exists"] and r["split"] == split.name]
    return {
        "utterances": len(split.utterances),
        "speakers": len(split.speakers),
        "hours": round(sum(r["duration_s"] for r in present) / 3600, 3),
        "gender": dict(Counter(dict(split.raw["spk2gender"]).values())),
    }


# --- main -----------------------------------------------------------------------------------------

def main():
    """Download and verify the corpus, build the manifest, run the checks and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="where the archive is downloaded and extracted")
    args = parser.parse_args()

    corpus, archive_digest = download_corpus(args.data_dir)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    scores = json.loads((corpus / "resource" / "scores.json").read_text(encoding="utf-8"))
    detail = json.loads((corpus / "resource" / "scores-detail.json").read_text(encoding="utf-8"))

    splits = [read_split(corpus, split) for split in SPLITS]
    rows = [row for split in splits for row in manifest_rows(split, corpus, scores, detail)]
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": URL,
        "archive_sha256": archive_digest,
        "splits": {split.name: split_summary(split, rows) for split in splits},
        "checks": run_checks(splits, rows, corpus, scores, detail),
    }

    write_dict_csv(MANIFEST_PATH, sorted(rows, key=lambda r: (r["split"] != "train", r["utt_id"])))
    write_json(OUTPUTS_DIR / "speechocean762_checks.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
    print(f"Wrote manifest and checks to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
