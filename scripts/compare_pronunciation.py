"""Compare a recording with its target sentence at phoneme level and show possible pronunciation errors.

    python scripts/compare_pronunciation.py kayit.wav "DO YOU HAVE ANY QUESTIONS FOR HER"

1. The target sentence is converted to phonemes with resources/lexicon.tsv (most frequent
   speechocean762 transcription of each word; other variants are accepted under rule R8).
2. The recording is passed through the pretrained wav2vec 2.0 phoneme model.
3. Both sequences are converted to the common phoneme set (phoneme_map.py) and aligned
   (alignment.py). Each difference is a substitution, deletion or insertion.
4. Differences covered by an acceptable-variant rule are listed separately; all others are
   reported as "possible errors" (olası hata). They are not final error decisions: the model
   itself makes mistakes, and its accuracy is measured in step 6.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from alignment import align
from phoneme_map import RULES, build_reference, ipa_token_to_common, stress_free

ROOT = Path(__file__).resolve().parent.parent
OP_NAMES = {"substitution": "değiştirme", "deletion": "silme", "insertion": "ekleme"}


def normalize_text(text):
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def load_lexicon(path=ROOT / "resources" / "lexicon.tsv"):
    """word -> [phones string, ...], most frequent first."""
    lexicon = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            word, phones, _ = line.rstrip("\n").split("\t")
            lexicon[word].append(phones)
    return lexicon


def target_words(text, lexicon):
    """Sentence -> (word entries with target phones, R8 variants). Raises KeyError for unknown words."""
    tokens = normalize_text(text)
    unknown = [t for t in tokens if t not in lexicon]
    if unknown:
        raise KeyError(f"not in the lexicon: {' '.join(unknown)}")
    words = [{"text": t, "phones": lexicon[t][0].split()} for t in tokens]
    variants = {t: {stress_free(p.split()) for p in lexicon[t]} for t in tokens}
    return words, variants


def compare(ipa, words, variants=None):
    """Align model output `ipa` with the target `words`; returns a JSON-serialisable result."""
    hyp, hyp_ipa = [], []
    for token in ipa.split():
        for p in ipa_token_to_common(token):
            hyp.append(p)
            hyp_ipa.append(token)
    ref, ref_word = build_reference(words, variants)

    per_word = [{"word": w["text"], "target": stress_free(w["phones"]), "heard": [], "heard_ipa": [],
                 "possible_errors": [], "accepted": []} for w in words]
    w_i = 0
    for op, ri, hi, rule in align(ref, hyp):
        if ri is not None:
            w_i = ref_word[ri]
        entry = per_word[w_i]
        if hi is not None:
            entry["heard"].append(hyp[hi])
            if not entry["heard_ipa"] or entry["heard_ipa"][-1] != hyp_ipa[hi]:
                entry["heard_ipa"].append(hyp_ipa[hi])
        if op == "match":
            continue
        kind = "substitution" if ri is not None and hi is not None else "deletion" if hi is None else "insertion"
        diff = {
            "type": OP_NAMES[kind],
            "expected": ref[ri][0] if ri is not None else None,
            "heard": hyp[hi] if hi is not None else None,
            "heard_ipa": hyp_ipa[hi] if hi is not None else None,
        }
        if op == "accepted":
            entry["accepted"].append({**diff, "rule": rule})
        else:
            entry["possible_errors"].append(diff)

    for entry in per_word:
        entry["target"] = list(entry["target"])
        entry["status"] = ("olası hata" if entry["possible_errors"]
                           else "kabul edilebilir fark" if entry["accepted"] else "doğru")
    n_errors = sum(len(e["possible_errors"]) for e in per_word)
    return {
        "model_ipa": ipa,
        "words": per_word,
        "summary": {
            "target_phonemes": len(ref),
            "possible_errors": n_errors,
            "accepted_differences": sum(len(e["accepted"]) for e in per_word),
            "words_with_possible_errors": sum(e["status"] == "olası hata" for e in per_word),
            "match_rate": round(1 - n_errors / len(ref), 3) if ref else None,
        },
    }


def describe(diff):
    heard = f"{diff['heard']} (model: {diff['heard_ipa']})" if diff["heard"] else "—"
    if diff["type"] == "değiştirme":
        return f"{diff['expected']} yerine {heard}"
    if diff["type"] == "silme":
        return f"{diff['expected']} duyulmadı"
    return f"fazladan {heard}"


def format_result(text, result):
    s = result["summary"]
    lines = [
        f"Hedef cümle : {' '.join(normalize_text(text))}",
        f"Model (IPA) : {result['model_ipa']}",
        f"Özet        : {s['target_phonemes']} hedef fonem, {s['possible_errors']} olası hata, "
        f"{s['accepted_differences']} kabul edilebilir fark, "
        f"{s['words_with_possible_errors']}/{len(result['words'])} kelimede olası hata",
        "",
        f"{'Kelime':<14} {'Durum':<22} {'Hedef':<22} Duyulan",
    ]
    for e in result["words"]:
        lines.append(f"{e['word']:<14} {e['status']:<22} {' '.join(e['target']):<22} {' '.join(e['heard']) or '—'}")
        for d in e["possible_errors"]:
            lines.append(f"{'':<14}   olası hata ({d['type']}): {describe(d)}")
        for d in e["accepted"]:
            lines.append(f"{'':<14}   kabul edilebilir ({d['rule']}): {describe(d)}")
    used = sorted({d["rule"] for e in result["words"] for d in e["accepted"]})
    if used:
        lines += ["", "Kurallar:"] + [f"  {r}: {RULES[r]}" for r in used]
    lines += ["", "Not: \"olası hata\" kesin bir hata kararı değildir; modelin kendi hataları da bu listede yer alabilir."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="ses dosyası (WAV/FLAC; 16 kHz değilse dönüştürülür)")
    parser.add_argument("text", help="okunması gereken hedef cümle")
    parser.add_argument("--json", type=Path, help="sonucu JSON olarak da bu dosyaya yaz")
    parser.add_argument("--cpu", action="store_true", help="GPU olsa da CPU kullan")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        words, variants = target_words(args.text, load_lexicon())
    except KeyError as e:
        sys.exit(f"Hata: hedef cümledeki bazı kelimeler sözlükte yok ({e.args[0]}).")

    from recognizer import PhonemeRecognizer, load_audio  # imported late: loading torch is slow
    ipa = PhonemeRecognizer(cpu=args.cpu)(load_audio(args.audio))
    result = compare(ipa, words, variants)
    print(format_result(args.text, result))
    if args.json:
        payload = {"audio": str(args.audio), "text": args.text, **result}
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
