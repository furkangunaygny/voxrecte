"""Step 5: compare a recording with its target sentence and show possible pronunciation errors.

    python scripts/compare_pronunciation.py kayit.wav "DO YOU HAVE ANY QUESTIONS FOR HER"

1. The target sentence is converted to phonemes with resources/lexicon.tsv (most frequent
   speechocean762 transcription of each word; other variants are accepted under rule R8).
2. The recording is passed through the pretrained wav2vec 2.0 phoneme model.
3. Both sequences are converted to the common phoneme set and aligned. Each difference is a
   substitution, deletion or insertion.
4. Differences covered by an acceptable-variant rule are listed separately; all others are
   reported as possible errors ("olası hata"), not final error decisions.
"""

import argparse
import sys
from pathlib import Path

from voxrecte.comparison import compare, format_result
from voxrecte.files import write_json
from voxrecte.lexicon import UnknownWordsError, load_lexicon, target_words


def parse_args():
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="audio file (WAV/FLAC; resampled to 16 kHz if needed)")
    parser.add_argument("text", help="target sentence the speaker read")
    parser.add_argument("--json", type=Path, help="also write the result as JSON to this file")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    return parser.parse_args()


def main():
    """Convert the target sentence, transcribe the recording, compare and print the result."""
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # IPA symbols on Windows consoles

    try:
        words, variants = target_words(args.text, load_lexicon())
    except UnknownWordsError as e:
        sys.exit(f"Hata: hedef cümledeki bazı kelimeler sözlükte yok ({' '.join(e.words)}).")

    from voxrecte.recognizer import PhonemeRecognizer, load_audio  # imported late: loading torch is slow
    ipa = PhonemeRecognizer(cpu=args.cpu).transcribe(load_audio(args.audio))
    result = compare(ipa, words, variants)
    print(format_result(args.text, result))
    if args.json:
        write_json(args.json, {"audio": str(args.audio), "text": args.text, **result})


if __name__ == "__main__":
    main()
