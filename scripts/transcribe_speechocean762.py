"""Run the pretrained wav2vec 2.0 phoneme model on every speechocean762 recording.

The model output is greedy CTC decoding: space-separated eSpeak IPA tokens, no word boundaries.

Input:   outputs/speechocean762_manifest.csv and data/speechocean762/ (prepare_speechocean762.py)
Output:  data/predictions/speechocean762_ipa.csv  (utt_id, split, ipa); data/ is not tracked by git
"""

import argparse
import csv
import time
from pathlib import Path

from recognizer import PhonemeRecognizer, load_audio

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    corpus_dir = args.data_dir / "speechocean762"
    out_path = args.data_dir / "predictions" / "speechocean762_ipa.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "outputs" / "speechocean762_manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    recognize = PhonemeRecognizer(cpu=args.cpu)
    start = time.perf_counter()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "split", "ipa"])
        for i, row in enumerate(rows):
            w.writerow([row["utt_id"], row["split"], recognize(load_audio(corpus_dir / row["wav_path"]))])
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                print(f"{i + 1}/{len(rows)} utterances, {time.perf_counter() - start:.0f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
