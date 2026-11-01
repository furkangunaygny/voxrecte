"""Run the pretrained wav2vec 2.0 phoneme model on every speechocean762 recording.

The output (greedy CTC decoding: space-separated eSpeak IPA tokens, no word boundaries) is used
by check_phoneme_map.py (step 4) and the evaluation (step 6).

Input:   outputs/speechocean762_manifest.csv and data/speechocean762/ (prepare_speechocean762.py)
Output:  data/predictions/speechocean762_ipa.csv  (utt_id, split, ipa); data/ is not tracked by git
"""

import argparse
import csv
import time
from pathlib import Path

from voxrecte.paths import DATA_DIR, corpus_dir, predictions_path
from voxrecte.recognizer import PhonemeRecognizer, load_audio
from voxrecte.speechocean import read_manifest

PROGRESS_EVERY = 500


def main():
    """Transcribe every recording in manifest order and write the model output."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    rows = read_manifest()
    out_path = predictions_path(args.data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer = PhonemeRecognizer(cpu=args.cpu)

    start = time.perf_counter()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "split", "ipa"])
        for done, row in enumerate(rows, 1):
            audio = load_audio(corpus_dir(args.data_dir) / row["wav_path"])
            writer.writerow([row["utt_id"], row["split"], recognizer.transcribe(audio)])
            if done % PROGRESS_EVERY == 0 or done == len(rows):
                print(f"{done}/{len(rows)} utterances, {time.perf_counter() - start:.0f}s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
