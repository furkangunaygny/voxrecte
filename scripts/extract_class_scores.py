"""Save the phoneme model's frame-level class scores for every speechocean762 recording.

The model's log posteriors over its 392 output symbols are pooled into the phoneme classes of
voxrecte/gop.py (blank, 42 common phonemes, other) and stored as float16 per recording. The
error classifier (train_error_classifier.py) reads them; the model itself is not changed.

Input:   outputs/speechocean762_manifest.csv and data/speechocean762/ (prepare_speechocean762.py)
Output:  data/posteriors/speechocean762_class_scores.npz  (one (frames, classes) array per utt_id)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from voxrecte.gop import class_log_scores, class_membership
from voxrecte.paths import DATA_DIR, class_scores_path, corpus_dir
from voxrecte.recognizer import PhonemeRecognizer, load_audio
from voxrecte.speechocean import read_manifest

PROGRESS_EVERY = 500


def main():
    """Run the model on every recording in manifest order and store the class scores."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    rows = read_manifest()
    recognizer = PhonemeRecognizer(cpu=args.cpu)
    membership = class_membership(recognizer.tokens, recognizer.special_ids)

    scores, start = {}, time.perf_counter()
    for done, row in enumerate(rows, 1):
        audio = load_audio(corpus_dir(args.data_dir) / row["wav_path"])
        scores[row["utt_id"]] = class_log_scores(recognizer.log_posteriors(audio), membership).astype(np.float16)
        if done % PROGRESS_EVERY == 0 or done == len(rows):
            print(f"{done}/{len(rows)} utterances, {time.perf_counter() - start:.0f}s")

    out_path = class_scores_path(args.data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **scores)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
