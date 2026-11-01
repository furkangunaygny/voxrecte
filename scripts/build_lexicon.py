"""Build the pronunciation lexicon used by compare_pronunciation.py from speechocean762.

Every word's reference transcriptions (ARPAbet with stress) are counted over the whole corpus.
The lexicon is small and kept in the repository, so the comparison tool works on new recordings
without downloading the corpus.

Input:   data/speechocean762/resource/scores.json (prepare_speechocean762.py)
Output:  resources/lexicon.tsv   word, phones, count  (derived from speechocean762, CC BY 4.0)
"""

import argparse
from pathlib import Path

from voxrecte.lexicon import count_pronunciations, write_lexicon
from voxrecte.paths import DATA_DIR, LEXICON_PATH
from voxrecte.speechocean import load_scores


def main():
    """Count every word's reference pronunciations and write the lexicon."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    counts = count_pronunciations(load_scores(args.data_dir))
    write_lexicon(counts, LEXICON_PATH)
    print(f"Wrote {len({word for word, _ in counts})} words, {len(counts)} pronunciations to {LEXICON_PATH}")


if __name__ == "__main__":
    main()
