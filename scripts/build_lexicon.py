"""Build a pronunciation lexicon from the speechocean762 reference transcriptions.

Every word's reference transcriptions (ARPAbet with stress) are counted over the whole corpus.
compare_pronunciation.py uses the most frequent one as the target pronunciation and accepts the
other same-length variants (rule R8). The lexicon is small and kept in the repository, so the
comparison tool works on new recordings without downloading the corpus.

Input:   data/speechocean762/resource/scores.json (prepare_speechocean762.py)
Output:  resources/lexicon.tsv   word, phones, count  (derived from speechocean762, CC BY 4.0)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    scores = json.loads((args.data_dir / "speechocean762" / "resource" / "scores.json").read_text(encoding="utf-8"))
    counts = Counter((w["text"].upper(), " ".join(w["phones"])) for v in scores.values() for w in v["words"])

    out = ROOT / "resources" / "lexicon.tsv"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("word\tphones\tcount\n")
        for (word, phones), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
            f.write(f"{word}\t{phones}\t{n}\n")
    print(f"Wrote {len({w for w, _ in counts})} words, {len(counts)} pronunciations to {out}")


if __name__ == "__main__":
    main()
