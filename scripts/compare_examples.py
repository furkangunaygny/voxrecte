"""Run compare_pronunciation on a few speechocean762 training recordings and set the result next to
the experts' phoneme marks, as a demonstration of step 5.

Input:   data/speechocean762/ (prepare_speechocean762.py), resources/lexicon.tsv
Output:  outputs/compare_examples.md
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from compare_pronunciation import compare, format_result, load_lexicon, target_words
from recognizer import PhonemeRecognizer, load_audio

ROOT = Path(__file__).resolve().parent.parent
# Training recordings (test recordings are kept for step 6): two the experts scored as fully
# correct, three with phonemes the experts marked as wrong or accented.
EXAMPLES = ["000360013", "001350219", "005940234", "007390062", "026210157"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    corpus = args.data_dir / "speechocean762"
    scores = json.loads((corpus / "resource" / "scores.json").read_text(encoding="utf-8"))
    with open(ROOT / "outputs" / "speechocean762_manifest.csv", encoding="utf-8") as f:
        manifest = {r["utt_id"]: r for r in csv.DictReader(f)}
    lexicon = load_lexicon()
    recognize = PhonemeRecognizer(cpu=args.cpu)

    lines = [
        "# Adım 5 — Telaffuz karşılaştırma örnekleri\n",
        "`scripts/compare_pronunciation.py` aracının speechocean762 train kayıtlarındaki çıktısı. Her örnekte "
        "önce uzmanların işaretlediği fonemler (fonem puanı 2'nin altında: 0 yanlış/eksik, 1 aksanlı), ardından "
        "aracın çıktısı verilir. Araç uzman puanlarını kullanmaz; yalnızca ses dosyası ve hedef metinle çalışır.\n",
        "Bu örnekler yalnızca gösterim içindir; aracın başarısı Adım 6'da test kayıtlarında ölçülecek.\n",
    ]
    for utt_id in EXAMPLES:
        row = manifest[utt_id]
        words, variants = target_words(row["text"], lexicon)
        result = compare(recognize(load_audio(corpus / row["wav_path"])), words, variants)
        marked = [f"{w['text']}: " + " ".join(f"{p} ({a:g})" for p, a in zip(w["phones"], w["phones-accuracy"]) if a < 2)
                  for w in scores[utt_id]["words"] if min(w["phones-accuracy"]) < 2]
        lines += [
            f"## {utt_id} — \"{row['text']}\"\n",
            f"Konuşmacı {row['speaker_id']} ({row['gender']}, {row['age']} yaş) · uzman puanları: doğruluk "
            f"{row['accuracy']}, total {row['total']} (0–10)\n",
            "**Uzmanların işaretlediği fonemler:** " + ("; ".join(marked) if marked else "yok (tüm fonemler doğru)") + "\n",
            "```text",
            format_result(row["text"], result),
            "```\n",
        ]
        print(f"{utt_id}: {result['summary']}")
    (ROOT / "outputs" / "compare_examples.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
