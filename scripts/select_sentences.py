"""Select a small set of reading sentences from speechocean762 that covers the English phonemes.

1. Sentences are deduplicated after upper-casing and removing punctuation.
2. Phonemes come from the corpus' own reference transcriptions (resource/scores.json,
   ARPAbet). Stress digits are dropped for counting, leaving the 39 CMU phonemes.
   The reference is written per recording, so a sentence read twice can have two
   different transcriptions (e.g. "FOR" as F AO or F AO R). A phoneme then counts for
   the sentence only as often as it occurs in every transcription.
3. Greedy selection. At each step the sentence with the best score is added, compared in order:
     a. number of phonemes not yet covered by any selected sentence
     b. progress towards every phoneme occurring in at least K selected sentences
     c. diminishing-returns gain: each phoneme adds weight / (1 + sentences already containing it),
        with weight higher for phonemes that are rare in the corpus; then fewer words, then text
   The first --core sentences use K = --core-k; the next --backup sentences use K = --core-k + 1,
   so the backups strengthen the phonemes that occur in the fewest core sentences.
4. Sentences listed in resources/excluded_sentences.csv (broken grammar or sensitive content,
   reviewed by hand because the sentences will be read by study participants) are not candidates.
5. A pilot subset of --pilot core sentences is picked with the same greedy rule (K = 1).

Input:   outputs/speechocean762_manifest.csv, data/speechocean762/resource/scores.json,
         resources/excluded_sentences.csv
Outputs (written to outputs/):
  sentence_selection.csv   selected sentences, phonemes and source recording IDs
  phoneme_coverage.csv     per-phoneme counts in the corpus and in the selection
  sentence_selection.md    report
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The 39 phonemes of the CMU Pronouncing Dictionary (ARPAbet, without stress).
PHONEMES = (
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW "
    "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH"
).split()
VOWELS = set(PHONEMES[:15])


def normalize(text):
    return " ".join(re.sub(r"[^A-Z' ]", " ", text.upper()).split())


def strip_stress(phone):
    return re.sub(r"\d", "", phone)


def load_sentences(manifest, scores):
    """Group recordings by normalized sentence text."""
    sentences = defaultdict(lambda: {"utt_ids": [], "splits": [], "totals": [], "transcriptions": []})
    for r in manifest:
        s = sentences[normalize(r["text"])]
        words = scores[r["utt_id"]]["words"]
        s["utt_ids"].append(r["utt_id"])
        s["splits"].append(r["split"])
        s["totals"].append(float(r["total"]))
        s["transcriptions"].append(" | ".join(" ".join(strip_stress(p) for p in w["phones"]) for w in words))
    for text, s in sentences.items():
        counts = [Counter(t.replace("|", " ").split()) for t in s["transcriptions"]]
        s["text"] = text
        s["num_words"] = len(text.split())
        s["phonemes"] = Counter({p: min(c[p] for c in counts) for p in counts[0]})
        s["phonemes"] = +s["phonemes"]  # drop zero counts
        s["variants"] = sorted(set(s["transcriptions"]))
    return sentences


def select(candidates, n, k, selected, weight, tag="new_phonemes"):
    """Greedily add n sentences to `selected`; the phonemes each one newly covers are stored under `tag`."""
    in_sentences = Counter(p for s in selected for p in s["phonemes"])
    chosen = {s["text"] for s in selected}
    for _ in range(n):
        def score(s):
            phones = s["phonemes"]
            new = sum(1 for p in phones if in_sentences[p] == 0)
            progress = sum(1 for p in phones if in_sentences[p] < k)
            balance = sum(weight[p] / (1 + in_sentences[p]) for p in phones)
            return (new, progress, balance, -s["num_words"], s["text"])

        best = max((s for s in candidates if s["text"] not in chosen), key=score)
        best[tag] = sorted(p for p in best["phonemes"] if in_sentences[p] == 0)
        selected.append(best)
        chosen.add(best["text"])
        in_sentences.update(best["phonemes"].keys())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--core", type=int, default=20)
    parser.add_argument("--backup", type=int, default=5)
    parser.add_argument("--pilot", type=int, default=5)
    parser.add_argument("--core-k", type=int, default=3, help="target number of core sentences per phoneme")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=10)
    args = parser.parse_args()

    out_dir = ROOT / "outputs"
    with open(out_dir / "speechocean762_manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    scores = json.loads((args.data_dir / "speechocean762" / "resource" / "scores.json").read_text(encoding="utf-8"))

    sentences = load_sentences(manifest, scores)
    unknown = {p for s in sentences.values() for p in s["phonemes"]} - set(PHONEMES)
    if unknown:
        raise ValueError(f"phonemes outside the 39-phoneme set: {sorted(unknown)}")

    corpus_tokens = Counter()  # over unique sentences, so repeated sentences are not counted twice
    corpus_sentences = Counter()
    for s in sentences.values():
        corpus_tokens.update(s["phonemes"])
        corpus_sentences.update(s["phonemes"].keys())
    weight = {p: 1 / math.log(1 + corpus_sentences[p]) for p in PHONEMES}

    with open(ROOT / "resources" / "excluded_sentences.csv", encoding="utf-8") as f:
        excluded = {normalize(r["text"]): r["reason"] for r in csv.DictReader(f)}
    missing = set(excluded) - set(sentences)
    if missing:
        raise ValueError(f"excluded sentences not found in the corpus: {sorted(missing)}")

    candidates = sorted(
        (s for s in sentences.values()
         if args.min_words <= s["num_words"] <= args.max_words and s["text"] not in excluded),
        key=lambda s: s["text"],
    )
    selected = select(candidates, args.core, args.core_k, [], weight)
    select(candidates, args.backup, args.core_k + 1, selected, weight)
    core, backup = selected[:args.core], selected[args.core:]
    pilot = select(core, args.pilot, 1, [], weight, tag="pilot_new_phonemes")
    pilot_rank = {s["text"]: i for i, s in enumerate(pilot, 1)}

    def tally(group):
        tokens, sents = Counter(), Counter()
        for s in group:
            tokens.update(s["phonemes"])
            sents.update(s["phonemes"].keys())
        return tokens, sents

    core_tokens, core_sents = tally(core)
    all_tokens, all_sents = tally(selected)
    _, pilot_sents = tally(pilot)

    with open(out_dir / "sentence_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "role", "pilot_rank", "text", "num_words", "num_phonemes", "distinct_phonemes",
                    "new_phonemes", "transcription", "utt_ids", "splits", "mean_total_score"])
        for i, s in enumerate(selected, 1):
            w.writerow([
                i, "core" if i <= args.core else "backup", pilot_rank.get(s["text"], ""), s["text"], s["num_words"],
                sum(s["phonemes"].values()), len(s["phonemes"]), " ".join(s["new_phonemes"]),
                " || ".join(s["variants"]), " ".join(s["utt_ids"]), " ".join(s["splits"]),
                round(sum(s["totals"]) / len(s["totals"]), 1),
            ])

    total_corpus, total_core = sum(corpus_tokens.values()), sum(core_tokens.values())
    coverage_rows = []
    for p in sorted(PHONEMES, key=lambda p: (-corpus_tokens[p], p)):
        coverage_rows.append({
            "phoneme": p,
            "type": "vowel" if p in VOWELS else "consonant",
            "corpus_count": corpus_tokens[p],
            "corpus_pct": round(100 * corpus_tokens[p] / total_corpus, 2),
            "corpus_sentences": corpus_sentences[p],
            "core_count": core_tokens[p],
            "core_pct": round(100 * core_tokens[p] / total_core, 2),
            "core_sentences": core_sents[p],
            "core_backup_count": all_tokens[p],
            "core_backup_sentences": all_sents[p],
            "sentences": " ".join(str(i) for i, s in enumerate(selected, 1) if p in s["phonemes"]),
        })
    with open(out_dir / "phoneme_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_rows[0]))
        w.writeheader()
        w.writerows(coverage_rows)

    covered = sum(1 for p in PHONEMES if core_sents[p])
    weakest = min(PHONEMES, key=lambda p: (core_sents[p], p))
    multi = sum(1 for s in sentences.values() if len(s["variants"]) > 1)
    lines = [
        "# Okutulacak cümlelerin seçimi\n",
        f"Kaynak: speechocean762, {len(manifest)} kayıt → {len(sentences)} tekrarsız cümle "
        f"({len(manifest) - len(sentences)} tekrar çıkarıldı). Aday havuzu: {args.min_words}–{args.max_words} kelimelik, "
        f"elle çıkarılan {len(excluded)} cümle hariç {len(candidates)} cümle.\n",
        "Fonemler veri setinin kendi referans transkripsiyonlarından (ARPAbet) alındı; vurgu rakamları "
        f"sayım için atıldı ve CMU sözlüğünün 39 fonemi kaldı. Aynı cümlenin iki kaydında farklı transkripsiyon "
        f"olan {multi} cümlede bir fonem, yalnızca tüm transkripsiyonlarda geçtiği kadar sayıldı.\n",
        f"Seçim açgözlü algoritmayla yapıldı: önce henüz kapsanmayan fonemler, ardından her fonemin en az "
        f"{args.core_k} temel cümlede geçmesi, ardından seçimde az geçen ve veri setinde nadir olan fonemlere "
        "öncelik veren azalan getirili bir puan; eşitlikte daha kısa cümle. "
        f"Yedek cümleler aynı yöntemle hedef {args.core_k + 1} cümleye çıkarılarak seçildi.\n",
        "## Özet\n",
        f"- Temel {len(core)} cümle 39 fonemin **{covered}**'unu kapsıyor.",
        f"- En az geçen fonem: **{weakest}**, {core_sents[weakest]} temel cümlede.",
        f"- Her fonem en az {min(core_sents[p] for p in PHONEMES)} temel cümlede, yedeklerle birlikte en az "
        f"{min(all_sents[p] for p in PHONEMES)} cümlede geçiyor.",
        f"- Temel set: {total_core} fonem, {sum(s['num_words'] for s in core)} kelime.",
        f"- Pilot ({len(pilot)} temel cümle): 39 fonemin {len(pilot_sents)}'unu kapsıyor"
        + (f"; eksik: {' '.join(sorted(set(PHONEMES) - set(pilot_sents)))}." if len(pilot_sents) < 39 else ".")
        + "\n",
        "## Cümleler\n",
        "| # | Rol | Pilot | Cümle | Kelime | Yeni kapsanan fonemler | Kaynak kayıtlar | Ort. total puanı |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in enumerate(selected, 1):
        role = "temel" if i <= args.core else "yedek"
        sources = ", ".join(f"{u} ({sp})" for u, sp in zip(s["utt_ids"], s["splits"]))
        lines.append(f"| {i} | {role} | {pilot_rank.get(s['text'], '')} | {s['text']} | {s['num_words']} | "
                     f"{' '.join(s['new_phonemes']) or '—'} | {sources} | {sum(s['totals']) / len(s['totals']):.1f} |")
    lines += [
        "\n## Elle çıkarılan cümleler\n",
        "Cümleler çalışma katılımcılarına okutulacağı için dilbilgisi bozuk ya da hassas içerikli (şiddet, ölüm, "
        "siyaset) olanlar `resources/excluded_sentences.csv` listesiyle aday havuzundan çıkarıldı.\n",
        "| Cümle | Gerekçe |",
        "| --- | --- |",
    ] + [f"| {t} | {r} |" for t, r in excluded.items()]
    lines += [
        "\n## Fonem kapsama tablosu\n",
        "Sıralama: tüm veri setindeki sıklık. Oranlar fonem sayısına göre; \"cümle\" sütunları fonemin kaç cümlede "
        "geçtiğini gösterir.\n",
        "| Fonem | Tür | Veri seti (adet) | Veri seti % | Temel (adet) | Temel % | Temel cümle | "
        "Temel+yedek cümle | Geçtiği cümleler (#) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in coverage_rows:
        kind = "ünlü" if r["type"] == "vowel" else "ünsüz"
        lines.append(f"| {r['phoneme']} | {kind} | {r['corpus_count']} | {r['corpus_pct']:.2f} | {r['core_count']} | "
                     f"{r['core_pct']:.2f} | {r['core_sentences']} | {r['core_backup_sentences']} | {r['sentences']} |")
    lines += [
        "\n## Not: referans aksan\n",
        "Referans transkripsiyonlar kayıt bazında yazılmış ve tek bir aksana bağlı değil. Örneğin \"FOR\" 278 "
        "kayıtta `F AO` (r'siz, İngiliz), 40 kayıtta `F AO R` (Amerikan) olarak geçiyor. Bu nedenle özellikle "
        "R ve ER fonemlerinin sayıları, cümleyi okuyacak kişinin aksanına göre değişebilir.",
    ]
    (out_dir / "sentence_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(sentences)} unique sentences, {len(candidates)} candidates; "
          f"core covers {covered}/39, min core sentences per phoneme {min(core_sents[p] for p in PHONEMES)}")
    print(f"pilot covers {len(pilot_sents)}/39")
    for i, s in enumerate(selected, 1):
        print(f"{i:2d} {'core  ' if i <= args.core else 'backup'} p{pilot_rank.get(s['text'], '-')} {s['text']}")


if __name__ == "__main__":
    main()
