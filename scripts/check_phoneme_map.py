"""Check the ARPAbet <-> eSpeak IPA mapping on real speechocean762 recordings and write the mapping table.

Only the training split is used, so that the rules are not tuned on the test recordings used later
for evaluation. Checks:
  1. every model token seen on train: how it is mapped (table entry, reduced, or unknown);
  2. on "clean" recordings, where the experts scored every reference phoneme as correct (2.0),
     the model output should match the reference apart from model errors and acceptable variants:
     the phoneme error rate is measured without and with the acceptable-variant rules;
  3. word examples for each rule, and the most frequent remaining differences.

Input:   data/predictions/speechocean762_ipa.csv (transcribe_speechocean762.py),
         data/speechocean762/resource/scores.json, outputs/speechocean762_manifest.csv
Outputs (written to outputs/):
  phoneme_mapping.csv         symbol mapping table (model IPA token -> common phonemes)
  phoneme_mapping.md          mapping table, rules, check results and word examples
  phoneme_mapping_check.json  the same check results in machine-readable form
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from alignment import align
from phoneme_map import (
    EXTRA, IPA_TO_COMMON, RULES, _reduce, build_reference, ipa_to_common, reference_variants,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PER_RULE = 4


def mapping_kind(token):
    if token in IPA_TO_COMMON:
        return "tablo"
    return "indirgenmiş" if _reduce(token) in IPA_TO_COMMON else "bilinmiyor"


def count_ops(ops):
    c = Counter(op for op, *_ in ops)
    return c["substitution"], c["deletion"], c["insertion"], c["accepted"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    out_dir = ROOT / "outputs"
    scores = json.loads((args.data_dir / "speechocean762" / "resource" / "scores.json").read_text(encoding="utf-8"))
    with open(args.data_dir / "predictions" / "speechocean762_ipa.csv", encoding="utf-8") as f:
        preds = {r["utt_id"]: r["ipa"] for r in csv.DictReader(f) if r["split"] == "train"}

    # 1. model tokens on train
    tokens = Counter(t for ipa in preds.values() for t in ipa.split())
    total_tokens = sum(tokens.values())
    by_kind = Counter()
    for t, c in tokens.items():
        by_kind[mapping_kind(t)] += c

    # 2. alignment on all train recordings, reported separately for clean ones
    stats = {name: Counter() for name in ("all", "clean")}
    rule_use = Counter()
    residual = {"substitution": Counter(), "deletion": Counter(), "insertion": Counter()}
    examples = defaultdict(list)
    variants = reference_variants(w for u in preds for w in scores[u]["words"])  # train only
    for utt_id, ipa in sorted(preds.items()):
        words = scores[utt_id]["words"]
        ref, ref_word = build_reference(words, variants)
        clean = all(a == 2.0 for w in words for a in w["phones-accuracy"])
        hyp = ipa_to_common(ipa)
        strict = align(ref, hyp, use_rules=False)
        ruled = align(ref, hyp)
        for name in ("all", "clean") if clean else ("all",):
            s, d, i, _ = count_ops(strict)
            stats[name]["ref"] += len(ref)
            stats[name]["utts"] += 1
            stats[name]["strict_errors"] += s + d + i
            s, d, i, a = count_ops(ruled)
            stats[name]["rule_errors"] += s + d + i
            stats[name]["accepted"] += a
        if not clean:
            continue

        # word-level view of the rule-based alignment (insertions belong to the preceding word)
        word_hyp, word_ops = defaultdict(list), defaultdict(list)
        last_word = 0
        for op, ri, hi, rule in ruled:
            if ri is not None:
                last_word = ref_word[ri]
            if hi is not None:
                word_hyp[last_word].append(hyp[hi])
            word_ops[last_word].append((op, rule))
            if op == "accepted":
                rule_use[rule] += 1
            elif op == "substitution":
                residual[op][f"{ref[ri][0]} → {hyp[hi]}"] += 1
            elif op == "deletion":
                residual[op][ref[ri][0]] += 1
            elif op == "insertion":
                residual[op][hyp[hi]] += 1
        # examples: words whose only differences are accepted ones
        for w_i, ops in word_ops.items():
            if any(op not in ("match", "accepted") for op, _ in ops):
                continue
            for rule in {r for op, r in ops if op == "accepted"}:
                if len(examples[rule]) < EXAMPLES_PER_RULE and words[w_i]["text"] not in {
                        e["word"] for e in examples[rule]}:
                    examples[rule].append({"word": words[w_i]["text"], "utt_id": utt_id,
                                           "ref": " ".join(words[w_i]["phones"]), "hyp": " ".join(word_hyp[w_i])})

    per = lambda s, key: 100 * s[key] / s["ref"] if s["ref"] else float("nan")
    check = {
        "split": "train",
        "model_tokens": {"distinct": len(tokens), "total": total_tokens, "by_mapping": dict(by_kind)},
        "unknown_tokens": {t: c for t, c in tokens.most_common() if mapping_kind(t) == "bilinmiyor"},
        "alignment": {
            name: {
                "utterances": s["utts"], "reference_phonemes": s["ref"],
                "per_strict_pct": round(per(s, "strict_errors"), 2),
                "per_with_rules_pct": round(per(s, "rule_errors"), 2),
                "accepted_differences": s["accepted"],
            } for name, s in stats.items()
        },
        "rule_use_clean": dict(rule_use.most_common()),
        "residual_clean": {k: dict(v.most_common(15)) for k, v in residual.items()},
    }
    (out_dir / "phoneme_mapping_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "phoneme_mapping.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ipa_token", "common", "seen_on_train"])
        for t, common in IPA_TO_COMMON.items():
            w.writerow([t, " ".join(common), tokens[t]])

    seen = lambda t: tokens[t] if tokens[t] else "—"
    lines = [
        "# Adım 4 — Referans ve model fonemlerinin ortak biçimi\n",
        "Teslim: dönüşüm kodu (`scripts/phoneme_map.py`), sembol eşleştirme tablosu (bu rapor ve "
        "`phoneme_mapping.csv`) ve test örnekleri (`tests/test_phoneme_map.py` ve aşağıdaki kelime örnekleri). "
        "Bu rapor `scripts/check_phoneme_map.py` tarafından yalnızca **train** kayıtları kullanılarak üretilir; "
        "test kayıtları Adım 6'daki değerlendirmeye ayrıldı.\n",
        "## Özet\n",
        f"- Ortak gösterim: CMU sözlüğünün **39 ARPAbet fonemi** (vurgusuz) + modelin ürettiği ama ARPAbet'te "
        f"karşılığı olmayan **{len(EXTRA)} ses** ({', '.join(EXTRA)}).",
        f"- Model train kayıtlarında {len(tokens)} farklı sembol üretti; sembollerin "
        f"%{100 * by_kind['tablo'] / total_tokens:.2f}'i doğrudan tablodan, "
        f"%{100 * by_kind['indirgenmiş'] / total_tokens:.2f}'i indirgenerek eşlendi, "
        f"%{100 * by_kind['bilinmiyor'] / total_tokens:.2f}'i İngilizce karşılığı olmadığı için `?` olarak işaretlendi.",
    ]
    c, a = check["alignment"]["clean"], check["alignment"]["all"]
    lines += [
        f"- Uzmanların tüm fonemleri doğru saydığı **{c['utterances']} temiz kayıtta** fonem hata oranı "
        f"kurallar olmadan **%{c['per_strict_pct']:.1f}**, kabul edilebilir fark kurallarıyla **%{c['per_with_rules_pct']:.1f}**. "
        f"Kalan fark, modelin kendi hataları ile henüz kurala bağlanmamış farklardan oluşur (aşağıda).",
        f"- Tüm train kayıtlarında ({a['utterances']}): kurallar olmadan %{a['per_strict_pct']:.1f}, kurallarla "
        f"%{a['per_with_rules_pct']:.1f}. Bu fark gerçek telaffuz hatalarını da içerir.\n",
        "## 1. Ortak gösterim ve vurgu\n",
        "- Referans (speechocean762): ARPAbet, ünlülerde vurgu rakamı (0 vurgusuz, 1 birincil, 2 ikincil). "
        "Rakam fonemden ayrılır ve ayrı tutulur.",
        "- Model: eSpeak IPA; vurgu işareti üretmez, kelime sınırı vermez. Her sembol tablodaki bir veya birkaç "
        "ortak foneme çevrilir (ör. `ɑːɹ` → `AA R`).",
        "- Vurgu fonem eşleşmesinde karşılaştırılmaz (R1); yalnızca vurgusuz ünlülerin indirgenmesine izin vermek "
        "için kullanılır (R2). Kelime vurgusu speechocean762'de ayrıca puanlandığı için fonem düzeyinin dışında kalır.\n",
        "## 2. Kabul edilebilir telaffuz farkları\n",
        "Bu farklar hata sayılmaz; karşılaştırma adımında ayrıca gösterilir.\n",
        "| Kural | Açıklama | Temiz kayıtlarda kullanım |",
        "| --- | --- | --- |",
    ] + [f"| {r} | {d} | {rule_use[r] if r != 'R1' else '—'} |" for r, d in RULES.items()] + [
        "\nR5 ve R6, speechocean762 referanslarının kayıt bazında İngiliz (r'siz) veya Amerikan (r'li) "
        "yazılmasından dolayı gerekli: ör. \"FOR\" 278 kayıtta `F AO`, 40 kayıtta `F AO R`; model ise Amerikan "
        "tarzında `f ɔːɹ` üretir.\n",
        "## 3. Sembol eşleştirme tablosu\n",
        "Son sütun, sembolün model tarafından train kayıtlarında kaç kez üretildiği.\n",
        "| Model (IPA) | Ortak | Train'de | | Model (IPA) | Ortak | Train'de |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    items = list(IPA_TO_COMMON.items())
    half = (len(items) + 1) // 2
    for left, right in zip(items[:half], items[half:] + [None]):
        row = f"| `{left[0]}` | {' '.join(left[1])} | {seen(left[0])} |"
        row += f" | `{right[0]}` | {' '.join(right[1])} | {seen(right[0])} |" if right else " | | | |"
        lines.append(row)
    lines += [
        "\nTabloda olmayan semboller (modelin çok dilli sözlüğünden): uzunluk işareti, ton rakamı ve "
        "ikincil artikülasyon (ʰ, ʲ) atılarak tekrar aranır; yine bulunamazsa `?` olur.\n",
        "| Train'de görülen tablo dışı sembol | Eşleşme | Adet |",
        "| --- | --- | --- |",
    ] + [f"| `{t}` | {' '.join(ipa_to_common(t))} ({mapping_kind(t)}) | {n} |"
         for t, n in tokens.most_common() if t not in IPA_TO_COMMON] + [
        "\n## 4. Kelime örnekleri\n",
        "Temiz kayıtlardan, her kural için gerçek model çıktısı. Referans vurgu rakamlarıyla, model çıktısı ortak "
        "gösterimle verilmiştir.\n",
        "| Kural | Kelime | Referans | Model | Kayıt |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in RULES:
        for e in examples.get(r, []):
            lines.append(f"| {r} | {e['word']} | {e.get('ref', '')} | {e.get('hyp', '')} | {e['utt_id']} |")
    lines += [
        "\n## 5. Kurallardan sonra kalan en sık farklar (temiz kayıtlar)\n",
        "Bu kayıtlarda uzmanlar her fonemi doğru saydığı için kalan farklar çoğunlukla modelin hatası ya da "
        "kurala bağlanmamış bir varyanttır. Yeni kural eklemeden önce incelenmeli.\n",
        "| Değiştirme (referans → model) | Adet | Silme (referansta var, modelde yok) | Adet | "
        "Ekleme (modelde fazla) | Adet |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    subs, dels, inss = (list(residual[k].most_common(10)) for k in ("substitution", "deletion", "insertion"))
    for k in range(10):
        cell = lambda lst: f"{lst[k][0]} | {lst[k][1]}" if k < len(lst) else " | "
        lines.append(f"| {cell(subs)} | {cell(dels)} | {cell(inss)} |")
    lines += [
        "\n## Sınırlılıklar\n",
        "- R5 ve R6 bağlamdan bağımsızdır: r'siz okunuş her yerde kabul edilir. Bu, gerçek bir R hatasını da "
        "gizleyebilir; Türk konuşmacı verisinde hedef aksan belirlenince daraltılabilir.",
        "- Fonksiyon kelimelerinin zayıf biçimleri (ör. AND → `AH N`) ayrıca kurala bağlanmadı; R2 ünlü kısmını "
        "karşılar, ünsüz düşmesini karşılamaz.",
        "- Kontrol, modelin kendi hatalarını kural eksiklerinden ayıramaz; kalan farklar elle incelenmelidir.",
        "- Referansın İngiliz aksanındaki \"o\" sesini (ɒ) `AH` ile yazdığı kelimeler, veri setinde başka bir "
        "yazımı yoksa (ör. JOHN, POCKET) R8 kapsamına girmez; model bunlarda Amerikan `AA`/`AO` üretir. "
        "Hedef aksan belirlenince bu kelimeler için kural genişletilebilir.",
        "- Kalan farkların bir kısmı uzmanların hoş gördüğü aksan özellikleri olabilir (ör. GOOD/LOOK'ta `UH → UW`, "
        "kelime sonuna eklenen `AH`). Model bunları duyuyor ama uzman \"doğru\" saymış; Adım 6'da değerlendirme "
        "ölçütü seçilirken dikkate alınmalı.",
    ]
    (out_dir / "phoneme_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(check["alignment"], indent=1))
    print("rule use:", dict(rule_use.most_common()))
    for k, v in residual.items():
        print(k, v.most_common(12))
    print("unknown:", list(check["unknown_tokens"].items())[:15])


if __name__ == "__main__":
    main()
