"""Phoneme-level comparison of a recording's model output with its target sentence.

Differences covered by an acceptable-variant rule are listed separately; all others are
reported as possible errors ("olası hata"). They are not final error decisions: the model
itself makes mistakes. User-facing labels are Turkish because the tool is used in a Turkish study.
"""

from dataclasses import dataclass, field

from voxrecte.alignment import ACCEPTED, DELETION, INSERTION, MATCH, SUBSTITUTION, align
from voxrecte.phonemes import ipa_token_to_common, strip_stress
from voxrecte.rules import RULES, build_reference
from voxrecte.text import words_of

DIFFERENCE_LABELS = {SUBSTITUTION: "değiştirme", DELETION: "silme", INSERTION: "ekleme"}
STATUS_CORRECT, STATUS_ACCEPTED, STATUS_POSSIBLE_ERROR = "doğru", "kabul edilebilir fark", "olası hata"
POSSIBLE_ERROR_NOTE = ("Not: \"olası hata\" kesin bir hata kararı değildir; "
                       "modelin kendi hataları da bu listede yer alabilir.")


@dataclass
class _WordResult:
    """Comparison result of one target word (exported with as_dict)."""
    word: str
    target: list[str]
    heard: list[str] = field(default_factory=list)
    heard_ipa: list[str] = field(default_factory=list)
    possible_errors: list[dict] = field(default_factory=list)
    accepted: list[dict] = field(default_factory=list)

    def add_heard(self, phoneme: str, ipa_token: str) -> None:
        """Attach a model phoneme and the IPA token it came from to the word."""
        self.heard.append(phoneme)
        if not self.heard_ipa or self.heard_ipa[-1] != ipa_token:  # one token can give several phonemes
            self.heard_ipa.append(ipa_token)

    @property
    def status(self) -> str:
        """Word status: possible error, accepted difference or correct."""
        if self.possible_errors:
            return STATUS_POSSIBLE_ERROR
        return STATUS_ACCEPTED if self.accepted else STATUS_CORRECT

    def as_dict(self) -> dict:
        """JSON-serialisable form, keys in output order."""
        return {"word": self.word, "target": self.target, "heard": self.heard, "heard_ipa": self.heard_ipa,
                "possible_errors": self.possible_errors, "accepted": self.accepted, "status": self.status}


def _model_phonemes(ipa: str) -> tuple[list[str], list[str]]:
    """Common phonemes of the model output and, for each, the IPA token it came from."""
    phonemes, tokens = [], []
    for token in ipa.split():
        for phoneme in ipa_token_to_common(token):
            phonemes.append(phoneme)
            tokens.append(token)
    return phonemes, tokens


def compare(ipa: str, words: list[dict], variants: dict | None = None) -> dict:
    """Align model output `ipa` with the target `words` ({'text', 'phones'}); JSON-serialisable result.

    Model phonemes that have no reference counterpart (insertions) are attributed to the
    preceding word."""
    heard, heard_tokens = _model_phonemes(ipa)
    reference = build_reference(words, variants)
    results = [_WordResult(w["text"], list(strip_stress(w["phones"]))) for w in words]

    current = 0
    for pair in align(reference, heard):
        if pair.ref_index is not None:
            current = reference[pair.ref_index].word_index
        word = results[current]
        if pair.hyp_index is not None:
            word.add_heard(heard[pair.hyp_index], heard_tokens[pair.hyp_index])
        if pair.kind == MATCH:
            continue
        difference = {
            "type": DIFFERENCE_LABELS[_edit_kind(pair)],
            "expected": reference[pair.ref_index].phoneme if pair.ref_index is not None else None,
            "heard": heard[pair.hyp_index] if pair.hyp_index is not None else None,
            "heard_ipa": heard_tokens[pair.hyp_index] if pair.hyp_index is not None else None,
        }
        if pair.kind == ACCEPTED:
            word.accepted.append({**difference, "rule": pair.rule})
        else:
            word.possible_errors.append(difference)

    return {"model_ipa": ipa, "words": [w.as_dict() for w in results], "summary": _summary(results, len(reference))}


def _edit_kind(pair) -> str:
    """Substitution / deletion / insertion, also for accepted differences."""
    if pair.ref_index is not None and pair.hyp_index is not None:
        return SUBSTITUTION
    return DELETION if pair.hyp_index is None else INSERTION


def _summary(results: list[_WordResult], target_phonemes: int) -> dict:
    """Counts over all words of the sentence."""
    n_errors = sum(len(w.possible_errors) for w in results)
    return {
        "target_phonemes": target_phonemes,
        "possible_errors": n_errors,
        "accepted_differences": sum(len(w.accepted) for w in results),
        "words_with_possible_errors": sum(w.status == STATUS_POSSIBLE_ERROR for w in results),
        "match_rate": round(1 - n_errors / target_phonemes, 3) if target_phonemes else None,
    }


def describe_difference(difference: dict) -> str:
    """Short Turkish description: 'TH yerine S (model: s)', 'K duyulmadı', 'fazladan AH (model: ə)'."""
    heard = f"{difference['heard']} (model: {difference['heard_ipa']})" if difference["heard"] else "—"
    if difference["type"] == DIFFERENCE_LABELS[SUBSTITUTION]:
        return f"{difference['expected']} yerine {heard}"
    if difference["type"] == DIFFERENCE_LABELS[DELETION]:
        return f"{difference['expected']} duyulmadı"
    return f"fazladan {heard}"


def format_result(text: str, result: dict) -> str:
    """Plain-text report of compare() for the terminal."""
    summary = result["summary"]
    lines = [
        f"Hedef cümle : {' '.join(words_of(text))}",
        f"Model (IPA) : {result['model_ipa']}",
        f"Özet        : {summary['target_phonemes']} hedef fonem, {summary['possible_errors']} olası hata, "
        f"{summary['accepted_differences']} kabul edilebilir fark, "
        f"{summary['words_with_possible_errors']}/{len(result['words'])} kelimede olası hata",
        "",
        f"{'Kelime':<14} {'Durum':<22} {'Hedef':<22} Duyulan",
    ]
    for word in result["words"]:
        lines.append(f"{word['word']:<14} {word['status']:<22} {' '.join(word['target']):<22} "
                     f"{' '.join(word['heard']) or '—'}")
        lines += [f"{'':<14}   olası hata ({d['type']}): {describe_difference(d)}" for d in word["possible_errors"]]
        lines += [f"{'':<14}   kabul edilebilir ({d['rule']}): {describe_difference(d)}" for d in word["accepted"]]
    used_rules = sorted({d["rule"] for word in result["words"] for d in word["accepted"]})
    if used_rules:
        lines += ["", "Kurallar:"] + [f"  {rule}: {RULES[rule]}" for rule in used_rules]
    lines += ["", POSSIBLE_ERROR_NOTE]
    return "\n".join(lines)
