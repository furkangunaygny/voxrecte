"""Reference phonemes and the acceptable pronunciation differences between reference and model.

Differences matched by a rule are not errors; the comparison reports them separately with the
rule id. They cover allophones the model writes (DX, Q, IX), unstressed vowel reduction, the
British/American differences found in the speechocean762 references (rhotic R, ER/AH), and
alternative reference transcriptions of the same word in the corpus (R8).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from voxrecte.phonemes import VOWELS, parse_arpabet, strip_stress

# Rule descriptions are in Turkish because the comparison tool prints them for Turkish users.
RULES = {
    "R1": "Vurgu karşılaştırılmaz: model vurgu işareti üretmez, referanstaki vurgu rakamı yalnızca R2 ve R7'de "
          "kullanılır. Tek heceli kelimelerde rakam bilgi taşımadığı için (veride %98'i 0) vurgu bilinmiyor sayılır.",
    "R2": "Vurgusuz ünlü indirgenmesi: çok heceli bir kelimede vurgusuz (0) ünlü yerine AH, IH veya IX kabul edilir.",
    "R3": "Dil ucu çırpması: DX, referanstaki T veya D yerine kabul edilir.",
    "R4": "Gırtlak durması: Q, referanstaki T yerine kabul edilir.",
    "R5": "r'li/r'siz aksan: ünlüden sonra gelen R referansta olup tahminde olmayabilir ya da tersi.",
    "R6": "r'li/r'siz aksan: çok heceli bir kelimenin vurgusuz hecesinde ER ile AH birbirinin yerine kabul edilir "
          "(\"mother\": M AH DH AH / m ʌ ð ɚ).",
    "R7": "Hecelik ünsüz: L, N veya M'den önce vurgusuz AH olmayabilir (\"little\" → l ɪ ɾ əl / L IH T L).",
    "R8": "Referans varyantı: aynı kelime veri setinde başka bir referans yazımıyla da geçiyorsa, o yazımdaki "
          "ses kabul edilir (ör. NOT: N AH T ve N AA T).",
}

REDUCED_VOWELS = ("AH", "IH", "IX")
SYLLABIC_CONSONANTS = ("L", "N", "M")


@dataclass(frozen=True)
class ReferencePhone:
    """One phoneme of the target pronunciation.

    stress:        0/1/2 for vowels of multi-syllable words, otherwise None
    alternatives:  phonemes that other reference transcriptions of the word have at this position (R8)
    word_index:    position of the word in the sentence
    """
    phoneme: str
    stress: int | None = None
    alternatives: frozenset[str] = field(default_factory=frozenset)
    word_index: int = 0


def reference_variants(words: Iterable[dict]) -> dict[str, set[tuple[str, ...]]]:
    """Word -> stress-free reference transcriptions, from speechocean762-style word entries."""
    variants: dict[str, set[tuple[str, ...]]] = {}
    for word in words:
        variants.setdefault(word["text"].upper(), set()).add(strip_stress(word["phones"]))
    return variants


def build_reference(words: list[dict], variants: dict | None = None) -> list[ReferencePhone]:
    """Word entries ({'text', 'phones'}) -> the reference phoneme sequence of the sentence.

    speechocean762 marks the vowel of 98% of one-syllable words with stress 0, so the digit
    carries no information there: their stress is set to None and R2/R7 do not apply."""
    reference = []
    for word_index, word in enumerate(words):
        own = strip_stress(word["phones"])
        others = [v for v in (variants or {}).get(word["text"].upper(), ()) if len(v) == len(own) and v != own]
        parsed = parse_arpabet(word["phones"])
        one_syllable = sum(phoneme in VOWELS for phoneme, _ in parsed) == 1
        for position, (phoneme, stress) in enumerate(parsed):
            reference.append(ReferencePhone(
                phoneme=phoneme,
                stress=None if one_syllable else stress,
                alternatives=frozenset(v[position] for v in others) - {phoneme},
                word_index=word_index,
            ))
    return reference


def substitution_rule(ref: ReferencePhone, heard: str) -> str | None:
    """Rule id if `heard` is an acceptable realisation of the reference phoneme, else None."""
    if heard == ref.phoneme:
        return None
    if heard in ref.alternatives:
        return "R8"
    if ref.phoneme in VOWELS and ref.stress == 0 and heard in REDUCED_VOWELS:
        return "R2"
    if heard == "DX" and ref.phoneme in ("T", "D"):
        return "R3"
    if heard == "Q" and ref.phoneme == "T":
        return "R4"
    if {ref.phoneme, heard} == {"ER", "AH"} and ref.stress == 0:
        return "R6"
    return None


def deletion_rule(ref: ReferencePhone, previous: str | None, following: str | None) -> str | None:
    """Rule id if the reference phoneme may be missing from the model output, else None.

    previous / following are the neighbouring reference phonemes."""
    if ref.phoneme == "R" and previous in VOWELS:
        return "R5"
    if ref.phoneme == "AH" and ref.stress == 0 and following in SYLLABIC_CONSONANTS:
        return "R7"
    return None


def insertion_rule(heard: str, previous_heard: str | None) -> str | None:
    """Rule id if a model phoneme that has no reference counterpart is acceptable, else None."""
    if heard == "R" and previous_heard in VOWELS:
        return "R5"
    return None
