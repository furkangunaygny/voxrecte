"""Convert speechocean762 ARPAbet references and wav2vec 2.0 eSpeak IPA output to one phoneme set.

Common representation
---------------------
The 39 ARPAbet phonemes of the CMU dictionary (no stress digits), plus three symbols for
sounds the model writes but ARPAbet has no phoneme for (names as in TIMIT):
  DX  flap            (ɾ, as in American "water")      acceptable for T or D
  Q   glottal stop    (ʔ)                              acceptable for T
  IX  reduced vowel   (ᵻ, as in "roses")               acceptable for unstressed IH or AH
References keep their stress digit separately, because the acceptable-variant rules use it.

Acceptable pronunciation differences
------------------------------------
The rules in RULES are not errors; the comparison step should report them separately.
They cover allophones the model writes (DX, Q, IX), unstressed vowel reduction, the
British/American differences found in the speechocean762 references (rhotic R, ER/AH), and
alternative reference transcriptions of the same word in the corpus (R8).

The model is multilingual and writes some sounds of these Mandarin-L1 speakers with Mandarin
eSpeak symbols (ph, th, ai5, ...); those with a close English counterpart are in the table.
Other tokens are first reduced (length marks, tone marks and secondary articulation removed)
and looked up again; anything still unknown becomes "?" so that it is visible instead of
silently dropped.
"""

import re
import unicodedata

PHONEMES = (
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW "
    "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH"
).split()
VOWELS = set(PHONEMES[:15]) | {"IX"}
EXTRA = ("DX", "Q", "IX")
UNKNOWN = "?"

# eSpeak IPA token -> common phonemes. Multi-phoneme tokens (r-coloured vowels, syllabic
# consonants) are split so that both sides use the same units.
IPA_TO_COMMON = {
    # monophthongs
    "i": ["IY"], "iː": ["IY"], "ɪ": ["IH"], "ɪː": ["IY"], "ᵻ": ["IX"],
    "e": ["EH"], "eː": ["EH"], "ɛ": ["EH"], "ɛː": ["EH"], "æ": ["AE"], "a": ["AE"], "aː": ["AA"],
    "ɑ": ["AA"], "ɑː": ["AA"], "ɒ": ["AA"], "ɔ": ["AO"], "ɔː": ["AO"], "o": ["OW"], "oː": ["AO"],
    "ʊ": ["UH"], "u": ["UW"], "uː": ["UW"], "ʌ": ["AH"], "ə": ["AH"], "ɐ": ["AH"],
    "ɚ": ["ER"], "ɜ": ["ER"], "ɜː": ["ER"],
    # diphthongs
    "eɪ": ["EY"], "aɪ": ["AY"], "aʊ": ["AW"], "ɔɪ": ["OY"], "oʊ": ["OW"], "əʊ": ["OW"],
    # r-coloured vowels and centring diphthongs
    "ɑːɹ": ["AA", "R"], "ɔːɹ": ["AO", "R"], "oːɹ": ["AO", "R"], "ɪɹ": ["IH", "R"], "ɛɹ": ["EH", "R"],
    "ʊɹ": ["UH", "R"], "aɪɚ": ["AY", "ER"], "aɪə": ["AY", "AH"], "iə": ["IY", "AH"], "eə": ["EH", "AH"],
    "ʊə": ["UH", "AH"],
    # syllabic consonants
    "əl": ["AH", "L"], "l̩": ["AH", "L"], "n̩": ["AH", "N"],
    # consonants
    "p": ["P"], "b": ["B"], "t": ["T"], "d": ["D"], "k": ["K"], "g": ["G"], "ɡ": ["G"],
    "f": ["F"], "v": ["V"], "θ": ["TH"], "ð": ["DH"], "s": ["S"], "z": ["Z"], "ʃ": ["SH"], "ʒ": ["ZH"],
    "h": ["HH"], "tʃ": ["CH"], "dʒ": ["JH"], "m": ["M"], "n": ["N"], "ŋ": ["NG"],
    "l": ["L"], "ɫ": ["L"], "ɹ": ["R"], "r": ["R"], "ɻ": ["R"], "w": ["W"], "j": ["Y"],
    # allophones without an ARPAbet phoneme
    "ɾ": ["DX"], "ʔ": ["Q"],
    # Mandarin (eSpeak cmn) symbols the multilingual model uses for these Mandarin-L1 speakers.
    # Tone marks are removed before lookup (see _reduce). Aspirated stops are what English
    # p/t/k normally sound like; the other entries are the nearest English sound sequence.
    "ph": ["P"], "th": ["T"], "kh": ["K"], "x": ["HH"], "tɕ": ["JH"], "tɕh": ["CH"], "ts.h": ["CH"],
    "ɕ": ["SH"], "ʉ": ["UW"], "ai": ["AY"], "ei": ["EY"], "au": ["AW"], "ɑu": ["AW"], "ou": ["OW"],
    "oɪ": ["OY"], "iɛ": ["Y", "EH"], "iɑ": ["Y", "AA"], "iou": ["Y", "OW"], "uo": ["W", "AO"],
    "ua": ["W", "AA"], "uei": ["W", "EY"], "onɡ": ["UH", "NG"],
}

# (rule id, short Turkish description) — used in reports and in the comparison step.
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


def stress_free(phones):
    return tuple(re.sub(r"\d", "", p) for p in phones)


def reference_variants(words):
    """Word -> set of stress-free reference transcriptions, from speechocean762-style word entries."""
    variants = {}
    for w in words:
        variants.setdefault(w["text"].upper(), set()).add(stress_free(w["phones"]))
    return variants


def build_reference(words, variants=None):
    """Word entries -> ([(phoneme, stress, alternatives)], [word index per phoneme]).

    `alternatives` are the phonemes other reference transcriptions of the same word (same
    length) have at this position; they are accepted under rule R8.

    speechocean762 marks the vowel of 98% of one-syllable words with stress 0, so the digit
    carries no information there: their stress is set to None (R2/R7 do not apply)."""
    ref, word_index = [], []
    for w_i, w in enumerate(words):
        own = stress_free(w["phones"])
        others = [v for v in (variants or {}).get(w["text"].upper(), ()) if len(v) == len(own) and v != own]
        parsed = parse_arpabet(w["phones"])
        if sum(p in VOWELS for p, _ in parsed) == 1:
            parsed = [(p, None) for p, _ in parsed]
        for k, (p, s) in enumerate(parsed):
            ref.append((p, s, frozenset(v[k] for v in others) - {p}))
            word_index.append(w_i)
    return ref, word_index


def parse_arpabet(phones):
    """['W', 'IY0'] or 'W IY0' -> [('W', None), ('IY', 0)]."""
    if isinstance(phones, str):
        phones = phones.split()
    out = []
    for p in phones:
        m = re.fullmatch(r"([A-Z]+)([012])?", p)
        if not m or m.group(1) not in PHONEMES:
            raise ValueError(f"not an ARPAbet phoneme: {p!r}")
        base, stress = m.group(1), m.group(2)
        out.append((base, int(stress) if stress is not None and base in VOWELS else None))
    return out


def _reduce(token):
    """Drop length marks, tone digits and secondary-articulation diacritics from an unknown token."""
    token = re.sub(r"[0-9ɜ:ː.^\"\[ʲʰˤ]+$", "", token) or token
    token = "".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c))
    return token.replace("ʲ", "").replace("ʰ", "").replace("ː", "")


def ipa_token_to_common(token):
    """One model token -> list of common phonemes ([UNKNOWN] if it has no English counterpart)."""
    if token in IPA_TO_COMMON:
        return list(IPA_TO_COMMON[token])
    reduced = _reduce(token)
    if reduced in IPA_TO_COMMON:
        return list(IPA_TO_COMMON[reduced])
    return [UNKNOWN]


def ipa_to_common(ipa):
    """Model output 'h ɛ l oʊ' -> ['HH', 'EH', 'L', 'OW']."""
    tokens = ipa.split() if isinstance(ipa, str) else ipa
    return [p for t in tokens for p in ipa_token_to_common(t)]


def substitution_rule(ref, ref_stress, hyp, alternatives=()):
    """Rule id if `hyp` is an acceptable realisation of reference phoneme `ref`, else None."""
    if ref == hyp:
        return None
    if hyp in alternatives:
        return "R8"
    if ref in VOWELS and ref_stress == 0 and hyp in ("AH", "IH", "IX"):
        return "R2"
    if hyp == "DX" and ref in ("T", "D"):
        return "R3"
    if hyp == "Q" and ref == "T":
        return "R4"
    if {ref, hyp} == {"ER", "AH"} and ref_stress == 0:
        return "R6"
    return None


def deletion_rule(ref, ref_stress, prev_ref, next_ref):
    """Rule id if reference phoneme `ref` may be missing from the model output, else None."""
    if ref == "R" and prev_ref in VOWELS:
        return "R5"
    if ref == "AH" and ref_stress == 0 and next_ref in ("L", "N", "M"):
        return "R7"
    return None


def insertion_rule(hyp, prev_hyp):
    """Rule id if model phoneme `hyp`, absent from the reference, is acceptable, else None."""
    if hyp == "R" and prev_hyp in VOWELS:
        return "R5"
    return None
