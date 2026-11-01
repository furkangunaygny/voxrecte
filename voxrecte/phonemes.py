"""Phoneme inventory and conversion of the model's eSpeak IPA output to the common phoneme set.

Common representation
---------------------
The 39 ARPAbet phonemes of the CMU Pronouncing Dictionary (no stress digits), plus three
symbols for sounds the model writes but ARPAbet has no phoneme for (names as in TIMIT):

    DX  flap            (ɾ, as in American "water")   acceptable for T or D
    Q   glottal stop    (ʔ)                           acceptable for T
    IX  reduced vowel   (ᵻ, as in "roses")            acceptable for unstressed IH or AH

The model is multilingual and writes some sounds of the Mandarin-L1 speakers of speechocean762
with Mandarin eSpeak symbols (ph, th, ai5, ...); those with a close English counterpart are in
the table. Other tokens are first reduced (length, tone and secondary-articulation marks
removed) and looked up again; anything still unknown becomes UNKNOWN ("?"), so that it stays
visible instead of being silently dropped.
"""

import re
import unicodedata

ARPABET_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
ARPABET_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
PHONEMES = ARPABET_VOWELS + ARPABET_CONSONANTS  # the 39 CMU phonemes
EXTRA_SYMBOLS = ("DX", "Q", "IX")
VOWELS = frozenset(ARPABET_VOWELS) | {"IX"}
UNKNOWN = "?"

# eSpeak IPA token -> common phonemes. Multi-phoneme tokens (r-coloured vowels, syllabic
# consonants) are split so that the reference and the model use the same units.
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
    # Mandarin (eSpeak cmn) symbols. Tone marks are removed before lookup (see reduce_token).
    # Aspirated stops are what English p/t/k normally sound like; the other entries are the
    # nearest English sound sequence.
    "ph": ["P"], "th": ["T"], "kh": ["K"], "x": ["HH"], "tɕ": ["JH"], "tɕh": ["CH"], "ts.h": ["CH"],
    "ɕ": ["SH"], "ʉ": ["UW"], "ai": ["AY"], "ei": ["EY"], "au": ["AW"], "ɑu": ["AW"], "ou": ["OW"],
    "oɪ": ["OY"], "iɛ": ["Y", "EH"], "iɑ": ["Y", "AA"], "iou": ["Y", "OW"], "uo": ["W", "AO"],
    "ua": ["W", "AA"], "uei": ["W", "EY"], "onɡ": ["UH", "NG"],
}

# How a model token was mapped (see mapping_source).
FROM_TABLE, FROM_REDUCED, NOT_MAPPED = "table", "reduced", "unknown"

_ARPABET = re.compile(r"([A-Z]+)([012])?")
_TRAILING_MARKS = re.compile(r"[0-9ɜ:ː.^\"\[ʲʰˤ]+$")  # length, tone (digits, ɜ) and similar suffixes


def strip_stress(phones) -> tuple[str, ...]:
    """('W', 'IY0') -> ('W', 'IY')."""
    return tuple(re.sub(r"\d", "", p) for p in phones)


def parse_arpabet(phones) -> list[tuple[str, int | None]]:
    """['W', 'IY0'] or 'W IY0' -> [('W', None), ('IY', 0)]; consonants never carry stress."""
    if isinstance(phones, str):
        phones = phones.split()
    parsed = []
    for phone in phones:
        match = _ARPABET.fullmatch(phone)
        if not match or match.group(1) not in PHONEMES:
            raise ValueError(f"not an ARPAbet phoneme: {phone!r}")
        base, stress = match.groups()
        parsed.append((base, int(stress) if stress is not None and base in VOWELS else None))
    return parsed


def reduce_token(token: str) -> str:
    """Drop length, tone and secondary-articulation marks: 'tʃʰ' -> 'tʃ', 'ai5' -> 'ai', 'dː' -> 'd'."""
    token = _TRAILING_MARKS.sub("", token) or token
    token = "".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c))
    return token.replace("ʲ", "").replace("ʰ", "").replace("ː", "")


def mapping_source(token: str) -> str:
    """FROM_TABLE, FROM_REDUCED or NOT_MAPPED, depending on how ipa_token_to_common maps the token."""
    if token in IPA_TO_COMMON:
        return FROM_TABLE
    return FROM_REDUCED if reduce_token(token) in IPA_TO_COMMON else NOT_MAPPED


def ipa_token_to_common(token: str) -> list[str]:
    """One model token -> common phonemes; [UNKNOWN] if it has no English counterpart."""
    for key in (token, reduce_token(token)):
        if key in IPA_TO_COMMON:
            return list(IPA_TO_COMMON[key])
    return [UNKNOWN]


def ipa_to_common(ipa) -> list[str]:
    """Model output 'h ɛ l oʊ' (or a token list) -> ['HH', 'EH', 'L', 'OW']."""
    tokens = ipa.split() if isinstance(ipa, str) else ipa
    return [phoneme for token in tokens for phoneme in ipa_token_to_common(token)]
