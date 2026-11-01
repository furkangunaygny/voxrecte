"""VoxRecte: phoneme-level English pronunciation analysis.

The package holds the reusable logic; the command-line tools in scripts/ only read inputs,
call these modules and write outputs.

    paths        project file locations
    text         sentence normalisation
    speechocean  speechocean762 corpus readers
    phonemes     phoneme inventory and model IPA -> common phoneme conversion
    rules        reference phonemes and acceptable pronunciation differences
    alignment    reference / model phoneme sequence alignment
    lexicon      pronunciation lexicon built from speechocean762
    comparison   recording vs. target sentence comparison
    evaluation   evaluation against the speechocean762 expert phoneme labels
    recognizer   audio loading and the wav2vec 2.0 phoneme recognizer
    selection    greedy phoneme-coverage sentence selection
    files        CSV / JSON output helpers
"""
