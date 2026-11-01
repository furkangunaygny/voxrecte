"""Project-wide file locations.

Large or generated data (the corpus, model predictions, features) lives under data/ and is
not tracked by git; results meant to be read live under outputs/; small hand-made or derived
inputs that the tools need live under resources/.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESOURCES_DIR = PROJECT_ROOT / "resources"

MANIFEST_PATH = OUTPUTS_DIR / "speechocean762_manifest.csv"
LEXICON_PATH = RESOURCES_DIR / "lexicon.tsv"
EXCLUDED_SENTENCES_PATH = RESOURCES_DIR / "excluded_sentences.csv"


def corpus_dir(data_dir: Path = DATA_DIR) -> Path:
    """Extracted speechocean762 corpus."""
    return data_dir / "speechocean762"


def predictions_path(data_dir: Path = DATA_DIR) -> Path:
    """Model phoneme output for every speechocean762 recording."""
    return data_dir / "predictions" / "speechocean762_ipa.csv"


def features_dir(data_dir: Path = DATA_DIR) -> Path:
    """Utterance-level wav2vec 2.0 features."""
    return data_dir / "features"
