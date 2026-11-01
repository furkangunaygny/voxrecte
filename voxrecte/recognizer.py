"""Audio loading and the pretrained wav2vec 2.0 phoneme recognizer.

Model: facebook/wav2vec2-lv-60-espeak-cv-ft (CTC, eSpeak IPA phonemes). The weights are used
as published; nothing is trained here.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16_000


def model_revision() -> str:
    """Commit hash of the model on the Hugging Face Hub, recorded for reproducibility."""
    return HfApi().model_info(MODEL_ID).sha


def load_audio(path: Path | str) -> np.ndarray:
    """Read an audio file as mono float32 at 16 kHz (other rates are resampled)."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa  # only needed for non-16 kHz input
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


class PhonemeRecognizer:
    """Wraps feature extractor, model and tokenizer; runs on GPU when available."""

    def __init__(self, cpu: bool = False):
        """Load the model on the GPU if available (or on the CPU when cpu=True)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        # do_phonemize=False: we only decode model output, so the phonemizer/eSpeak backend is not needed.
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_ID, do_phonemize=False)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device).eval()

    @property
    def num_layers(self) -> int:
        """Hidden states returned per utterance: CNN/projection output + each transformer layer."""
        return self.model.config.num_hidden_layers + 1

    @property
    def hidden_size(self) -> int:
        """Size of each hidden state vector."""
        return self.model.config.hidden_size

    def _inputs(self, audio: np.ndarray) -> torch.Tensor:
        """Normalised model input for one recording, on the model's device."""
        features = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return features.input_values.to(self.device)

    def transcribe(self, audio: np.ndarray) -> str:
        """Greedy CTC decoding: space-separated eSpeak IPA tokens, without word boundaries."""
        with torch.inference_mode():
            logits = self.model(self._inputs(audio)).logits
        return self.tokenizer.decode(torch.argmax(logits, dim=-1)[0].cpu())

    def layer_means(self, audio: np.ndarray) -> np.ndarray:
        """Time-averaged hidden state of every layer, shape (num_layers, hidden_size)."""
        with torch.inference_mode():
            hidden = self.model(self._inputs(audio), output_hidden_states=True).hidden_states
        return torch.stack(hidden)[:, 0].mean(dim=1).float().cpu().numpy()
