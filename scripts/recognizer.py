"""Load audio and run the pretrained wav2vec 2.0 phoneme recognizer."""

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16_000


def load_audio(path):
    """Read an audio file as mono float32 at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


class PhonemeRecognizer:
    """Greedy CTC decoding: returns space-separated eSpeak IPA tokens without word boundaries."""

    def __init__(self, cpu=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        # do_phonemize=False: we only decode model output, so the phonemizer/eSpeak backend is not needed.
        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_ID, do_phonemize=False)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device).eval()

    def __call__(self, audio):
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
        return self.tokenizer.decode(torch.argmax(logits, dim=-1)[0].cpu())
