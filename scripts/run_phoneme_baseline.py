"""Step 1: run the pretrained wav2vec 2.0 phoneme recognizer on sample recordings.

Model:   facebook/wav2vec2-lv-60-espeak-cv-ft (CTC, eSpeak IPA phonemes)
Samples: hf-internal-testing/librispeech_asr_dummy (LibriSpeech clean/validation)

Outputs (written to outputs/):
  phoneme_baseline.json  per-sample predictions + timing
  environment.json       versions of Python, libraries, CUDA, GPU and model/dataset revisions
"""

import argparse
import io
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download

from voxrecte.files import write_json
from voxrecte.paths import OUTPUTS_DIR
from voxrecte.recognizer import MODEL_ID, SAMPLE_RATE, PhonemeRecognizer, model_revision

DATASET_ID = "hf-internal-testing/librispeech_asr_dummy"
DATASET_FILE = "clean/validation-00000-of-00001.parquet"
PACKAGES = [
    "torch", "torchaudio", "transformers", "accelerate", "datasets", "huggingface-hub",
    "tokenizers", "safetensors", "librosa", "soundfile", "numpy", "scipy", "pyarrow",
]


def load_samples(n: int) -> list[dict]:
    """First n LibriSpeech samples as {'id', 'text', 'audio'} (read from Parquet, no datasets cache)."""
    path = hf_hub_download(DATASET_ID, DATASET_FILE, repo_type="dataset")
    samples = []
    for row in pq.read_table(path).slice(0, n).to_pylist():
        audio, sample_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{row['id']}: expected {SAMPLE_RATE} Hz, got {sample_rate}")
        samples.append({"id": row["id"], "text": row["text"], "audio": audio})
    return samples


def collect_environment(device: torch.device) -> dict:
    """Everything needed to tell under which conditions the outputs were produced."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: version(p) for p in PACKAGES},
        "cuda": {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "run_device": str(device),
        "model": {"id": MODEL_ID, "revision": model_revision()},
        "dataset": {"id": DATASET_ID, "file": DATASET_FILE, "revision": HfApi().dataset_info(DATASET_ID).sha},
    }


def transcribe_timed(recognizer: PhonemeRecognizer, audio) -> tuple[str, float]:
    """Model output and wall-clock inference time (GPU work included)."""
    start = time.perf_counter()
    phonemes = recognizer.transcribe(audio)
    if recognizer.device.type == "cuda":
        torch.cuda.synchronize()
    return phonemes, time.perf_counter() - start


def main():
    """Transcribe the samples, record timing and environment, write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    recognizer = PhonemeRecognizer(cpu=args.cpu)
    results = []
    for sample in load_samples(args.num_samples):
        phonemes, elapsed = transcribe_timed(recognizer, sample["audio"])
        duration = len(sample["audio"]) / SAMPLE_RATE
        results.append({
            "id": sample["id"],
            "duration_s": round(duration, 2),
            "inference_s": round(elapsed, 3),
            "reference_text": sample["text"],
            "predicted_phonemes": phonemes,
        })
        print(f"[{sample['id']}] {duration:.1f}s audio, {elapsed:.2f}s inference\n"
              f"  text: {sample['text']}\n  ipa:  {phonemes}\n")

    environment = collect_environment(recognizer.device)
    write_json(OUTPUTS_DIR / "environment.json", environment)
    write_json(OUTPUTS_DIR / "phoneme_baseline.json",
               {"model": MODEL_ID, "device": str(recognizer.device), "results": results})
    print(f"Wrote results and environment to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
