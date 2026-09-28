"""Exact runner for the recorded Linux VM experiment.

Place this file in an isolated directory with manifest.json, audio/, hub/, and
hyprwhspr/lib/src/ (the checkout under test). See the adjacent benchmark report.
This is a historical experiment recipe, not an installed CLI or startup hook.
"""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_CACHE"] = str(ROOT / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT / "hyprwhspr/lib/src"))
# Load the real backend and its relative helpers without initializing unrelated backends.
pkg = types.ModuleType("backends")
pkg.__path__ = [str(ROOT / "hyprwhspr/lib/src/backends")]
sys.modules["backends"] = pkg
import numpy as np
import onnxruntime as ort
import soundfile as sf
import soxr
from backends.onnx_asr_backend import OnnxAsrBackend
from rapidfuzz.distance import Levenshtein
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


class Config:
    def __init__(self, model):
        self.model = model

    def get_setting(self, key, default=None):
        return {
            "onnx_asr_model": self.model,
            "onnx_asr_quantization": "int8",
            "onnx_asr_use_vad": True,
        }.get(key, default)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


manifest = json.loads((ROOT / "manifest.json").read_text())
models = {}
loads = {}
for key, name in [("orukeet", "orukeet"), ("parakeet", "nemo-parakeet-tdt-0.6b-v3")]:
    b = OnnxAsrBackend(
        types.SimpleNamespace(config=Config(name), ready=False, current_model=None)
    )
    start = time.perf_counter()
    assert b.initialize(), key
    loads[key] = time.perf_counter() - start
    models[key] = b
print("BOTH MODELS LOADED", loads, flush=True)


def audio(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim == 2:
        a = a.mean(axis=1)
    if sr != 16000:
        a = soxr.resample(a, sr, 16000)
    return a, 16000


# One untimed warm-up per model, excluded from all measured rows.
a, sr = audio(ROOT / manifest["clips"][25]["path"])
for b in models.values():
    assert b.transcribe(a, sr), "Warm-up unexpectedly empty"
receipt = {
    "source_head": "72207dbaac0008bb5d1537e4a0c24c0454e6ff0e",
    "scope": "Actual unchanged hyprwhspr OnnxAsrBackend initialization and transcription on Linux aarch64 Colima VM, 4 vCPUs / 8 GiB on Apple M5 Max. No compositor, microphone, or text injection. Common source normalization outside inference timing: stereo downmix by mean, soxr resample to mono 16kHz (the backend microphone contract). Network blocked in an isolated Linux network namespace after genuine cached asset acquisition. Default CPU ORT session settings; identical process, sequential alternating model order; one warm-up per model excluded. VAD enabled with unchanged 30-second threshold (all selected clips shorter).",
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "versions": {
        p: importlib.metadata.version(p)
        for p in [
            "onnx-asr",
            "onnxruntime",
            "numpy",
            "soundfile",
            "whisper-normalizer",
            "rapidfuzz",
        ]
    },
    "providers": ort.get_available_providers(),
    "load_seconds": loads,
    "manifest_sha256": sha(ROOT / "manifest.json"),
    "model_hashes": {},
    "backend_sha256": sha(ROOT / "hyprwhspr/lib/src/backends/onnx_asr_backend.py"),
    "rows": [],
}
for folder in (ROOT / "hub").glob("models*"):
    receipt["model_hashes"][folder.name] = {
        str(f.relative_to(folder)): sha(f) for f in folder.rglob("*") if f.is_file()
    }
(ROOT / "receipt-start.json").write_text(json.dumps(receipt, indent=2))
with (ROOT / "paired-results.jsonl").open("w") as log:
    for i, c in enumerate(manifest["clips"]):
        p = ROOT / c["path"]
        assert sha(p) == c["sha256"]
        a, sr = audio(p)
        assert a.ndim == 1
        row = {**c, "results": {}}
        for key in ["orukeet", "parakeet"] if i % 2 == 0 else ["parakeet", "orukeet"]:
            output = io.StringIO()
            start = time.perf_counter()
            with contextlib.redirect_stdout(output):
                text = models[key].transcribe(a, sr)
            seconds = time.perf_counter() - start
            ref = normalizer(c["reference"]).split()
            hyp = normalizer(text).split()
            row["results"][key] = {
                "text": text,
                "seconds": seconds,
                "words": len(ref),
                "errors": Levenshtein.distance(ref, hyp),
                "backend_error": "Transcription failed:" in output.getvalue(),
            }
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()
        receipt["rows"].append(row)
        if (i + 1) % 20 == 0:
            print("PROGRESS", i + 1, "/", len(manifest["clips"]), flush=True)
# Stability: repeated speech, empty silence, cold offline reload, and actual VAD path.
receipt["stability"] = {}
for key, b in models.items():
    a, sr = audio(ROOT / manifest["clips"][25]["path"])
    t1 = b.transcribe(a, sr)
    t2 = b.transcribe(a, sr)
    quiet = [
        b.transcribe(np.zeros(int(s * 16000), dtype=np.float32), 16000)
        for s in [0.1, 1, 5]
    ]
    # Exercise VAD on a 31-second input without changing production threshold.
    long = np.concatenate([a, np.zeros(max(0, 31 * sr - len(a)), dtype=np.float32)])
    vad_text = b.transcribe(long, sr)
    b.unload()
    assert b.initialize()
    again = b.transcribe(a, sr)
    receipt["stability"][key] = {
        "repeat_equal": t1 == t2,
        "silence": quiet,
        "offline_reload_equal": again == t1,
        "vad_path_text": vad_text,
    }
    assert t1 == t2 == again and quiet == ["", "", ""]


def aggregate(rows):
    out = {}
    for key in models:
        words = sum(r["results"][key]["words"] for r in rows)
        errors = sum(r["results"][key]["errors"] for r in rows)
        elapsed = sum(r["results"][key]["seconds"] for r in rows)
        audio = sum(r["audio_seconds"] for r in rows)
        out[key] = {
            "words": words,
            "errors": errors,
            "wer_percent": errors / words * 100,
            "inference_seconds": elapsed,
            "audio_seconds": audio,
            "rtf": elapsed / audio,
            "median_ms": statistics.median(r["results"][key]["seconds"] for r in rows)
            * 1000,
            "backend_errors": sum(r["results"][key]["backend_error"] for r in rows),
            "empty_outputs": sum(not r["results"][key]["text"] for r in rows),
        }
    return out


receipt["overall"] = aggregate(receipt["rows"])
receipt["by_dataset"] = {
    d: aggregate([r for r in receipt["rows"] if r["dataset"] == d])
    for d in sorted({r["dataset"] for r in receipt["rows"]})
}
receipt["status"] = (
    "passed"
    if not any(x["backend_errors"] for x in receipt["overall"].values())
    else "completed_with_backend_errors"
)
(ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
print(
    json.dumps(
        {
            k: v
            for k, v in receipt.items()
            if k in ["status", "overall", "by_dataset", "stability"]
        },
        indent=2,
    ),
    flush=True,
)
