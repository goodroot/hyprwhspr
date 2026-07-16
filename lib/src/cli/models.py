"""
Model management commands for hyprwhspr — pywhispercpp (whisper.cpp),
ONNX-ASR, faster-whisper and Cohere transcribe model families
"""

import os
import sys
import subprocess
from pathlib import Path

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..paths import RECORDING_CONTROL_FILE, MODEL_UNLOADED_FILE
except ImportError:
    from paths import RECORDING_CONTROL_FILE, MODEL_UNLOADED_FILE

try:
    from ..backend_installer import VENV_DIR, PYWHISPERCPP_MODELS_DIR
except ImportError:
    from backend_installer import VENV_DIR, PYWHISPERCPP_MODELS_DIR

try:
    from ..backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

try:
    from ..output_control import (log_info, log_success, log_warning, log_error,
                                  run_command)
except ImportError:
    from output_control import (log_info, log_success, log_warning, log_error,
                                run_command)


# Selectable Whisper models, shared by the setup prompt and `model list`.
# Single source of truth so the two never drift.
MULTILINGUAL_MODELS = [
    ('tiny', 'Fastest, least accurate'),
    ('base', 'Good balance (recommended)'),
    ('small', 'Better accuracy'),
    ('medium', 'High accuracy'),
    ('large-v3-turbo', 'Fast, near-large-v3 accuracy, requires GPU'),
    ('large', 'Best accuracy, requires GPU'),
    ('large-v3', 'Latest large model, requires GPU'),
]
ENGLISH_ONLY_MODELS = [
    ('tiny.en', 'Fastest, least accurate (English only)'),
    ('base.en', 'Good balance (English only, recommended)'),
    ('small.en', 'Better accuracy (English only)'),
    ('medium.en', 'High accuracy (English only)'),
]


# ==================== Model Commands ====================

def _send_model_control(command: str) -> bool:
    """Send a model_unload or model_reload command to the running service via FIFO."""
    import stat, select
    if not RECORDING_CONTROL_FILE.exists():
        log_error("Recording control file not found — is the hyprwhspr service running?")
        log_info("Start it with: systemctl --user start hyprwhspr")
        return False
    try:
        file_stat = RECORDING_CONTROL_FILE.stat()
        is_fifo = stat.S_ISFIFO(file_stat.st_mode)
    except Exception:
        is_fifo = False
    try:
        if is_fifo:
            fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
            fd_closed = False
            try:
                _, ready, _ = select.select([], [fd], [], 2.0)
                if not ready:
                    os.close(fd)
                    fd_closed = True
                    log_error("Service not responding (timeout)")
                    return False
                os.write(fd, (command + '\n').encode())
            finally:
                if not fd_closed:
                    os.close(fd)
        else:
            RECORDING_CONTROL_FILE.write_text(command + '\n')
        return True
    except OSError as e:
        if e.errno == 6:
            log_error("Service not listening on control FIFO — is hyprwhspr running?")
        else:
            log_error(f"Failed to send command: {e}")
        return False
    except Exception as e:
        log_error(f"Failed to send command: {e}")
        return False


def model_command(action: str, model_name: str = 'base'):
    """Handle model subcommands"""
    config = ConfigManager()
    backend = normalize_backend(config.get_setting('transcription_backend', 'pywhispercpp'))

    # unload / reload are valid for any local-model backend
    if action == 'unload':
        if backend in ('rest-api', 'realtime-ws'):
            log_error(f"Model unload not applicable for backend: {backend}")
            log_info("Only local backends (pywhispercpp, faster-whisper, onnx-asr, cohere-transcribe) hold GPU memory.")
            return
        if MODEL_UNLOADED_FILE.exists():
            log_warning("Model is already unloaded.")
            log_info("Reload it with: hyprwhspr model reload")
            return
        log_info("Sending model unload request to service...")
        if _send_model_control('model_unload'):
            log_success("Model unload requested — GPU resources will be freed.")
        return

    if action == 'reload':
        if backend in ('rest-api', 'realtime-ws'):
            log_error(f"Model reload not applicable for backend: {backend}")
            return
        if not MODEL_UNLOADED_FILE.exists():
            log_warning("Model does not appear to be unloaded.")
            log_info("Use this after: hyprwhspr model unload")
            return
        log_info("Sending model reload request to service...")
        if _send_model_control('model_reload'):
            log_success("Model reload requested — service will load model back into memory.")
        return

    if backend == 'faster-whisper':
        if action == 'download':
            download_faster_whisper_model(model_name)
        elif action == 'list':
            list_faster_whisper_models()
        elif action == 'status':
            faster_whisper_model_status()
        else:
            log_error(f"Unknown model action: {action}")
    elif backend == 'onnx-asr':
        if action == 'list':
            list_onnx_asr_models()
        elif action == 'status':
            onnx_asr_model_status()
        elif action == 'download':
            log_info("Parakeet model is downloaded during setup.")
            log_info("If the model is missing, re-run: hyprwhspr setup")
        else:
            log_error(f"Unknown model action: {action}")
    elif backend == 'cohere-transcribe':
        if action == 'list':
            list_cohere_transcribe_models()
        elif action == 'status':
            cohere_transcribe_model_status()
        elif action == 'download':
            download_cohere_transcribe_model()
        else:
            log_error(f"Unknown model action: {action}")
    else:
        if action == 'download':
            download_model(model_name)
        elif action == 'list':
            list_models()
        elif action == 'status':
            model_status()
        else:
            log_error(f"Unknown model action: {action}")


def download_model(model_name: str = 'base'):
    """Download pywhispercpp model with progress feedback"""
    try:
        from .backend_installer import download_pywhispercpp_model
    except ImportError:
        from backend_installer import download_pywhispercpp_model

    return download_pywhispercpp_model(model_name)


def list_models():
    """List available models"""
    print("Available models:\n")

    print("Multilingual models (support all languages, auto-detect):")
    for name, desc in MULTILINGUAL_MODELS:
        print(f"  - {name} - {desc}")

    print("\nEnglish-only models (smaller, faster, English only):")
    for name, desc in ENGLISH_ONLY_MODELS:
        print(f"  - {name} - {desc}")

    print("\nNote: Use multilingual models for non-English languages or mixed-language content.")
    print("      Use English-only (.en) models for English-only content (smaller file size).")


def model_status():
    """Check installed pywhispercpp (Whisper.cpp) models in ~/.local/share/pywhispercpp/models"""
    if not PYWHISPERCPP_MODELS_DIR.exists():
        log_warning("Models directory does not exist")
        return

    models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))

    if not models:
        log_warning("No models installed")
        return

    print("Installed models:")
    for model in sorted(models):
        size = model.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {model.name} ({size:.1f} MB)")


def onnx_asr_model_status():
    """Check Parakeet/onnx-asr model in Hugging Face cache (~/.cache/huggingface/hub/)"""
    hf_hub_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    if not hf_hub_dir.exists():
        log_warning("Hugging Face cache directory does not exist (~/.cache/huggingface/hub/)")
        log_info("Parakeet model is downloaded on first use when the backend starts.")
        return

    # Parakeet TDT 0.6B v3: HF repo is nvidia/parakeet-tdt-0.6b-v3 -> cache: models--nvidia--parakeet-tdt-0.6b-v3
    parakeet_patterns = ['models--nvidia--parakeet-tdt-0.6b-v3', 'models--*parakeet*']
    found = []
    seen = set()
    for pattern in parakeet_patterns:
        for model_dir in sorted(hf_hub_dir.glob(pattern)):
            if model_dir.is_dir() and model_dir.resolve() not in seen:
                seen.add(model_dir.resolve())
                found.append(model_dir)
    if not found:
        log_warning("No Parakeet model found in ~/.cache/huggingface/hub/")
        log_info("Model not found. Re-run: hyprwhspr setup to download it.")
        return

    print("Parakeet (onnx-asr) model cache:")
    for model_dir in found:
        total_bytes = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_mb = total_bytes / (1024 * 1024)
        if size_mb >= 1024:
            size_str = f"{size_mb / 1024:.1f} GB"
        else:
            size_str = f"{size_mb:.0f} MB"
        log_success(f"  {model_dir.name} ({size_str})")


def list_onnx_asr_models():
    """List Parakeet/onnx-asr model option (single supported model for now)."""
    print("Parakeet (onnx-asr) model:\n")
    print("  - nemo-parakeet-tdt-0.6b-v3  (~1 GB, downloaded during setup)")
    print()
    print("Storage: ~/.cache/huggingface/hub/")
    print("To check if the model is cached: hyprwhspr model status")
    print("The model downloads automatically when the onnx-asr backend starts.")


# ==================== Cohere Transcribe Model Commands ====================

def cohere_transcribe_model_status():
    """Check Cohere Transcribe model in Hugging Face cache (~/.cache/huggingface/hub/)"""
    hf_hub_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    model_cache = hf_hub_dir / 'models--CohereLabs--cohere-transcribe-03-2026'

    if not model_cache.exists():
        log_warning("Cohere Transcribe model not found in ~/.cache/huggingface/hub/")
        log_info("Re-run: hyprwhspr setup and select cohere-transcribe to download the model.")
        return

    total_bytes = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
    size_gb = total_bytes / (1024 ** 3)
    log_success(f"  {model_cache.name} ({size_gb:.1f} GB)")


def list_cohere_transcribe_models():
    """List Cohere Transcribe model (single model)."""
    print("Cohere Transcribe model:\n")
    print("  - CohereLabs/cohere-transcribe-03-2026  (~4 GB, downloaded during setup)")
    print()
    print("Storage: ~/.cache/huggingface/hub/")
    print("To check cache status: hyprwhspr model status")
    print("Precision: bfloat16 on GPU (default), float32 on CPU")


def download_cohere_transcribe_model():
    """Download (or re-download) the Cohere Transcribe model weights."""
    try:
        from credential_manager import get_credential
    except ImportError:
        try:
            from .credential_manager import get_credential
        except ImportError:
            get_credential = lambda _: None

    hf_token = get_credential('huggingface') or None
    if not hf_token:
        log_warning("No HuggingFace token found.")
        log_info("Run hyprwhspr setup to provide your token, or accept the model license at:")
        log_info("  https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")
        return

    try:
        from backend_installer import install_backend
    except ImportError:
        from .backend_installer import install_backend

    from paths import VENV_DIR
    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        log_error("Cohere Transcribe venv not found. Run: hyprwhspr setup")
        return

    import subprocess, os
    download_script = '''
try:
    from huggingface_hub import enable_progress_bars
    enable_progress_bars()
except ImportError:
    pass
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch, os, sys
model_id = "CohereLabs/cohere-transcribe-03-2026"
token = os.environ.get("HF_TOKEN") or None
print("Downloading processor...", flush=True)
AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
print("Downloading model weights (~4 GB)...", flush=True)
sys.stdout.flush()
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.bfloat16,
    low_cpu_mem_usage=True, token=token,
)
del model
print("Done.", flush=True)
'''
    env = {**os.environ, 'HF_TOKEN': hf_token, 'PYTHONUNBUFFERED': '1'}
    log_info("Downloading Cohere Transcribe model (~4 GB)...")
    result = subprocess.run([str(venv_python), '-c', download_script], env=env)
    if result.returncode == 0:
        log_success("Model downloaded and cached successfully")
    else:
        log_error("Download failed — check your HuggingFace token and license acceptance")


# ==================== faster-whisper Model Commands ====================

FASTER_WHISPER_MODELS = [
    ('tiny',            '~75 MB',   'Fastest, lowest accuracy'),
    ('base',            '~145 MB',  'Good balance (recommended for CPU)'),
    ('small',           '~484 MB',  'Better accuracy, still fast'),
    ('medium',          '~1.5 GB',  'High accuracy'),
    ('large-v3',        '~3.1 GB',  'Best accuracy (INT8 on GPU ~3.1 GB VRAM)'),
    ('large-v3-turbo',  '~1.6 GB',  'Fast + accurate, recommended for NVIDIA GPUs'),
    ('distil-large-v3', '~1.5 GB',  'Distilled large-v3, great CPU/GPU balance'),
]


def list_faster_whisper_models():
    """List available faster-whisper models"""
    print("Available faster-whisper models:\n")
    print(f"  {'Model':<22} {'Size (INT8)':<12} Notes")
    print(f"  {'-'*22} {'-'*12} {'-'*35}")
    for model_name, size, notes in FASTER_WHISPER_MODELS:
        print(f"  {model_name:<22} {size:<12} {notes}")
    print()
    print("Models are downloaded automatically from HuggingFace on first load.")
    print("Storage: ~/.cache/huggingface/hub/")
    print()
    print("To download a model manually:")
    print("  hyprwhspr model download large-v3-turbo")


def download_faster_whisper_model(model_name: str = 'base'):
    """Download a faster-whisper model via the venv Python (triggers HuggingFace download)"""
    log_info(f"Downloading faster-whisper model: {model_name}")
    print("Models are fetched from HuggingFace. This may take a while for large models.")

    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        log_error("faster-whisper venv not found. Run: hyprwhspr setup and select faster-whisper")
        return False

    download_script = (
        'from faster_whisper import WhisperModel; '
        f'print("Downloading {model_name}...", flush=True); '
        f'WhisperModel("{model_name}", device="cpu", compute_type="float32"); '
        'print("Download complete", flush=True)'
    )
    try:
        run_command([str(venv_python), '-c', download_script], check=True)
        log_success(f"Model '{model_name}' downloaded successfully.")
        print("Storage: ~/.cache/huggingface/hub/")
        return True
    except Exception as e:
        log_error(f"Failed to download model '{model_name}': {e}")
        return False


def faster_whisper_model_status():
    """Report which faster-whisper models are installed"""
    hf_hub_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    if not hf_hub_dir.exists():
        log_warning("HuggingFace cache directory does not exist (~/.cache/huggingface/hub/)")
        log_warning("No faster-whisper models downloaded yet.")
        return

    model_dirs = sorted(hf_hub_dir.glob('models--Systran--faster-whisper-*'))
    if not model_dirs:
        log_warning("No faster-whisper models found in ~/.cache/huggingface/hub/")
        return

    print("Installed faster-whisper models:")
    for model_dir in model_dirs:
        model_name = model_dir.name.replace('models--Systran--faster-whisper-', '')
        # Calculate total size
        total_bytes = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_mb = total_bytes / (1024 * 1024)
        if size_mb >= 1024:
            size_str = f"{size_mb / 1024:.1f} GB"
        else:
            size_str = f"{size_mb:.0f} MB"
        print(f"  - {model_name} ({size_str})")
