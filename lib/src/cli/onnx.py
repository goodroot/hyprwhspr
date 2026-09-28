"""Prepare ONNX models in the active backend environment and record the selection."""
import json
import os
from pathlib import Path

try:
    from .. import backend_installer
    from ..onnx_model import resolve_model, NEEDS_DOWNLOAD
    from ..output_control import run_command, log_error, log_info, log_success, log_warning
except ImportError:
    import backend_installer
    from onnx_model import resolve_model, NEEDS_DOWNLOAD
    from output_control import run_command, log_error, log_info, log_success, log_warning


def prepare_model(selection):
    """Load the model once in the backend venv, downloading or repairing only when needed."""
    model = selection[0]
    python = backend_installer.VENV_DIR / 'bin' / 'python'
    script = (
        'import sys,json; '
        f'sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); '
        'from src.onnx_model import prepare; '
        'sys.exit(prepare(json.loads(sys.argv[1]), offline=sys.argv[2] == "1"))'
    )
    command = [str(python), '-c', script, json.dumps(selection)]
    # An explicit env bypasses run_command's own mise handling, so apply it here.
    env = (backend_installer._create_mise_free_environment()
           if backend_installer._check_mise_active() else os.environ.copy())
    try:
        cached = run_command(command + ['1'], check=False, capture_output=True,
                             show_output_on_error=False,
                             env=dict(env, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        if cached.returncode == NEEDS_DOWNLOAD:
            log_info(f'Downloading {model}; this can take several minutes...')
            run_command(command + ['0'], check=True, env=env, verbose=True)
        elif cached.returncode != 0:
            # Not a cache problem, so downloading would not help; show why.
            detail = (cached.stderr or cached.stdout or '').strip().splitlines()
            raise RuntimeError(detail[-1] if detail else f'exit code {cached.returncode}')
        log_success(f'Model ready: {model}')
        return True
    except Exception as exc:
        log_error(f'Model preparation failed: {exc}')
        log_info("Retry with 'hyprwhspr model download'; finished files are kept.")
        return False


def prepare_selection(selection, current=None):
    """Return the selection setup should save; setup always continues.

    current is the working selection when the backend is unchanged, else None.
    A new backend replaces the environment, so its selection is saved even if
    the model is not ready yet; the service downloads it on start.
    """
    if prepare_model(selection):
        return selection
    if current is None or current[0] == selection[0]:
        log_warning("Model not ready; saved anyway. It downloads when the service starts, "
                    "or run 'hyprwhspr model download'.")
        return selection
    log_warning(f"Keeping {current[0]}; retry {selection[0]} with 'hyprwhspr model download'.")
    return current


def orukeet_needs_onnx(model, backend):
    """Reject Orukeet for other backends, which would look for a same-named model."""
    if model == 'orukeet' and backend != 'onnx-asr':
        log_error("Orukeet needs the onnx-asr backend: run 'hyprwhspr setup', pick Parakeet → Orukeet.")
        return True
    return False


def apply_selection(config, selection):
    # Quantization is left alone: it is already the user's value, and Orukeet ignores it.
    config.set_setting('transcription_backend', 'onnx-asr')
    config.set_setting('onnx_asr_model', selection[0])


def save_selection(config, selection):
    apply_selection(config, selection)
    if not config.save_config():
        raise OSError('Could not save model selection; retry setup. Downloaded files are retained.')
