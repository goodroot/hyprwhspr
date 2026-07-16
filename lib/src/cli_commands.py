"""
CLI command implementations for hyprwhspr
"""

import os
import sys
import json
import paths
import subprocess
import getpass
import shutil
import socket
import re
from pathlib import Path
from typing import Optional

try:
    from rich.prompt import Prompt, Confirm
    from rich.console import Console
    from rich.table import Table
except (ImportError, ModuleNotFoundError) as e:
    # Hard fail – rich is required for the CLI
    print("ERROR: python-rich is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print(f"\nPython being used: {sys.executable}", file=sys.stderr)
    print(f"Python version:    {sys.version.split()[0]}", file=sys.stderr)
    print("\nThis usually means python-rich is installed for a different Python version.", file=sys.stderr)
    print("The CLI requires system Python with distro packages installed.", file=sys.stderr)
    print("\nTry installing it using your package manager:", file=sys.stderr)
    print("  Arch:          pacman -S python-rich", file=sys.stderr)
    print("  Debian/Ubuntu: apt install python3-rich", file=sys.stderr)
    print("  Fedora:        dnf install python3-rich", file=sys.stderr)
    print("  Or via pip:    pip install rich>=13.0.0", file=sys.stderr)
    print("\nIf using a Python version manager (pyenv, mise, asdf), ensure", file=sys.stderr)
    print("python-rich is installed for your system Python (/usr/bin/python3).", file=sys.stderr)
    sys.exit(1)

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .paths import CONFIG_DIR, CONFIG_FILE, RECORDING_CONTROL_FILE, SOCKET_FILE, RECORDING_STATUS_FILE, MODEL_UNLOADED_FILE
except ImportError:
    from paths import CONFIG_DIR, CONFIG_FILE, RECORDING_CONTROL_FILE, SOCKET_FILE, RECORDING_STATUS_FILE, MODEL_UNLOADED_FILE

try:
    from .backend_utils import (BACKEND_DISPLAY_NAMES, BACKEND_IMPORT_MODULES,
                                LOCAL_INSTALL_BACKENDS, normalize_backend)
except ImportError:
    from backend_utils import (BACKEND_DISPLAY_NAMES, BACKEND_IMPORT_MODULES,
                               LOCAL_INSTALL_BACKENDS, normalize_backend)

try:
    from .backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        USER_BASE, PYWHISPERCPP_SRC_DIR,
        PYWHISPERCPP_MODELS_DIR, MAX_COMPATIBLE_PYTHON, _get_python_version
    )
except ImportError:
    from backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        USER_BASE, PYWHISPERCPP_SRC_DIR,
        PYWHISPERCPP_MODELS_DIR, MAX_COMPATIBLE_PYTHON, _get_python_version
    )

try:
    from .provider_registry import (
        PROVIDERS, get_provider, list_providers, get_provider_models,
        get_model_config, validate_api_key
    )
except ImportError:
    from provider_registry import (
        PROVIDERS, get_provider, list_providers, get_provider_models,
        get_model_config, validate_api_key
    )

try:
    from .credential_manager import (
        save_credential, get_credential, mask_api_key, CREDENTIALS_FILE
    )
except ImportError:
    from credential_manager import (
        save_credential, get_credential, mask_api_key, CREDENTIALS_FILE
    )

try:
    from .output_control import (
        log_info, log_success, log_warning, log_error, log_debug, log_verbose,
        run_command, run_sudo_command, OutputController, VerbosityLevel
    )
except ImportError:
    from output_control import (
        log_info, log_success, log_warning, log_error, log_debug, log_verbose,
        run_command, run_sudo_command, OutputController, VerbosityLevel
    )

try:
    from .global_shortcuts import get_available_keyboards, test_key_accessibility
except ImportError:
    from global_shortcuts import get_available_keyboards, test_key_accessibility


# Shared constants and environment helpers live in cli._shared; re-imported
# here so not-yet-moved sections and their tests keep working during the split.
try:
    from .cli._shared import (
        HYPRWHSPR_ROOT, SERVICE_NAME, RESUME_SERVICE_NAME, YDOTOOL_UNIT,
        _YDOTOOL_MARKERS, USER_HOME, USER_CONFIG_DIR, USER_SYSTEMD_DIR,
        _is_niri_session, _is_gnome_or_mutter_session,
        _check_mise_active, _create_mise_free_environment,
        _check_python_compatibility, _check_ydotool_version,
        _strip_jsonc, _load_jsonc, _validate_hyprwhspr_root,
    )
except ImportError:
    from cli._shared import (
        HYPRWHSPR_ROOT, SERVICE_NAME, RESUME_SERVICE_NAME, YDOTOOL_UNIT,
        _YDOTOOL_MARKERS, USER_HOME, USER_CONFIG_DIR, USER_SYSTEMD_DIR,
        _is_niri_session, _is_gnome_or_mutter_session,
        _check_mise_active, _create_mise_free_environment,
        _check_python_compatibility, _check_ydotool_version,
        _strip_jsonc, _load_jsonc, _validate_hyprwhspr_root,
    )

# Model commands live in cli.models; names still read by not-yet-moved sections
# are re-imported here until the split completes.
try:
    from .cli.models import (
        MULTILINGUAL_MODELS, ENGLISH_ONLY_MODELS, FASTER_WHISPER_MODELS,
        download_model, download_faster_whisper_model, model_status,
        onnx_asr_model_status, faster_whisper_model_status,
        cohere_transcribe_model_status,
    )
except ImportError:
    from cli.models import (
        MULTILINGUAL_MODELS, ENGLISH_ONLY_MODELS, FASTER_WHISPER_MODELS,
        download_model, download_faster_whisper_model, model_status,
        onnx_asr_model_status, faster_whisper_model_status,
        cohere_transcribe_model_status,
    )


# ==================== Setup Command ====================

def _detect_current_backend(existing_cfg: Optional[dict] = None) -> Optional[str]:
    """
    Detect currently installed backend.

    Returns:
        'cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr', 'rest-api', or None if not detected
    """
    # First check config file
    try:
        backend = (
            existing_cfg.get('transcription_backend')
            if existing_cfg is not None
            else ConfigManager(verbose=False).get_setting('transcription_backend', None)
        )
        
        # Backward compatibility: map old values
        if backend == 'remote':
            return 'rest-api'
        if backend == 'local':
            # Old 'local' - try to detect from venv or default to 'cpu'
            venv_python = VENV_DIR / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '-c', 'import pywhispercpp; print("ok")'],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return 'cpu'  # Default to cpu for old 'local'
                except Exception:
                    pass
            return 'cpu'  # Fallback
        
        if backend == 'rest-api':
            return 'rest-api'
        if backend == 'realtime-ws':
            return 'realtime-ws'
        if backend == 'onnx-asr':
            # Verify onnx-asr is actually installed in venv
            venv_python = VENV_DIR / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '-c', 'import onnx_asr; print("ok")'],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return 'onnx-asr'
                except Exception:
                    pass
            # onnx-asr configured but not installed - fall through to return None
        if backend == 'faster-whisper':
            # Verify faster-whisper is actually installed in venv
            venv_python = VENV_DIR / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '-c', 'import faster_whisper; print("ok")'],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return 'faster-whisper'
                except Exception:
                    pass
            # faster-whisper configured but not installed - fall through to return None
        if backend == 'cohere-transcribe':
            # Verify transformers is installed in venv
            venv_python = VENV_DIR / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '-c', 'from transformers import AutoModelForSpeechSeq2Seq; print("ok")'],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return 'cohere-transcribe'
                except Exception:
                    pass
            # cohere-transcribe configured but not installed - fall through to return None
        if backend in ['cpu', 'nvidia', 'amd', 'vulkan', 'pywhispercpp']:
            # Verify it's actually installed in venv
            venv_python = VENV_DIR / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '-c', 'import pywhispercpp; print("ok")'],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        # Normalize backend before returning (handles 'amd' -> 'vulkan')
                        return normalize_backend(backend)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Fallback: check if venv exists and has pywhispercpp
    venv_python = VENV_DIR / 'bin' / 'python'
    if venv_python.exists():
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'import pywhispercpp; print("ok")'],
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Venv has pywhispercpp but no config - assume CPU (safest)
                return 'cpu'
        except Exception:
            pass
    
    return None


def _cleanup_backend(backend_type: str) -> bool:
    """
    Clean up an installed backend.

    Args:
        backend_type: 'cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr', or 'remote'

    Returns:
        True if cleanup succeeded
    """
    if backend_type == 'onnx-asr':
        log_info("Cleaning up ONNX-ASR backend...")
        venv_python = VENV_DIR / 'bin' / 'python'
        if not venv_python.exists():
            log_info("No venv found, nothing to clean")
            return True
        try:
            pip_bin = VENV_DIR / 'bin' / 'pip'
            if pip_bin.exists():
                # Uninstall onnx-asr
                subprocess.run(
                    [str(pip_bin), 'uninstall', '-y', 'onnx-asr'],
                    check=False,
                    capture_output=True
                )
                log_success("ONNX-ASR backend cleaned up")
            return True
        except Exception as e:
            log_warning(f"Cleanup warning: {e}")
            return True  # Don't fail on cleanup errors

    if backend_type == 'faster-whisper':
        log_info("Cleaning up faster-whisper backend...")
        pip_bin = VENV_DIR / 'bin' / 'pip'
        if pip_bin.exists():
            try:
                subprocess.run(
                    [str(pip_bin), 'uninstall', '-y', 'faster-whisper'],
                    check=False,
                    capture_output=True
                )
                log_success("faster-whisper backend cleaned up")
            except Exception as e:
                log_warning(f"Cleanup warning: {e}")
        return True

    if backend_type == 'cohere-transcribe':
        log_info("Cleaning up Cohere Transcribe backend...")
        pip_bin = VENV_DIR / 'bin' / 'pip'
        if pip_bin.exists():
            try:
                subprocess.run(
                    [str(pip_bin), 'uninstall', '-y', 'transformers', 'sentencepiece', 'protobuf', 'librosa'],
                    check=False,
                    capture_output=True
                )
                log_success("Cohere Transcribe backend cleaned up")
            except Exception as e:
                log_warning(f"Cleanup warning: {e}")
        return True

    if backend_type in ['rest-api', 'remote']:
        # REST API doesn't have venv, nothing to clean
        return True

    log_info(f"Cleaning up {backend_type.upper()} backend...")

    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        log_info("No venv found, nothing to clean")
        return True

    try:
        pip_bin = VENV_DIR / 'bin' / 'pip'
        if pip_bin.exists():
            # Uninstall pywhispercpp
            subprocess.run(
                [str(pip_bin), 'uninstall', '-y', 'pywhispercpp'],
                check=False,
                capture_output=True
            )
            log_success("Backend cleaned up")
        return True
    except Exception as e:
        log_warning(f"Cleanup warning: {e}")
        return True  # Don't fail on cleanup errors


def _load_existing_setup_config() -> dict:
    """Return current config settings (merged over defaults) for seeding setup
    prompt defaults, or {} when no config exists yet / on any error."""
    try:
        if CONFIG_FILE.exists():
            return ConfigManager().get_all_settings()
    except Exception:
        pass
    return {}


def _bool_default(cfg: dict, key: str, fallback: bool) -> bool:
    """Default for a Confirm prompt: the stored bool, else `fallback`."""
    val = cfg.get(key, fallback)
    return val if isinstance(val, bool) else fallback


def _choice_default(value, ordered_values, fallback: str) -> str:
    """1-based choice string for `value` within `ordered_values`, else `fallback`."""
    try:
        return str(ordered_values.index(value) + 1)
    except (ValueError, AttributeError):
        return fallback


# Reverse of the backend_map inside _prompt_backend_selection: name -> choice.
_BACKEND_CHOICE = {
    'onnx-asr': '1', 'faster-whisper': '2', 'pywhispercpp': '3', 'cpu': '3',
    'nvidia': '4', 'vulkan': '5', 'cohere-transcribe': '6', 'rest-api': '7',
    'realtime-ws': '8',
}


def _prompt_backend_selection(existing_cfg: Optional[dict] = None):
    """Prompt user for backend selection with current state detection"""
    current_backend = _detect_current_backend(existing_cfg)

    print("\n" + "="*60)
    print("Backend Selection")
    print("="*60)

    if current_backend:
        backend_names = BACKEND_DISPLAY_NAMES
        print(f"\nCurrent backend: {backend_names.get(current_backend, current_backend)}")
    else:
        print("\nNo backend currently configured.")

    print("\nChoose your transcription backend:")
    print()
    print("Local In-Memory Backends:")
    print("  [1] Parakeet TDT V3       - Optimized for light hardware (autodetects CPU/GPU)")
    print("  [2] faster-whisper        - CTranslate2 + INT8 quantization, CPU or NVIDIA GPU")
    print("  [3] Whisper (CPU)         - whisper.cpp, works everywhere")
    print("  [4] Whisper (NVIDIA)      - whisper.cpp + CUDA, premium local transcription for NVIDIA GPUs")
    print("  [5] Whisper (Vulkan)      - whisper.cpp + Vulkan, ideal for AMD/Intel GPUs")
    print("  [6] Cohere Transcribe     - #1 Open ASR leaderboard, 14 languages, ~4-5 GB VRAM (fp16)")
    print()
    print("Cloud/REST Backends:")
    print("  [7] REST API              - OpenAI, Groq, or custom endpoint")
    print("  [8] Realtime WS           - Low-latency streaming")
    print()

    # Seed the prompt default with the currently configured/installed backend so
    # a re-run keeps it (Enter = keep). Prefer the verified install; fall back to
    # the raw config value so a configured backend still preselects even if the
    # venv check returned None.
    default_src = current_backend
    if not default_src:
        try:
            default_src = (
                existing_cfg.get('transcription_backend')
                if existing_cfg is not None
                else ConfigManager(verbose=False).get_setting('transcription_backend', None)
            )
        except Exception:
            default_src = None
    default_backend_choice = (_BACKEND_CHOICE.get(normalize_backend(default_src), '1')
                              if default_src else '1')

    while True:
        try:
            choice = Prompt.ask("Select backend", choices=['1', '2', '3', '4', '5', '6', '7', '8'], default=default_backend_choice)
            backend_map = {
                '1': 'onnx-asr',
                '2': 'faster-whisper',
                '3': 'cpu',
                '4': 'nvidia',
                '5': 'vulkan',
                '6': 'cohere-transcribe',
                '7': 'rest-api',
                '8': 'realtime-ws',
            }
            selected = backend_map[choice]

            # Backend display names for warnings/messages
            backend_names = {
                'onnx-asr': 'Parakeet TDT V3 (onnx-asr)',
                'cpu': 'Whisper CPU',
                'nvidia': 'Whisper NVIDIA (CUDA)',
                'faster-whisper': 'faster-whisper (CTranslate2)',
                'cohere-transcribe': 'Cohere Transcribe 2B',
                'amd': 'Whisper AMD/Intel (Vulkan)',
                'vulkan': 'Whisper AMD/Intel (Vulkan)',
                'rest-api': 'REST API',
                'realtime-ws': 'Realtime WebSocket',
                'pywhispercpp': 'pywhispercpp'
            }

            # Warn if switching to different backend
            switching_local_backends = False
            if current_backend and current_backend != selected:
                print(f"\n⚠️  Switching from {backend_names.get(current_backend, current_backend)} to {backend_names.get(selected, selected)}")

                if current_backend not in ['rest-api', 'remote', 'realtime-ws'] and selected not in ['rest-api', 'remote', 'realtime-ws']:
                    print("This will recreate the venv and install the new backend cleanly.")
                    if not Confirm.ask("Continue?", default=True):
                        continue
                    switching_local_backends = True
                elif selected in ['rest-api', 'realtime-ws']:
                    backend_type_name = 'REST API' if selected == 'rest-api' else 'Realtime WebSocket'
                    print(f"Switching to {backend_type_name} backend.")
                    print("The local backend venv will no longer be needed.")
                    cleanup_venv = Confirm.ask("Remove the venv to free up space?", default=False)
                    print(f"\n✓ Selected: {BACKEND_DISPLAY_NAMES.get(selected, selected)}")
                    return (selected, cleanup_venv, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)

            # If re-selecting same backend, offer reinstall option
            if current_backend == selected and selected in LOCAL_INSTALL_BACKENDS:
                print(f"\n{BACKEND_DISPLAY_NAMES.get(selected, selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reinstall, continue to return with wants_reinstall=True
            elif current_backend == selected and selected == 'realtime-ws':
                print(f"\n{BACKEND_DISPLAY_NAMES.get(selected, selected)} backend is already configured.")
                reconfigure = Confirm.ask("Reconfigure backend?", default=False)
                if not reconfigure:
                    print("Keeping existing configuration.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reconfigure, continue to return with wants_reinstall=True for correct state tracking
            elif current_backend == selected and selected in ['rest-api', 'remote']:
                print(f"\n{BACKEND_DISPLAY_NAMES.get(selected, selected)} backend is already configured.")
                reconfigure = Confirm.ask("Reconfigure backend?", default=False)
                if not reconfigure:
                    print("Keeping existing configuration.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reconfigure, continue to return with wants_reinstall=True for correct state tracking

            print(f"\n✓ Selected: {backend_names.get(selected, selected)}")
            # Force rebuild when: reinstalling same backend OR switching between local backends
            # This ensures a clean venv without stale packages from the previous backend
            wants_reinstall = (current_backend == selected) or switching_local_backends
            return (selected, False, wants_reinstall)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_model_selection(current_model: Optional[str] = None):
    """Prompt user for model selection"""
    print("\n" + "="*60)
    print("Model Selection")
    print("="*60)
    print("\nChoose your default Whisper model:")
    print()
    print("Multilingual models (support all languages, auto-detect):")
    for i, (model, desc) in enumerate(MULTILINGUAL_MODELS, 1):
        print(f"  [{i}] {model:15} - {desc}")

    print("\nEnglish-only models (smaller, faster, English only):")
    for i, (model, desc) in enumerate(ENGLISH_ONLY_MODELS, len(MULTILINGUAL_MODELS) + 1):
        print(f"  [{i}] {model:15} - {desc}")
    print()

    all_models = [m[0] for m in MULTILINGUAL_MODELS] + [m[0] for m in ENGLISH_ONLY_MODELS]
    choices = [str(i) for i in range(1, len(all_models) + 1)]
    
    while True:
        try:
            choice = Prompt.ask("Select model", choices=choices,
                                default=_choice_default(current_model, all_models, '2'))
            selected_model = all_models[int(choice) - 1]
            print(f"\n✓ Selected: {selected_model}")
            return selected_model
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_faster_whisper_model_selection(current_model: Optional[str] = None) -> str:
    """Prompt user for faster-whisper model selection"""
    print("\n" + "="*60)
    print("Model Selection")
    print("="*60)
    print("\nChoose a faster-whisper model:")
    print()
    for i, (model_name, size, notes) in enumerate(FASTER_WHISPER_MODELS, 1):
        print(f"  [{i}] {model_name:<22} {size:<10} {notes}")
    print()

    choices = [str(i) for i in range(1, len(FASTER_WHISPER_MODELS) + 1)]
    while True:
        try:
            choice = Prompt.ask("Select model", choices=choices,
                                default=_choice_default(current_model, [m[0] for m in FASTER_WHISPER_MODELS], '2'))
            selected = FASTER_WHISPER_MODELS[int(choice) - 1][0]
            print(f"\n✓ Selected: {selected}")
            return selected
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_realtime_provider_model_selection():
    """
    Prompt user for realtime provider and model selection in one flat list.

    Returns:
        Tuple of (provider_id, model_id, api_key, custom_config) or None if cancelled.
        custom_config contains websocket_url for custom WebSocket endpoints.
    """
    print("\n" + "="*60)
    print("Realtime Provider and Model Selection")
    print("="*60)
    print("\nChoose a realtime streaming provider and model:")
    print()

    realtime_options = []
    for provider_id, provider in PROVIDERS.items():
        if not provider.get('websocket_endpoint'):
            continue

        for model_id, model_data in provider.get('models', {}).items():
            if not model_data.get('realtime_model', False):
                continue
            realtime_options.append((provider_id, provider, model_id, model_data))
            print(
                f"  [{len(realtime_options)}] "
                f"{provider['name']}: {model_data['name']} - {model_data['description']}"
            )

    custom_choice = len(realtime_options) + 1
    print(f"  [{custom_choice}] Custom WebSocket endpoint")
    print()

    choices = [str(i) for i in range(1, custom_choice + 1)]

    while True:
        try:
            choice = Prompt.ask("Select provider and model", choices=choices, default='1')
            choice_num = int(choice)

            if choice_num == custom_choice:
                print("\n" + "="*60)
                print("Custom WebSocket Configuration")
                print("="*60)
                print("\nConfigure a custom realtime WebSocket backend.")
                print()

                websocket_url = Prompt.ask("WebSocket URL (e.g., wss://api.example.com/v1/realtime)", default="")
                if not websocket_url:
                    log_error("WebSocket URL is required for custom realtime backends")
                    if not Confirm.ask("Try again?", default=True):
                        return None
                    continue

                if not websocket_url.startswith('wss://') and not websocket_url.startswith('ws://'):
                    log_warning("WebSocket URL should start with wss:// or ws://")
                    if not Confirm.ask("Continue anyway?", default=True):
                        continue

                model_id = Prompt.ask("Model identifier", default="")
                if not model_id:
                    log_error("Model identifier is required for custom realtime backends")
                    if not Confirm.ask("Try again?", default=True):
                        return None
                    continue

                has_api_key = Confirm.ask("Do you have an API key?", default=False)
                api_key = None
                if has_api_key:
                    api_key = getpass.getpass("Enter API key: ")
                    if api_key:
                        save_credential('custom', api_key)

                custom_config = {'websocket_url': websocket_url}
                return ('custom', model_id, api_key, custom_config)

            provider_id, provider, model_id, model_data = realtime_options[choice_num - 1]
            print(f"\n✓ Selected: {provider['name']}: {model_data['name']}")

            existing_key = get_credential(provider_id)
            if existing_key:
                masked = mask_api_key(existing_key)
                print(f"\nFound existing API key: {masked}")
                use_existing = Confirm.ask("Use existing API key?", default=True)
                if use_existing:
                    api_key = existing_key
                else:
                    api_key = getpass.getpass(f"Enter {provider['api_key_description']}: ")
            else:
                api_key = getpass.getpass(f"Enter {provider['api_key_description']}: ")

            is_valid, error_msg = validate_api_key(provider_id, api_key)
            if not is_valid:
                log_warning(f"API key validation: {error_msg}")
                if not Confirm.ask("Continue anyway?", default=True):
                    continue

            if save_credential(provider_id, api_key):
                log_success("API key saved securely")
            else:
                log_warning("Failed to save API key, but continuing with configuration")

            return (provider_id, model_id, api_key, None)

        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_remote_provider_selection(filter_realtime: bool = False):
    """
    Prompt user for remote provider and model selection.
    
    Args:
        filter_realtime: If True, only show realtime-compatible models (for realtime-ws backend)
    
    Returns:
        Tuple of (provider_id, model_id, api_key, custom_config) or None if cancelled.
        custom_config is a dict with custom endpoint/headers/body if custom backend selected.
    """
    print("\n" + "="*60)
    print("Remote Provider Selection")
    print("="*60)
    print("\nChoose a cloud transcription provider:")
    print()
    
    # Build provider list
    all_providers_list = list_providers()
    
    # Filter providers if this is for realtime-ws (only show providers with websocket_endpoint)
    if filter_realtime:
        providers_list = []
        for provider_id, provider_name, _model_ids in all_providers_list:
            provider = get_provider(provider_id)
            if not (provider and provider.get('websocket_endpoint')):
                continue

            # Only show providers that have realtime-capable models
            models = get_provider_models(provider_id) or {}
            realtime_model_ids = [
                model_id
                for model_id, model_data in models.items()
                if model_data.get('realtime_model', False)
            ]
            if realtime_model_ids:
                providers_list.append((provider_id, provider_name, realtime_model_ids))
    else:
        # REST providers: only include providers with at least one REST-visible model
        # (i.e. not marked hidden in provider_registry)
        providers_list = []
        for provider_id, provider_name, _model_ids in all_providers_list:
            models = get_provider_models(provider_id) or {}
            visible_model_ids = [
                model_id
                for model_id, model_data in models.items()
                if not model_data.get('hidden', False)
            ]
            if visible_model_ids:
                providers_list.append((provider_id, provider_name, visible_model_ids))
    
    provider_choices = []
    
    for i, (provider_id, provider_name, model_ids) in enumerate(providers_list, 1):
        model_list = ', '.join(model_ids)
        print(f"  [{i}] {provider_name} ({model_list})")
        provider_choices.append((str(i), provider_id))
    
    print(f"  [{len(providers_list) + 1}] Customize your own backend")
    provider_choices.append((str(len(providers_list) + 1), 'custom'))
    
    print()
    
    choices = [str(i) for i in range(1, len(providers_list) + 2)]
    
    while True:
        try:
            choice = Prompt.ask("Select provider", choices=choices, default='1')
            choice_num = int(choice)
            
            if choice_num <= len(providers_list):
                # Known provider selected
                _, provider_id = provider_choices[choice_num - 1]
                provider = get_provider(provider_id)
                
                # Show models for this provider
                print("\n" + "="*60)
                print(f"{provider['name']} Models")
                print("="*60)
                print()
                
                models = get_provider_models(provider_id)
                model_list = []
                
                # Filter models based on backend type
                for model_id, model_data in models.items():
                    if filter_realtime:
                        # Only include realtime models (marked with realtime_model flag)
                        if not model_data.get('realtime_model', False):
                            continue
                    else:
                        # For REST API, hide models marked as hidden
                        if model_data.get('hidden', False):
                            continue
                    model_list.append((model_id, model_data))
                    print(f"  [{len(model_list)}] {model_data['name']} - {model_data['description']}")
                
                if not model_list:
                    if filter_realtime:
                        print("\n⚠ No realtime-compatible models found for this provider.")
                        if not Confirm.ask("Select a different provider?", default=True):
                            return None
                        continue
                    else:
                        print("\n⚠ No models found for this provider.")
                        if not Confirm.ask("Select a different provider?", default=True):
                            return None
                        continue
                
                print()
                model_choices = [str(i) for i in range(1, len(model_list) + 1)]
                model_choice = Prompt.ask("Select model", choices=model_choices, default='1')
                selected_model_id, selected_model_data = model_list[int(model_choice) - 1]
                
                print(f"\n✓ Selected: {selected_model_data['name']}")
                
                # Check for existing credential
                existing_key = get_credential(provider_id)
                if existing_key:
                    masked = mask_api_key(existing_key)
                    print(f"\nFound existing API key: {masked}")
                    use_existing = Confirm.ask("Use existing API key?", default=True)
                    if use_existing:
                        api_key = existing_key
                    else:
                        # Use getpass for secure password input (masks input, doesn't echo)
                        api_key = getpass.getpass(f"Enter {provider['api_key_description']}: ")
                else:
                    # Use getpass for secure password input (masks input, doesn't echo)
                    api_key = getpass.getpass(f"Enter {provider['api_key_description']}: ")
                
                # Validate API key
                is_valid, error_msg = validate_api_key(provider_id, api_key)
                if not is_valid:
                    log_warning(f"API key validation: {error_msg}")
                    if not Confirm.ask("Continue anyway?", default=True):
                        continue
                
                # Save credential
                if save_credential(provider_id, api_key):
                    log_success("API key saved securely")
                else:
                    log_warning("Failed to save API key, but continuing with configuration")
                
                return (provider_id, selected_model_id, api_key, None)
            
            else:
                # Custom backend selected
                print("\n" + "="*60)
                print("Custom Backend Configuration")
                print("="*60)
                print("\nConfigure a custom REST API backend.")
                print()

                endpoint_url = Prompt.ask("Endpoint URL", default="")
                if not endpoint_url:
                    log_error("Endpoint URL is required")
                    if not Confirm.ask("Try again?", default=True):
                        return None
                    continue

                # Validate URL format
                if not endpoint_url.startswith('http://') and not endpoint_url.startswith('https://'):
                    log_warning("URL should start with http:// or https://")
                    if not Confirm.ask("Continue anyway?", default=True):
                        continue

                # Model name - required for REST APIs, optional for WebSocket
                print("\nModel name (required for REST APIs, leave blank for WebSocket)")
                print("Examples: whisper-1, whisper-large-v3, distil-whisper-large-v3-en")
                model_name = Prompt.ask("Model name", default="") or None

                # Optional API key
                has_api_key = Confirm.ask("Do you have an API key?", default=False)
                api_key = None
                if has_api_key:
                    # Use getpass for secure password input (masks input, doesn't echo to terminal)
                    api_key = getpass.getpass("Enter API key: ")
                    # Save as 'custom' provider
                    if api_key:
                        save_credential('custom', api_key)

                # Optional custom headers
                has_headers = Confirm.ask("Add custom HTTP headers?", default=False)
                custom_headers = {}
                if has_headers:
                    headers_json = Prompt.ask("Enter headers as JSON (e.g., {\"authorization\": \"Bearer token\"})", default="{}")
                    try:
                        custom_headers = json.loads(headers_json)
                        if not isinstance(custom_headers, dict):
                            log_error("Headers must be a JSON object")
                            custom_headers = {}
                    except json.JSONDecodeError as e:
                        log_error(f"Invalid JSON: {e}")
                        custom_headers = {}

                # Optional additional body fields
                has_body = Confirm.ask("Add additional body fields?", default=False)
                custom_body = {'model': model_name} if model_name else {}
                if has_body:
                    body_json = Prompt.ask("Enter additional body fields as JSON (e.g., {\"language\": \"en\"})", default="{}")
                    try:
                        extra_body = json.loads(body_json)
                        if isinstance(extra_body, dict):
                            custom_body.update(extra_body)
                        else:
                            log_error("Body fields must be a JSON object")
                    except json.JSONDecodeError as e:
                        log_error(f"Invalid JSON: {e}")

                custom_config = {
                    'endpoint': endpoint_url,
                    'headers': custom_headers,
                    'body': custom_body
                }

                return ('custom', model_name, api_key, custom_config)
                
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _generate_remote_config(provider_id: str, model_id: Optional[str], api_key: str, custom_config: Optional[dict] = None, backend_type: str = 'rest-api') -> dict:
    """
    Generate remote backend configuration based on provider/model selection.
    
    Args:
        provider_id: Provider identifier (e.g., 'openai', 'groq', 'custom')
        model_id: Model identifier (None for custom backends)
        api_key: API key to use
        custom_config: Custom config dict for custom backends
        backend_type: Backend type ('rest-api' or 'realtime-ws')
    
    Returns:
        Configuration dictionary ready to be saved
    """
    if backend_type == 'realtime-ws':
        config = {
            'transcription_backend': 'realtime-ws',
            'websocket_provider': provider_id,
            'websocket_model': model_id
        }
        # For custom backends, include websocket_url from custom_config
        if provider_id == 'custom' and custom_config and 'websocket_url' in custom_config:
            config['websocket_url'] = custom_config['websocket_url']
        return config
    
    config = {
        'transcription_backend': 'rest-api'
    }
    
    if custom_config:
        # Custom backend
        config['rest_endpoint_url'] = custom_config['endpoint']
        if api_key:
            # Store provider identifier instead of API key
            # API key is already saved securely via credential_manager
            config['rest_api_provider'] = 'custom'
        if custom_config.get('headers'):
            config['rest_headers'] = custom_config['headers']
        if custom_config.get('body'):
            config['rest_body'] = custom_config['body']
    else:
        # Known provider
        model_config = get_model_config(provider_id, model_id)
        if not model_config:
            raise ValueError(f"Invalid provider/model combination: {provider_id}/{model_id}")
        
        config['rest_endpoint_url'] = model_config['endpoint']
        # Store provider identifier instead of API key
        # API key is already saved securely via credential_manager
        config['rest_api_provider'] = provider_id
        config['rest_body'] = model_config['body'].copy()
    
    return config


def _setup_command_symlink():
    """Offer to create ~/.local/bin/hyprwhspr symlink for git clone installs"""
    # Only relevant for non-package installs (git clones)
    if HYPRWHSPR_ROOT == '/usr/lib/hyprwhspr':
        return  # Package install, symlink not needed

    local_bin = USER_HOME / '.local' / 'bin'
    symlink_path = local_bin / 'hyprwhspr'
    source_path = Path(HYPRWHSPR_ROOT) / 'bin' / 'hyprwhspr'

    # Check if symlink already exists and points to correct location
    if symlink_path.is_symlink():
        try:
            if symlink_path.resolve() == source_path.resolve():
                log_info(f"Command symlink already configured: {symlink_path}")
                return
        except Exception:
            pass

    # Check if there's already a hyprwhspr in PATH that's not our symlink
    existing = shutil.which('hyprwhspr')
    if existing and Path(existing).resolve() != source_path.resolve():
        log_info(f"hyprwhspr already in PATH: {existing}")
        return

    print("\n" + "="*60)
    print("Command Setup")
    print("="*60)
    print(f"\nInstallation detected at: {HYPRWHSPR_ROOT}")
    print(f"Create symlink so 'hyprwhspr' command works from anywhere?")
    print(f"  {symlink_path} -> {source_path}")

    if Confirm.ask("Create command symlink?", default=True):
        try:
            local_bin.mkdir(parents=True, exist_ok=True)
            # Remove existing symlink if it points elsewhere
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(source_path)
            log_success(f"Created symlink: {symlink_path}")

            # Check if ~/.local/bin is in PATH
            path_dirs = os.environ.get('PATH', '').split(':')
            if str(local_bin) not in path_dirs:
                log_warning(f"{local_bin} is not in your PATH")
                log_info("Add this to ~/.bashrc or ~/.zshrc:")
                log_info(f'  export PATH="$HOME/.local/bin:$PATH"')
        except Exception as e:
            log_warning(f"Failed to create symlink: {e}")
            log_info(f"You can create it manually:")
            log_info(f"  ln -sf {source_path} {symlink_path}")


def setup_command(python_path: Optional[str] = None):
    """Interactive full initial setup

    Args:
        python_path: Optional path to Python executable to use for venv creation.
                     If None, auto-detects a compatible Python (3.14 or earlier).
    """
    print("\n" + "="*60)
    print("hyprwhspr setup")
    print("="*60)
    print("\nThis setup will guide you through configuring hyprwhspr.")
    print("Skip any step by answering 'no'.\n")

    # Check for MISE interference and handle automatically
    mise_active, _ = _check_mise_active()
    if mise_active:
        # Try to deactivate MISE (may be a shell function)
        if shutil.which('mise'):
            try:
                run_command(['bash', '-c', 'mise deactivate'], check=False, capture_output=True)
            except Exception:
                pass
        log_info("MISE deactivated for installation")

    # Early Python version compatibility check
    # This warns users upfront if their Python is too new for local ML backends
    if not python_path:  # Only check if user didn't specify a custom Python
        is_compatible, version_str, guidance = _check_python_compatibility()
        if guidance:
            # Either found alternative Python or need to warn user
            if is_compatible:
                log_info(f"Note: {guidance}")
            else:
                print("\n" + "="*60)
                print("Python Version Warning")
                print("="*60)
                log_warning(f"System Python {version_str} detected")
                print()
                print(guidance)
                print()
                if not Confirm.ask("Continue anyway? (Cloud backends will still work)", default=True):
                    log_info("Setup cancelled.")
                    log_info("Install Python 3.14 or 3.13, then re-run: hyprwhspr setup")
                    return
                print()

    # Setup command symlink for git clone installs
    _setup_command_symlink()

    # Load the existing config (if any) so each prompt below can default to the
    # current value — re-running setup then keeps settings on Enter.
    existing_cfg = _load_existing_setup_config()

    # Step 1: Backend selection (now returns tuple: (backend, cleanup_venv))
    backend_result = _prompt_backend_selection(existing_cfg)
    if not backend_result:
        log_error("Backend selection is required. Exiting.")
        return
    
    # Handle tuple or string return (backward compatibility)
    if isinstance(backend_result, tuple):
        if len(backend_result) == 3:
            backend, cleanup_venv, wants_reinstall = backend_result
        elif len(backend_result) == 2:
            backend, cleanup_venv = backend_result
            wants_reinstall = False
        else:
            backend = backend_result[0]
            cleanup_venv = False
            wants_reinstall = False
    else:
        backend = backend_result
        cleanup_venv = False
        wants_reinstall = False
    
    current_backend = _detect_current_backend(existing_cfg)
    
    # Normalize backends for comparison (handles 'amd' -> 'vulkan' mapping)
    if current_backend:
        current_backend = normalize_backend(current_backend)
    backend_normalized = normalize_backend(backend)
    
    # Handle backend switching
    if current_backend and current_backend != backend_normalized:
        if current_backend not in ['rest-api', 'remote', 'realtime-ws']:
            # Switching from local to something else
            if not _cleanup_backend(current_backend):
                log_warning("Failed to clean up old backend, continuing anyway...")
        
        
        if cleanup_venv and backend_normalized in ['rest-api', 'remote', 'realtime-ws']:
            # User wants to remove venv when switching to cloud backend
            if VENV_DIR.exists():
                log_info("Removing venv as requested...")
                shutil.rmtree(VENV_DIR)
                log_success("Venv removed")
    
    # Step 1.5: Backend installation (if not cloud backend)
    backend_install_skipped = False
    if backend_normalized not in ['rest-api', 'remote', 'realtime-ws']:
        # Skip installation section if user selected the same backend and declined reinstalling
        if current_backend == backend_normalized and not wants_reinstall:
            # User already said "no" to reinstalling in the selection step, skip installation section
            pass
        else:
            # New backend selected, or user wants to reinstall existing backend
            print("\n" + "="*60)
            print("Backend Installation")
            print("="*60)
            if backend_normalized == 'onnx-asr':
                print("\nThis will install the ONNX-ASR backend for hyprwhspr.")
                print("This backend automatically detects and uses GPU acceleration when available,")
                print("or falls back to CPU-optimized mode. Uses ONNX runtime for fast transcription.")
                print("This may take several minutes as it downloads models and dependencies.")
            elif backend_normalized == 'faster-whisper':
                print("\nThis will install the faster-whisper backend (CTranslate2).")
                print("Works on CPU and NVIDIA GPUs. INT8 quantization is fast and memory-efficient.")
                print("On NVIDIA: large-v3-turbo runs in ~3.1 GB VRAM. On CPU: INT8 is faster than pywhispercpp.")
                print("Built-in Silero VAD reduces hallucination loops on longer recordings.")
                print("Models are downloaded automatically on first use.")
            elif backend_normalized == 'cohere-transcribe':
                print("\nThis will install the Cohere Transcribe backend (transformers).")
                print("2B parameter Conformer model — #1 on the Open ASR Leaderboard (English), 14 languages.")
                print("Requires ~4-5 GB VRAM with fp16 (default) or ~8-9 GB with fp32.")
                print("Falls back to CPU if no GPU is available (slow — not recommended for live dictation).")
                print("Model weights (~4 GB) will be downloaded from HuggingFace during setup.")
                print("This may take several minutes depending on your connection speed.")
            else:
                print(f"\nThis will install the {backend_normalized.upper()} backend for pywhispercpp.")
                print("This may take several minutes as it compiles from source.")
            if not Confirm.ask("Proceed with backend installation?", default=True):
                log_warning("Skipping backend installation. You can install it later.")
                log_warning("Backend installation is required for local transcription to work.")
                backend_install_skipped = True
            else:
                backend_install_skipped = False

                # Cohere Transcribe is a gated HuggingFace model — each user must accept
                # the license and authenticate with their own token before downloading.
                if backend_normalized == 'cohere-transcribe':
                    existing_hf_token = get_credential('huggingface')
                    if existing_hf_token:
                        log_success("HuggingFace token already saved — skipping prompt")
                    else:
                        print("\nCohere Transcribe is a gated model on HuggingFace.")
                        print("Before continuing:")
                        print("  1. Accept the license at: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026")
                        print("  2. Generate a read token at: https://huggingface.co/settings/tokens")
                        hf_token = Prompt.ask("\nHuggingFace token", password=True)
                        if hf_token and hf_token.strip():
                            token_clean = hf_token.strip()
                            # Basic format check: HF read tokens start with "hf_"
                            if not token_clean.startswith('hf_'):
                                log_warning("Token doesn't start with 'hf_' — double-check you copied a read token")
                            else:
                                masked = token_clean[:6] + '*' * (len(token_clean) - 9) + token_clean[-3:]
                                log_success(f"Token received: {masked} ({len(token_clean)} chars)")
                            save_credential('huggingface', token_clean)
                            log_success("HuggingFace token securely saved")
                        else:
                            log_warning("No token provided — model download will likely fail")

                # Pass force_rebuild=True when reinstalling to ensure clean venv
                # Use normalized backend to ensure 'amd' -> 'vulkan' for new installs
                if not install_backend(backend_normalized, force_rebuild=wants_reinstall, custom_python=python_path):
                    log_error("Backend installation failed. Setup cannot continue.")
                    return
                
    
    # Step 2: Provider/model selection (if REST API backend)
    remote_config = None
    selected_model = None
    faster_whisper_model = None
    if backend_normalized in ['rest-api', 'remote']:
        # Prompt for remote provider selection
        provider_result = _prompt_remote_provider_selection()
        if not provider_result:
            log_error("Provider selection cancelled. Exiting.")
            return
        
        provider_id, model_id, api_key, custom_config = provider_result
        
        # Generate remote configuration
        try:
            remote_config = _generate_remote_config(provider_id, model_id, api_key, custom_config, backend_type='rest-api')
            log_success("Remote configuration generated")
        except Exception as e:
            log_error(f"Failed to generate remote configuration: {e}")
            return
    elif backend_normalized == 'realtime-ws':
        provider_result = _prompt_realtime_provider_model_selection()
        if not provider_result:
            log_error("Provider selection cancelled. Exiting.")
            return
        
        provider_id, model_id, api_key, custom_config = provider_result
        
        # Generate realtime-ws configuration
        try:
            remote_config = _generate_remote_config(provider_id, model_id, api_key, custom_config, backend_type='realtime-ws')
            log_success("Realtime WebSocket configuration generated")
        except Exception as e:
            log_error(f"Failed to generate realtime configuration: {e}")
            return
        
        remote_config['realtime_mode'] = 'transcribe'
        log_info("Realtime mode: transcribe")
    
    # Step 1.4: Ensure venv and base dependencies for cloud backends
    if backend_normalized in ['rest-api', 'remote', 'realtime-ws']:
        print("\n" + "="*60)
        print("Python Environment Setup")
        print("="*60)
        log_info("Ensuring Python virtual environment and dependencies are installed...")
        
        try:
            from .backend_installer import setup_python_venv, compute_file_hash, get_state, set_state, HYPRWHSPR_ROOT
        except ImportError:
            from backend_installer import setup_python_venv, compute_file_hash, get_state, set_state, HYPRWHSPR_ROOT
        
        # Setup venv (creates if needed, updates if exists)
        pip_bin = setup_python_venv()
        
        # Check if requirements.txt has changed
        requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
        cur_req_hash = compute_file_hash(requirements_file)
        stored_req_hash = get_state("requirements_hash")
        
        # Check if base dependencies are installed (excluding pywhispercpp)
        deps_installed = False
        try:
            python_bin = VENV_DIR / 'bin' / 'python'
            result = run_command([
                'timeout', '5s', str(python_bin), '-c',
                'import sounddevice, numpy, requests; import websocket; import elevenlabs'
            ], check=False, capture_output=True, show_output_on_error=False)
            deps_installed = result.returncode == 0
        except Exception:
            pass
        
        # Install base dependencies if needed (excluding pywhispercpp)
        if cur_req_hash != stored_req_hash or not stored_req_hash or not deps_installed:
            if not stored_req_hash:
                # First time setup - no stored hash means venv is new
                log_info("Installing base Python dependencies (excluding pywhispercpp)...")
            elif cur_req_hash != stored_req_hash:
                # Requirements actually changed
                log_info("Requirements.txt has changed. Updating base Python dependencies...")
            else:
                # Dependencies missing but hash matches (shouldn't happen often)
                log_info("Installing missing base Python dependencies...")
            
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_req:
                temp_req_path = Path(temp_req.name)
                try:
                    with open(requirements_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            # Skip pywhispercpp - not needed for cloud backends
                            if not line.strip().startswith('pywhispercpp'):
                                temp_req.write(line)
                    
                    temp_req.flush()
                    
                    if temp_req_path.stat().st_size > 0:
                        run_command([str(pip_bin), 'install', '-r', str(temp_req_path)], check=True)
                    else:
                        log_warning("No dependencies to install (all excluded)")
                except Exception as e:
                    log_error(f"Failed to install base dependencies: {e}")
                    log_warning("Continuing anyway - dependencies may be missing")
                finally:
                    # Clean up temp file
                    if temp_req_path.exists():
                        temp_req_path.unlink()
            
            set_state("requirements_hash", cur_req_hash)
            log_success("Base Python dependencies installed")
        else:
            log_info("Base Python dependencies up to date")
    
    # Model selection for local backends
    if not backend_install_skipped:
        if backend_normalized not in ['rest-api', 'remote', 'realtime-ws', 'onnx-asr', 'faster-whisper', 'cohere-transcribe']:
            # Local backend - prompt for model selection
            # Note: ONNX-ASR, faster-whisper, and cohere-transcribe don't use Whisper.cpp models
            selected_model = _prompt_model_selection(current_model=existing_cfg.get('model'))
        elif backend_normalized == 'faster-whisper':
            faster_whisper_model = _prompt_faster_whisper_model_selection(current_model=existing_cfg.get('faster_whisper_model'))
            if download_faster_whisper_model(faster_whisper_model):
                log_success(f"Model {faster_whisper_model} downloaded successfully")
            else:
                log_warning(f"Model download failed. You can download it later with:")
                log_warning(f"  hyprwhspr model download {faster_whisper_model}")
    
    # Step 3: Bar integration - offer whichever supported shells are detected
    print("\n" + "="*60)
    print("Bar Integration")
    print("="*60)
    waybar_config_path = Path.home() / '.config' / 'waybar' / 'config.jsonc'
    waybar_style_path = Path.home() / '.config' / 'waybar' / 'style.css'
    waybar_installed = waybar_config_path.exists() or waybar_style_path.exists()
    noctalia_installed = _noctalia_detected()

    setup_waybar_choice = False
    setup_noctalia_choice = False
    if waybar_installed:
        print(f"\nWaybar configuration detected at: {waybar_config_path.parent}")
        setup_waybar_choice = Confirm.ask("Configure Waybar integration?", default=True)
    if noctalia_installed:
        print("\nNoctalia shell detected.")
        setup_noctalia_choice = Confirm.ask(
            "Configure Noctalia integration (bar widget + mic-OSD theme sync)?",
            default=True)
    if not waybar_installed and not noctalia_installed:
        print("\nNo supported bar detected (Waybar, Noctalia).")
        setup_waybar_choice = Confirm.ask("Set up Waybar integration anyway?", default=False)
    
    # Step 3b: Recording status indicator setup
    print("\n" + "="*60)
    print("Recording Status Indicator")
    print("="*60)

    # GNOME/Mutter does not implement the layer-shell protocol the animated
    # overlay needs. There the overlay degrades to a focus-stealing toplevel
    # window that swallows the post-dictation paste keystroke, so recording
    # status is shown as desktop notifications instead (these never take focus).
    is_mutter_session = _is_gnome_or_mutter_session()

    if is_mutter_session:
        print("\nGNOME/Mutter detected. The animated overlay needs the layer-shell")
        print("protocol, which Mutter does not support, so recording status is shown")
        print("as desktop notifications instead (they never steal keyboard focus).")
        print("This needs 'notify-send' (libnotify) — no GTK4/gtk4-layer-shell required.")
        print("Note: ASCII text is typed directly on US layouts; set prefer_clipboard_paste: true")
        print("to route dictation through clipboard managers such as Cliphist or CopyQ.")

        if shutil.which('notify-send') is None:
            if Path('/etc/debian_version').exists():
                notify_pkg = "libnotify-bin"
            elif Path('/etc/fedora-release').exists():
                notify_pkg = "libnotify"
            elif Path('/etc/os-release').exists() and 'suse' in Path('/etc/os-release').read_text(encoding='utf-8').lower():
                notify_pkg = "libnotify-tools"
            else:
                notify_pkg = "libnotify (Arch naming)"
            print(f"\nNote: 'notify-send' not found. Install: {notify_pkg}")

        setup_mic_osd_choice = Confirm.ask("Show recording status as desktop notifications?", default=_bool_default(existing_cfg, 'mic_osd_enabled', True))

        # GNOME picks the paste shortcut per app (terminal → Ctrl+Shift+V, GUI → Ctrl+V)
        # by detecting the focused window via AT-SPI, which needs the GNOME accessibility
        # bridge enabled. Without it, detection fails and paste falls back to Ctrl+V.
        print("\nGNOME can auto-pick the paste shortcut per app (terminal vs. GUI) via the")
        print("accessibility bridge. Without it, paste falls back to Ctrl+V (wrong in terminals).")
        if shutil.which('gsettings') and Confirm.ask(
            "Enable the GNOME accessibility bridge for automatic paste-shortcut detection?",
            default=True,
        ):
            result = run_command(
                ['gsettings', 'set', 'org.gnome.desktop.interface', 'toolkit-accessibility', 'true'],
                check=False,
            )
            if result.returncode == 0:
                print("Enabled (revert: gsettings set org.gnome.desktop.interface toolkit-accessibility false).")
                print("Apps started before this may need a restart to be detected.")
            else:
                print("Could not enable it automatically — run this manually:")
                print("  gsettings set org.gnome.desktop.interface toolkit-accessibility true")
    else:
        print("\nShows a visual overlay during recording with animated bars")
        print("and a pulsing indicator. Requires GTK4, PyCairo, and gtk4-layer-shell.")
        print("(On compositors without layer-shell support — e.g. GNOME/Mutter — the")
        print("status automatically falls back to desktop notifications instead.)")

        # Check if dependencies are available using service's Python
        mic_osd_available, mic_osd_reason = _check_mic_osd_availability()
        if not mic_osd_available:
            print(f"\nNote: {mic_osd_reason}")

        if mic_osd_available:
            setup_mic_osd_choice = Confirm.ask("Enable mic-osd visualization?", default=_bool_default(existing_cfg, 'mic_osd_enabled', True))
        else:
            # Provide distro-appropriate package names
            if Path('/etc/debian_version').exists():
                pkg_hint = "python3-gi python3-cairo gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0"
            elif Path('/etc/fedora-release').exists():
                pkg_hint = "python3-gobject python3-cairo gtk4 gtk4-layer-shell"
            elif Path('/etc/os-release').exists() and 'suse' in Path('/etc/os-release').read_text(encoding='utf-8').lower():
                pkg_hint = "python3-gobject python3-pycairo typelib-1_0-Gtk-4_0 (gtk4-layer-shell from community repo)"
            else:
                pkg_hint = "python-gobject python-cairo gtk4 gtk4-layer-shell (Arch naming)"
            print(f"\nDependencies not found. Install: {pkg_hint}")
            setup_mic_osd_choice = Confirm.ask("Enable mic-osd anyway (will work after installing deps)?", default=_bool_default(existing_cfg, 'mic_osd_enabled', False))

    # Step 3c: Audio ducking setup
    print("\n" + "="*60)
    print("Audio Ducking")
    print("="*60)
    print("\nAutomatically reduces system volume while recording to prevent")
    print("audio interference with your microphone.")

    setup_audio_ducking_choice = Confirm.ask("Enable audio ducking?", default=_bool_default(existing_cfg, 'audio_ducking', True))
    audio_ducking_percent = 50  # Default
    if setup_audio_ducking_choice:
        print("\nHow much to reduce volume BY during recording?")
        print("  50 = reduce to 50% of original (recommended)")
        print("  70 = reduce to 30% of original (aggressive)")
        print("  30 = reduce to 70% of original (subtle)")
        ducking_input = Prompt.ask("Reduction percentage", default=str(existing_cfg.get('audio_ducking_percent', 50)))
        try:
            audio_ducking_percent = max(0, min(100, int(ducking_input)))
        except ValueError:
            audio_ducking_percent = 50
            log_warning("Invalid input, using default 50%")

    # Step 3d: Hyprland compositor bindings
    # Detect if user is running Hyprland
    is_hyprland_session = os.environ.get('HYPRLAND_INSTANCE_SIGNATURE') is not None
    current_desktop = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
    hypr_config_dir = USER_HOME / '.config' / 'hypr'
    hypr_config_exists = (hypr_config_dir / 'hyprland.conf').exists() or (hypr_config_dir / 'bindings.conf').exists()

    # Only show Hyprland section if relevant
    if is_hyprland_session or hypr_config_exists or 'hyprland' in current_desktop:
        print("\n" + "="*60)
        print("Hyprland Compositor Bindings")
        print("="*60)
        print("\nUse Hyprland's native compositor bindings instead of evdev keyboard grabbing.")
        print("Better compatibility with keyboard remappers.")
        print("Requires adding bindings to ~/.config/hypr/hyprland.conf or bindings.conf")

        if is_hyprland_session:
            print("\nHyprland session detected.")
            setup_hyprland_choice = Confirm.ask("Configure Hyprland compositor bindings?", default=_bool_default(existing_cfg, 'use_hypr_bindings', True))
        elif hypr_config_exists:
            print(f"\nHyprland configuration detected at: {hypr_config_dir}")
            setup_hyprland_choice = Confirm.ask("Configure Hyprland compositor bindings?", default=_bool_default(existing_cfg, 'use_hypr_bindings', True))
        else:
            print("\nHyprland configuration not found.")
            setup_hyprland_choice = Confirm.ask("Set up Hyprland compositor bindings anyway?", default=_bool_default(existing_cfg, 'use_hypr_bindings', False))
    else:
        # Not a Hyprland system - skip this section entirely
        setup_hyprland_choice = False

    # Step 3e: Keyboard device allowlist (which keyboards the shortcut listens to).
    # Gate this so routine setup re-runs do not always show the device table.
    keyboard_cfg = existing_cfg
    current_allowlist = keyboard_cfg.get('keyboard_device_names') or []
    keyboard_allowlist_choice = None
    if Confirm.ask("Configure keyboard allowlist?",
                   default=not bool(current_allowlist)):
        # Returns names / [] (auto-detect) / None (unchanged). Prints its own header.
        keyboard_allowlist_choice = _run_keyboard_selection(keyboard_cfg)

    # Step 4: Systemd setup
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)
    print("\nSystemd user service will run hyprwhspr in the background.")
    print("This will enable/configure:")
    print("  • hyprwhspr.service (main application)")
    setup_systemd_choice = Confirm.ask("Set up systemd user service?", default=True)
    
    # Step 5: Permissions setup
    print("\n" + "="*60)
    print("Permissions Setup")
    print("="*60)
    print("\nAdds you to required groups (input, audio, tty)")
    print("and configures uinput device permissions.")
    print("Note: Requires sudo access.")
    setup_permissions_choice = Confirm.ask("Set up permissions?", default=True)
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print(f"\nBackend: {backend_normalized}")
    if remote_config:
        if backend_normalized == 'realtime-ws':
            provider_id = remote_config.get('websocket_provider')
            print(f"Provider: {provider_id or 'N/A'}")
            print(f"Model: {remote_config.get('websocket_model', 'N/A')}")
            if remote_config.get('websocket_url'):
                print(f"WebSocket URL: {remote_config['websocket_url']}")
            print("Realtime mode: transcribe (change with: hyprwhspr config edit)")
            if provider_id:
                api_key = get_credential(provider_id)
                if api_key:
                    masked = mask_api_key(api_key)
                    print(f"API Key: {masked}")
                elif provider_id != 'custom':
                    print(f"API Key: not found in credential store for provider {provider_id}")
        else:
            print(f"Endpoint: {remote_config.get('rest_endpoint_url', 'N/A')}")
            if remote_config.get('rest_body'):
                model_name = remote_config['rest_body'].get('model', 'N/A')
                print(f"Model: {model_name}")
            provider_id = remote_config.get('rest_api_provider')
            if provider_id:
                # Retrieve and mask API key for display
                api_key = get_credential(provider_id)
                if api_key:
                    masked = mask_api_key(api_key)
                    print(f"Provider: {provider_id} (API Key: {masked})")
                else:
                    print(f"Provider: {provider_id} (API Key: not found in credential store)")
            # Backward compatibility: check for old rest_api_key
            elif remote_config.get('rest_api_key'):
                api_key = remote_config.get('rest_api_key')
                masked = mask_api_key(api_key)
                print(f"API Key: {masked} (legacy - will be migrated)")
    elif faster_whisper_model:
        print(f"Model: {faster_whisper_model}")
    elif selected_model:
        print(f"Model: {selected_model}")
    elif backend_normalized == 'onnx-asr':
        print("Model: nemo-parakeet-tdt-0.6b-v3 (~1 GB, downloaded during setup)")
    elif backend_normalized == 'cohere-transcribe':
        print("Model: CohereLabs/cohere-transcribe-03-2026 (~4 GB, downloaded during setup)")
    print(f"Waybar integration: {'Yes' if setup_waybar_choice else 'No'}")
    if noctalia_installed:
        print(f"Noctalia integration: {'Yes' if setup_noctalia_choice else 'No'}")
    _status_label = "Recording status (notifications)" if is_mutter_session else "Mic-OSD visualization"
    if setup_mic_osd_choice and not is_mutter_session and not mic_osd_available:
        print(f"{_status_label}: Yes (enabled, but dependencies missing: {mic_osd_reason})")
    else:
        print(f"{_status_label}: {'Yes' if setup_mic_osd_choice else 'No'}")
    if setup_audio_ducking_choice:
        print(f"Audio ducking: Yes ({audio_ducking_percent}% reduction)")
    else:
        print("Audio ducking: No")
    print(f"Hyprland compositor bindings: {'Yes' if setup_hyprland_choice else 'No'}")
    if keyboard_allowlist_choice:
        print(f"Keyboard allowlist: {', '.join(keyboard_allowlist_choice)}")
    elif keyboard_allowlist_choice == []:
        print("Keyboard allowlist: cleared (auto-detect)")
    if setup_systemd_choice:
        print("Systemd service: Yes (hyprwhspr)")
    else:
        print("Systemd service: No")
    print(f"Permissions: {'Yes' if setup_permissions_choice else 'No'}")

    # Paste mode detection notice — only shown when auto-detection won't work at runtime.
    # Per-app paste detection: Hyprland uses hyprctl, Niri uses niri msg
    # (NIRI_SOCKET), XWayland uses xdotool, and GNOME uses the AT-SPI accessibility
    # bridge (offered above). A pure-Wayland compositor with none of these can't
    # tell terminals from other apps, so terminal paste (Ctrl+Shift+V) isn't
    # auto-selected there.
    _has_hyprctl = bool(shutil.which('hyprctl'))
    _has_niri = bool(shutil.which('niri')) and bool(os.environ.get('NIRI_SOCKET'))
    _has_xdotool = bool(shutil.which('xdotool'))
    if not is_mutter_session and not _has_hyprctl and not _has_niri and not _has_xdotool:
        print("\nNote: Window detection unavailable on this system (no hyprctl, niri session, or xdotool).")
        print("Paste will default to Ctrl+V, which works in most apps but not terminals.")
        print("If a paste ever lands in the wrong place, you can override the key combo")
        print("with paste_mode — see docs/CONFIGURATION.md (Paste mode).")

    print()

    # Final confirmation
    if not Confirm.ask("Proceed with setup?", default=True):
        print("\nSetup cancelled.")
        return
    
    print("\n" + "="*60)
    print("Running Setup")
    print("="*60 + "\n")
    
    # Execute selected steps
    try:
        # Step 1: Config
        if remote_config:
            setup_config(backend=backend_normalized, remote_config=remote_config)
        else:
            setup_config(backend=backend_normalized, model=selected_model)
        if faster_whisper_model:
            config = ConfigManager()
            config.set_setting('faster_whisper_model', faster_whisper_model)
            config.save_config()
        
        # Step 2: Bar integration
        if setup_waybar_choice:
            setup_waybar('install')
        else:
            log_info("Skipping Waybar integration")
        if setup_noctalia_choice:
            setup_noctalia('install')
        elif noctalia_installed:
            log_info("Skipping Noctalia integration")
        
        # Step 2b: Mic-OSD
        if setup_mic_osd_choice:
            mic_osd_enable()
            if not is_mutter_session and not mic_osd_available:
                log_warning(f"Mic-OSD is enabled but unavailable until dependencies are installed: {mic_osd_reason}")
        else:
            mic_osd_disable()
            log_info("Mic-OSD visualization disabled")

        # Step 2c: Audio ducking
        config = ConfigManager()
        config.set_setting('audio_ducking', setup_audio_ducking_choice)
        if setup_audio_ducking_choice:
            config.set_setting('audio_ducking_percent', audio_ducking_percent)
            log_success(f"Audio ducking enabled ({audio_ducking_percent}% reduction)")
        else:
            log_info("Audio ducking disabled")
        config.save_config()

        # Step 2c-kb: Keyboard device allowlist
        if keyboard_allowlist_choice is not None:
            kbcfg = ConfigManager()
            kbcfg.set_setting('keyboard_device_names', keyboard_allowlist_choice or None)
            kbcfg.save_config()
            if keyboard_allowlist_choice:
                log_success(f"Keyboard allowlist set ({len(keyboard_allowlist_choice)} device(s))")
            else:
                log_info("Keyboard allowlist cleared — using auto-detection")
            if not setup_systemd_choice:
                log_info("Restart hyprwhspr to apply: systemctl --user restart hyprwhspr.service")

        # Step 2d: Hyprland compositor bindings
        if setup_hyprland_choice:
            print("\n" + "="*60)
            print("Hyprland Compositor Bindings")
            print("="*60)
            # Update config to use Hyprland bindings
            config = ConfigManager()
            config.set_setting('use_hypr_bindings', True)
            config.set_setting('grab_keys', False)
            config.save_config()
            log_success("Configuration updated for Hyprland compositor bindings")
            
            # Add bindings to Hyprland config file
            if _setup_hyprland_bindings():
                log_success("Hyprland bindings added to config file")
            else:
                log_warning("Could not add Hyprland bindings automatically")
                log_warning("See README for manual setup instructions")
        else:
            log_info("Skipping Hyprland compositor bindings setup")
        
        # Step 3: Systemd
        if setup_systemd_choice:
            setup_systemd('install')
            
            # Check if service was already running and restart to pick up config changes
            try:
                result = subprocess.run(
                    ['systemctl', '--user', 'is-active', SERVICE_NAME],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False
                )
                if result.returncode == 0:
                    # Service was already running, restart it
                    log_info("Restarting hyprwhspr service to apply configuration changes...")
                    systemd_restart()
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass  # Service check failed, continue
        else:
            log_info("Skipping systemd setup")
            # Warn if the service is running and won't pick up the new config automatically
            try:
                result = subprocess.run(
                    ['systemctl', '--user', 'is-active', SERVICE_NAME],
                    capture_output=True, text=True, timeout=2, check=False
                )
                if result.returncode == 0:
                    log_warning("Systemd service is running but setup was skipped.")
                    log_warning("Restart it for configuration changes to take effect:")
                    log_warning("  systemctl --user restart hyprwhspr")
                elif _is_running_manually():
                    log_warning("hyprwhspr appears to be running manually (not via systemd).")
                    log_warning("Restart it manually for configuration changes to take effect.")
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        # Step 4: Permissions
        if setup_permissions_choice:
            setup_permissions()
        else:
            log_info("Skipping permissions setup")
        
        # Step 5: Model download (if local backend)
        if backend not in ['rest-api', 'remote'] and selected_model:
            # If we got here and backend != 'remote', backend installation succeeded
            # (or was skipped, but user selected a model, so they want to use it)
            # Just download the model - we don't need to check if pywhispercpp is importable
            # (it's in the venv, and the service will use venv Python)
            print(f"\nDownloading model: {selected_model}")
            if download_model(selected_model):
                log_success(f"Model {selected_model} downloaded successfully")
            else:
                log_warning(f"Model download failed. You can download it later with:")
                log_warning(f"  hyprwhspr model download {selected_model}")
        
        # Step 6: Validation
        print("\n" + "="*60)
        print("Validation")
        print("="*60 + "\n")
        validate_command()
        
        print("\n" + "="*60)
        log_success("Setup completed!")
        print("="*60)
        print("\nNext steps:")
        if setup_permissions_choice:
            print("  If initial install, log out and back in (for group permissions)")
        if setup_systemd_choice:
            print("  Press hotkey to start dictation!")
        else:
            print("  Run hyprwhspr manually or set up systemd service later")
            print("  Press hotkey to start dictation!")
        print()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        log_info("Partial setup completed. You can resume by running 'hyprwhspr setup' again.")
        sys.exit(1)
    except Exception as e:
        log_error(f"Setup failed: {e}")
        log_debug(f"Full error traceback: {sys.exc_info()}")
        log_info("You can try running 'hyprwhspr backend repair' to fix issues, or 'hyprwhspr state reset' to start fresh.")
        sys.exit(1)


# ==================== Install Commands ====================

def _auto_download_model(model: str = 'base'):
    """Auto-download Whisper model without prompts

    Args:
        model: Model name to download (default: 'base')
    """
    try:
        from .backend_installer import download_pywhispercpp_model
    except ImportError:
        from backend_installer import download_pywhispercpp_model

    log_info(f"Downloading {model} Whisper model...")
    if download_pywhispercpp_model(model):
        log_success("Model downloaded")
    else:
        log_warning(f"Model download failed - can download later with: hyprwhspr model download {model}")


def _setup_hyprland_bindings() -> bool:
    """
    Set up Hyprland compositor bindings in config file.
    
    Returns:
        True if bindings were added successfully, False otherwise
    """
    hypr_config_dir = USER_HOME / '.config' / 'hypr'
    bindings_file = hypr_config_dir / 'bindings.conf'
    hyprland_conf = hypr_config_dir / 'hyprland.conf'
    
    # Determine which file to use
    target_file = None
    if bindings_file.exists():
        target_file = bindings_file
        log_info(f"Found bindings file: {bindings_file}")
    elif hyprland_conf.exists():
        target_file = hyprland_conf
        log_info(f"Found hyprland.conf, using it instead: {hyprland_conf}")
    else:
        # Create bindings.conf if neither exists
        target_file = bindings_file
        try:
            hypr_config_dir.mkdir(parents=True, exist_ok=True)
            log_info(f"Creating bindings file: {bindings_file}")
        except Exception as e:
            log_warning(f"Could not create Hyprland config directory: {e}")
            log_warning("Skipping Hyprland bindings setup - see README for manual setup")
            return False
    
    if target_file:
        # Check if bindings already exist
        bindings_exist = False
        try:
            if target_file.exists():
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check for existing hyprwhspr bindings
                    # Just check for the command - keybind could be anything
                    if 'hyprwhspr-tray.sh record' in content or \
                       '# added by hyprwhspr' in content:
                        bindings_exist = True
        except Exception as e:
            log_warning(f"Could not read {target_file}: {e}")
            log_warning("Skipping duplicate check - will attempt to add bindings")
        
        if bindings_exist:
            log_info("Hyprland bindings already exist, skipping")
            return True
        else:
            # Append bindings to file
            try:
                with open(target_file, 'a', encoding='utf-8') as f:
                    f.write('\n# hyprwhspr - Toggle mode (added by hyprwhspr setup)\n')
                    f.write('# Press once to start, press again to stop\n')
                    f.write(f'bindd = SUPER ALT, D, Speech-to-text, exec, {HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh record\n')
                log_success(f"Added Hyprland bindings to {target_file}")
                if shutil.which('hyprctl') and os.environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
                    run_command(['hyprctl', 'reload'], check=False)
                    log_success("Reloaded Hyprland config to apply bindings")
                else:
                    log_info("Restart Hyprland or reload config to apply bindings")
                return True
            except PermissionError:
                log_warning(f"Permission denied writing to {target_file}")
                log_warning("Could not add bindings automatically - see README for manual setup")
                return False
            except Exception as e:
                log_warning(f"Could not write to {target_file}: {e}")
                log_warning("Could not add bindings automatically - see README for manual setup")
                return False
    
    return False


def _verify_installation_step(step_name: str, verify_func) -> bool:
    """
    Generic helper to verify an installation step.
    
    Args:
        step_name: Human-readable name of the step
        verify_func: Function that returns True if verification passes, False otherwise
        
    Returns:
        True if verification passes, False otherwise
    """
    try:
        if verify_func():
            log_success(f"✓ {step_name} verified")
            return True
        else:
            log_error(f"✗ {step_name} verification failed")
            return False
    except Exception as e:
        log_error(f"✗ {step_name} verification error: {e}")
        return False


def _verify_backend_installation(backend: str) -> bool:
    """
    Verify that backend is actually importable from venv.

    Args:
        backend: Backend name (e.g., 'nvidia', 'vulkan', 'cpu', 'onnx-asr')

    Returns:
        True if backend is importable, False otherwise
    """
    if backend not in LOCAL_INSTALL_BACKENDS:
        # For non-local backends, skip import check
        return True

    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        return False

    import_module = BACKEND_IMPORT_MODULES[backend]

    try:
        result = subprocess.run(
            [str(venv_python), '-c', f'import {import_module}'],
            check=False,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _verify_config_created() -> bool:
    """
    Verify that config file exists and contains expected settings.
    
    Returns:
        True if config is valid, False otherwise
    """
    config_file = CONFIG_FILE
    if not config_file.exists():
        return False
    
    try:
        config = ConfigManager()
        # Check that essential settings exist
        backend = config.get_setting('transcription_backend')
        recording_mode = config.get_setting('recording_mode')
        return backend is not None and recording_mode is not None
    except Exception:
        return False


def _verify_service_running() -> bool:
    """
    Verify that systemd service is actually running.
    
    Returns:
        True if service is active, False otherwise
    """
    return _is_service_running_via_systemd()


def _verify_model_downloaded(model_name: str = 'base') -> bool:
    """
    Verify that model file exists and is readable.
    
    Args:
        model_name: Model name (default: 'base')
        
    Returns:
        True if model file exists, False otherwise
    """
    model_file = PYWHISPERCPP_MODELS_DIR / f'ggml-{model_name}.bin'
    return model_file.exists() and model_file.is_file()


def omarchy_command(args=None):
    """
    Automated setup

    This command:
    1. Auto-detects GPU hardware (NVIDIA/AMD/Intel/CPU) or uses specified backend
    2. Installs appropriate backend (CUDA for NVIDIA, Vulkan for others, CPU fallback)
    3. Configures defaults (auto recording mode, bar integration for detected shells)
    4. Sets up and starts systemd service
    5. Validates installation

    All without user interaction.

    Args:
        args: Optional argparse namespace with:
            - backend: 'nvidia', 'vulkan', 'cpu', or 'onnx-asr' (default: auto-detect)
            - model: Model name to download (default: 'base' for whisper, auto for onnx-asr)
            - no_waybar: Skip bar integration (Waybar/Noctalia)
            - no_mic_osd: Disable mic-osd visualization
            - no_systemd: Skip systemd service setup
            - hypr_bindings: Enable Hyprland compositor bindings
            - python_path: Path to Python executable for venv creation

    Note: Hyprland compositor bindings are NOT configured by default.
    Use 'hyprwhspr setup' for interactive setup with Hyprland compositor options,
    or use --hypr-bindings flag.
    """
    # Import functions we need
    try:
        from .backend_installer import detect_gpu_type, install_backend
        from .config_manager import ConfigManager
    except ImportError:
        from backend_installer import detect_gpu_type, install_backend
        from config_manager import ConfigManager

    # Extract CLI options with defaults for backwards compatibility
    explicit_backend = getattr(args, 'backend', None) if args else None
    explicit_model = getattr(args, 'model', None) if args else None
    skip_waybar = getattr(args, 'no_waybar', False) if args else False
    skip_mic_osd = getattr(args, 'no_mic_osd', False) if args else False
    skip_systemd = getattr(args, 'no_systemd', False) if args else False
    enable_hypr_bindings = getattr(args, 'hypr_bindings', False) if args else False
    python_path = getattr(args, 'python_path', None) if args else None

    # 1. Print banner
    print("\n" + "="*60)
    print("hyprwhspr - automated setup")
    print("="*60)

    # 2. Check and handle MISE
    mise_active, mise_details = _check_mise_active()
    mise_free_env = None
    if mise_active:
        log_warning("MISE detected - will be temporarily deactivated for installation")
        log_warning(f"Details:\n    {mise_details}")
        mise_free_env = _create_mise_free_environment()
        # Note: install_backend() already handles MISE warnings

    # 3. Determine backend (explicit or auto-detect)
    if explicit_backend:
        backend = explicit_backend
        log_info(f"Using specified backend: {backend.upper()}")
    else:
        log_info("Detecting hardware...")
        gpu_type = detect_gpu_type()  # Returns 'nvidia', 'vulkan', or 'cpu'
        backend = gpu_type

        gpu_descriptions = {
            'nvidia': 'NVIDIA GPU with CUDA acceleration',
            'vulkan': 'GPU with Vulkan acceleration (AMD/Intel/other)',
            'cpu': 'CPU-only (no GPU detected)'
        }

        log_success(f"Detected: {gpu_descriptions[gpu_type]}")

    log_info(f"Installing: {backend.upper()} backend")

    # 4. Install backend
    print("\n" + "="*60)
    print("Backend Installation")
    print("="*60)

    if not install_backend(backend, force_rebuild=False, custom_python=python_path):
        log_error("Backend installation failed")
        return False
    
    # 4.5. Verify backend installation
    print("\n" + "="*60)
    print("Verifying Backend Installation")
    print("="*60)
    if not _verify_installation_step("Backend installation", lambda: _verify_backend_installation(backend)):
        log_error("Backend installation verification failed - installation may be incomplete")
        return False

    # 5. Configure defaults
    log_info("Configuring defaults...")
    config = ConfigManager()
    config.set_setting('recording_mode', 'auto')

    # Configure backend-specific settings
    if backend == 'onnx-asr':
        config.set_setting('transcription_backend', 'onnx-asr')
        # Set onnx-asr model (defaults to parakeet)
        onnx_model = explicit_model or 'nemo-parakeet-tdt-0.6b-v3'
        config.set_setting('onnx_asr_model', onnx_model)
        log_info(f"Configured onnx-asr with model: {onnx_model}")
    else:
        config.set_setting('transcription_backend', 'pywhispercpp')
        # Set whisper model (defaults to 'base')
        whisper_model = explicit_model or 'base'
        config.set_setting('model', whisper_model)
        log_info(f"Configured pywhispercpp with model: {whisper_model}")

    # Configure mic-osd (enabled unless --no-mic-osd specified)
    config.set_setting('mic_osd_enabled', not skip_mic_osd)

    # Configure Hyprland bindings if requested
    if enable_hypr_bindings:
        config.set_setting('use_hypr_bindings', True)
        config.set_setting('grab_keys', False)
        log_info("Hyprland compositor bindings enabled")

    config.save_config()
    log_success("Configuration saved")
    
    # 5.5. Verify config creation
    if not _verify_installation_step("Config creation", _verify_config_created):
        log_error("Config verification failed - configuration may be incomplete")
        return False

    # 6. Download model (for local whisper backends only)
    # onnx-asr models download automatically on first use
    if backend in ['cpu', 'nvidia', 'vulkan']:
        model_to_download = explicit_model or 'base'
        _auto_download_model(model_to_download)
        # 6.5. Verify model download
        if not _verify_installation_step("Model download", lambda: _verify_model_downloaded(model_to_download)):
            log_warning("Model download verification failed - model may not be available")
            log_warning(f"You can download it later with: hyprwhspr model download {model_to_download}")
    elif backend == 'onnx-asr':
        log_info("onnx-asr model downloaded during setup")

    # 7. Bar integration (whichever supported shells are detected, unless skipped)
    print("\n" + "="*60)
    print("Bar Integration")
    print("="*60)

    if skip_waybar:
        log_info("Bar integration skipped (--no-waybar)")
    else:
        waybar_config = Path.home() / '.config' / 'waybar' / 'config.jsonc'
        if waybar_config.exists():
            log_info("Waybar detected - installing integration...")
            waybar_command('install')
        else:
            log_info("Waybar not detected - skipping")
        if _noctalia_detected():
            log_info("Noctalia detected - installing integration...")
            noctalia_command('install')
        else:
            log_info("Noctalia not detected - skipping")

    # 8. Systemd service (unless skipped)
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)

    if skip_systemd:
        log_info("Systemd service setup skipped (--no-systemd)")
    else:
        systemd_command('install')

        try:
            # Use MISE-free environment if MISE was detected
            env = mise_free_env if mise_free_env else None
            run_command(['systemctl', '--user', 'enable', 'hyprwhspr.service'], check=True, env=env)
            run_command(['systemctl', '--user', 'start', 'hyprwhspr.service'], check=True, env=env)
            log_success("Service enabled and started")
        except Exception as e:
            log_warning(f"Could not start service: {e}")

        # 8.5. Verify service is running
        print("\n" + "="*60)
        print("Verifying Service Status")
        print("="*60)
        if not _verify_installation_step("Service running", _verify_service_running):
            log_warning("Service verification failed - service may not be running")
            log_warning("Check service status with: systemctl --user status hyprwhspr")

    # 9. Validate
    print("\n" + "="*60)
    print("Validation")
    print("="*60)
    validate_command()

    # 10. Restart service for clean initialization (only if systemd was set up)
    if not skip_systemd:
        print("\n" + "="*60)
        print("Service Restart")
        print("="*60)
        log_info("Restarting service to ensure clean initialization...")

        # Check if service is actually running before restarting
        if _is_service_running_via_systemd():
            systemd_restart()
            log_success("Service restarted with clean state")
        else:
            log_warning("Service not running - skipping restart")

    # 11. Completion
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nAutomated setup completed successfully!")
    print("\nNext steps:")
    print("  1. Log out and back in (for group permissions)")
    print("  2. Press Super+Alt+D to start dictating")
    print("  3. Tap (<400ms) to toggle, hold (>=400ms) for push-to-talk")
    print("\nFor help: hyprwhspr --help")

    return True


# Config commands live in cli.config; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.config import setup_config
except ImportError:
    from cli.config import setup_config
# Systemd commands live in cli.systemd; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.systemd import (systemd_command, setup_systemd, systemd_restart,
                              _is_running_manually, _is_service_running_via_systemd,
                              _is_hyprwhspr_managed_ydotool_unit)
except ImportError:
    from cli.systemd import (systemd_command, setup_systemd, systemd_restart,
                             _is_running_manually, _is_service_running_via_systemd,
                             _is_hyprwhspr_managed_ydotool_unit)

# Waybar commands live in cli.waybar; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.waybar import waybar_command, setup_waybar, waybar_status
except ImportError:
    from cli.waybar import waybar_command, setup_waybar, waybar_status
# Noctalia commands live in cli.noctalia; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.noctalia import _noctalia_detected, noctalia_command, setup_noctalia
except ImportError:
    from cli.noctalia import _noctalia_detected, noctalia_command, setup_noctalia
# Mic-OSD commands live in cli.mic_osd; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.mic_osd import (_check_mic_osd_availability, mic_osd_enable,
                              mic_osd_disable)
except ImportError:
    from cli.mic_osd import (_check_mic_osd_availability, mic_osd_enable,
                             mic_osd_disable)
# Status command lives in cli.status.

# ==================== Permissions Setup ====================

def setup_permissions():
    """Setup permissions (requires sudo)"""
    log_info("Setting up permissions...")

    # Safer way to get username
    username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
    if not username:
        log_error("Could not determine username for permissions setup.")
        return False

    any_failures = False

    # Add user to required groups
    try:
        result = run_sudo_command(['usermod', '-a', '-G', 'input,audio,tty', username], check=False)
        if result.returncode == 0:
            log_success("Added user to required groups")
        else:
            log_warning(f"Failed to add user to groups (exit code {result.returncode})")
            log_info("You may need to run manually: sudo usermod -a -G input,audio,tty $USER")
            any_failures = True
    except Exception as e:
        log_warning(f"Failed to add user to groups: {e}")
        any_failures = True

    # Load uinput module
    if not Path('/dev/uinput').exists():
        log_info("Loading uinput module...")
        try:
            result = run_sudo_command(['modprobe', 'uinput'], check=False)
            if result.returncode != 0:
                log_warning("Failed to load uinput module")
                any_failures = True
        except Exception as e:
            log_warning(f"Failed to load uinput module: {e}")
            any_failures = True
        import time
        time.sleep(2)

    # Create udev rule.
    # On modern systemd the active-session user already gets /dev/uinput via a
    # `uaccess` ACL (so user-scope ydotoold runs rootless without this). This rule
    # is a stable fallback: uaccess only covers the *active* session. The `input`
    # group is needed mainly for the global hotkey, which reads /dev/input/event*
    # via evdev (no uaccess ACL there) — not for the ydotool paste path.
    udev_rule = Path('/etc/udev/rules.d/99-uinput.rules')
    if not udev_rule.exists():
        log_info("Creating udev rule...")
        rule_content = '# Allow members of the input group to access uinput device\nKERNEL=="uinput", GROUP="input", MODE="0660"\n'
        try:
            result = run_sudo_command(['tee', str(udev_rule)], input_data=rule_content.encode(), check=False)
            if result.returncode == 0:
                log_success("udev rule created")
            else:
                log_warning(f"Failed to create udev rule (exit code {result.returncode})")
                log_info("You may need to run manually: sudo tee /etc/udev/rules.d/99-uinput.rules")
                any_failures = True
        except Exception as e:
            log_warning(f"Failed to create udev rule: {e}")
            any_failures = True
    else:
        log_info("udev rule already exists")

    # Reload udev
    try:
        result1 = run_sudo_command(['udevadm', 'control', '--reload-rules'], check=False)
        result2 = run_sudo_command(['udevadm', 'trigger', '--name-match=uinput'], check=False)
        if result1.returncode == 0 and result2.returncode == 0:
            log_success("udev rules reloaded")
        else:
            log_warning("Failed to reload udev rules")
            log_info("You may need to run manually: sudo udevadm control --reload-rules && sudo udevadm trigger")
            any_failures = True
    except Exception as e:
        log_warning(f"Failed to reload udev rules: {e}")
        any_failures = True

    if any_failures:
        log_warning("Some permission setup commands failed. You may need to run them manually as root.")
    log_warning("You may need to log out/in for new group memberships to apply")


# Maintenance commands (backend repair/reset, state, validate) live in
# cli.maintenance; names still read by not-yet-moved sections are re-imported
# here until the split completes.
try:
    from .cli.maintenance import validate_command
except ImportError:
    from cli.maintenance import validate_command
# Test command lives in cli.test_cmd.

# Keyboard commands live in cli.keyboard; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.keyboard import _run_keyboard_selection
except ImportError:
    from cli.keyboard import _run_keyboard_selection

# ==================== Uninstall Command ====================

def uninstall_command(keep_models: bool = False, remove_permissions: bool = False,
                     skip_permissions: bool = False, yes: bool = False):
    """Completely remove hyprwhspr and all user data"""
    print("\n" + "="*60)
    print("hyprwhspr Uninstall")
    print("="*60)
    
    # Build summary of what will be removed
    items_to_remove = []
    
    # Systemd services
    if (USER_SYSTEMD_DIR / SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {SERVICE_NAME}")
    if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {RESUME_SERVICE_NAME} (deprecated)")
    ydotool_unit_path = USER_SYSTEMD_DIR / YDOTOOL_UNIT
    if _is_hyprwhspr_managed_ydotool_unit(ydotool_unit_path):
        items_to_remove.append(f"Systemd service: {YDOTOOL_UNIT}")

    # Waybar integration
    waybar_module = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    if waybar_module.exists():
        items_to_remove.append("Waybar integration")
    
    # Plain filesystem targets: declared once here, listed in the summary and
    # removed in one pass below (dirs rmtree'd, files/symlinks unlinked)
    fs_targets = []
    if USER_CONFIG_DIR.exists():
        fs_targets.append((f"User configuration: {USER_CONFIG_DIR}", USER_CONFIG_DIR))
    if VENV_DIR.exists():
        fs_targets.append((f"Main backend venv: {VENV_DIR}", VENV_DIR))
    if PYWHISPERCPP_SRC_DIR.exists():
        fs_targets.append((f"pywhispercpp source: {PYWHISPERCPP_SRC_DIR}", PYWHISPERCPP_SRC_DIR))
    src_dir = USER_BASE / 'src'
    if (src_dir / '.git').exists():
        fs_targets.append((f"Managed clone: {src_dir}", src_dir))
    cmd_symlink = USER_HOME / '.local' / 'bin' / 'hyprwhspr'
    if cmd_symlink.is_symlink() and os.readlink(cmd_symlink).endswith('/bin/hyprwhspr'):
        fs_targets.append((f"Command symlink: {cmd_symlink}", cmd_symlink))
    if STATE_DIR.exists():
        fs_targets.append((f"State files: {STATE_DIR}", STATE_DIR))
    if CREDENTIALS_FILE.exists():
        fs_targets.append(("Stored API credentials", CREDENTIALS_FILE))
    temp_dir = USER_BASE / 'temp'
    if temp_dir.exists():
        fs_targets.append((f"Temporary files: {temp_dir}", temp_dir))
    items_to_remove.extend(label for label, _ in fs_targets)

    # Models
    if not keep_models and PYWHISPERCPP_MODELS_DIR.exists():
        models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
        if models:
            items_to_remove.append(f"Whisper models: {len(models)} model(s) in {PYWHISPERCPP_MODELS_DIR}")
    
    # Permissions (if not skipped)
    if not skip_permissions:
        items_to_remove.append("System permissions (groups, udev rules) - optional")
    
    if not items_to_remove:
        log_info("Nothing to remove - hyprwhspr appears to be already uninstalled")
        return
    
    # Show summary
    print("\nThe following will be removed:")
    for item in items_to_remove:
        print(f"  • {item}")
    print()
    
    # Confirmation
    if not yes:
        log_warning("This will permanently delete all hyprwhspr data and configuration.")
        if not Confirm.ask("Are you sure you want to continue?", default=False):
            print("\nUninstall cancelled.")
            return
    
    print("\n" + "="*60)
    print("Removing Components")
    print("="*60 + "\n")
    
    errors = []
    
    # 1. Stop and remove systemd services
    log_info("Stopping and removing systemd services...")
    try:
        # Stop and disable hyprwhspr service
        if (USER_SYSTEMD_DIR / SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {SERVICE_NAME}")


        # Stop and disable deprecated resume service
        if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', RESUME_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', RESUME_SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {RESUME_SERVICE_NAME}")

        if _is_hyprwhspr_managed_ydotool_unit(ydotool_unit_path):
            run_command(['systemctl', '--user', 'stop', YDOTOOL_UNIT], check=False)
            run_command(['systemctl', '--user', 'disable', YDOTOOL_UNIT], check=False)
            ydotool_unit_path.unlink(missing_ok=True)
            log_success(f"Removed {YDOTOOL_UNIT}")

        # Reload systemd daemon
        run_command(['systemctl', '--user', 'daemon-reload'], check=False)
    except Exception as e:
        error_msg = f"Failed to remove systemd services: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 2. Remove Waybar integration
    log_info("Removing Waybar integration...")
    try:
        setup_waybar('remove')
    except Exception as e:
        error_msg = f"Failed to remove Waybar integration: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 3. Backend-specific cleanup (while config still exists for detection)
    log_info("Cleaning up backend...")
    try:
        current_backend = _detect_current_backend()
        if current_backend:
            _cleanup_backend(current_backend)
    except Exception as e:
        error_msg = f"Backend cleanup failed: {e}"
        log_warning(error_msg)
        errors.append(error_msg)

    # 4. Remove filesystem targets from the summary manifest
    log_info("Removing files...")
    for label, target in fs_targets:
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            log_success(f"Removed {label}")
        except Exception as e:
            error_msg = f"Failed to remove {label}: {e}"
            log_warning(error_msg)
            errors.append(error_msg)

    # 5. Remove models (if not keeping)
    if not keep_models:
        log_info("Removing Whisper models...")
        try:
            if PYWHISPERCPP_MODELS_DIR.exists():
                models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
                if models:
                    shutil.rmtree(PYWHISPERCPP_MODELS_DIR, ignore_errors=True)
                    log_success(f"Removed {len(models)} model(s) from {PYWHISPERCPP_MODELS_DIR}")
                else:
                    # Remove empty directory
                    PYWHISPERCPP_MODELS_DIR.rmdir()
        except Exception as e:
            error_msg = f"Failed to remove models: {e}"
            log_warning(error_msg)
            errors.append(error_msg)
    else:
        log_info("Keeping Whisper models (--keep-models flag)")
    
    # 6. Remove the base directory if it's empty or only contains empty subdirs
    try:
        if USER_BASE.exists():
            has_content = False
            for item in USER_BASE.iterdir():
                if item.is_file():
                    has_content = True
                    break
                elif item.is_dir():
                    try:
                        if any(item.iterdir()):
                            has_content = True
                            break
                    except Exception:
                        pass

            if not has_content:
                shutil.rmtree(USER_BASE, ignore_errors=True)
                log_success(f"Removed {USER_BASE}")
    except Exception:
        pass  # Ignore errors when trying to remove base directory

    # 7. Remove system permissions (if requested)
    permissions_removed = False
    if not skip_permissions:
        log_info("Checking system permissions...")
        
        should_remove = remove_permissions
        if not remove_permissions and not yes:
            should_remove = Confirm.ask(
                "Remove system permissions (remove user from input/audio/tty groups and udev rules)?",
                default=False
            )
        
        if should_remove:
            permissions_removed = True
            try:
                username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
                if not username:
                    log_warning("Could not determine username for permission removal")
                else:
                    # Remove from groups
                    groups_to_remove = ['input', 'audio', 'tty']
                    for group in groups_to_remove:
                        try:
                            run_sudo_command(['gpasswd', '-d', username, group], check=False)
                            log_success(f"Removed user from '{group}' group")
                        except Exception as e:
                            log_warning(f"Failed to remove user from '{group}' group: {e}")
                    
                    # Remove udev rule (only if it exists and was created by hyprwhspr)
                    udev_rule = Path('/etc/udev/rules.d/99-uinput.rules')
                    if udev_rule.exists():
                        # Check if it's our rule by reading it
                        try:
                            with open(udev_rule, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if 'hyprwhspr' in content.lower() or 'input' in content.lower():
                                run_sudo_command(['rm', str(udev_rule)], check=False)
                                log_success("Removed udev rule")
                                # Reload udev
                                run_sudo_command(['udevadm', 'control', '--reload-rules'], check=False)
                                run_sudo_command(['udevadm', 'trigger', '--name-match=uinput'], check=False)
                        except Exception as e:
                            log_warning(f"Failed to remove udev rule: {e}")
            except Exception as e:
                error_msg = f"Failed to remove system permissions: {e}"
                log_warning(error_msg)
                errors.append(error_msg)
        else:
            log_info("Skipping permission removal")
    else:
        log_info("Skipping permission removal (--skip-permissions flag)")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        log_warning("Uninstall completed with some errors:")
        for error in errors:
            log_warning(f"  • {error}")
        print("="*60)
    else:
        log_success("Uninstall completed successfully!")
        print("="*60)
    
    print("\nAll hyprwhspr user data has been removed.")
    hf_cache = Path(os.environ.get('XDG_CACHE_HOME', USER_HOME / '.cache')) / 'huggingface'
    if hf_cache.exists():
        print(f"Note: {hf_cache} (shared model cache) was not removed.")
        print("      Delete it manually if no other apps use Hugging Face models.")
    if not skip_permissions and not permissions_removed:
        print("Note: System permissions (group memberships, udev rules) were not removed.")
        print("      You may want to remove them manually if needed.")
    print()


