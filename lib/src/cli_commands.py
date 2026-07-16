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


# Setup command and permissions setup live in cli.setup.


# Install commands live in cli.install; names still read by not-yet-moved
# sections are re-imported here until the split completes.
try:
    from .cli.install import _setup_hyprland_bindings
except ImportError:
    from cli.install import _setup_hyprland_bindings


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

# Uninstall command lives in cli.uninstall.
