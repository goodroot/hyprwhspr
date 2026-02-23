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
from pathlib import Path
from typing import Optional

try:
    from rich.prompt import Prompt, Confirm
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
    from .paths import CONFIG_DIR, CONFIG_FILE, RECORDING_CONTROL_FILE, RECORDING_STATUS_FILE
except ImportError:
    from paths import CONFIG_DIR, CONFIG_FILE, RECORDING_CONTROL_FILE, RECORDING_STATUS_FILE

try:
    from .backend_utils import BACKEND_DISPLAY_NAMES, normalize_backend
except ImportError:
    from backend_utils import BACKEND_DISPLAY_NAMES, normalize_backend

try:
    from .backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT, USER_BASE, PYWHISPERCPP_SRC_DIR,
        MAX_COMPATIBLE_PYTHON, _get_python_version
    )
except ImportError:
    from backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT, USER_BASE, PYWHISPERCPP_SRC_DIR,
        MAX_COMPATIBLE_PYTHON, _get_python_version
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


# Constants
HYPRWHSPR_ROOT = os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr')
SERVICE_NAME = 'hyprwhspr.service'
RESUME_SERVICE_NAME = 'hyprwhspr-resume.service'  # Deprecated, kept for cleanup in uninstall/status
PARAKEET_SERVICE_NAME = 'parakeet-tdt-0.6b-v3.service'
YDOTOOL_UNIT = 'ydotool.service'
USER_HOME = Path.home()
USER_CONFIG_DIR = CONFIG_DIR  # Use centralized path constant
USER_SYSTEMD_DIR = USER_HOME / '.config' / 'systemd' / 'user'
PYWHISPERCPP_MODELS_DIR = Path(os.environ.get('XDG_DATA_HOME', USER_HOME / '.local' / 'share')) / 'pywhispercpp' / 'models'


def _check_mise_active() -> tuple[bool, str]:
    """
    Check if MISE (runtime version manager) is active in the current environment.

    Returns:
        Tuple of (is_active, details_message)
    """
    indicators = []

    # Check for MISE environment variables
    if os.environ.get('MISE_SHELL'):
        indicators.append(f"MISE_SHELL={os.environ['MISE_SHELL']}")
    if os.environ.get('__MISE_ACTIVATE'):
        indicators.append("__MISE_ACTIVATE is set")

    # Check if Python is being managed by MISE
    python_path = shutil.which('python3') or shutil.which('python')
    if python_path and '.local/share/mise' in python_path:
        indicators.append(f"Python path: {python_path}")

    # Check if mise binary is managing this session
    if shutil.which('mise') and os.environ.get('MISE_DATA_DIR'):
        indicators.append(f"MISE_DATA_DIR={os.environ['MISE_DATA_DIR']}")

    is_active = len(indicators) > 0
    details = "\n    ".join(indicators) if indicators else ""

    return is_active, details


def _create_mise_free_environment() -> dict:
    """
    Create environment with MISE deactivated for subprocesses.

    This prevents MISE from interfering with Python version detection
    during pip install operations.

    Returns:
        Environment dict suitable for subprocess.run(env=...)
    """
    env = os.environ.copy()

    # Remove MISE-related environment variables
    mise_vars = ['MISE_SHELL', '__MISE_ACTIVATE', 'MISE_DATA_DIR']
    for var in mise_vars:
        env.pop(var, None)

    # Clean PATH of MISE entries
    path = env.get('PATH', '')
    if '.local/share/mise' in path:
        paths = path.split(':')
        paths = [p for p in paths if '.local/share/mise' not in p]
        env['PATH'] = ':'.join(paths)

    return env


def _check_python_compatibility() -> tuple[bool, str, str]:
    """
    Check if a compatible Python version is available for local ML backends.

    ML packages (onnxruntime, etc.) require Python 3.14 or earlier.
    This check warns users early if their system Python is too new.

    Returns:
        Tuple of (is_compatible, current_version_str, guidance_message)
    """
    max_str = f"{MAX_COMPATIBLE_PYTHON[0]}.{MAX_COMPATIBLE_PYTHON[1]}"

    # Get current system Python version
    python_path = shutil.which('python3') or shutil.which('python') or sys.executable
    current_version = _get_python_version(python_path)

    if not current_version:
        return (True, "unknown", "")  # Can't detect, let it proceed

    version_str = f"{current_version[0]}.{current_version[1]}"

    if current_version <= MAX_COMPATIBLE_PYTHON:
        return (True, version_str, "")

    # Python is too new - search for a compatible alternative directly
    # (Don't call _find_compatible_python() as it prints errors and exits on failure)
    for minor in [14, 13, 12, 11]:
        if (3, minor) > MAX_COMPATIBLE_PYTHON:
            continue
        for prefix in ['/usr/bin', '/usr/local/bin']:
            alt_path = f"{prefix}/python3.{minor}"
            if os.path.isfile(alt_path) and os.access(alt_path, os.X_OK):
                test_version = _get_python_version(alt_path)
                if test_version and test_version <= MAX_COMPATIBLE_PYTHON:
                    # Found a compatible Python
                    guidance = (
                        f"System Python {version_str} is too new for ML packages.\n"
                        f"Found compatible Python: python3.{minor} ({alt_path})\n"
                        f"The installer will use this automatically."
                    )
                    return (True, version_str, guidance)

    # No compatible Python available
    guidance = (
        f"System Python {version_str} is too new for ML packages (onnxruntime, etc.).\n"
        f"Local transcription backends require Python {max_str} or earlier.\n"
        f"\n"
        f"Options:\n"
        f"  1. Install Python 3.14 or 3.13:\n"
        f"     Fedora:     sudo dnf install python3.14\n"
        f"     Arch:       yay -S python314  # or python313\n"
        f"     Ubuntu/Deb: sudo apt install python3.13\n"
        f"\n"
        f"  2. Use cloud transcription (no local Python requirement):\n"
        f"     Select 'REST API' or 'Realtime WS' during backend selection\n"
        f"\n"
        f"  3. Specify Python path manually:\n"
        f"     hyprwhspr setup --python /path/to/python3.14"
    )
    return (False, version_str, guidance)


def _check_ydotool_version() -> tuple[bool, str, str]:
    """
    Check if ydotool is installed and has a compatible version.

    hyprwhspr requires ydotool 1.0+ for paste injection. Ubuntu/Debian apt
    repositories contain an outdated 0.1.x version that uses incompatible syntax.
    Arch-based distros (Arch, Manjaro, CachyOS) typically have 1.0+ in their repos.

    Returns:
        Tuple of (is_compatible, version_string, message)
        - is_compatible: True if ydotool 1.0+ is available
        - version_string: The detected version or empty string
        - message: Human-readable status message
    """
    MIN_VERSION = "1.0.0"

    # Check if ydotool is installed
    if not shutil.which('ydotool'):
        return False, "", "ydotool not found"

    # Get version - try dpkg first (ydotool 1.0+ has no --version flag)
    import re
    version = None

    # Try dpkg (Debian/Ubuntu)
    try:
        result = subprocess.run(
            ['dpkg', '-l', 'ydotool'],
            capture_output=True,
            text=True,
            timeout=5
        )
        match = re.search(r'ii\s+ydotool\s+(\d+\.\d+\.?\d*)', result.stdout)
        if match:
            version = match.group(1)
    except Exception:
        pass

    # Try pacman (Arch/Manjaro/CachyOS)
    if not version:
        try:
            result = subprocess.run(
                ['pacman', '-Q', 'ydotool'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Output format: "ydotool 1.0.4-2.1"
            match = re.search(r'ydotool\s+(\d+\.\d+\.?\d*)', result.stdout)
            if match:
                version = match.group(1)
        except Exception:
            pass

    # Fallback: try --version (old ydotool 0.1.x supports this)
    if not version:
        try:
            result = subprocess.run(
                ['ydotool', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_output = result.stdout + result.stderr
            match = re.search(r'(\d+\.\d+\.?\d*)', version_output)
            if match:
                version = match.group(1)
        except Exception:
            pass

    # If still no version, assume old
    if not version:
        version = "0.1.0"

    # Compare versions
    def version_tuple(v):
        return tuple(map(int, (v.split('.') + ['0', '0'])[:3]))

    try:
        is_compatible = version_tuple(version) >= version_tuple(MIN_VERSION)
    except ValueError:
        is_compatible = False

    if is_compatible:
        return True, version, f"ydotool {version} (compatible)"
    else:
        return False, version, f"ydotool {version} is too old (requires {MIN_VERSION}+)"


def _strip_jsonc(text: str) -> str:
    """Strip // and /* */ comments from JSONC while preserving strings."""
    result = []
    i = 0
    in_str = False
    esc = False
    in_line = False
    in_block = False
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line:
            if ch == "\n":
                in_line = False
                result.append(ch)
            i += 1
            continue

        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
            else:
                i += 1
            continue

        if not in_str:
            if ch == "/" and nxt == "/":
                in_line = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue

        if ch == '"' and not esc:
            in_str = not in_str

        if ch == "\\" and in_str:
            esc = not esc
        else:
            esc = False

        result.append(ch)
        i += 1

    return "".join(result)


def _load_jsonc(path: Path):
    """Load JSONC file by stripping comments first."""
    with open(path, 'r', encoding='utf-8') as f:
        stripped = _strip_jsonc(f.read())
    return json.loads(stripped)


def _validate_hyprwhspr_root() -> bool:
    """Validate that HYPRWHSPR_ROOT exists and contains expected files"""
    root_path = Path(HYPRWHSPR_ROOT)
    is_development = root_path != Path('/usr/lib/hyprwhspr')
    
    if not root_path.exists():
        log_error(f"HYPRWHSPR_ROOT does not exist: {HYPRWHSPR_ROOT}")
        log_error("")
        if is_development:
            log_error("Development installation detected (not /usr/lib/hyprwhspr)")
            log_error("Check that HYPRWHSPR_ROOT environment variable is set correctly.")
            log_error("For development, ensure you're running from the repository root.")
            log_error("")
            log_error("Example: export HYPRWHSPR_ROOT=$(pwd)")
        else:
            log_error("This appears to be an AUR installation issue.")
            log_error("Try reinstalling: yay -S hyprwhspr")
        log_error("")
        return False
    
    # Check for expected files
    required_files = [
        root_path / 'bin' / 'hyprwhspr',
        root_path / 'lib' / 'main.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path.relative_to(root_path)))
    
    if missing_files:
        log_error(f"HYPRWHSPR_ROOT is missing required files: {', '.join(missing_files)}")
        log_error(f"Root path: {HYPRWHSPR_ROOT}")
        log_error("")
        if is_development:
            log_error("Development installation detected (not /usr/lib/hyprwhspr)")
            log_error("This may be a development installation issue.")
            log_error("Ensure you're running from the repository root.")
            log_error("")
            log_error("Expected structure:")
            log_error(f"  {root_path}/bin/hyprwhspr")
            log_error(f"  {root_path}/lib/main.py")
        else:
            log_error("This appears to be a corrupted AUR installation.")
            log_error("Try reinstalling: yay -S hyprwhspr")
        log_error("")
        return False
    
    return True


# ==================== Setup Command ====================

def _detect_current_backend() -> Optional[str]:
    """
    Detect currently installed backend.

    Returns:
        'cpu', 'nvidia', 'amd', 'vulkan', 'parakeet', 'onnx-asr', 'rest-api', or None if not detected
    """
    # First check config file
    try:
        config_manager = ConfigManager()
        backend = config_manager.get_setting('transcription_backend', None)
        
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
    if backend_type == 'parakeet':
        log_info("Cleaning up Parakeet backend...")
        
        # Clean up Parakeet systemd service
        parakeet_service_dest = USER_SYSTEMD_DIR / PARAKEET_SERVICE_NAME
        if parakeet_service_dest.exists():
            log_info("Removing Parakeet systemd service...")
            # Stop and disable service
            run_command(['systemctl', '--user', 'stop', PARAKEET_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', PARAKEET_SERVICE_NAME], check=False)
            # Remove the service file
            try:
                parakeet_service_dest.unlink()
                log_success("Parakeet service file removed")
            except Exception as e:
                log_warning(f"Failed to remove Parakeet service file: {e}")
            # Reload systemd daemon
            run_command(['systemctl', '--user', 'daemon-reload'], check=False)
        
        # Clean up Parakeet venv
        if PARAKEET_VENV_DIR.exists():
            import shutil
            try:
                shutil.rmtree(PARAKEET_VENV_DIR, ignore_errors=True)
                log_success("Parakeet venv removed")
            except Exception as e:
                log_warning(f"Cleanup warning: {e}")
        
        log_success("Parakeet backend cleaned up")
        return True

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


def _prompt_backend_selection():
    """Prompt user for backend selection with current state detection"""
    current_backend = _detect_current_backend()

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
    print("  [1] Parakeet TDT V3       - Solid performance for most people (Autodetects CPU/GPU)")
    print("  [2] Whisper (CPU)         - whisper.cpp, works everywhere")
    print("  [3] Whisper (NVIDIA)      - whisper.cpp + CUDA, perfect for NVIDIA GPUs")
    print("  [4] faster-whisper        - CTranslate2 + INT8 quantization, CPU or NVIDIA GPU")
    print("  [5] Whisper (Vulkan)      - whisper.cpp + Vulkan, AMD/Intel GPUs")
    print()
    print("Cloud/REST Backends:")
    print("  [6] REST API              - OpenAI, Groq, or custom endpoint")
    print("  [7] Realtime WS           - Low-latency streaming (experimental)")
    print()

    while True:
        try:
            choice = Prompt.ask("Select backend", choices=['1', '2', '3', '4', '5', '6', '7'], default='1')
            backend_map = {
                '1': 'onnx-asr',
                '2': 'cpu',
                '3': 'nvidia',
                '4': 'faster-whisper',
                '5': 'vulkan',
                '6': 'rest-api',
                '7': 'realtime-ws'
            }
            selected = backend_map[choice]

            # Backend display names for warnings/messages
            backend_names = {
                'onnx-asr': 'Parakeet TDT V3 (onnx-asr)',
                'cpu': 'Whisper CPU',
                'nvidia': 'Whisper NVIDIA (CUDA)',
                'faster-whisper': 'faster-whisper (CTranslate2)',
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
            # Local backends that need installation: cpu, nvidia, vulkan, onnx-asr, faster-whisper
            local_install_backends = ['cpu', 'nvidia', 'vulkan', 'onnx-asr', 'faster-whisper']
            if current_backend == selected and selected in local_install_backends:
                print(f"\n{BACKEND_DISPLAY_NAMES.get(selected, selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reinstall, continue to return with wants_reinstall=True
            elif current_backend == selected and selected == 'parakeet':
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


def _prompt_model_selection():
    """Prompt user for model selection"""
    multilingual_models = [
        ('tiny', 'Fastest, least accurate'),
        ('base', 'Good balance (recommended)'),
        ('small', 'Better accuracy'),
        ('medium', 'High accuracy'),
        ('large', 'Best accuracy, requires GPU'),
        ('large-v3', 'Latest large model, requires GPU')
    ]
    
    english_only_models = [
        ('tiny.en', 'Fastest, least accurate (English only)'),
        ('base.en', 'Good balance (English only, recommended)'),
        ('small.en', 'Better accuracy (English only)'),
        ('medium.en', 'High accuracy (English only)')
    ]
    
    print("\n" + "="*60)
    print("Model Selection")
    print("="*60)
    print("\nChoose your default Whisper model:")
    print()
    print("Multilingual models (support all languages, auto-detect):")
    for i, (model, desc) in enumerate(multilingual_models, 1):
        print(f"  [{i}] {model:12} - {desc}")
    
    print("\nEnglish-only models (smaller, faster, English only):")
    for i, (model, desc) in enumerate(english_only_models, len(multilingual_models) + 1):
        print(f"  [{i}] {model:12} - {desc}")
    print()
    
    all_models = [m[0] for m in multilingual_models] + [m[0] for m in english_only_models]
    choices = [str(i) for i in range(1, len(all_models) + 1)]
    
    while True:
        try:
            choice = Prompt.ask("Select model", choices=choices, default='2')  # default to base
            selected_model = all_models[int(choice) - 1]
            print(f"\n✓ Selected: {selected_model}")
            return selected_model
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise
        except (ValueError, IndexError):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_faster_whisper_model_selection() -> str:
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
            choice = Prompt.ask("Select model", choices=choices, default='2')  # default to 'base'
            selected = FASTER_WHISPER_MODELS[int(choice) - 1][0]
            print(f"\n✓ Selected: {selected}")
            return selected
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

    # Step 1: Backend selection (now returns tuple: (backend, cleanup_venv))
    backend_result = _prompt_backend_selection()
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
    
    current_backend = _detect_current_backend()
    
    # Normalize backends for comparison (handles 'amd' -> 'vulkan' mapping)
    if current_backend:
        current_backend = normalize_backend(current_backend)
    backend_normalized = normalize_backend(backend)
    
    # Check for Parakeet REST backend migration
    # Detect parakeet by checking endpoint URL and venv existence
    # (since _detect_current_backend() no longer returns 'parakeet')
    is_parakeet_config = False
    if current_backend == 'rest-api':
        config_manager = ConfigManager()
        endpoint = config_manager.get_setting('rest_endpoint_url', '')
        if endpoint == 'http://127.0.0.1:8080/transcribe' and PARAKEET_VENV_DIR.exists():
            is_parakeet_config = True
    
    if is_parakeet_config:
        log_info("Parakeet REST backend detected.")
        if Confirm.ask("\nMigrate to in-process onnx-asr backend (GPU-accelerated, no REST server)?", default=True):
            backend_normalized = 'onnx-asr'
            backend = 'onnx-asr'
            log_info("Migrating to onnx-asr backend...")
            # Clean up Parakeet REST service
            if _cleanup_backend('parakeet'):
                log_success("Parakeet REST backend cleaned up")
            log_info("Will install onnx-asr backend instead")
        else:
            log_warning("Keeping Parakeet REST backend. Note: Parakeet REST is deprecated.")
            log_warning("Consider migrating to onnx-asr for better performance and simpler setup.")
            # Preserve Parakeet backend to prevent cleanup
            backend_normalized = 'parakeet'
            backend = 'parakeet'
    
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
            else:
                print(f"\nThis will install the {backend_normalized.upper()} backend for pywhispercpp.")
                print("This may take several minutes as it compiles from source.")
            if not Confirm.ask("Proceed with backend installation?", default=True):
                log_warning("Skipping backend installation. You can install it later.")
                log_warning("Backend installation is required for local transcription to work.")
            else:
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
        # Prompt for remote provider selection (filter for realtime models)
        provider_result = _prompt_remote_provider_selection(filter_realtime=True)
        if not provider_result:
            log_error("Provider selection cancelled. Exiting.")
            return
        
        provider_id, model_id, api_key, custom_config = provider_result
        
        # Validate that selected model is realtime-compatible (for known providers)
        if provider_id != 'custom':
            models = get_provider_models(provider_id)
            if models:
                # Check if selected model exists
                if model_id not in models:
                    log_error(f"Selected model {model_id} not found in provider models")
                    return
                
                # Verify it's a realtime-compatible model
                model_data = models.get(model_id, {})
                is_realtime_model = model_data.get('realtime_model', False) or 'realtime' in model_id.lower()
                
                if not is_realtime_model:
                    log_warning(f"Selected model {model_id} may not be realtime-compatible")
                    # Try to find a realtime model
                    realtime_models = [mid for mid in models.keys() 
                                      if models[mid].get('realtime_model', False) or 'realtime' in mid.lower()]
                    if realtime_models:
                        if not Confirm.ask(f"Use realtime model '{realtime_models[0]}' instead?", default=True):
                            log_error("Realtime-compatible model required for realtime-ws backend")
                            return
                        model_id = realtime_models[0]
                        log_info(f"Using realtime model: {model_id}")
                    else:
                        log_error(f"No realtime-compatible models found for provider {provider_id}")
                        return
        
        # Handle custom backends specially - require websocket_url
        if provider_id == 'custom':
            if not custom_config:
                custom_config = {}
            
            # Prompt for WebSocket URL (required for custom realtime backends)
            websocket_url = Prompt.ask("WebSocket URL (e.g., wss://api.example.com/v1/realtime)", default="")
            if not websocket_url:
                log_error("WebSocket URL is required for custom realtime backends")
                return
            
            # Validate URL format
            if not websocket_url.startswith('wss://') and not websocket_url.startswith('ws://'):
                log_warning("WebSocket URL should start with wss:// or ws://")
                if not Confirm.ask("Continue anyway?", default=True):
                    return
            
            # Store websocket_url in custom_config
            custom_config['websocket_url'] = websocket_url
            
            # Model is required for custom realtime backends
            if not model_id:
                model_id = Prompt.ask("Model identifier (for URL query parameter)", default="")
                if not model_id:
                    log_error("Model identifier is required for custom realtime backends")
                    return
        
        # Generate realtime-ws configuration
        try:
            remote_config = _generate_remote_config(provider_id, model_id, api_key, custom_config, backend_type='realtime-ws')
            log_success("Realtime WebSocket configuration generated")
        except Exception as e:
            log_error(f"Failed to generate realtime configuration: {e}")
            return
        
        # Prompt for realtime mode
        # NOTE: Only OpenAI (and custom OpenAI-compatible endpoints) support "converse".
        if provider_id in ('openai', 'custom'):
            print("\nRealtime Mode:")
            print("  1. Transcribe - Convert speech to text (default)")
            print("  2. Converse - Voice-to-AI: speak and get AI responses")
            mode_choice = Prompt.ask("Select mode", choices=['1', '2'], default='1')
            realtime_mode = 'transcribe' if mode_choice == '1' else 'converse'
        else:
            realtime_mode = 'transcribe'
        remote_config['realtime_mode'] = realtime_mode
        log_info(f"Realtime mode: {realtime_mode}")
    
    # Step 1.4: Ensure venv and base dependencies for cloud backends
    if backend_normalized in ['rest-api', 'remote', 'realtime-ws']:
        print("\n" + "="*60)
        print("Python Environment Setup")
        print("="*60)
        log_info("Ensuring Python virtual environment and dependencies are installed...")
        
        try:
            from .backend_installer import setup_python_venv, compute_file_hash, get_state, set_state, HYPRWHSPR_ROOT
            from .output_control import run_command
        except ImportError:
            from backend_installer import setup_python_venv, compute_file_hash, get_state, set_state, HYPRWHSPR_ROOT
            from output_control import run_command
        
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
    
    # Step 2: Model selection for local backends (always prompt, regardless of install/reinstall)
    if backend_normalized not in ['rest-api', 'remote', 'realtime-ws', 'parakeet', 'onnx-asr', 'faster-whisper']:
        # Local backend - prompt for model selection
        # Note: ONNX-ASR and faster-whisper don't use Whisper.cpp models
        selected_model = _prompt_model_selection()
    elif backend_normalized == 'faster-whisper':
        faster_whisper_model = _prompt_faster_whisper_model_selection()
        if download_faster_whisper_model(faster_whisper_model):
            log_success(f"Model {faster_whisper_model} downloaded successfully")
        else:
            log_warning(f"Model download failed. You can download it later with:")
            log_warning(f"  hyprwhspr model download {faster_whisper_model}")
    
    # Step 3: Waybar integration
    print("\n" + "="*60)
    print("Waybar Integration")
    print("="*60)
    waybar_config_path = Path.home() / '.config' / 'waybar' / 'config.jsonc'
    waybar_style_path = Path.home() / '.config' / 'waybar' / 'style.css'
    waybar_installed = waybar_config_path.exists() or waybar_style_path.exists()
    
    if waybar_installed:
        print(f"\nWaybar configuration detected at: {waybar_config_path.parent}")
        setup_waybar_choice = Confirm.ask("Configure Waybar integration?", default=True)
    else:
        print("\nWaybar configuration not found.")
        setup_waybar_choice = Confirm.ask("Set up Waybar integration anyway?", default=False)
    
    # Step 3b: Mic-OSD setup
    print("\n" + "="*60)
    print("Mic-OSD Visualization")
    print("="*60)
    print("\nShows a visual overlay during recording with animated bars")
    print("and a pulsing indicator. Requires GTK4 and gtk4-layer-shell.")
    
    # Check if dependencies are available using service's Python
    mic_osd_available, mic_osd_reason = _check_mic_osd_availability()
    if not mic_osd_available:
        print(f"\nNote: {mic_osd_reason}")
    
    if mic_osd_available:
        setup_mic_osd_choice = Confirm.ask("Enable mic-osd visualization?", default=True)
    else:
        # Provide distro-appropriate package names
        if Path('/etc/debian_version').exists():
            pkg_hint = "python3-gi gir1.2-gtk-4.0 gir1.2-gtk4layershell-1.0"
        elif Path('/etc/fedora-release').exists():
            pkg_hint = "python3-gobject gtk4 gtk4-layer-shell"
        elif Path('/etc/os-release').exists() and 'suse' in Path('/etc/os-release').read_text().lower():
            pkg_hint = "python3-gobject typelib-1_0-Gtk-4_0 (gtk4-layer-shell from community repo)"
        else:
            pkg_hint = "python-gobject gtk4 gtk4-layer-shell (Arch naming)"
        print(f"\nDependencies not found. Install: {pkg_hint}")
        setup_mic_osd_choice = Confirm.ask("Enable mic-osd anyway (will work after installing deps)?", default=False)

    # Step 3c: Audio ducking setup
    print("\n" + "="*60)
    print("Audio Ducking")
    print("="*60)
    print("\nAutomatically reduces system volume while recording to prevent")
    print("audio interference with your microphone.")

    setup_audio_ducking_choice = Confirm.ask("Enable audio ducking?", default=True)
    audio_ducking_percent = 50  # Default
    if setup_audio_ducking_choice:
        print("\nHow much to reduce volume BY during recording?")
        print("  50 = reduce to 50% of original (recommended)")
        print("  70 = reduce to 30% of original (aggressive)")
        print("  30 = reduce to 70% of original (subtle)")
        ducking_input = Prompt.ask("Reduction percentage", default="50")
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
            setup_hyprland_choice = Confirm.ask("Configure Hyprland compositor bindings?", default=True)
        elif hypr_config_exists:
            print(f"\nHyprland configuration detected at: {hypr_config_dir}")
            setup_hyprland_choice = Confirm.ask("Configure Hyprland compositor bindings?", default=True)
        else:
            print("\nHyprland configuration not found.")
            setup_hyprland_choice = Confirm.ask("Set up Hyprland compositor bindings anyway?", default=False)
    else:
        # Not a Hyprland system - skip this section entirely
        setup_hyprland_choice = False
    
    # Step 4: Systemd setup
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)
    print("\nSystemd user service will run hyprwhspr in the background.")
    print("This will enable/configure:")
    print("  • ydotool.service (required dependency, provides paste)")
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
    print(f"\nBackend: {backend}")
    if remote_config:
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
    print(f"Waybar integration: {'Yes' if setup_waybar_choice else 'No'}")
    print(f"Mic-OSD visualization: {'Yes' if setup_mic_osd_choice else 'No'}")
    if setup_audio_ducking_choice:
        print(f"Audio ducking: Yes ({audio_ducking_percent}% reduction)")
    else:
        print("Audio ducking: No")
    print(f"Hyprland compositor bindings: {'Yes' if setup_hyprland_choice else 'No'}")
    if setup_systemd_choice:
        print("Systemd services: Yes (ydotool + hyprwhspr)")
    else:
        print("Systemd service: No")
    print(f"Permissions: {'Yes' if setup_permissions_choice else 'No'}")
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
        
        # Check if running manually before systemd setup
        if _is_running_manually():
            log_warning("hyprwhspr appears to be running manually (not via systemd).")
            log_warning("Please restart it manually for configuration changes to take effect.")
        
        # Step 2: Waybar
        if setup_waybar_choice:
            setup_waybar('install')
        else:
            log_info("Skipping Waybar integration")
        
        # Step 2b: Mic-OSD
        if setup_mic_osd_choice:
            mic_osd_enable()
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
            # Check if service is running anyway and warn user
            try:
                result = subprocess.run(
                    ['systemctl', '--user', 'is-active', SERVICE_NAME],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False
                )
                if result.returncode == 0:
                    log_warning("Systemd service is running but setup was skipped.")
                    log_warning("You may need to manually restart the service for changes to take effect:")
                    log_warning("  systemctl --user restart hyprwhspr")
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
            if backend == 'parakeet':
                print("  Parakeet server will start automatically via systemd")
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
                    f.write('bindd = SUPER ALT, D, Speech-to-text, exec, /usr/lib/hyprwhspr/config/hyprland/hyprwhspr-tray.sh record\n')
                log_success(f"Added Hyprland bindings to {target_file}")
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
    if backend not in ['cpu', 'nvidia', 'vulkan', 'onnx-asr']:
        # For non-local backends, skip import check
        return True

    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        return False

    # Choose the module to import based on backend
    if backend == 'onnx-asr':
        import_module = 'onnx_asr'
    else:
        import_module = 'pywhispercpp'

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
    3. Configures defaults (auto recording mode, Waybar integration)
    4. Sets up and starts systemd service
    5. Validates installation

    All without user interaction.

    Args:
        args: Optional argparse namespace with:
            - backend: 'nvidia', 'vulkan', 'cpu', or 'onnx-asr' (default: auto-detect)
            - model: Model name to download (default: 'base' for whisper, auto for onnx-asr)
            - no_waybar: Skip waybar integration
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
        log_info("onnx-asr model will be downloaded automatically on first use")

    # 7. Waybar integration (if detected and not skipped)
    print("\n" + "="*60)
    print("Waybar Integration")
    print("="*60)

    if skip_waybar:
        log_info("Waybar integration skipped (--no-waybar)")
    else:
        waybar_config = Path.home() / '.config' / 'waybar' / 'config.jsonc'
        if waybar_config.exists():
            log_info("Waybar detected - installing integration...")
            waybar_command('install')
        else:
            log_info("Waybar not detected - skipping")

    # 8. Systemd service (unless skipped)
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)

    if skip_systemd:
        log_info("Systemd service setup skipped (--no-systemd)")
    else:
        systemd_command('install')

        try:
            from .output_control import run_command
        except ImportError:
            from output_control import run_command

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


# ==================== Config Commands ====================

def config_command(action: str, show_all: bool = False):
    """Handle config subcommands"""
    if action == 'init':
        setup_config()
    elif action == 'show':
        show_config(show_all=show_all)
    elif action == 'edit':
        edit_config()
    elif action == 'secondary-shortcut':
        configure_secondary_shortcut()
    else:
        log_error(f"Unknown config action: {action}")


def setup_config(backend: Optional[str] = None, model: Optional[str] = None, remote_config: Optional[dict] = None):
    """Create or update user config"""
    log_info("Setting up user config...")
    
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        # Create default config using ConfigManager
        config = ConfigManager()
        # Override with user selections if provided
        if backend:
            config.set_setting('transcription_backend', backend)
        if model:
            config.set_setting('model', model)
        
        # Apply remote configuration if provided
        if remote_config:
            for key, value in remote_config.items():
                config.set_setting(key, value)
        
        if config.save_config():
            log_success(f"Created {config_file}")
        else:
            log_error(f"Failed to create {config_file}")
    else:
        log_info(f"Config already exists at {config_file}")
        # Update existing config via ConfigManager (handles migrations, sparse save)
        try:
            config = ConfigManager()

            if backend:
                # Map old values for backward compatibility
                if backend == 'local':
                    backend = 'cpu'
                elif backend == 'remote':
                    backend = 'rest-api'
                config.set_setting('transcription_backend', backend)

            if remote_config:
                for key, value in remote_config.items():
                    config.set_setting(key, value)

            if model:
                config.set_setting('model', model)

            if config.save_config():
                log_success("Updated existing config")
            else:
                log_error("Failed to save updated config")
        except Exception as e:
            log_error(f"Failed to update config: {e}")


def show_config(show_all: bool = False):
    """Display current config"""
    try:
        if show_all:
            config = ConfigManager().get_all_settings()
        else:
            config_file = USER_CONFIG_DIR / 'config.json'
            if not config_file.exists():
                log_error("Config file not found. Run 'hyprwhspr config init' first.")
                return
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        print(json.dumps(config, indent=2))
    except (json.JSONDecodeError, IOError) as e:
        log_error(f"Failed to read config: {e}")


def edit_config():
    """Open config in editor"""
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        log_error("Config file not found. Run 'hyprwhspr config init' first.")
        return
    
    editor = os.environ.get('EDITOR', 'nano')
    try:
        subprocess.run([editor, str(config_file)], check=True)
        log_success("Config edited")
    except Exception as e:
        log_error(f"Failed to open editor: {e}")


def configure_secondary_shortcut():
    """Configure secondary shortcut and language"""
    from rich.prompt import Prompt, Confirm
    
    config = ConfigManager()
    
    print("\n" + "="*60)
    print("Secondary Shortcut Configuration")
    print("="*60)
    print("\nConfigure a second hotkey that will use a specific language for transcription.")
    print("The primary shortcut will continue to use the default language from config.")
    print()
    
    # Check if already configured
    current_shortcut = config.get_setting('secondary_shortcut')
    current_language = config.get_setting('secondary_language')
    
    if current_shortcut:
        print(f"Current secondary shortcut: {current_shortcut}")
        if current_language:
            print(f"Current secondary language: {current_language}")
        print()
        if not Confirm.ask("Do you want to change the secondary shortcut?", default=False):
            return
    
    # Prompt for shortcut
    print("\nEnter the secondary shortcut key combination.")
    print("Examples: SUPER+ALT+I, CTRL+SHIFT+L, F11")
    print("Leave blank to disable secondary shortcut.")
    shortcut = Prompt.ask("Secondary shortcut", default=current_shortcut or "")
    
    if not shortcut or shortcut.strip() == "":
        # Disable secondary shortcut
        config.set_setting('secondary_shortcut', None)
        config.set_setting('secondary_language', None)
        config.save_config()
        log_success("Secondary shortcut disabled")
        return
    
    # Prompt for language
    print("\nEnter the language code for this shortcut.")
    print("Examples: 'it' (Italian), 'en' (English), 'fr' (French), 'de' (German), 'es' (Spanish)")
    print("Leave blank to disable secondary shortcut.")
    language = Prompt.ask("Language code", default=current_language or "")
    
    if not language or language.strip() == "":
        log_warning("Language code is required. Secondary shortcut not configured.")
        return
    
    # Validate language code (basic check - 2-3 letter code)
    language = language.strip().lower()
    if len(language) < 2 or len(language) > 3:
        log_warning("Language code should be 2-3 letters (e.g., 'it', 'en', 'fr')")
        if not Confirm.ask("Continue anyway?", default=False):
            return
    
    # Save configuration
    config.set_setting('secondary_shortcut', shortcut.strip())
    config.set_setting('secondary_language', language)
    config.save_config()
    
    log_success(f"Secondary shortcut configured: {shortcut.strip()} (language: {language})")
    print("\nNote: Restart hyprwhspr service for changes to take effect:")
    print("  systemctl --user restart hyprwhspr")


# ==================== Systemd Commands ====================

def systemd_command(action: str):
    """Handle systemd subcommands"""
    if action == 'install':
        setup_systemd('install')
    elif action == 'enable':
        setup_systemd('enable')
    elif action == 'disable':
        setup_systemd('disable')
    elif action == 'status':
        systemd_status()
    elif action == 'restart':
        systemd_restart()
    else:
        log_error(f"Unknown systemd action: {action}")


def setup_systemd(mode: str = 'install'):
    """Setup systemd user service"""
    log_info("Configuring systemd user services...")
    
    # Validate HYPRWHSPR_ROOT
    if not _validate_hyprwhspr_root():
        return False
    
    # Validate main executable exists
    main_exec = Path(HYPRWHSPR_ROOT) / 'bin' / 'hyprwhspr'
    if not main_exec.exists() or not os.access(main_exec, os.X_OK):
        log_error(f"Main executable not found or not executable: {main_exec}")
        return False
    
    # Create user systemd directory
    USER_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read hyprwhspr service file template and substitute paths
    service_source = Path(HYPRWHSPR_ROOT) / 'config' / 'systemd' / SERVICE_NAME
    service_dest = USER_SYSTEMD_DIR / SERVICE_NAME
    
    if not service_source.exists():
        log_error(f"Service file not found: {service_source}")
        return False
    
    # Read template and substitute HYPRWHSPR_ROOT
    try:
        with open(service_source, 'r', encoding='utf-8') as f:
            service_content = f.read()
        
        # Substitute hardcoded path with actual HYPRWHSPR_ROOT
        service_content = service_content.replace('/usr/lib/hyprwhspr', HYPRWHSPR_ROOT)
        
        # Write substituted content to user directory
        with open(service_dest, 'w', encoding='utf-8') as f:
            f.write(service_content)

        log_success("User service file created with correct paths")
    except IOError as e:
        log_error(f"Failed to read/write service file: {e}")
        return False

    # Reload systemd daemon
    run_command(['systemctl', '--user', 'daemon-reload'], check=False)
    
    if mode in ('install', 'enable'):
        # Enable & start services
        run_command(['systemctl', '--user', 'enable', '--now', YDOTOOL_UNIT], check=False)

        # Check if hyprwhspr service was already running before enabling
        service_was_running = False
        try:
            result = subprocess.run(
                ['systemctl', '--user', 'is-active', SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=2,
                check=False
            )
            service_was_running = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        run_command(['systemctl', '--user', 'enable', '--now', SERVICE_NAME], check=False)
        
        # If service was already running, restart it to pick up any config changes
        if service_was_running:
            log_info("Service was already running, restarting to apply configuration changes...")
            systemd_restart()
        else:
            log_success("Systemd user services enabled and started")
    elif mode == 'disable':
        run_command(['systemctl', '--user', 'disable', '--now', SERVICE_NAME], check=False)
        # Disable suspend/resume service if it exists
        log_success("Systemd user service disabled")
    
    return True


def systemd_status():
    """Show systemd service status"""
    try:

        log_info("hyprwhspr service status:")
        run_command(['systemctl', '--user', 'status', SERVICE_NAME], check=False)
        print()  # Add spacing

        # Show suspend/resume service status if it exists
        if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
            log_info("Suspend/resume handler status:")
            run_command(['systemctl', '--user', 'status', RESUME_SERVICE_NAME], check=False)
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to get status: {e}")


def _is_service_running_via_systemd() -> bool:
    """Check if hyprwhspr service is running via systemd"""
    try:
        from .instance_detection import is_service_active_via_systemd
        return is_service_active_via_systemd(SERVICE_NAME)
    except ImportError:
        # Fallback if import fails
        try:
            result = subprocess.run(
                ['systemctl', '--user', 'is-active', SERVICE_NAME],
                capture_output=True,
                text=True,
                timeout=2,
                check=False
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False


def _is_running_manually() -> bool:
    """Check if hyprwhspr is running manually (not via systemd)"""
    try:
        from .instance_detection import is_running_manually
        return is_running_manually()
    except ImportError:
        # Fallback if import fails
        # Check if there's a process but systemd service is not active
        try:
            pgrep_result = subprocess.run(
                ['pgrep', '-f', 'hyprwhspr.*main.py'],
                capture_output=True,
                timeout=2,
                check=False
            )
            if pgrep_result.returncode == 0:
                # Process exists, check if it's via systemd
                if not _is_service_running_via_systemd():
                    return True
        except Exception:
            pass
        return False


def systemd_restart():
    """Restart systemd service"""
    log_info("Restarting service...")
    try:
        run_command(['systemctl', '--user', 'restart', SERVICE_NAME], check=False)
        log_success("Service restarted")
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to restart service: {e}")


# ==================== Waybar Commands ====================

def waybar_command(action: str):
    """Handle waybar subcommands"""
    if action == 'install':
        setup_waybar('install')
    elif action == 'remove':
        setup_waybar('remove')
    elif action == 'status':
        waybar_status()
    else:
        log_error(f"Unknown waybar action: {action}")


def setup_waybar(mode: str = 'install'):
    """Setup or remove waybar integration"""
    if mode == 'install':
        log_info("Setting up Waybar integration...")
    else:
        log_info("Removing Waybar integration...")
    
    # Validate HYPRWHSPR_ROOT
    if not _validate_hyprwhspr_root():
        return False
    
    # Validate required files exist
    tray_script = Path(HYPRWHSPR_ROOT) / 'config' / 'hyprland' / 'hyprwhspr-tray.sh'
    css_file = Path(HYPRWHSPR_ROOT) / 'config' / 'waybar' / 'hyprwhspr-style.css'
    
    if not tray_script.exists():
        log_error(f"Tray script not found: {tray_script}")
        return False
    
    if not css_file.exists():
        log_error(f"Waybar CSS not found: {css_file}")
        return False
    
    waybar_config = USER_HOME / '.config' / 'waybar' / 'config.jsonc'
    waybar_style = USER_HOME / '.config' / 'waybar' / 'style.css'
    user_module_config = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    
    if mode == 'install':
        # Create waybar config directory
        waybar_config.parent.mkdir(parents=True, exist_ok=True)
        
        # Create basic waybar config if it doesn't exist
        if not waybar_config.exists():
            log_info("Creating basic Waybar config...")
            basic_config = {
                "layer": "top",
                "position": "top",
                "height": 30,
                "modules-left": ["hyprland/workspaces"],
                "modules-center": ["hyprland/window"],
                "modules-right": ["custom/hyprwhspr", "clock", "tray"],
                "include": [str(user_module_config)]
            }
            with open(waybar_config, 'w', encoding='utf-8') as f:
                json.dump(basic_config, f, indent=2)
            log_success("Created basic Waybar config")
        
        # Create user module config
        module_config = {
            "custom/hyprwhspr": {
                "format": "{}",
                "exec": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh status",
                "interval": 1,
                "return-type": "json",
                "exec-on-event": True,
                "on-click": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh record",
                "on-click-right": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh restart",
                "tooltip": True
            }
        }
        
        with open(user_module_config, 'w', encoding='utf-8') as f:
            json.dump(module_config, f, indent=2)
        
        # Update main waybar config
        try:
            config = _load_jsonc(waybar_config)
            
            # Add include if not present
            if 'include' not in config:
                config['include'] = []
            
            if str(user_module_config) not in config['include']:
                config['include'].append(str(user_module_config))
            
            # Add module to modules-right if not present
            if 'modules-right' not in config:
                config['modules-right'] = []
            
            if 'custom/hyprwhspr' not in config['modules-right']:
                # Try to insert after group/tray-expander
                try:
                    tray_index = config['modules-right'].index('group/tray-expander')
                    config['modules-right'].insert(tray_index + 1, 'custom/hyprwhspr')
                except ValueError:
                    # group/tray-expander not found, append to end
                    config['modules-right'].append('custom/hyprwhspr')
            
            # Write back
            with open(waybar_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, separators=(',', ': '))
            
            log_success("Waybar config updated")
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
            log_error("Please check your config.jsonc for JSON syntax errors.")
            return False
        except IOError as e:
            log_error(f"Failed to read/write waybar config: {e}")
            return False
        
        # Add CSS import
        if waybar_style.exists():
            with open(waybar_style, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            import_line = f'@import "{css_file}";'
            
            if import_line not in css_content:
                with open(waybar_style, 'w', encoding='utf-8') as f:
                    f.write(import_line + '\n' + css_content)
                log_success("CSS import added to waybar style.css")
            else:
                log_info("CSS import already present")
        else:
            log_warning("No waybar style.css found - user will need to add CSS import manually")
        
        log_success("Waybar integration installed")
    
    elif mode == 'remove':
        # Remove from config
        if waybar_config.exists():
            try:
                config = _load_jsonc(waybar_config)
                
                # Remove from include
                if 'include' in config and str(user_module_config) in config['include']:
                    config['include'].remove(str(user_module_config))
                
                # Remove from modules-right
                if 'modules-right' in config and 'custom/hyprwhspr' in config['modules-right']:
                    config['modules-right'].remove('custom/hyprwhspr')
                
                with open(waybar_config, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                
                log_success("Removed from waybar config")
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
                log_error("Please check your config.jsonc for JSON syntax errors.")
            except IOError as e:
                log_error(f"Failed to read/write waybar config: {e}")
        
        # Remove module config file
        if user_module_config.exists():
            user_module_config.unlink()
            log_success("Removed waybar module config")
        
        # Remove CSS import
        if waybar_style.exists():
            try:
                with open(waybar_style, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                
                import_line = f'@import "{css_file}";'
                if import_line in css_content:
                    # Remove with newline
                    css_content = css_content.replace(import_line + '\n', '')
                    # Remove without newline (in case it's at the end)
                    css_content = css_content.replace(import_line, '')
                    
                    with open(waybar_style, 'w', encoding='utf-8') as f:
                        f.write(css_content)
                    log_success("Removed CSS import")
            except IOError as e:
                log_error(f"Failed to remove CSS import: {e}")
    
    return True


def waybar_status():
    """Check if waybar is configured"""
    waybar_config = USER_HOME / '.config' / 'waybar' / 'config.jsonc'
    user_module_config = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    
    if not waybar_config.exists():
        log_warning("Waybar config not found")
        return False
    
    try:
        config = _load_jsonc(waybar_config)
        
        has_module = 'custom/hyprwhspr' in config.get('modules-right', [])
        has_include = str(user_module_config) in config.get('include', [])
        has_module_file = user_module_config.exists()
        
        if has_module and has_include and has_module_file:
            log_success("Waybar is configured for hyprwhspr")
            return True
        else:
            log_warning("Waybar is partially configured")
            if not has_module:
                log_info("  - Module not in modules-right")
            if not has_include:
                log_info("  - Module config not in include")
            if not has_module_file:
                log_info("  - Module config file missing")
            return False
    except json.JSONDecodeError as e:
        log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
        return False
    except IOError as e:
        log_error(f"Failed to check waybar status: {e}")
        return False


# ==================== Mic-OSD Commands ====================

def _check_mic_osd_availability():
    """Check mic-osd availability using the same Python the service will use.
    
    Returns:
        tuple: (is_available: bool, reason: str)
    """
    # First, try with venv Python (same as service uses)
    venv_python = VENV_DIR / 'bin' / 'python'
    if venv_python.exists():
        try:
            lib_path = Path(__file__).parent.parent
            # Use repr() to safely escape the path (handles quotes, backslashes, etc.)
            lib_path_str = repr(str(lib_path))
            check_code = f"""
import sys
sys.path.insert(0, {lib_path_str})
from mic_osd import MicOSDRunner
if MicOSDRunner.is_available():
    print('AVAILABLE')
else:
    print('UNAVAILABLE:', MicOSDRunner.get_unavailable_reason())
"""
            result = subprocess.run(
                [str(venv_python), '-c', check_code],
                check=False,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output == 'AVAILABLE':
                    return True, ""
                elif output.startswith('UNAVAILABLE:'):
                    return False, output.replace('UNAVAILABLE:', '').strip()
        except Exception as e:
            # Fall through to current Python check
            pass
    
    # Fallback: check with current Python environment
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mic_osd import MicOSDRunner
        
        if MicOSDRunner.is_available():
            return True, ""
        else:
            return False, MicOSDRunner.get_unavailable_reason()
    except ImportError:
        return False, "mic-osd module not found"


def mic_osd_command(action: str):
    """Handle mic-osd subcommands"""
    if action == 'enable':
        mic_osd_enable()
    elif action == 'disable':
        mic_osd_disable()
    elif action == 'status':
        mic_osd_status()
    else:
        log_error(f"Unknown mic-osd action: {action}")


def mic_osd_enable():
    """Enable the mic-osd visualization overlay"""
    # Check if dependencies are available using service's Python
    is_available, reason = _check_mic_osd_availability()
    
    if not is_available:
        log_error(f"Cannot enable mic-osd: {reason}")
        return False
    
    # Update config
    config = ConfigManager()
    config.set_setting('mic_osd_enabled', True)
    config.save_config()
    log_success("Mic-OSD visualization enabled")
    log_info("The overlay will show during recording when the service is running")
    return True


def mic_osd_disable():
    """Disable the mic-osd visualization overlay"""
    config = ConfigManager()
    config.set_setting('mic_osd_enabled', False)
    config.save_config()
    log_success("Mic-OSD visualization disabled")
    return True


def mic_osd_status():
    """Check mic-osd status"""
    config = ConfigManager()
    enabled = config.get_setting('mic_osd_enabled', True)
    
    # Check dependencies using service's Python
    deps_available, deps_reason = _check_mic_osd_availability()
    
    print("\nMic-OSD Status:")
    print(f"  Enabled in config: {'Yes' if enabled else 'No'}")
    print(f"  Dependencies available: {'Yes' if deps_available else 'No'}")
    
    if deps_available:
        if enabled:
            log_success("Mic-OSD will show during recording")
        else:
            log_info("Mic-OSD is disabled (use 'hyprwhspr mic-osd enable' to enable)")
    else:
        log_warning(f"Mic-OSD cannot run: {deps_reason}")

    return enabled and deps_available


# ==================== Model Commands ====================

def model_command(action: str, model_name: str = 'base'):
    """Handle model subcommands"""
    config = ConfigManager()
    backend = normalize_backend(config.get_setting('transcription_backend', 'pywhispercpp'))

    if backend == 'faster-whisper':
        if action == 'download':
            download_faster_whisper_model(model_name)
        elif action == 'list':
            list_faster_whisper_models()
        elif action == 'status':
            faster_whisper_model_status()
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
    log_info(f"Downloading pywhispercpp model: {model_name}")
    
    PYWHISPERCPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_file = PYWHISPERCPP_MODELS_DIR / f'ggml-{model_name}.bin'
    model_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    if model_file.exists():
        file_size = model_file.stat().st_size
        if file_size > 100000000:  # > 100MB, probably valid
            log_success(f"Model already exists: {model_file}")
            return True
        else:
            log_warning("Existing model appears invalid, re-downloading...")
            model_file.unlink()
    
    log_info(f"Fetching {model_url}")
    try:
        import urllib.request
        
        def show_progress(block_num, block_size, total_size):
            """Callback to show download progress"""
            if not OutputController.is_progress_enabled():
                return
            
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size) if total_size > 0 else 0
            size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            downloaded_mb = downloaded / (1024 * 1024)
            
            # Show progress on same line
            progress_msg = f"\r[INFO] Downloading: {downloaded_mb:.1f}/{size_mb:.1f} MB ({percent}%)"
            OutputController.write(progress_msg, VerbosityLevel.NORMAL, flush=True)
            
            if downloaded >= total_size and total_size > 0:
                OutputController.write("\n", VerbosityLevel.NORMAL, flush=True)  # New line when complete
        
        urllib.request.urlretrieve(model_url, model_file, reporthook=show_progress)
        log_success(f"Model downloaded: {model_file}")
        return True
    except (urllib.error.URLError, IOError) as e:
        log_error(f"Failed to download model: {e}")
        return False


def list_models():
    """List available models"""
    # Multilingual models (support all languages, auto-detect)
    multilingual_models = [
        'tiny',      # Fastest, least accurate
        'base',      # Good balance (recommended)
        'small',     # Better accuracy
        'medium',    # High accuracy
        'large',     # Best accuracy, requires GPU
        'large-v3'   # Latest large model, requires GPU
    ]
    
    # English-only models (smaller, faster, English only)
    english_only_models = [
        'tiny.en',   # Fastest, least accurate
        'base.en',   # Good balance
        'small.en',  # Better accuracy
        'medium.en'  # High accuracy
    ]
    
    print("Available models:\n")
    
    print("Multilingual models (support all languages, auto-detect):")
    for model in multilingual_models:
        size_note = " (requires GPU)" if model in ('large', 'large-v3') else ""
        print(f"  - {model}{size_note}")
    
    print("\nEnglish-only models (smaller, faster, English only):")
    for model in english_only_models:
        print(f"  - {model}")
    
    print("\nNote: Use multilingual models for non-English languages or mixed-language content.")
    print("      Use English-only (.en) models for English-only content (smaller file size).")


def model_status():
    """Check installed models"""
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


# ==================== Status Command ====================

def status_command():
    """Overall status check"""
    log_info("Checking hyprwhspr status...")
    
    # Check systemd service
    print("\n[Systemd Service]")
    try:
        result = run_command(
            ['systemctl', '--user', 'is-active', SERVICE_NAME],
            check=False,
            capture_output=True
        )
        if result.returncode == 0:
            log_success(f"Service is active")
        else:
            log_warning(f"Service is not active")
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to check service: {e}")
    
    # Check waybar config
    print("\n[Waybar Integration]")
    waybar_status()
    
    # Check user config
    print("\n[User Config]")
    config_file = paths.CONFIG_FILE
    if config_file.exists():
        log_success(f"Config exists: {config_file}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            log_warning(f"Config file is invalid, hyprwhspr will be using default config. Please check config line {e.lineno}, column  {e.colno}.")
    else:
        log_warning("Config file not found")
    
    # Check models
    print("\n[Models]")
    model_status()
    
    # Check permissions
    print("\n[Permissions]")
    check_permissions()


def check_permissions():
    """Check user permissions"""
    import grp
    
    # Get username safely
    username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
    if not username:
        log_error("Could not determine username for permissions check")
        return
    
    # Check groups
    user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
    user_groups.append(grp.getgrgid(os.getgid()).gr_name)
    
    required_groups = ['input', 'audio', 'tty']
    for group in required_groups:
        if group in user_groups:
            log_success(f"User in '{group}' group")
        else:
            log_warning(f"User NOT in '{group}' group")
    
    # Check uinput
    uinput_path = Path('/dev/uinput')
    if uinput_path.exists():
        if os.access(uinput_path, os.R_OK | os.W_OK):
            log_success("/dev/uinput is accessible")
        else:
            log_warning("/dev/uinput exists but is not accessible")
    else:
        log_warning("/dev/uinput does not exist")


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

    # Create udev rule
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


# ==================== Backend Management Commands ====================

def backend_repair_command():
    """Repair corrupted installation"""
    log_info("Checking for installation issues...")
    
    # Check venv
    venv_python = VENV_DIR / 'bin' / 'python'
    venv_corrupted = False
    
    if VENV_DIR.exists():
        if not venv_python.exists():
            log_warning("Venv exists but Python binary is missing")
            venv_corrupted = True
        else:
            # Test if Python works
            try:
                result = subprocess.run(
                    [str(venv_python), '--version'],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    log_warning("Venv Python binary is not working")
                    venv_corrupted = True
            except Exception:
                log_warning("Venv Python binary cannot be executed")
                venv_corrupted = True
    
    # Check backend module installation based on configured backend
    backend_missing = False
    backend_module = None
    configured_backend = None

    # Get the configured backend to know which module to check
    try:
        config_manager = ConfigManager()
        configured_backend = config_manager.get_setting('transcription_backend', 'pywhispercpp')
        configured_backend = normalize_backend(configured_backend)
    except Exception:
        pass

    # Determine which module to check based on backend
    if configured_backend == 'onnx-asr':
        backend_module = 'onnx_asr'
    elif configured_backend == 'faster-whisper':
        backend_module = 'faster_whisper'
    elif configured_backend in ['cpu', 'nvidia', 'vulkan', 'amd', 'pywhispercpp']:
        backend_module = 'pywhispercpp'
    # For rest-api, realtime-ws, parakeet - no local module to check

    if backend_module and venv_python.exists() and not venv_corrupted:
        try:
            result = subprocess.run(
                [str(venv_python), '-c', f'import {backend_module}'],
                check=False,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                log_warning(f"{backend_module} is not installed in venv (required for {configured_backend} backend)")
                backend_missing = True
        except Exception:
            pass

    if not venv_corrupted and not backend_missing:
        log_success("No issues detected")
        return True
    
    print("\n" + "="*60)
    print("Repair Options")
    print("="*60)
    
    if venv_corrupted:
        print("\nIssues found:")
        print("  • Virtual environment is corrupted")
        print("\nOptions:")
        print("  [1] Recreate venv (recommended)")
        print("  [2] Skip (manual repair required)")
        
        choice = Prompt.ask("Select option", choices=['1', '2'], default='1')
        if choice == '1':
            log_info("Recreating venv...")
            import shutil
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            # Recreate by calling setup_python_venv
            try:
                from .backend_installer import setup_python_venv
            except ImportError:
                from backend_installer import setup_python_venv
            setup_python_venv()
            log_success("Venv recreated")
    
    if backend_missing:
        print("\nIssues found:")
        print(f"  • {backend_module} is not installed (required for {configured_backend} backend)")
        print("\nOptions:")
        print("  [1] Reinstall backend")
        print("  [2] Skip (manual repair required)")

        choice = Prompt.ask("Select option", choices=['1', '2'], default='1')
        if choice == '1':
            # Use the configured backend for reinstallation
            if configured_backend and configured_backend in ['cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr']:
                log_info(f"Reinstalling {configured_backend.upper()} backend...")
                # Use force_rebuild=True to ensure clean reinstall
                if install_backend(configured_backend, force_rebuild=True):
                    log_success("Backend reinstalled successfully")
                else:
                    log_error("Backend reinstallation failed")
                    return False
            else:
                log_warning("Could not detect backend type. Please run 'hyprwhspr setup'")
                return False
    
    log_success("Repair completed")
    return True


def backend_reset_command():
    """Reset installation state"""
    log_warning("This will reset the installation state.")
    log_warning("This does NOT remove installed files, only state tracking.")
    if not Confirm.ask("Are you sure?", default=False):
        log_info("Reset cancelled")
        return False
    
    init_state()
    set_install_state('not_started')
    log_success("Installation state reset")
    return True


# ==================== State Management Commands ====================

def state_show_command():
    """Show current installation state"""
    init_state()
    state, error = get_install_state()
    all_state = get_all_state()
    
    print("\n" + "="*60)
    print("Installation State")
    print("="*60)
    print(f"\nStatus: {state}")
    
    if error:
        print(f"Last error: {error}")
        error_time = all_state.get('last_error_time')
        if error_time:
            print(f"Error time: {error_time}")
    
    # Show other state info
    if all_state:
        print("\nState details:")
        for key, value in all_state.items():
            if key not in ['install_state', 'last_error', 'last_error_time']:
                print(f"  {key}: {value}")
    
    # Check actual installation
    print("\nActual installation status:")
    venv_python = VENV_DIR / 'bin' / 'python'
    if venv_python.exists():
        log_success("Venv exists")
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'import pywhispercpp'],
                check=False,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                log_success("pywhispercpp is installed")
            else:
                log_warning("pywhispercpp is NOT installed")
        except Exception:
            log_warning("Could not check pywhispercpp installation")
    else:
        log_warning("Venv does not exist")
    
    print()


def state_validate_command():
    """Validate state consistency"""
    log_info("Validating state consistency...")
    init_state()
    
    issues = []
    
    # Check state file is valid JSON
    try:
        all_state = get_all_state()
    except Exception as e:
        log_error(f"State file is corrupted: {e}")
        issues.append("State file corruption")
        print("\nTo fix: Run 'hyprwhspr state reset'")
        return False
    
    # Check if state matches actual installation
    state, _ = get_install_state()
    venv_python = VENV_DIR / 'bin' / 'python'
    
    if state == 'completed':
        if not venv_python.exists():
            issues.append("State says 'completed' but venv does not exist")
        else:
            try:
                result = subprocess.run(
                    [str(venv_python), '-c', 'import pywhispercpp'],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0:
                    issues.append("State says 'completed' but pywhispercpp is not installed")
            except Exception:
                pass
    
    if issues:
        log_warning("State validation found issues:")
        for issue in issues:
            log_warning(f"  • {issue}")
        print("\nTo fix: Run 'hyprwhspr backend repair' or 'hyprwhspr state reset'")
        return False
    else:
        log_success("State is consistent")
        return True


def state_reset_command(remove_all: bool = False):
    """Reset state file"""
    if remove_all:
        log_warning("This will:")
        log_warning("  • Clear state file")
        log_warning("  • Remove venv directory")
        log_warning("  • Remove pywhispercpp source directory")
        if not Confirm.ask("Are you sure? This cannot be undone!", default=False):
            log_info("Reset cancelled")
            return False
        
        # Remove venv
        if VENV_DIR.exists():
            log_info("Removing venv...")
            import shutil
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            log_success("Venv removed")
        
        # Remove pywhispercpp source
        try:
            from .backend_installer import PYWHISPERCPP_SRC_DIR
        except ImportError:
            from backend_installer import PYWHISPERCPP_SRC_DIR
        
        if PYWHISPERCPP_SRC_DIR.exists():
            log_info("Removing pywhispercpp source...")
            import shutil
            shutil.rmtree(PYWHISPERCPP_SRC_DIR, ignore_errors=True)
            log_success("Source directory removed")
    else:
        log_warning("This will clear the state file (installations will remain)")
        if not Confirm.ask("Are you sure?", default=False):
            log_info("Reset cancelled")
            return False
    
    # Reset state file
    init_state()
    set_install_state('not_started')
    log_success("State reset complete")
    return True


def validate_command():
    """Validate installation"""
    log_info("Validating installation...")
    
    all_ok = True
    
    # Validate HYPRWHSPR_ROOT first
    if not _validate_hyprwhspr_root():
        all_ok = False
        return all_ok
    
    # Detect current backend to determine what to validate
    current_backend = _detect_current_backend()
    is_rest_api = current_backend in ['rest-api', 'parakeet', 'remote', 'realtime-ws']
    is_onnx_asr = current_backend == 'onnx-asr'
    is_pywhispercpp = current_backend in ['cpu', 'nvidia', 'amd', 'vulkan', 'pywhispercpp']
    
    # Check static files
    required_files = [
        Path(HYPRWHSPR_ROOT) / 'bin' / 'hyprwhspr',
        Path(HYPRWHSPR_ROOT) / 'lib' / 'main.py',
        Path(HYPRWHSPR_ROOT) / 'config' / 'systemd' / SERVICE_NAME,
    ]
    
    for file_path in required_files:
        if file_path.exists():
            log_success(f"✓ {file_path.name} exists")
        else:
            log_error(f"✗ {file_path.name} missing")
            all_ok = False
    
    # Check Python imports
    # Check packages in venv first, then fallback to current environment
    venv_python = VENV_DIR / 'bin' / 'python'
    
    # Check sounddevice (always needed)
    sounddevice_available = False
    if venv_python.exists():
        # Check in venv using subprocess
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'import sounddevice'],
                check=False,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                sounddevice_available = True
        except Exception:
            pass
    
    # Fallback: check in current environment if not found in venv
    if not sounddevice_available:
        try:
            import sounddevice  # noqa: F401
            sounddevice_available = True
        except ImportError:
            sounddevice_available = False
    
    if sounddevice_available:
        log_success("✓ sounddevice available")
    else:
        log_error("✗ sounddevice not available")
        all_ok = False
    
    # Check backend-specific packages
    if is_onnx_asr:
        # Check onnx-asr availability
        onnx_asr_available = False
        if venv_python.exists():
            try:
                result = subprocess.run(
                    [str(venv_python), '-c', 'import onnx_asr'],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    onnx_asr_available = True
            except Exception:
                pass
        
        # Fallback: check in current environment if not found in venv
        if not onnx_asr_available:
            try:
                import onnx_asr  # noqa: F401
                onnx_asr_available = True
            except ImportError:
                onnx_asr_available = False
        
        if onnx_asr_available:
            log_success("✓ onnx-asr available")
        else:
            log_warning("⚠ onnx-asr not available")
            print("")
            print("To use ONNX-ASR backend, run: hyprwhspr setup")
            print("This will install the ONNX-ASR backend.")
            print("")
        
        # Skip model file check for onnx-asr (uses different model format)
        
    elif is_pywhispercpp:
        # Check pywhispercpp (only for pywhispercpp backends)
        pywhispercpp_available = False
        
        if venv_python.exists():
            # Check in venv using subprocess - try both import styles
            try:
                # Try modern layout first
                result = subprocess.run(
                    [str(venv_python), '-c', 'from pywhispercpp.model import Model'],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    pywhispercpp_available = True
                else:
                    # Fallback to flat layout
                    result = subprocess.run(
                        [str(venv_python), '-c', 'from pywhispercpp import Model'],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        pywhispercpp_available = True
            except Exception:
                pass
        
        # Fallback: check in current environment if not found in venv
        if not pywhispercpp_available:
            try:
                # Try modern layout first
                try:
                    from pywhispercpp.model import Model  # noqa: F401
                    pywhispercpp_available = True
                except ImportError:
                    # Fallback for flat layout
                    try:
                        from pywhispercpp import Model  # noqa: F401
                        pywhispercpp_available = True
                    except ImportError:
                        pywhispercpp_available = False
            except ImportError:
                pywhispercpp_available = False
        
        if pywhispercpp_available:
            log_success("✓ pywhispercpp available")
        else:
            log_warning("⚠ pywhispercpp not available")
            print("")
            print("To use local transcription, run: hyprwhspr setup")
            print("This will install the backend (CPU/NVIDIA/AMD) of your choice.")
            print("(or use REST API backend by setting 'transcription_backend': 'rest-api' in config.json)")
            print("")
        
        # Check base model (only for pywhispercpp backends)
        model_file = PYWHISPERCPP_MODELS_DIR / 'ggml-base.bin'
        if model_file.exists():
            log_success(f"✓ Base model exists: {model_file}")
        else:
            log_warning(f"⚠ Base model missing: {model_file}")
    else:
        # For REST API backends, check Parakeet if applicable
        if current_backend == 'parakeet':
            if PARAKEET_VENV_DIR.exists():
                log_success("✓ Parakeet venv exists")
            else:
                log_warning("⚠ Parakeet venv not found")
            
            if PARAKEET_SCRIPT.exists():
                log_success("✓ Parakeet script exists")
            else:
                log_error("✗ Parakeet script missing")
                all_ok = False

    # Check ydotool version (required for paste injection)
    ydotool_ok, ydotool_version, ydotool_msg = _check_ydotool_version()
    if ydotool_ok:
        log_success(f"✓ {ydotool_msg}")
    elif ydotool_version:
        log_error(f"✗ {ydotool_msg}")
        log_error("  Paste injection will output garbage with this version.")
        log_error("  Ubuntu/Debian users: Run scripts/install-deps.sh to fix this,")
        log_error("  or manually install ydotool 1.0+ from Debian backports:")
        log_error("  wget http://deb.debian.org/debian/pool/main/y/ydotool/ydotool_1.0.4-2~bpo13+1_amd64.deb")
        log_error("  sudo dpkg -i ydotool_1.0.4-2~bpo13+1_amd64.deb")
        all_ok = False
    else:
        log_error(f"✗ {ydotool_msg}")
        log_error("  ydotool is required for paste injection.")
        all_ok = False

    # Validate configuration for potential conflicts
    try:
        from .config_manager import ConfigManager
        config = ConfigManager()
        use_hypr_bindings = config.get_setting('use_hypr_bindings', False)
        grab_keys = config.get_setting('grab_keys', False)

        if use_hypr_bindings:
            log_info("ℹ Using Hyprland compositor bindings (evdev disabled)")
            if grab_keys:
                log_warning("⚠ Warning: use_hypr_bindings=true but grab_keys=true")
                log_warning("  Recommendation: Set grab_keys=false when using compositor bindings")
    except Exception:
        pass  # Config validation is optional, don't fail if it errors

    # Check graphical session readiness
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'is-active', 'graphical-session.target'],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.stdout.strip() == 'active':
            log_success("✓ graphical-session.target is active")
        else:
            log_warning("⚠ graphical-session.target is not active")
            print("  hyprwhspr starts with the graphical session.")
            print("  A session manager like uwsm is needed to activate graphical-session.target.")
            print("  See: https://github.com/Vladimir-csp/uwsm")
    except Exception:
        pass

    # Check WAYLAND_DISPLAY in systemd user environment
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'show-environment'],
            capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and 'WAYLAND_DISPLAY=' in result.stdout:
            log_success("✓ WAYLAND_DISPLAY set in systemd user environment")
        else:
            log_warning("⚠ WAYLAND_DISPLAY not found in systemd user environment")
            print("  Add to ~/.config/hypr/hyprland.conf:")
            print("    exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE")
    except Exception:
        pass

    if all_ok:
        log_success("Validation passed")
    else:
        log_error("Validation failed - some components are missing")

    return all_ok


# ==================== Test Command ====================

def test_command(live: bool = False, mic_only: bool = False):
    """Test microphone and backend connectivity end-to-end"""
    import time
    import wave
    from io import BytesIO

    print("\n" + "="*60)
    print("hyprwhspr Diagnostic Test")
    print("="*60)

    all_passed = True

    # ===== MICROPHONE TEST =====
    print("\n[Microphone]")

    try:
        from .audio_capture import AudioCapture
    except ImportError:
        from audio_capture import AudioCapture

    # Ensure audio is defined on all code paths (e.g., no devices found)
    audio = None

    try:
        # Check for available devices
        devices = AudioCapture.get_available_input_devices()
        if not devices:
            log_error("No input devices found")
            all_passed = False
        else:
            log_success(f"Found {len(devices)} input device(s)")

            # Get configured device from config
            config = ConfigManager()
            device_id = config.get_setting('audio_device_id', None)

            # Initialize audio capture
            audio = AudioCapture(device_id=device_id, config_manager=config)

            if audio.is_available():
                device_info = audio.get_current_device_info()
                if device_info:
                    log_success(f"Using: {device_info['name']}")
                else:
                    log_success("Audio device available")
            else:
                log_error("Failed to initialize audio capture")
                all_passed = False

    except Exception as e:
        log_error(f"Microphone test failed: {e}")
        all_passed = False
        audio = None

    # If mic-only, stop here
    if mic_only:
        print("\n" + "-"*60)
        if all_passed:
            log_success("Microphone test passed")
        else:
            log_error("Microphone test failed")
        return all_passed

    # ===== BACKEND TEST =====
    print("\n[Backend]")

    config = ConfigManager()
    backend = config.get_setting('transcription_backend', 'pywhispercpp')
    backend = normalize_backend(backend)

    log_info(f"Configured backend: {backend}")

    backend_ready = False

    if backend == 'rest-api':
        # Test REST API connectivity
        endpoint_url = config.get_setting('rest_endpoint_url')
        if not endpoint_url:
            log_error("REST endpoint URL not configured")
            all_passed = False
        else:
            log_success(f"Endpoint: {endpoint_url}")

            # Check credentials
            provider_id = config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if api_key:
                    log_success(f"Credentials configured (provider: {provider_id})")
                    backend_ready = True
                else:
                    log_error(f"API key not found for provider: {provider_id}")
                    all_passed = False
            else:
                # Check for legacy api key
                api_key = config.get_setting('rest_api_key')
                if api_key:
                    log_success("Credentials configured (legacy)")
                    backend_ready = True
                else:
                    log_warning("No API credentials configured")
                    # May still work if endpoint doesn't require auth
                    backend_ready = True

    elif backend == 'realtime-ws':
        # Test WebSocket configuration
        provider_id = config.get_setting('websocket_provider')
        model_id = config.get_setting('websocket_model')

        if not provider_id:
            log_error("WebSocket provider not configured")
            all_passed = False
        elif not model_id:
            log_error("WebSocket model not configured")
            all_passed = False
        else:
            api_key = get_credential(provider_id)
            if api_key:
                log_success(f"Provider: {provider_id}, Model: {model_id}")
                log_success("Credentials configured")
                backend_ready = True
            else:
                log_error(f"API key not found for provider: {provider_id}")
                all_passed = False

    elif backend == 'onnx-asr':
        # Test ONNX-ASR model availability
        try:
            import onnx_asr
            model_name = config.get_setting('onnx_asr_model', 'nemo-parakeet-tdt-0.6b-v3')
            log_success(f"onnx-asr available, model: {model_name}")
            backend_ready = True
        except ImportError:
            log_error("onnx-asr not installed")
            all_passed = False

    elif backend == 'faster-whisper':
        # Test faster-whisper availability
        try:
            import faster_whisper  # noqa: F401
            model_name = config.get_setting('faster_whisper_model', 'base')
            log_success(f"faster-whisper available, model: {model_name}")
            backend_ready = True
        except ImportError:
            log_error("faster-whisper not installed. Run: hyprwhspr setup")
            all_passed = False

    elif backend in ('pywhispercpp', 'nvidia', 'cpu', 'vulkan'):
        # Test pywhispercpp model availability (covers all local whisper variants)
        try:
            try:
                from pywhispercpp.model import Model
            except ImportError:
                from pywhispercpp import Model

            model_name = config.get_setting('model', 'base')
            model_file = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.bin"

            # Try English-only variant if base not found
            if not model_file.exists() and not model_name.endswith('.en'):
                model_file = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.en.bin"

            if model_file.exists():
                log_success(f"pywhispercpp available, model: {model_name}")
                backend_ready = True
            else:
                log_error(f"Model file not found: {model_file}")
                log_info(f"Download with: hyprwhspr model download {model_name}")
                all_passed = False
        except ImportError:
            log_error("pywhispercpp not installed")
            all_passed = False
    else:
        log_warning(f"Unknown backend: {backend}")
        all_passed = False

    # ===== TRANSCRIPTION TEST =====
    print("\n[Transcription]")

    if not backend_ready:
        log_warning("Skipping transcription test (backend not ready)")
    else:
        # Get audio data - either from test.wav or live recording
        audio_data = None
        audio_source = None

        if live:
            # Record live audio
            if audio and audio.is_available():
                print("  Recording for 3 seconds... speak now!")
                try:
                    audio.start_recording()
                    time.sleep(3.0)
                    audio_data = audio.stop_recording()

                    if audio_data is not None and len(audio_data) > 0:
                        # Calculate audio level
                        import numpy as np
                        rms = np.sqrt(np.mean(audio_data**2))
                        db = 20 * np.log10(max(rms, 1e-10))
                        log_success(f"Recorded {len(audio_data)/16000:.1f}s audio (level: {db:.0f}dB)")
                        audio_source = "live recording"

                        # Warn if audio is very quiet (likely silence)
                        if db < -40:
                            log_warning("Audio level very low - check microphone")
                    else:
                        log_error("No audio data captured")
                        all_passed = False
                except Exception as e:
                    log_error(f"Recording failed: {e}")
                    all_passed = False
            else:
                log_error("Cannot record - audio capture not available")
                all_passed = False
        else:
            # Use test.wav
            test_wav_path = Path(HYPRWHSPR_ROOT) / 'share' / 'assets' / 'test.wav'

            if not test_wav_path.exists():
                log_error(f"Test audio file not found: {test_wav_path}")
                log_info("Use --live to record audio instead")
                all_passed = False
            else:
                try:
                    import numpy as np
                    with wave.open(str(test_wav_path), 'rb') as wf:
                        # Read audio data
                        frames = wf.readframes(wf.getnframes())
                        sample_rate = wf.getframerate()

                        # Convert to float32 numpy array
                        sample_width = wf.getsampwidth()
                        if sample_width == 2:  # 16-bit
                            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit
                            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            audio_data = np.frombuffer(frames, dtype=np.float32)

                        # Resample to 16kHz if needed
                        if sample_rate != 16000:
                            from scipy import signal
                            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

                        duration = len(audio_data) / 16000
                        log_success(f"Loaded test.wav ({duration:.1f}s)")
                        audio_source = "test.wav"

                except Exception as e:
                    log_error(f"Failed to load test.wav: {e}")
                    all_passed = False

        # Transcribe if we have audio
        if audio_data is not None and len(audio_data) > 0:
            try:
                from .whisper_manager import WhisperManager
            except ImportError:
                from whisper_manager import WhisperManager

            try:
                log_info("Initializing backend...")
                whisper = WhisperManager(config_manager=config)

                if not whisper.initialize():
                    log_error("Failed to initialize transcription backend")
                    all_passed = False
                else:
                    duration = len(audio_data) / 16000
                    if duration > 5:
                        log_info(f"Transcribing {duration:.0f}s of audio (this may take a moment)...")
                    else:
                        log_info("Transcribing...")

                    # For realtime-ws, we need to handle differently
                    if backend == 'realtime-ws':
                        # Realtime requires streaming - not ideal for test
                        # Just verify connection worked during initialize()
                        log_success("WebSocket connected successfully")
                        log_info("(Realtime transcription requires streaming audio)")
                        whisper.cleanup()
                    else:
                        result = whisper.transcribe_audio(audio_data)

                        if result:
                            # Clean up the result for display
                            result_clean = result.strip()
                            if result_clean:
                                log_success("Transcription successful")
                                print(f"  -> \"{result_clean}\"")
                            else:
                                log_warning("Transcription returned empty result")
                                log_info("This may be normal if audio was silence")
                        else:
                            log_error("Transcription returned no result")
                            all_passed = False

                        # Cleanup
                        if hasattr(whisper, 'cleanup'):
                            whisper.cleanup()

            except Exception as e:
                log_error(f"Transcription test failed: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    # ===== SUMMARY =====
    print("\n" + "-"*60)
    if all_passed:
        log_success("All tests passed!")
    else:
        log_error("Some tests failed")

    return all_passed


# ==================== Keyboard Command ====================

def keyboard_command(action: str):
    """Handle keyboard subcommands"""
    if action == 'list':
        list_keyboards()
    elif action == 'test':
        test_keyboard_access()
    else:
        log_error(f"Unknown keyboard action: {action}")


def list_keyboards():
    """List available keyboard devices"""
    log_info("Discovering available keyboard devices...")
    
    try:
        # Get current config to show selected device
        config = ConfigManager()
        shortcut = config.get_setting("primary_shortcut", "Super+Alt+D")
        selected_device_name = config.get_setting("selected_device_name", None)
        selected_device_path = config.get_setting("selected_device_path", None)
        
        # Get available keyboards
        keyboards = get_available_keyboards(shortcut)
        
        if not keyboards:
            log_warning("No accessible keyboard devices found")
            log_info("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
            return
        
        print("\nAvailable keyboard devices:")
        print("-" * 70)
        
        # Find which device would actually be selected (matching GlobalShortcuts logic)
        selected_device_index = None
        if selected_device_name:
            search_name_lower = selected_device_name.lower()
            for i, kb in enumerate(keyboards):
                kb_name_lower = kb['name'].lower()
                # Match GlobalShortcuts logic: exact match OR partial match
                if kb_name_lower == search_name_lower or search_name_lower in kb_name_lower:
                    selected_device_index = i
                    break  # Use first match, same as GlobalShortcuts
        elif selected_device_path:
            for i, kb in enumerate(keyboards):
                if kb['path'] == selected_device_path:
                    selected_device_index = i
                    break
        
        for i, kb in enumerate(keyboards, 1):
            # Mark only the device that would actually be selected
            marker = " [SELECTED]" if (i - 1) == selected_device_index else ""
            print(f"  {i}. {kb['name']}")
            print(f"     Path: {kb['path']}{marker}")
        
        print("-" * 70)
        print(f"\nTotal: {len(keyboards)} accessible device(s)")
        
        if selected_device_name:
            print(f"\nCurrently selected by name: '{selected_device_name}'")
        elif selected_device_path:
            print(f"\nCurrently selected by path: {selected_device_path}")
        else:
            print("\nNo specific device selected - using auto-detection")
        
        print("\nTo select a device, add to your config (~/.config/hyprwhspr/config.json):")
        print('  "selected_device_name": "Device Name"')
        print('  or')
        print('  "selected_device_path": "/dev/input/eventX"')
        
    except Exception as e:
        log_error(f"Error listing keyboards: {e}")
        import traceback
        traceback.print_exc()


def test_keyboard_access():
    """Test keyboard device accessibility"""
    log_info("Testing keyboard device accessibility...")
    
    try:
        results = test_key_accessibility()
        
        print("\n" + "=" * 70)
        print("Keyboard Device Accessibility Test")
        print("=" * 70)
        
        print(f"\nTotal devices found: {results['total_devices']}")
        print(f"Accessible devices: {len(results['accessible_devices'])}")
        print(f"Inaccessible devices: {len(results['inaccessible_devices'])}")
        
        if results['accessible_devices']:
            print("\n✓ Accessible devices:")
            for dev in results['accessible_devices']:
                print(f"  - {dev['name']}")
                print(f"    Path: {dev['path']}")
        
        if results['inaccessible_devices']:
            print("\n✗ Inaccessible devices:")
            for dev in results['inaccessible_devices']:
                print(f"  - {dev['name']}")
                print(f"    Path: {dev['path']}")
            print("\nNote: Inaccessible devices may be in use by another process")
            print("      (e.g., Espanso, keyd, kmonad) or require permissions")
        
        if not results['accessible_devices']:
            print("\n⚠ No accessible devices found!")
            print("Solutions:")
            print("  1. Add yourself to 'input' group: sudo usermod -aG input $USER")
            print("     (then log out and back in)")
            print("  2. Check if devices are grabbed by other tools:")
            print("     sudo fuser /dev/input/event*")
            print("  3. Consider using 'selected_device_name' in config to avoid conflicts")
        
    except Exception as e:
        log_error(f"Error testing keyboard access: {e}")
        import traceback
        traceback.print_exc()


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
    if (USER_SYSTEMD_DIR / PARAKEET_SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {PARAKEET_SERVICE_NAME}")
    if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {RESUME_SERVICE_NAME} (deprecated)")

    # Waybar integration
    waybar_module = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    if waybar_module.exists():
        items_to_remove.append("Waybar integration")
    
    # User configuration
    if USER_CONFIG_DIR.exists():
        items_to_remove.append(f"User configuration: {USER_CONFIG_DIR}")
    
    # Backend installations
    if VENV_DIR.exists():
        items_to_remove.append(f"Main backend venv: {VENV_DIR}")
    if PARAKEET_VENV_DIR.exists():
        items_to_remove.append(f"Parakeet venv: {PARAKEET_VENV_DIR}")
    if PYWHISPERCPP_SRC_DIR.exists():
        items_to_remove.append(f"pywhispercpp source: {PYWHISPERCPP_SRC_DIR}")
    
    # Models
    if not keep_models and PYWHISPERCPP_MODELS_DIR.exists():
        models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
        if models:
            items_to_remove.append(f"Whisper models: {len(models)} model(s) in {PYWHISPERCPP_MODELS_DIR}")
    
    # State files
    if STATE_DIR.exists():
        items_to_remove.append(f"State files: {STATE_DIR}")
    
    # Credentials
    if CREDENTIALS_FILE.exists():
        items_to_remove.append("Stored API credentials")
    
    # Temp files
    temp_dir = USER_BASE / 'temp'
    if temp_dir.exists():
        items_to_remove.append(f"Temporary files: {temp_dir}")
    
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

        # Stop and disable Parakeet service
        if (USER_SYSTEMD_DIR / PARAKEET_SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', PARAKEET_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', PARAKEET_SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / PARAKEET_SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {PARAKEET_SERVICE_NAME}")

        # Stop and disable deprecated resume service
        if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', RESUME_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', RESUME_SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {RESUME_SERVICE_NAME}")

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
    
    # 3. Remove user configuration
    log_info("Removing user configuration...")
    try:
        if USER_CONFIG_DIR.exists():
            shutil.rmtree(USER_CONFIG_DIR, ignore_errors=True)
            log_success(f"Removed {USER_CONFIG_DIR}")
    except Exception as e:
        error_msg = f"Failed to remove user configuration: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 4. Remove backend installations
    log_info("Removing backend installations...")
    try:
        # Detect current backend and cleanup
        current_backend = _detect_current_backend()
        if current_backend:
            _cleanup_backend(current_backend)
        
        # Remove main venv
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR, ignore_errors=True)
            log_success(f"Removed {VENV_DIR}")
        
        # Remove Parakeet venv
        if PARAKEET_VENV_DIR.exists():
            shutil.rmtree(PARAKEET_VENV_DIR, ignore_errors=True)
            log_success(f"Removed {PARAKEET_VENV_DIR}")
        
        # Remove pywhispercpp source
        if PYWHISPERCPP_SRC_DIR.exists():
            shutil.rmtree(PYWHISPERCPP_SRC_DIR, ignore_errors=True)
            log_success(f"Removed {PYWHISPERCPP_SRC_DIR}")
    except Exception as e:
        error_msg = f"Failed to remove backend installations: {e}"
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
    
    # 6. Remove state files
    log_info("Removing state files...")
    try:
        if STATE_DIR.exists():
            shutil.rmtree(STATE_DIR, ignore_errors=True)
            log_success(f"Removed {STATE_DIR}")
    except Exception as e:
        error_msg = f"Failed to remove state files: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 7. Remove credentials
    log_info("Removing stored credentials...")
    try:
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink(missing_ok=True)
            log_success("Removed stored API credentials")
    except Exception as e:
        error_msg = f"Failed to remove credentials: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 8. Remove temp files
    log_info("Removing temporary files...")
    try:
        temp_dir = USER_BASE / 'temp'
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            log_success(f"Removed {temp_dir}")
        
        # Also try to remove the entire USER_BASE directory if it's empty or only contains empty dirs
        if USER_BASE.exists():
            try:
                # Check if directory is empty or only contains empty subdirs
                has_content = False
                for item in USER_BASE.iterdir():
                    if item.is_file():
                        has_content = True
                        break
                    elif item.is_dir():
                        # Check if subdirectory has content
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
    except Exception as e:
        error_msg = f"Failed to remove temporary files: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 9. Remove system permissions (if requested)
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
    if not skip_permissions and not permissions_removed:
        print("Note: System permissions (group memberships, udev rules) were not removed.")
        print("      You may want to remove them manually if needed.")
    print()


def record_command(action: str, language: str = None):
    """
    Control recording via CLI - useful when keyboard grab is not possible.

    This writes to the recording control FIFO to trigger start/stop/cancel/toggle
    without requiring keyboard grab. Useful for users with:
    - External hotkey systems (KDE, GNOME, sxhkd, etc.)
    - Keyboard remappers that grab devices (Espanso, keyd, kmonad)
    - Multiple keyboard tools that conflict with grab_keys

    Args:
        action: The action to perform (start, stop, cancel, toggle, status)
        language: Optional language code for transcription (e.g., 'en', 'it', 'de')
    """
    import stat

    def is_recording() -> bool:
        """Check if currently recording (status file exists with 'true')"""
        if not RECORDING_STATUS_FILE.exists():
            return False
        try:
            content = RECORDING_STATUS_FILE.read_text().strip().lower()
            return content == 'true'
        except Exception:
            return False

    def send_control(command: str) -> bool:
        """Send a command to the recording control FIFO"""
        if not RECORDING_CONTROL_FILE.exists():
            log_error("Recording control file not found.")
            log_error("Is the hyprwhspr service running?")
            log_info("Start it with: systemctl --user start hyprwhspr")
            return False

        # Check if it's a FIFO (named pipe)
        try:
            file_stat = RECORDING_CONTROL_FILE.stat()
            is_fifo = stat.S_ISFIFO(file_stat.st_mode)
        except Exception:
            is_fifo = False

        try:
            if is_fifo:
                # Open FIFO in non-blocking mode with timeout
                import select
                fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
                fd_closed = False
                try:
                    # Wait for FIFO to be ready for writing (service is listening)
                    _, ready, _ = select.select([], [fd], [], 2.0)
                    if not ready:
                        os.close(fd)
                        fd_closed = True
                        log_error("Service not responding (timeout waiting for FIFO)")
                        log_info("The service may be busy or not running properly")
                        return False
                    os.write(fd, (command + '\n').encode())
                finally:
                    if not fd_closed:
                        os.close(fd)
            else:
                # Fall back to regular file write
                RECORDING_CONTROL_FILE.write_text(command + '\n')
            return True
        except OSError as e:
            if e.errno == 6:  # ENXIO - no reader on FIFO
                log_error("Service not listening on control FIFO")
                log_info("Is the hyprwhspr service running?")
                log_info("Start it with: systemctl --user start hyprwhspr")
            else:
                log_error(f"Failed to send command: {e}")
            return False
        except Exception as e:
            log_error(f"Failed to send command: {e}")
            return False

    # Build start command with optional language
    start_cmd = f'start:{language}' if language else 'start'

    if action == 'start':
        if is_recording():
            log_warning("Already recording")
            return
        if send_control(start_cmd):
            msg = f"Recording started (language: {language})" if language else "Recording started"
            log_success(msg)

    elif action == 'stop':
        if not is_recording():
            log_warning("Not currently recording")
            return
        if send_control('stop'):
            log_success("Recording stopped")

    elif action == 'cancel':
        if not is_recording():
            log_warning("Not currently recording")
            return
        if send_control('cancel'):
            log_success("Recording cancelled (audio discarded)")

    elif action == 'toggle':
        if is_recording():
            if send_control('stop'):
                log_success("Recording stopped")
        else:
            if send_control(start_cmd):
                msg = f"Recording started (language: {language})" if language else "Recording started"
                log_success(msg)

    elif action == 'status':
        if is_recording():
            log_info("Status: Recording in progress")
        else:
            log_info("Status: Idle")

    else:
        log_error(f"Unknown action: {action}")
        log_info("Available actions: start, stop, cancel, toggle, status")

