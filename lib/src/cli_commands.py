"""
CLI command implementations for hyprwhspr
"""

import os
import sys
import json
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
    print("\nTry installing it, for example:", file=sys.stderr)
    print("  pacman -S python-rich    # system-wide on Arch", file=sys.stderr)
    print("  pip install rich>=13.0.0  # or via pip", file=sys.stderr)
    sys.exit(1)

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .paths import CONFIG_DIR, CONFIG_FILE
except ImportError:
    from paths import CONFIG_DIR, CONFIG_FILE

try:
    from .backend_utils import BACKEND_DISPLAY_NAMES, normalize_backend
except ImportError:
    from backend_utils import BACKEND_DISPLAY_NAMES, normalize_backend

try:
    from .backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT, USER_BASE, PYWHISPERCPP_SRC_DIR
    )
except ImportError:
    from backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT, USER_BASE, PYWHISPERCPP_SRC_DIR
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
        'cpu', 'nvidia', 'amd', 'vulkan', 'parakeet', 'rest-api', or None if not detected
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
            # Check if it's parakeet by checking endpoint URL
            endpoint = config_manager.get_setting('rest_endpoint_url', '')
            if endpoint == 'http://127.0.0.1:8080/transcribe':
                # Check if parakeet venv exists
                if PARAKEET_VENV_DIR.exists():
                    return 'parakeet'
            return 'rest-api'
        if backend == 'realtime-ws':
            return 'realtime-ws'
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
        backend_type: 'cpu', 'nvidia', 'amd', 'vulkan', 'parakeet', or 'remote'
    
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
    print("PyWhisperCPP (Local in-memory default, very fast):")
    print("  [1] CPU - CPU-only, works on all systems")
    print("  [2] NVIDIA - NVIDIA GPU acceleration (CUDA)")
    print("  [3] AMD - AMD GPU acceleration (Vulkan)")
    print()
    print("REST API (Remote - Cloud API or localhost):")
    print("  [4] Configure cloud provider or custom backend")
    print()
    print("Realtime WebSocket (Low-latency cloud streaming transcription):")
    print("  [5] Realtime WebSocket (OpenAI)")
    print()
    print("Parakeet (Local REST Server - Latest NVIDIA Parakeet model):")
    print("  [6] NVIDIA Parakeet model")
    print()
    
    while True:
        try:
            choice = Prompt.ask("Select backend", choices=['1', '2', '3', '4', '5', '6'], default='1')
            backend_map = {
                '1': 'cpu',
                '2': 'nvidia',
                '3': 'vulkan',
                '4': 'rest-api',
                '5': 'realtime-ws',
                '6': 'parakeet'
            }
            selected = backend_map[choice]
            
            # Warn if switching to different backend
            if current_backend and current_backend != selected:
                backend_names = {
                    'cpu': 'CPU',
                    'nvidia': 'NVIDIA (CUDA)',
                    'amd': 'AMD (Vulkan)',
                    'vulkan': 'AMD (Vulkan)',
                    'parakeet': 'Parakeet',
                    'rest-api': 'REST API',
                    'realtime-ws': 'Realtime WebSocket',
                    'pywhispercpp': 'pywhispercpp'
                }
                print(f"\n⚠️  Switching from {backend_names.get(current_backend, current_backend)} to {backend_names.get(selected, selected)}")
                
                if selected == 'parakeet':
                    print("Parakeet uses a separate venv and runs as a local REST server.")
                    if current_backend not in ['rest-api', 'remote', 'realtime-ws', 'parakeet']:
                        print("This will uninstall the current backend.")
                        if not Confirm.ask("Continue?", default=True):
                            continue
                elif current_backend not in ['rest-api', 'remote', 'realtime-ws', 'parakeet'] and selected not in ['rest-api', 'remote', 'realtime-ws', 'parakeet']:
                    print("This will uninstall the current backend and install the new one.")
                    if not Confirm.ask("Continue?", default=True):
                        continue
                elif selected in ['rest-api', 'realtime-ws']:
                    backend_type_name = 'REST API' if selected == 'rest-api' else 'Realtime WebSocket'
                    print(f"Switching to {backend_type_name} backend.")
                    if current_backend == 'parakeet':
                        print("The Parakeet venv will no longer be needed.")
                        cleanup_venv = Confirm.ask("Remove the Parakeet venv to free up space?", default=False)
                    else:
                        print("The local backend venv will no longer be needed.")
                        cleanup_venv = Confirm.ask("Remove the venv to free up space?", default=False)
                    backend_names = BACKEND_DISPLAY_NAMES
                    print(f"\n✓ Selected: {backend_names[selected]}")
                    return (selected, cleanup_venv, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
            
            # If re-selecting same backend, offer reinstall option
            if current_backend == selected and selected not in ['rest-api', 'remote', 'realtime-ws', 'parakeet']:
                backend_names = BACKEND_DISPLAY_NAMES
                print(f"\n{backend_names.get(selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reinstall, continue to return with wants_reinstall=True
            elif current_backend == selected and selected == 'parakeet':
                backend_names = BACKEND_DISPLAY_NAMES
                print(f"\n{backend_names.get(selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reinstall, continue to return with wants_reinstall=True
            elif current_backend == selected and selected == 'realtime-ws':
                backend_names = BACKEND_DISPLAY_NAMES
                print(f"\n{backend_names.get(selected)} backend is already configured.")
                reconfigure = Confirm.ask("Reconfigure backend?", default=False)
                if not reconfigure:
                    print("Keeping existing configuration.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reconfigure, continue to return with wants_reinstall=True for correct state tracking
            elif current_backend == selected and selected in ['rest-api', 'remote']:
                backend_names = BACKEND_DISPLAY_NAMES
                print(f"\n{backend_names.get(selected)} backend is already configured.")
                reconfigure = Confirm.ask("Reconfigure backend?", default=False)
                if not reconfigure:
                    print("Keeping existing configuration.")
                    return (selected, False, False)  # Return tuple: (backend, cleanup_venv, wants_reinstall)
                # If yes to reconfigure, continue to return with wants_reinstall=True for correct state tracking
            
            backend_names = {
                'cpu': 'CPU',
                'nvidia': 'NVIDIA (CUDA)',
                'amd': 'AMD (Vulkan)',
                'vulkan': 'AMD (Vulkan)',
                'parakeet': 'Parakeet',
                'rest-api': 'REST API',
                'realtime-ws': 'Realtime WebSocket'
            }
            print(f"\n✓ Selected: {backend_names[selected]}")
            # Check if user wants to reinstall/reconfigure (same backend selected and they said yes)
            # For local backends: wants_reinstall means reinstall
            # For cloud backends: wants_reinstall means reconfigure (correctly tracks user intent)
            wants_reinstall = (current_backend == selected)
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
        for provider_id, provider_name, model_ids in all_providers_list:
            provider = get_provider(provider_id)
            if provider and provider.get('websocket_endpoint'):
                providers_list.append((provider_id, provider_name, model_ids))
    else:
        providers_list = all_providers_list
    
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
                
                # Optional custom body fields
                has_body = Confirm.ask("Add custom body fields?", default=False)
                custom_body = {}
                if has_body:
                    body_json = Prompt.ask("Enter body fields as JSON (e.g., {\"model\": \"custom-model\"})", default="{}")
                    try:
                        custom_body = json.loads(body_json)
                        if not isinstance(custom_body, dict):
                            log_error("Body fields must be a JSON object")
                            custom_body = {}
                    except json.JSONDecodeError as e:
                        log_error(f"Invalid JSON: {e}")
                        custom_body = {}
                
                custom_config = {
                    'endpoint': endpoint_url,
                    'headers': custom_headers,
                    'body': custom_body
                }
                
                return ('custom', None, api_key, custom_config)
                
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


def setup_command():
    """Interactive full initial setup"""
    print("\n" + "="*60)
    print("hyprwhspr setup")
    print("="*60)
    print("\nThis setup will guide you through configuring hyprwhspr.")
    print("Skip any step by answering 'no'.\n")

    # Check for MISE interference
    mise_active, mise_details = _check_mise_active()
    if mise_active:
        log_warning("Warning! MISE is active. This may cause installation errors.")
        log_warning("To fix: mise deactivate (or: mise unuse -g python)")
        print() 

        if not Confirm.ask("Continue anyway? (not recommended)", default=False):
            log_info("Setup cancelled. Please deactivate MISE and try again.")
            return
    
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
    
    # Handle backend switching
    if current_backend and current_backend != backend_normalized:
        if current_backend not in ['rest-api', 'remote', 'realtime-ws']:
            # Switching from local to something else
            if not _cleanup_backend(current_backend):
                log_warning("Failed to clean up old backend, continuing anyway...")
        
        # Also handle cleanup when switching TO Parakeet FROM local
        if backend_normalized == 'parakeet' and current_backend in ['cpu', 'nvidia', 'amd', 'vulkan']:
            if VENV_DIR.exists():
                cleanup_main_venv = Confirm.ask(
                    "Remove the old local backend venv to free up space?", 
                    default=False
                )
                if cleanup_main_venv:
                    import shutil
                    log_info("Removing main venv...")
                    shutil.rmtree(VENV_DIR)
                    log_success("Main venv removed")
        
        if cleanup_venv and backend_normalized in ['rest-api', 'remote', 'realtime-ws']:
            # User wants to remove venv when switching to cloud backend
            if VENV_DIR.exists():
                log_info("Removing venv as requested...")
                import shutil
                shutil.rmtree(VENV_DIR)
                log_success("Venv removed")
    
    # Step 1.5: Backend installation (if not cloud backend)
    parakeet_installed = False
    if backend_normalized not in ['rest-api', 'remote', 'realtime-ws']:
        # Skip installation section if user selected the same backend and declined reinstalling
        if current_backend == backend_normalized and not wants_reinstall:
            # User already said "no" to reinstalling in the selection step, skip installation section
            # If Parakeet is already installed, mark it as installed so REST API auto-configuration works
            if backend_normalized == 'parakeet' and current_backend == 'parakeet':
                parakeet_installed = True
        else:
            # New backend selected, or user wants to reinstall existing backend
            print("\n" + "="*60)
            print("Backend Installation")
            print("="*60)
            if backend_normalized == 'parakeet':
                print("\nThis will install the Parakeet backend.")
                print("This will create a separate virtual environment and install dependencies.")
                print("Parakeet runs as a local REST API server that hyprwhspr connects to.")
            else:
                print(f"\nThis will install the {backend_normalized.upper()} backend for pywhispercpp.")
                print("This may take several minutes as it compiles from source.")
            if not Confirm.ask("Proceed with backend installation?", default=True):
                log_warning("Skipping backend installation. You can install it later.")
                log_warning("Backend installation is required for local transcription to work.")
            else:
                # Pass force_rebuild=True when reinstalling to ensure clean venv
                # Use normalized backend to ensure 'amd' -> 'vulkan' for new installs
                if not install_backend(backend_normalized, force_rebuild=wants_reinstall):
                    log_error("Backend installation failed. Setup cannot continue.")
                    return
                
                if backend_normalized == 'parakeet':
                    parakeet_installed = True
    
    # Step 2: Provider/model selection (if REST API backend or parakeet)
    remote_config = None
    selected_model = None
    if backend_normalized == 'parakeet':
        # Auto-configure REST API for parakeet (always needed, regardless of installation status)
        # If installation was skipped, user can install Parakeet later, but config should be set up now
        log_info("Auto-configuring REST API for Parakeet...")
        remote_config = {
            'transcription_backend': 'rest-api',
            'rest_endpoint_url': 'http://127.0.0.1:8080/transcribe',
            'rest_headers': {},
            'rest_body': {}
        }
        log_success("Parakeet REST API configuration ready")
        if not parakeet_installed:
            log_warning("Parakeet backend is not installed. Install it later to use this configuration.")
        # Note about manual start will be shown later if systemd is not set up
    elif backend_normalized in ['rest-api', 'remote']:
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
        print("\nRealtime Mode:")
        print("  1. Transcribe - Convert speech to text (default)")
        print("  2. Converse - Voice-to-AI: speak and get AI responses")
        mode_choice = Prompt.ask("Select mode", choices=['1', '2'], default='1')
        realtime_mode = 'transcribe' if mode_choice == '1' else 'converse'
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
                'import sounddevice, numpy, requests; import websocket'
            ], check=False, capture_output=True, show_output_on_error=False)
            deps_installed = result.returncode == 0
        except Exception:
            pass
        
        # Install base dependencies if needed (excluding pywhispercpp)
        if cur_req_hash != stored_req_hash or not stored_req_hash or not deps_installed:
            log_info("Installing base Python dependencies (excluding pywhispercpp)...")
            
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
    
    elif backend not in ['rest-api', 'remote', 'realtime-ws', 'parakeet']:
        # Local backend - prompt for model selection
        # Note: Parakeet doesn't use Whisper models, it's handled via REST API auto-configuration
        selected_model = _prompt_model_selection()
    
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
    
    # Check if dependencies are available
    mic_osd_available = False
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from mic_osd import MicOSDRunner
        mic_osd_available = MicOSDRunner.is_available()
        if not mic_osd_available:
            print(f"\nNote: {MicOSDRunner.get_unavailable_reason()}")
    except ImportError:
        pass
    
    if mic_osd_available:
        setup_mic_osd_choice = Confirm.ask("Enable mic-osd visualization?", default=True)
    else:
        print("\nDependencies not found. Install python-gobject and gtk4-layer-shell to enable.")
        setup_mic_osd_choice = Confirm.ask("Enable mic-osd anyway (will work after installing deps)?", default=False)
    
    # Step 4: Systemd setup
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)
    if backend == 'parakeet':
        print("\nSystemd user services will run both hyprwhspr and the Parakeet server in the background.")
        print("This will enable/configure:")
        print("  • ydotool.service (required dependency, provides paste)")
        print("  • parakeet-tdt-0.6b-v3.service (Parakeet REST API server)")
        print("  • hyprwhspr.service (main application)")
    else:
        print("\nSystemd user service will run hyprwhspr in the background.")
        print("This will enable/configure:")
        print("  • ydotool.service (required dependency, provides paste)")
        print("  • hyprwhspr.service (main application)")
    setup_systemd_choice = Confirm.ask("Set up systemd user service" + ("s" if backend == 'parakeet' else ""), default=True)
    
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
    if backend == 'parakeet' and remote_config:
        print(f"Endpoint: {remote_config.get('rest_endpoint_url', 'N/A')}")
        print("Model: parakeet-tdt-0.6b-v3")
    elif remote_config:
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
    elif selected_model:
        print(f"Model: {selected_model}")
    print(f"Waybar integration: {'Yes' if setup_waybar_choice else 'No'}")
    print(f"Mic-OSD visualization: {'Yes' if setup_mic_osd_choice else 'No'}")
    if backend == 'parakeet':
        if setup_systemd_choice:
            print("Systemd services: Yes (ydotool + Parakeet + hyprwhspr)")
        else:
            print("Systemd services: No")
    else:
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
            if backend == 'parakeet':
                print("  Start the Parakeet server manually:")
                print(f"    {PARAKEET_VENV_DIR / 'bin' / 'python'} {PARAKEET_SCRIPT}")
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


# ==================== Omarchy Commands ====================

def _auto_download_model():
    """Auto-download base model without prompts"""
    try:
        from .backend_installer import download_pywhispercpp_model
    except ImportError:
        from backend_installer import download_pywhispercpp_model

    log_info("Downloading base Whisper model...")
    if download_pywhispercpp_model('base'):
        log_success("Model downloaded")
    else:
        log_warning("Model download failed - can download later with: hyprwhspr model download")


def omarchy_command():
    """
    Automated Omarchy setup

    This command:
    1. Auto-detects GPU hardware (NVIDIA/AMD/Intel/CPU)
    2. Installs appropriate backend (CUDA for NVIDIA, Vulkan for others, CPU fallback)
    3. Configures Omarchy defaults (auto recording mode, Waybar integration)
    4. Sets up and starts systemd service
    5. Validates installation

    All without user interaction.
    """
    # Import functions we need
    try:
        from .backend_installer import detect_gpu_type, install_backend
        from .config_manager import ConfigManager
    except ImportError:
        from backend_installer import detect_gpu_type, install_backend
        from config_manager import ConfigManager

    # 1. Print banner
    print("\n" + "="*60)
    print("hyprwhspr + omarchy - native stt")
    print("="*60)

    # 2. Check and handle MISE
    mise_active, mise_details = _check_mise_active()
    mise_free_env = None
    if mise_active:
        log_warning("MISE detected - will be temporarily deactivated for installation")
        log_warning(f"Details:\n    {mise_details}")
        mise_free_env = _create_mise_free_environment()
        # Note: install_backend() already handles MISE warnings

    # 3. Auto-detect GPU type
    log_info("Detecting hardware...")
    gpu_type = detect_gpu_type()  # Returns 'nvidia', 'vulkan', or 'cpu'

    # gpu_type is already the backend name
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

    if not install_backend(backend, force_rebuild=False):
        log_error("Backend installation failed")
        return False

    # 5. Configure Omarchy defaults
    log_info("Configuring Omarchy defaults...")
    config = ConfigManager()
    config.set_setting('recording_mode', 'auto')
    config.set_setting('grab_keys', True)
    config.set_setting('transcription_backend', 'pywhispercpp')
    config.set_setting('mic_osd_enabled', True)
    config.save_config()
    log_success("Configuration saved")

    # 6. Download model (for local backends)
    if backend in ['cpu', 'nvidia', 'vulkan']:
        _auto_download_model()

    # 7. Waybar integration (if detected)
    print("\n" + "="*60)
    print("Waybar Integration")
    print("="*60)

    waybar_config = Path.home() / '.config' / 'waybar' / 'config.jsonc'
    if waybar_config.exists():
        log_info("Waybar detected - installing integration...")
        waybar_command('install')
    else:
        log_info("Waybar not detected - skipping")

    # 8. Systemd service
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)

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

    # 9. Validate
    print("\n" + "="*60)
    print("Validation")
    print("="*60)
    validate_command()

    # 10. Restart service for clean initialization
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
    print("\nOmarchy setup completed successfully!")
    print("\nNext steps:")
    print("  1. Log out and back in (for group permissions)")
    print("  2. Press Super+Alt+D to start dictating")
    print("  3. Tap (<400ms) to toggle, hold (>=400ms) for push-to-talk")
    print("\nFor help: hyprwhspr --help")

    return True


# ==================== Config Commands ====================

def config_command(action: str):
    """Handle config subcommands"""
    if action == 'init':
        setup_config()
    elif action == 'show':
        show_config()
    elif action == 'edit':
        edit_config()
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
        
        config.save_config()
        log_success(f"Created {config_file}")
    else:
        log_info(f"Config already exists at {config_file}")
        # Update existing config if needed
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            
            # Update backend if provided (accept both old 'local'/'remote' and new backend types)
            if backend:
                # Map old values for backward compatibility
                if backend == 'local':
                    backend = 'cpu'  # Map old 'local' to 'cpu'
                elif backend == 'remote':
                    backend = 'rest-api'  # Map old 'remote' to 'rest-api'
                existing_config['transcription_backend'] = backend
            
            # Apply remote configuration if provided
            if remote_config:
                for key, value in remote_config.items():
                    existing_config[key] = value
            
            # Update model if provided, otherwise default to base if missing
            if model:
                existing_config['model'] = model
            elif 'model' not in existing_config and not remote_config:
                # Only set default model if not using remote backend
                existing_config['model'] = 'base'
            
            # Add audio_feedback if missing
            if 'audio_feedback' not in existing_config:
                existing_config['audio_feedback'] = True
                existing_config['start_sound_volume'] = 0.5
                existing_config['stop_sound_volume'] = 0.5
                existing_config['start_sound_path'] = 'ping-up.ogg'
                existing_config['stop_sound_path'] = 'ping-down.ogg'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2)
            
            log_success("Updated existing config")
        except Exception as e:
            log_error(f"Failed to update config: {e}")


def show_config():
    """Display current config"""
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        log_error("Config file not found. Run 'hyprwhspr config init' first.")
        return
    
    try:
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
    
    # Detect current backend to determine if Parakeet service is needed
    current_backend = _detect_current_backend()
    is_parakeet = current_backend == 'parakeet'
    
    # Create user systemd directory
    USER_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Handle Parakeet service if needed
    if is_parakeet:
        # Validate Parakeet components exist
        if not PARAKEET_SCRIPT.exists():
            log_error(f"Parakeet script not found at {PARAKEET_SCRIPT}")
            return False
        if not PARAKEET_VENV_DIR.exists():
            log_error(f"Parakeet venv not found at {PARAKEET_VENV_DIR}")
            return False
        
        # Clean old Parakeet service file first (following venv cleanup pattern)
        parakeet_service_dest = USER_SYSTEMD_DIR / PARAKEET_SERVICE_NAME
        if parakeet_service_dest.exists():
            log_info("Removing existing Parakeet service file...")
            # Stop and disable service if it's active
            run_command(['systemctl', '--user', 'stop', PARAKEET_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', PARAKEET_SERVICE_NAME], check=False)
            # Remove the service file
            try:
                parakeet_service_dest.unlink()
            except Exception as e:
                log_warning(f"Failed to remove old Parakeet service file: {e}")
        
        # Read Parakeet service template
        parakeet_service_source = Path(HYPRWHSPR_ROOT) / 'config' / 'systemd' / PARAKEET_SERVICE_NAME
        if not parakeet_service_source.exists():
            log_error(f"Parakeet service template not found: {parakeet_service_source}")
            return False
        
        try:
            with open(parakeet_service_source, 'r', encoding='utf-8') as f:
                parakeet_service_content = f.read()
            
            # Substitute paths
            parakeet_service_content = parakeet_service_content.replace('/usr/lib/hyprwhspr', HYPRWHSPR_ROOT)
            parakeet_service_content = parakeet_service_content.replace('/home/USER', str(USER_HOME))
            parakeet_service_content = parakeet_service_content.replace(
                '/home/USER/.local/share/hyprwhspr/parakeet-venv',
                str(PARAKEET_VENV_DIR)
            )
            
            # Write substituted content
            with open(parakeet_service_dest, 'w', encoding='utf-8') as f:
                f.write(parakeet_service_content)
            
            log_success("Parakeet service file created with correct paths")
        except IOError as e:
            log_error(f"Failed to read/write Parakeet service file: {e}")
            return False
    
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
        
        # Conditionally inject Parakeet dependencies if Parakeet is configured
        if is_parakeet:
            # Add Parakeet service to After= line
            if 'After=' in service_content:
                # Find the After= line and add parakeet service
                lines = service_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('After='):
                        # Check if parakeet service is already in the line
                        if PARAKEET_SERVICE_NAME not in line:
                            lines[i] = line.rstrip() + ' ' + PARAKEET_SERVICE_NAME
                        break
                service_content = '\n'.join(lines)
            
            # Add Parakeet service to Wants= line (or create it if it doesn't exist)
            if 'Wants=' in service_content:
                lines = service_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Wants='):
                        # Check if parakeet service is already in the line
                        if PARAKEET_SERVICE_NAME not in line:
                            lines[i] = line.rstrip() + ' ' + PARAKEET_SERVICE_NAME
                        break
                service_content = '\n'.join(lines)
            else:
                # Insert Wants= line after After= line
                lines = service_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('After='):
                        lines.insert(i + 1, f'Wants={PARAKEET_SERVICE_NAME}')
                        break
                service_content = '\n'.join(lines)
        
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
        if is_parakeet:
            run_command(['systemctl', '--user', 'enable', '--now', PARAKEET_SERVICE_NAME], check=False)

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
        if is_parakeet:
            run_command(['systemctl', '--user', 'disable', '--now', PARAKEET_SERVICE_NAME], check=False)
        run_command(['systemctl', '--user', 'disable', '--now', SERVICE_NAME], check=False)
        # Disable suspend/resume service if it exists
        log_success("Systemd user service disabled")
    
    return True


def systemd_status():
    """Show systemd service status"""
    # Check if Parakeet backend is configured
    current_backend = _detect_current_backend()
    is_parakeet = current_backend == 'parakeet'

    try:
        if is_parakeet:
            # Show status for all services
            log_info("Parakeet service status:")
            run_command(['systemctl', '--user', 'status', PARAKEET_SERVICE_NAME], check=False)
            print()  # Add spacing

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
    # Check if Parakeet backend is configured
    current_backend = _detect_current_backend()
    is_parakeet = current_backend == 'parakeet'
    
    log_info("Restarting service...")
    try:
        if is_parakeet:
            # Restart both services
            run_command(['systemctl', '--user', 'restart', PARAKEET_SERVICE_NAME], check=False)
            log_success("Parakeet service restarted")
            run_command(['systemctl', '--user', 'restart', SERVICE_NAME], check=False)
            log_success("hyprwhspr service restarted")
        else:
            # Restart hyprwhspr only
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
    # Check if dependencies are available
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mic_osd import MicOSDRunner
        
        if not MicOSDRunner.is_available():
            reason = MicOSDRunner.get_unavailable_reason()
            log_error(f"Cannot enable mic-osd: {reason}")
            log_info("Install required packages: python-gobject gtk4-layer-shell")
            return False
    except ImportError:
        log_error("mic-osd module not found")
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
    
    # Check dependencies
    deps_available = False
    deps_reason = ""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from mic_osd import MicOSDRunner
        deps_available = MicOSDRunner.is_available()
        if not deps_available:
            deps_reason = MicOSDRunner.get_unavailable_reason()
    except ImportError:
        deps_reason = "mic_osd module not found"
    
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
        log_info("Install required packages: python-gobject gtk4-layer-shell")
    
    return enabled and deps_available


# ==================== Model Commands ====================

def model_command(action: str, model_name: str = 'base'):
    """Handle model subcommands"""
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
    config_file = USER_CONFIG_DIR / 'config.json'
    if config_file.exists():
        log_success(f"Config exists: {config_file}")
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
    
    # Add user to required groups
    try:
        run_sudo_command(['usermod', '-a', '-G', 'input,audio,tty', username], check=False)
        log_success("Added user to required groups")
    except Exception as e:
        log_warning(f"Failed to add user to groups: {e}")
    
    # Load uinput module
    if not Path('/dev/uinput').exists():
        log_info("Loading uinput module...")
        run_sudo_command(['modprobe', 'uinput'], check=False)
        import time
        time.sleep(2)
    
    # Create udev rule
    udev_rule = Path('/etc/udev/rules.d/99-uinput.rules')
    if not udev_rule.exists():
        log_info("Creating udev rule...")
        rule_content = '# Allow members of the input group to access uinput device\nKERNEL=="uinput", GROUP="input", MODE="0660"\n'
        try:
            run_sudo_command(['tee', str(udev_rule)], input_data=rule_content.encode(), check=False)
            log_success("udev rule created")
        except Exception as e:
            log_warning(f"Failed to create udev rule: {e}")
    else:
        log_info("udev rule already exists")
    
    # Reload udev
    try:
        run_sudo_command(['udevadm', 'control', '--reload-rules'], check=False)
        run_sudo_command(['udevadm', 'trigger', '--name-match=uinput'], check=False)
        log_success("udev rules reloaded")
    except Exception as e:
        log_warning(f"Failed to reload udev rules: {e}")
    
    log_warning("You may need to log out/in for new group memberships to apply")


# ==================== Validate Command ====================

def cleanup_venv_command():
    """Remove the venv directory completely"""
    if not VENV_DIR.exists():
        log_info("No venv found")
        return True
    
    log_warning("This will remove the entire Python virtual environment.")
    log_warning("All installed packages (including pywhispercpp) will be deleted.")
    if not Confirm.ask("Are you sure?", default=False):
        log_info("Cleanup cancelled")
        return False
    
    try:
        import shutil
        shutil.rmtree(VENV_DIR)
        log_success("Venv removed successfully")
        return True
    except Exception as e:
        log_error(f"Failed to remove venv: {e}")
        return False


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
    
    # Check pywhispercpp installation
    pywhispercpp_missing = False
    if venv_python.exists() and not venv_corrupted:
        try:
            result = subprocess.run(
                [str(venv_python), '-c', 'import pywhispercpp'],
                check=False,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                log_warning("pywhispercpp is not installed in venv")
                pywhispercpp_missing = True
        except Exception:
            pass
    
    if not venv_corrupted and not pywhispercpp_missing:
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
    
    if pywhispercpp_missing:
        print("\nIssues found:")
        print("  • pywhispercpp is not installed")
        print("\nOptions:")
        print("  [1] Reinstall backend (detect and install)")
        print("  [2] Skip (manual repair required)")
        
        choice = Prompt.ask("Select option", choices=['1', '2'], default='1')
        if choice == '1':
            # Detect backend type from config
            current_backend = _detect_current_backend()
            if current_backend and current_backend in ['cpu', 'nvidia', 'amd', 'vulkan']:
                log_info(f"Reinstalling {current_backend.upper()} backend...")
                # Use force_rebuild=True to ensure clean reinstall
                if install_backend(current_backend, force_rebuild=True):
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
    
    # Check pywhispercpp (only for local backends)
    if not is_rest_api:
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
        
        # Check base model (only for local backends)
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

    if all_ok:
        log_success("Validation passed")
    else:
        log_error("Validation failed - some components are missing")

    return all_ok


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

