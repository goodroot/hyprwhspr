"""
CLI command implementations for hyprwhspr
"""

import os
import sys
import json
import subprocess
import getpass
from pathlib import Path
from typing import Optional

try:
    from rich.prompt import Prompt, Confirm
except ImportError:
    # Fallback if rich is not available (shouldn't happen in production)
    # This would cause errors, but rich is in requirements.txt
    Prompt = None
    Confirm = None

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT
    )
except ImportError:
    from backend_installer import (
        install_backend, VENV_DIR, STATE_FILE, STATE_DIR,
        get_install_state, set_install_state, get_all_state,
        init_state, _cleanup_partial_installation,
        PARAKEET_VENV_DIR, PARAKEET_SCRIPT
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
        save_credential, get_credential, mask_api_key
    )
except ImportError:
    from credential_manager import (
        save_credential, get_credential, mask_api_key
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
PARAKEET_SERVICE_NAME = 'parakeet-tdt-0.6b-v3.service'
YDOTOOL_UNIT = 'ydotool.service'
USER_HOME = Path.home()
USER_CONFIG_DIR = USER_HOME / '.config' / 'hyprwhspr'
USER_SYSTEMD_DIR = USER_HOME / '.config' / 'systemd' / 'user'
PYWHISPERCPP_MODELS_DIR = Path(os.environ.get('XDG_DATA_HOME', USER_HOME / '.local' / 'share')) / 'pywhispercpp' / 'models'


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
        'cpu', 'nvidia', 'amd', 'parakeet', 'rest-api', or None if not detected
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
        if backend in ['cpu', 'nvidia', 'amd', 'pywhispercpp']:
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
                        # Try to detect which variant by checking build artifacts or imports
                        # For now, trust config - could enhance later
                        return backend
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
        backend_type: 'cpu', 'nvidia', 'amd', 'parakeet', or 'remote'
    
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
        backend_names = {
            'cpu': 'CPU',
            'nvidia': 'NVIDIA (CUDA)',
            'amd': 'AMD (ROCm)',
            'parakeet': 'Parakeet',
            'rest-api': 'REST API',
            'pywhispercpp': 'pywhispercpp'
        }
        print(f"\nCurrent backend: {backend_names.get(current_backend, current_backend)}")
    else:
        print("\nNo backend currently configured.")
    
    print("\nChoose your transcription backend:")
    print()
    print("  [1] CPU - CPU-only, works on all systems")
    print("  [2] NVIDIA - NVIDIA GPU acceleration (CUDA)")
    print("  [3] AMD - AMD GPU acceleration (ROCm)")
    print("  [4] REST API - Use external API/backend (localhost or remote, requires network)")
    print("  [5] Parakeet - NVIDIA Parakeet model (local REST server)")
    print()
    
    while True:
        try:
            choice = Prompt.ask("Select backend", choices=['1', '2', '3', '4', '5'], default='1')
            backend_map = {
                '1': 'cpu',
                '2': 'nvidia',
                '3': 'amd',
                '4': 'rest-api',
                '5': 'parakeet'
            }
            selected = backend_map[choice]
            
            # Warn if switching to different backend
            if current_backend and current_backend != selected:
                backend_names = {
                    'cpu': 'CPU',
                    'nvidia': 'NVIDIA (CUDA)',
                    'amd': 'AMD (ROCm)',
                    'parakeet': 'Parakeet',
                    'rest-api': 'REST API',
                    'pywhispercpp': 'pywhispercpp'
                }
                print(f"\n⚠️  Switching from {backend_names.get(current_backend)} to {backend_names.get(selected)}")
                
                if selected == 'parakeet':
                    print("Parakeet uses a separate venv and runs as a local REST server.")
                    if current_backend not in ['rest-api', 'remote', 'parakeet']:
                        print("This will uninstall the current backend.")
                        if not Confirm.ask("Continue?", default=True):
                            continue
                elif current_backend not in ['rest-api', 'remote', 'parakeet'] and selected not in ['rest-api', 'remote', 'parakeet']:
                    print("This will uninstall the current backend and install the new one.")
                    if not Confirm.ask("Continue?", default=True):
                        continue
                elif selected == 'rest-api':
                    print("Switching to REST API backend.")
                    if current_backend == 'parakeet':
                        print("The Parakeet venv will no longer be needed.")
                        cleanup_venv = Confirm.ask("Remove the Parakeet venv to free up space?", default=False)
                    else:
                        print("The local backend venv will no longer be needed.")
                        cleanup_venv = Confirm.ask("Remove the venv to free up space?", default=False)
                    backend_names = {
                        'cpu': 'CPU',
                        'nvidia': 'NVIDIA (CUDA)',
                        'amd': 'AMD (ROCm)',
                        'parakeet': 'Parakeet',
                        'rest-api': 'REST API'
                    }
                    print(f"\n✓ Selected: {backend_names[selected]}")
                    return (selected, cleanup_venv)  # Return tuple: (backend, cleanup_venv)
            
            # If re-selecting same backend, offer reinstall option
            if current_backend == selected and selected not in ['rest-api', 'remote', 'parakeet']:
                backend_names = {
                    'cpu': 'CPU',
                    'nvidia': 'NVIDIA (CUDA)',
                    'amd': 'AMD (ROCm)'
                }
                print(f"\n{backend_names.get(selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False)
            elif current_backend == selected and selected == 'parakeet':
                backend_names = {
                    'parakeet': 'Parakeet'
                }
                print(f"\n{backend_names.get(selected)} backend is already installed.")
                reinstall = Confirm.ask("Reinstall backend?", default=False)
                if not reinstall:
                    print("Keeping existing installation.")
                    return (selected, False)
            
            backend_names = {
                'cpu': 'CPU',
                'nvidia': 'NVIDIA (CUDA)',
                'amd': 'AMD (ROCm)',
                'parakeet': 'Parakeet',
                'rest-api': 'REST API'
            }
            print(f"\n✓ Selected: {backend_names[selected]}")
            return (selected, False)  # Return tuple: (backend, cleanup_venv)
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


def _prompt_remote_provider_selection():
    """
    Prompt user for remote provider and model selection.
    
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
    providers_list = list_providers()
    provider_choices = []
    
    for i, (provider_id, provider_name, model_ids) in enumerate(providers_list, 1):
        model_list = ', '.join(model_ids)
        print(f"  [{i}] {provider_name} ({model_list})")
        provider_choices.append((str(i), provider_id))
    
    print(f"  [{len(providers_list) + 1}] Custom/Arbitrary Backend")
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
                for model_id, model_data in models.items():
                    model_list.append((model_id, model_data))
                    print(f"  [{len(model_list)}] {model_data['name']} - {model_data['description']}")
                
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


def _generate_remote_config(provider_id: str, model_id: Optional[str], api_key: str, custom_config: Optional[dict] = None) -> dict:
    """
    Generate REST API configuration based on provider/model selection.
    
    Args:
        provider_id: Provider identifier (e.g., 'openai', 'groq', 'custom')
        model_id: Model identifier (None for custom backends)
        api_key: API key to use
        custom_config: Custom config dict for custom backends
    
    Returns:
        Configuration dictionary ready to be saved
    """
    config = {
        'transcription_backend': 'rest-api'
    }
    
    if custom_config:
        # Custom backend
        config['rest_endpoint_url'] = custom_config['endpoint']
        if api_key:
            config['rest_api_key'] = api_key
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
        config['rest_api_key'] = api_key
        config['rest_body'] = model_config['body'].copy()
    
    return config


def setup_command():
    """Interactive full initial setup"""
    print("\n" + "="*60)
    print("hyprwhspr setup")
    print("="*60)
    print("\nThis setup will guide you through configuring hyprwhspr.")
    print("Skip any step by answering 'no'.\n")
    
    # Step 1: Backend selection (now returns tuple: (backend, cleanup_venv))
    backend_result = _prompt_backend_selection()
    if not backend_result:
        log_error("Backend selection is required. Exiting.")
        return
    
    # Handle tuple or string return (backward compatibility)
    if isinstance(backend_result, tuple):
        backend, cleanup_venv = backend_result
    else:
        backend = backend_result
        cleanup_venv = False
    
    current_backend = _detect_current_backend()
    
    # Handle backend switching
    if current_backend and current_backend != backend:
        if current_backend not in ['rest-api', 'remote']:
            # Switching from local to something else
            if not _cleanup_backend(current_backend):
                log_warning("Failed to clean up old backend, continuing anyway...")
        
        if cleanup_venv and backend in ['rest-api', 'remote']:
            # User wants to remove venv when switching to REST API
            if VENV_DIR.exists():
                log_info("Removing venv as requested...")
                import shutil
                shutil.rmtree(VENV_DIR)
                log_success("Venv removed")
    
    # Step 1.5: Backend installation (if not REST API)
    parakeet_installed = False
    if backend not in ['rest-api', 'remote']:
        print("\n" + "="*60)
        print("Backend Installation")
        print("="*60)
        if backend == 'parakeet':
            print(f"\nThis will install the Parakeet backend.")
            print("This will create a separate virtual environment and install dependencies.")
            print("The Parakeet server will need to be started manually after installation.")
        else:
            print(f"\nThis will install the {backend.upper()} backend for pywhispercpp.")
            print("This may take several minutes as it compiles from source.")
        if not Confirm.ask("Proceed with backend installation?", default=True):
            log_warning("Skipping backend installation. You can install it later.")
            log_warning("Backend installation is required for local transcription to work.")
        else:
            if not install_backend(backend):
                log_error("Backend installation failed. Setup cannot continue.")
                return
            
            if backend == 'parakeet':
                parakeet_installed = True
    
    # Step 2: Provider/model selection (if REST API backend or parakeet)
    remote_config = None
    selected_model = None
    if backend == 'parakeet' and parakeet_installed:
        # Auto-configure REST API for parakeet
        log_info("Auto-configuring REST API for Parakeet...")
        remote_config = {
            'transcription_backend': 'rest-api',
            'rest_endpoint_url': 'http://127.0.0.1:8080/transcribe',
            'rest_headers': {},
            'rest_body': {}
        }
        log_success("Parakeet REST API configuration ready")
        log_info("Note: Start the Parakeet server with:")
        log_info(f"  {PARAKEET_VENV_DIR / 'bin' / 'python'} {PARAKEET_SCRIPT}")
    elif backend in ['rest-api', 'remote']:
        # Prompt for remote provider selection
        provider_result = _prompt_remote_provider_selection()
        if not provider_result:
            log_error("Provider selection cancelled. Exiting.")
            return
        
        provider_id, model_id, api_key, custom_config = provider_result
        
        # Generate remote configuration
        try:
            remote_config = _generate_remote_config(provider_id, model_id, api_key, custom_config)
            log_success("Remote configuration generated")
        except Exception as e:
            log_error(f"Failed to generate remote configuration: {e}")
            return
    elif backend not in ['rest-api', 'remote']:
        # Local backend - prompt for model selection
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
    
    # Step 4: Systemd setup
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)
    print("\nSystemd user service will run hyprwhspr in the background.")
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
    if backend == 'parakeet' and remote_config:
        print(f"Endpoint: {remote_config.get('rest_endpoint_url', 'N/A')}")
        print("Model: parakeet-tdt-0.6b-v3")
    elif remote_config:
        print(f"Endpoint: {remote_config.get('rest_endpoint_url', 'N/A')}")
        if remote_config.get('rest_body'):
            model_name = remote_config['rest_body'].get('model', 'N/A')
            print(f"Model: {model_name}")
        api_key = remote_config.get('rest_api_key')
        if api_key:
            masked = mask_api_key(api_key)
            print(f"API Key: {masked}")
    elif selected_model:
        print(f"Model: {selected_model}")
    print(f"Waybar integration: {'Yes' if setup_waybar_choice else 'No'}")
    print(f"Systemd service: {'Yes' if setup_systemd_choice else 'No'}")
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
            setup_config(backend=backend, remote_config=remote_config)
        else:
            setup_config(backend=backend, model=selected_model)
        
        # Step 2: Waybar
        if setup_waybar_choice:
            setup_waybar('install')
        else:
            log_info("Skipping Waybar integration")
        
        # Step 3: Systemd
        if setup_systemd_choice:
            setup_systemd('install')
        else:
            log_info("Skipping systemd setup")
        
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
        if setup_systemd_choice:
            print("  1. Log out and back in (for group permissions to take effect)")
        else:
            print("  1. Log out and back in (if permissions were set up)")
            print("  2. Run hyprwhspr manually or set up systemd service later")
        
        # If backend was changed, suggest restarting service
        if current_backend and current_backend != backend:
            print("  3. Press hotkey to start dictation!")
        else:
            print("  3. Press hotkey to start dictation!")
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
            
            # Update model if provided, otherwise default to base.en if missing
            if model:
                existing_config['model'] = model
            elif 'model' not in existing_config and not remote_config:
                # Only set default model if not using remote backend
                existing_config['model'] = 'base.en'
            
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
        run_command(['systemctl', '--user', 'enable', '--now', SERVICE_NAME], check=False)
        log_success("Systemd user services enabled and started")
    elif mode == 'disable':
        if is_parakeet:
            run_command(['systemctl', '--user', 'disable', '--now', PARAKEET_SERVICE_NAME], check=False)
        run_command(['systemctl', '--user', 'disable', '--now', SERVICE_NAME], check=False)
        log_success("Systemd user service disabled")
    
    return True


def systemd_status():
    """Show systemd service status"""
    # Check if Parakeet backend is configured
    current_backend = _detect_current_backend()
    is_parakeet = current_backend == 'parakeet'
    
    try:
        if is_parakeet:
            # Show status for both services
            log_info("Parakeet service status:")
            run_command(['systemctl', '--user', 'status', PARAKEET_SERVICE_NAME], check=False)
            print()  # Add spacing
            log_info("hyprwhspr service status:")
            run_command(['systemctl', '--user', 'status', SERVICE_NAME], check=False)
        else:
            # Show status for hyprwhspr only
            run_command(['systemctl', '--user', 'status', SERVICE_NAME], check=False)
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to get status: {e}")


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
                "on-click": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh toggle",
                "on-click-right": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh start",
                "on-click-middle": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh restart",
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
                config['modules-right'].insert(0, 'custom/hyprwhspr')
            
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


# ==================== Model Commands ====================

def model_command(action: str, model_name: str = 'base.en'):
    """Handle model subcommands"""
    if action == 'download':
        download_model(model_name)
    elif action == 'list':
        list_models()
    elif action == 'status':
        model_status()
    else:
        log_error(f"Unknown model action: {action}")


def download_model(model_name: str = 'base.en'):
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
            if current_backend and current_backend in ['cpu', 'nvidia', 'amd']:
                log_info(f"Reinstalling {current_backend.upper()} backend...")
                if install_backend(current_backend):
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
    try:
        import sounddevice  # noqa: F401
        log_success("✓ sounddevice available")
    except ImportError:
        log_error("✗ sounddevice not available")
        all_ok = False
    
    # Check pywhispercpp (optional)
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
    
    # Check base model
    model_file = PYWHISPERCPP_MODELS_DIR / 'ggml-base.en.bin'
    if model_file.exists():
        log_success(f"✓ Base model exists: {model_file}")
    else:
        log_warning(f"⚠ Base model missing: {model_file}")
    
    if all_ok:
        log_success("Validation passed")
    else:
        log_error("Validation failed - some components are missing")
    
    return all_ok

