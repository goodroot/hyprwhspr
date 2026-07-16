"""
Install commands for hyprwhspr — omarchy automated install and its helpers
"""

import os
import shutil
import subprocess
from pathlib import Path

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..paths import CONFIG_FILE
except ImportError:
    from paths import CONFIG_FILE

try:
    from ..backend_utils import BACKEND_IMPORT_MODULES, LOCAL_INSTALL_BACKENDS
except ImportError:
    from backend_utils import BACKEND_IMPORT_MODULES, LOCAL_INSTALL_BACKENDS

try:
    from ..backend_installer import VENV_DIR, PYWHISPERCPP_MODELS_DIR
except ImportError:
    from backend_installer import VENV_DIR, PYWHISPERCPP_MODELS_DIR

try:
    from ..output_control import (
        log_info, log_success, log_warning, log_error, run_command
    )
except ImportError:
    from output_control import (
        log_info, log_success, log_warning, log_error, run_command
    )

from ._shared import (HYPRWHSPR_ROOT, USER_HOME, _check_mise_active,
                      _create_mise_free_environment)
from .maintenance import validate_command
from .noctalia import _noctalia_detected, noctalia_command
from .systemd import (systemd_command, systemd_restart,
                      _is_service_running_via_systemd)
from .waybar import waybar_command



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
