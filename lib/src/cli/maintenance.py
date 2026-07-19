"""
Maintenance commands for hyprwhspr — backend repair/reset, install-state
management and installation validation
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rich.prompt import Prompt, Confirm

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..backend_utils import (BACKEND_IMPORT_MODULES, LOCAL_INSTALL_BACKENDS,
                                 normalize_backend)
except ImportError:
    from backend_utils import (BACKEND_IMPORT_MODULES, LOCAL_INSTALL_BACKENDS,
                               normalize_backend)

try:
    from ..backend_installer import (
        install_backend, VENV_DIR, PYWHISPERCPP_SRC_DIR, PYWHISPERCPP_MODELS_DIR,
        get_install_state, set_install_state, get_all_state, get_state, init_state,
        resolve_dependency_plan, execute_dependency_plan,
    )
except ImportError:
    from backend_installer import (
        install_backend, VENV_DIR, PYWHISPERCPP_SRC_DIR, PYWHISPERCPP_MODELS_DIR,
        get_install_state, set_install_state, get_all_state, get_state, init_state,
        resolve_dependency_plan, execute_dependency_plan,
    )

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

try:
    from ..session_environment import classify_display_environment
except ImportError:
    from session_environment import classify_display_environment

from ._shared import (HYPRWHSPR_ROOT, SERVICE_NAME, _check_ydotool_version,
                      _is_niri_session, _validate_hyprwhspr_root)


# ==================== Backend Management Commands ====================

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
    required_imports = ()
    configured_backend = None
    config_manager = None

    # Get the configured backend to know which module to check
    try:
        config_manager = ConfigManager()
        configured_backend = config_manager.get_setting('transcription_backend', 'pywhispercpp')
        configured_backend = normalize_backend(configured_backend)
        provider_key = (
            'websocket_provider' if configured_backend == 'realtime-ws'
            else 'rest_api_provider'
        )
        provider = config_manager.get_setting(provider_key, None)
        plan = resolve_dependency_plan(configured_backend, provider)
        required_imports = plan.required_imports
    except Exception:
        pass

    # The resolved plan is authoritative for local and cloud import checks.
    if required_imports:
        backend_module = ', '.join(required_imports)
    else:
        backend_module = BACKEND_IMPORT_MODULES.get(configured_backend)

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
            if not _reinstall_configured_dependencies(config_manager, configured_backend):
                log_error("Venv repair failed")
                return False
            log_success("Venv recreated with configured dependencies")
            venv_corrupted = False
            backend_missing = False
    
    if backend_missing:
        print("\nIssues found:")
        print(f"  • {backend_module} is not installed (required for {configured_backend} backend)")
        print("\nOptions:")
        print("  [1] Reinstall backend")
        print("  [2] Skip (manual repair required)")

        choice = Prompt.ask("Select option", choices=['1', '2'], default='1')
        if choice == '1':
            log_info(f"Reinstalling {configured_backend.upper()} backend...")
            if not _reinstall_configured_dependencies(config_manager, configured_backend):
                log_error("Backend reinstallation failed")
                return False
            log_success("Backend reinstalled successfully")
    
    log_success("Repair completed")
    return True


def _reinstall_configured_dependencies(config_manager, configured_backend: str) -> bool:
    """Repair the configured dependency family through its normal transaction."""
    if configured_backend == 'pywhispercpp':
        installed = get_state('installed_backend')
        configured_backend = installed if installed in LOCAL_INSTALL_BACKENDS else 'cpu'

    if configured_backend in LOCAL_INSTALL_BACKENDS:
        return install_backend(configured_backend, force_rebuild=True)

    if configured_backend in ('rest-api', 'realtime-ws'):
        provider_key = (
            'websocket_provider' if configured_backend == 'realtime-ws'
            else 'rest_api_provider'
        )
        provider = config_manager.get_setting(provider_key, None)
        try:
            plan = resolve_dependency_plan(configured_backend, provider)
            execute_dependency_plan(plan, force_rebuild=True)
            return True
        except Exception as exc:
            log_error(f"Dependency repair failed: {exc}")
            return False

    log_warning("Could not detect backend type. Please run 'hyprwhspr setup'")
    return False


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
    is_rest_api = current_backend in ['rest-api', 'remote', 'realtime-ws']
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
    # (ConfigManager comes from the module-level import; a nested relative
    # re-import here used to fail silently and skip this whole check)
    try:
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

    # Check graphical session environment in the systemd user environment.
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'show-environment'],
            capture_output=True, text=True, timeout=5, check=False
        )
        env_output = result.stdout if result.returncode == 0 else ''
        display_kind = classify_display_environment(
            env_output, os.environ.get('XDG_SESSION_TYPE', '')
        )
        if display_kind == 'wayland':
            log_success("✓ Wayland display set in systemd user environment")
        elif display_kind == 'x11':
            log_success("✓ X11 DISPLAY set in systemd user environment")
        else:
            log_warning("⚠ No Wayland or X11 display found in systemd user environment")
            if os.environ.get('XDG_SESSION_TYPE', '').lower() == 'x11':
                print("  Import the X11 session environment:")
                print("    systemctl --user import-environment DISPLAY XAUTHORITY XDG_SESSION_TYPE")
            else:
                print("  Add the relevant compositor environment export to your startup config.")
                print("  Hyprland example:")
                print("    exec-once = dbus-update-activation-environment --systemd WAYLAND_DISPLAY XDG_CURRENT_DESKTOP HYPRLAND_INSTANCE_SIGNATURE")

        if _is_niri_session():
            if 'NIRI_SOCKET=' in env_output:
                log_success("✓ NIRI_SOCKET set in systemd user environment")
            else:
                log_warning("⚠ NIRI_SOCKET not found in systemd user environment")
                print("  Niri window detection needs NIRI_SOCKET in the systemd user environment.")
                print("  Add to your Niri startup config:")
                print("    spawn-at-startup \"dbus-update-activation-environment\" \"--systemd\" \"WAYLAND_DISPLAY\" \"XDG_CURRENT_DESKTOP\" \"NIRI_SOCKET\"")
    except Exception:
        pass

    if all_ok:
        log_success("Validation passed")
    else:
        log_error("Validation failed - some components are missing")

    return all_ok
