"""
Overall status command for hyprwhspr
"""

import os
import json
import getpass
import subprocess
from pathlib import Path

try:
    from .. import paths
except ImportError:
    import paths

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend

try:
    from ..output_control import (log_info, log_success, log_warning, log_error,
                                  run_command)
except ImportError:
    from output_control import (log_info, log_success, log_warning, log_error,
                                run_command)

from ._shared import SERVICE_NAME
from .models import (model_status, onnx_asr_model_status,
                     faster_whisper_model_status, cohere_transcribe_model_status)
from .waybar import waybar_status


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
    
    # Check models (backend-aware: pywhispercpp vs faster-whisper vs Parakeet/onnx-asr)
    print("\n[Models]")
    try:
        config = ConfigManager()
        backend = normalize_backend(config.get_setting('transcription_backend', 'pywhispercpp'))
        if backend == 'faster-whisper':
            faster_whisper_model_status()
        elif backend == 'onnx-asr':
            onnx_asr_model_status()
        elif backend == 'cohere-transcribe':
            cohere_transcribe_model_status()
        else:
            model_status()
    except Exception:
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
