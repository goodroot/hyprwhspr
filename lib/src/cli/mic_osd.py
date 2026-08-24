"""
Mic-OSD (microphone visualization overlay) commands for hyprwhspr
"""

import sys
import subprocess
from pathlib import Path

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..backend_installer import VENV_DIR
except ImportError:
    from backend_installer import VENV_DIR

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error


# ==================== Mic-OSD Commands ====================

def _query_mic_osd_availability():
    """Return availability, reason, and runtime source for the service Python."""
    # First, try with venv Python (same as service uses)
    venv_python = VENV_DIR / 'bin' / 'python'
    if venv_python.exists():
        try:
            # mic_osd package lives at lib/, three levels up from lib/src/cli/
            lib_path = Path(__file__).parent.parent.parent
            # Use repr() to safely escape the path (handles quotes, backslashes, etc.)
            lib_path_str = repr(str(lib_path))
            check_code = f"""
import sys
sys.path.insert(0, {lib_path_str})
from mic_osd import MicOSDRunner
source = MicOSDRunner.runtime_source()
if source != 'unavailable':
    print('AVAILABLE:', source)
else:
    print('UNAVAILABLE:', MicOSDRunner.get_unavailable_reason())
"""
            result = subprocess.run(
                [str(venv_python), '-c', check_code],
                check=False,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if output == 'AVAILABLE':
                    return True, "", "system"
                if output.startswith('AVAILABLE:'):
                    return True, "", output.replace('AVAILABLE:', '').strip()
                elif output.startswith('UNAVAILABLE:'):
                    return False, output.replace('UNAVAILABLE:', '').strip(), "unavailable"
        except Exception as e:
            # Fall through to current Python check
            pass
    
    # Fallback: check with current Python environment
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from mic_osd import MicOSDRunner
        
        if MicOSDRunner.is_available():
            return True, "", MicOSDRunner.runtime_source()
        else:
            return False, MicOSDRunner.get_unavailable_reason(), "unavailable"
    except ImportError:
        return False, "mic-osd module not found", "unavailable"


def _check_mic_osd_availability():
    """Check mic-osd availability using the same Python the service will use."""
    available, reason, _source = _query_mic_osd_availability()
    return available, reason


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
    deps_available, deps_reason, runtime_source = _query_mic_osd_availability()
    
    print("\nMic-OSD Status:")
    print(f"  Enabled in config: {'Yes' if enabled else 'No'}")
    print(f"  Dependencies available: {'Yes' if deps_available else 'No'}")
    print(f"  Layer-shell runtime: {runtime_source}")
    
    if deps_available:
        if enabled:
            log_success("Mic-OSD will show during recording")
        else:
            log_info("Mic-OSD is disabled (use 'hyprwhspr mic-osd enable' to enable)")
    else:
        log_warning(f"Mic-OSD cannot run: {deps_reason}")

    return enabled and deps_available
