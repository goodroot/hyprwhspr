"""
Systemd service management commands for hyprwhspr
"""

import os
import subprocess
from pathlib import Path

try:
    from ..output_control import (log_info, log_success, log_warning, log_error,
                                  log_debug, run_command)
except ImportError:
    from output_control import (log_info, log_success, log_warning, log_error,
                                log_debug, run_command)

from ._shared import (HYPRWHSPR_ROOT, SERVICE_NAME, RESUME_SERVICE_NAME,
                      YDOTOOL_UNIT, _YDOTOOL_MARKERS, USER_SYSTEMD_DIR,
                      _validate_hyprwhspr_root)


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


def _migrate_remove_managed_ydotool_unit():
    """Retire a user-scope ydotool.service that older hyprwhspr versions deployed.

    hyprwhspr now runs a *private* ydotoold child from the app itself
    (lib/src/ydotoold_session.py) instead of managing a systemd unit. Stop, disable
    and remove a unit we authored (one carrying a hyprwhspr marker); never touch a
    distro/admin/user-owned unit at that path.
    """
    target = USER_SYSTEMD_DIR / YDOTOOL_UNIT
    if not _is_hyprwhspr_managed_ydotool_unit(target):
        if target.exists():
            log_debug(f"Leaving foreign {YDOTOOL_UNIT} untouched at {target}")
        return
    run_command(['systemctl', '--user', 'disable', '--now', YDOTOOL_UNIT], check=False)
    try:
        target.unlink(missing_ok=True)
        (target.parent / (target.name + '.bak')).unlink(missing_ok=True)
        log_success(f"Retired previously-managed {YDOTOOL_UNIT} (hyprwhspr now runs a private ydotoold)")
        log_warning("hyprwhspr no longer deploys ydotoold as a service — if you use ydotool in your own keybinds, start ydotoold separately")
    except OSError as e:
        log_warning(f"Could not remove {YDOTOOL_UNIT}: {e}")
    run_command(['systemctl', '--user', 'daemon-reload'], check=False)


def _is_hyprwhspr_managed_ydotool_unit(target: Path) -> bool:
    """True only for ydotool.service files authored by older hyprwhspr setup."""
    if not target.exists():
        return False
    try:
        text = target.read_text(encoding='utf-8')
    except OSError:
        return False
    return any(marker in text for marker in _YDOTOOL_MARKERS)


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

    # hyprwhspr no longer deploys/manages a ydotool.service. The paste fallback now
    # runs a *private* ydotoold child from the app (lib/src/ydotoold_session.py) on
    # its own socket — no shared daemon, no managed unit. Retire any unit a previous
    # version of hyprwhspr deployed.
    _migrate_remove_managed_ydotool_unit()

    # Import the compositor environment visible to this setup process into the
    # systemd user manager. Niri's focused-window IPC needs NIRI_SOCKET; Hyprland
    # detection needs HYPRLAND_INSTANCE_SIGNATURE; wtype/wl-clipboard need the
    # Wayland display environment.
    run_command([
        'systemctl', '--user', 'import-environment',
        'WAYLAND_DISPLAY', 'XDG_CURRENT_DESKTOP',
        'HYPRLAND_INSTANCE_SIGNATURE', 'NIRI_SOCKET',
    ], check=False)

    # Reload systemd daemon
    run_command(['systemctl', '--user', 'daemon-reload'], check=False)
    
    if mode in ('install', 'enable'):
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


def _show_systemd_unit_status(unit_name: str):
    """Stream systemctl status output directly to terminal without capturing."""
    run_command(
        ['systemctl', '--user', 'status', unit_name],
        check=False,
        verbose=True,
        show_output_on_error=False,
    )


def systemd_status():
    """Show systemd service status"""
    try:
        log_info("hyprwhspr service status:")
        _show_systemd_unit_status(SERVICE_NAME)
        print()  # Add spacing

        # Show suspend/resume service status if it exists
        if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
            log_info("Suspend/resume handler status:")
            _show_systemd_unit_status(RESUME_SERVICE_NAME)
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
