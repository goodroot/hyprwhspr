"""
Shared utility functions for detecting if hyprwhspr is already running.
Used by both main.py and cli_commands.py to broker between local/remote state.
"""

import os
import subprocess
from typing import Tuple, Optional


def is_running_under_systemd() -> bool:
    """
    Check if the current process is running under systemd.
    Returns True if we're in a systemd service context.
    """
    try:
        # Method 1: Check if we're a child of systemd
        ppid = os.getppid()
        try:
            with open(f'/proc/{ppid}/comm', 'r', encoding='utf-8') as f:
                parent_comm = f.read().strip()
                if 'systemd' in parent_comm:
                    return True
        except (FileNotFoundError, IOError):
            pass
        
        # Method 2: Check environment variable (systemd sets this)
        if os.environ.get('INVOCATION_ID') or os.environ.get('JOURNAL_STREAM'):
            return True
    except Exception:
        pass
    
    return False


def is_service_active_via_systemd(service_name: str = 'hyprwhspr.service') -> bool:
    """
    Check if a systemd service is active.
    
    Args:
        service_name: Name of the systemd service to check
        
    Returns:
        True if the service is active, False otherwise
    """
    try:
        result = subprocess.run(
            ['systemctl', '--user', 'is-active', service_name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def find_hyprwhspr_processes() -> list[int]:
    """
    Find all running hyprwhspr processes.
    
    Returns:
        List of process IDs (excluding the current process)
    """
    current_pid = os.getpid()
    pids = []
    
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'hyprwhspr.*main.py'],
            capture_output=True,
            timeout=2,
            check=False
        )
        
        if result.returncode == 0:
            found_pids = [int(pid) for pid in result.stdout.decode().strip().split('\n') if pid]
            pids = [pid for pid in found_pids if pid != current_pid]
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    
    return pids


def verify_process_is_valid(pid: int) -> bool:
    """
    Verify that a process ID is valid and actually a hyprwhspr process.
    
    Args:
        pid: Process ID to verify
        
    Returns:
        True if the process is valid and running, False otherwise
    """
    try:
        # Check if process exists
        os.kill(pid, 0)
        
        # Check process state - ignore zombies
        try:
            with open(f'/proc/{pid}/stat', 'r', encoding='utf-8') as f:
                stat_data = f.read().split()
                if len(stat_data) >= 3:
                    state = stat_data[2]  # Process state: R=running, S=sleeping, Z=zombie, etc.
                    if state == 'Z':  # Zombie
                        return False
        except (FileNotFoundError, IOError, IndexError):
            pass
        
        # Verify it's actually a hyprwhspr/python process
        try:
            with open(f'/proc/{pid}/comm', 'r', encoding='utf-8') as f:
                comm = f.read().strip()
                if 'hyprwhspr' in comm or 'python' in comm:
                    return True
        except (FileNotFoundError, IOError):
            # If we can't read comm, assume it's valid if it passed other checks
            return True
        
        return False
    except (ProcessLookupError, PermissionError):
        # Process doesn't exist or we can't access it
        return False


def is_hyprwhspr_running() -> Tuple[bool, Optional[str]]:
    """
    Comprehensive check if hyprwhspr is already running.
    
    Returns:
        Tuple of (is_running: bool, how_detected: str or None)
        If is_running is True, how_detected describes how it was detected.
    """
    # Check if we're running under systemd
    # If we are, systemd already manages single instances - skip the check entirely
    # This avoids race conditions during restarts where the old process might still be shutting down
    if is_running_under_systemd():
        return False, None
    
    # We're running manually - check if systemd service is active
    if is_service_active_via_systemd():
        # Systemd service is active - check if there's actually a process
        pids = find_hyprwhspr_processes()
        if pids:
            # Verify at least one process is valid
            valid_pids = [pid for pid in pids if verify_process_is_valid(pid)]
            if valid_pids:
                return True, f"systemd service (PIDs: {', '.join(map(str, valid_pids))})"
            # If no valid PIDs but service is active, trust systemd
            return True, "systemd service"
    
    # Check for other manual processes
    pids = find_hyprwhspr_processes()
    if pids:
        # Verify processes are actually running (not zombies)
        valid_pids = [pid for pid in pids if verify_process_is_valid(pid)]
        
        if valid_pids:
            return True, f"process (PIDs: {', '.join(map(str, valid_pids))})"
    
    return False, None


def is_running_manually() -> bool:
    """
    Check if hyprwhspr is running manually (not via systemd).
    Used by CLI commands to determine if a manual instance is running.
    
    Returns:
        True if running manually, False otherwise
    """
    # Check if there's a process but systemd service is not active
    pids = find_hyprwhspr_processes()
    if pids:
        # Verify at least one process is valid
        valid_pids = [pid for pid in pids if verify_process_is_valid(pid)]
        if valid_pids and not is_service_active_via_systemd():
            return True
    return False

