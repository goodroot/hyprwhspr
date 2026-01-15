"""
Output control and verbosity management for hyprwhspr
"""

import sys
import subprocess
from enum import Enum
from typing import Optional, Callable, TextIO
from pathlib import Path


class VerbosityLevel(Enum):
    """Verbosity levels for output control"""
    QUIET = 0      # Only errors
    NORMAL = 1     # Errors, warnings, success, info
    VERBOSE = 2    # All output + command details
    DEBUG = 3      # Everything + debug info


class OutputController:
    """Manages output verbosity and redirection"""
    
    _instance: Optional['OutputController'] = None
    _verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    _log_file: Optional[Path] = None
    _log_file_handle: Optional[TextIO] = None
    _progress_enabled: bool = True
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def set_verbosity(cls, level: VerbosityLevel):
        """Set the global verbosity level"""
        cls._verbosity = level
    
    @classmethod
    def get_verbosity(cls) -> VerbosityLevel:
        """Get the current verbosity level"""
        return cls._verbosity
    
    @classmethod
    def set_log_file(cls, log_file: Optional[Path]):
        """Set a log file to write all output to"""
        if cls._log_file_handle:
            cls._log_file_handle.close()
            cls._log_file_handle = None
        
        cls._log_file = log_file
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            cls._log_file_handle = open(log_file, 'a', encoding='utf-8')
    
    @classmethod
    def set_progress_enabled(cls, enabled: bool):
        """Enable or disable progress indicators"""
        cls._progress_enabled = enabled
    
    @classmethod
    def is_progress_enabled(cls) -> bool:
        """Check if progress indicators are enabled"""
        return cls._progress_enabled
    
    @classmethod
    def should_show(cls, level: VerbosityLevel) -> bool:
        """Check if output at this level should be shown"""
        return cls._verbosity.value >= level.value
    
    @classmethod
    def write(cls, message: str, level: VerbosityLevel = VerbosityLevel.NORMAL, 
              file: Optional[TextIO] = None, flush: bool = False):
        """Write a message respecting verbosity settings"""
        if not cls.should_show(level):
            return
        
        target = file or sys.stdout
        target.write(message)
        if flush:
            target.flush()
        
        # Also write to log file if set
        if cls._log_file_handle:
            cls._log_file_handle.write(message)
            if flush:
                cls._log_file_handle.flush()
    
    @classmethod
    def cleanup(cls):
        """Clean up resources"""
        if cls._log_file_handle:
            cls._log_file_handle.close()
            cls._log_file_handle = None


def log_info(msg: str):
    """Print info message respecting verbosity"""
    OutputController.write(f"[INFO] {msg}\n", VerbosityLevel.NORMAL)


def log_success(msg: str):
    """Print success message respecting verbosity"""
    OutputController.write(f"[SUCCESS] {msg}\n", VerbosityLevel.NORMAL)


def log_warning(msg: str):
    """Print warning message respecting verbosity"""
    OutputController.write(f"[WARNING] {msg}\n", VerbosityLevel.NORMAL)


def log_error(msg: str):
    """Print error message (always shown)"""
    OutputController.write(f"[ERROR] {msg}\n", VerbosityLevel.QUIET, file=sys.stderr)


def log_debug(msg: str):
    """Print debug message (only in debug mode)"""
    OutputController.write(f"[DEBUG] {msg}\n", VerbosityLevel.DEBUG)


def log_verbose(msg: str):
    """Print verbose message (only in verbose/debug mode)"""
    OutputController.write(f"[VERBOSE] {msg}\n", VerbosityLevel.VERBOSE)


def run_command(cmd: list, check: bool = True, capture_output: bool = False,
                env: Optional[dict] = None, verbose: Optional[bool] = None,
                show_output_on_error: bool = True, use_mise_free_env: bool = True,
                timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """
    Run a shell command with output control.

    Args:
        cmd: Command to run
        check: Raise exception on non-zero exit
        capture_output: Whether to capture output
        env: Environment variables (if None and use_mise_free_env=True, auto-applies mise-free env when mise is active)
        verbose: Override verbosity for this command (None = use global)
        show_output_on_error: Show captured output if command fails
        use_mise_free_env: If True and env is None, automatically use mise-free environment when mise is active
        timeout: Timeout in seconds (None = no timeout). Uses Python subprocess timeout, not external 'timeout' command.

    Returns:
        CompletedProcess result
    """
    controller = OutputController()
    verbosity = controller.get_verbosity()
    
    # Auto-apply mise-free environment if needed
    if use_mise_free_env and env is None:
        # Lazy import to avoid circular dependencies
        try:
            from .backend_installer import _check_mise_active, _create_mise_free_environment
        except ImportError:
            try:
                from backend_installer import _check_mise_active, _create_mise_free_environment
            except ImportError:
                # If import fails, just proceed without mise handling
                _check_mise_active = None
                _create_mise_free_environment = None
        
        if _check_mise_active and _create_mise_free_environment:
            if _check_mise_active():
                env = _create_mise_free_environment()
    
    # Determine if we should show output
    if verbose is None:
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
    
    # Track if capture_output was explicitly requested
    # If capture_output=True is explicitly set, always respect it (even in verbose mode)
    explicit_capture = capture_output
    
    # If not verbose and capture_output not explicitly set, always capture output (we'll show on error if needed)
    if not verbose and not explicit_capture:
        capture_output = True
    
    try:
        # If capture_output was explicitly requested, always use it regardless of verbosity
        if explicit_capture:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )
        elif verbose:
            # Show real-time output
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=False,
                text=True,
                env=env,
                timeout=timeout
            )
        else:
            # Capture output, show on error if requested
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )

            # Show output on error if requested
            if result.returncode != 0 and show_output_on_error:
                if result.stdout:
                    log_error(f"Command stdout:\n{result.stdout}")
                if result.stderr:
                    log_error(f"Command stderr:\n{result.stderr}")

        return result
    except subprocess.TimeoutExpired:
        log_error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {' '.join(cmd)}")
        if hasattr(e, 'stdout') and e.stdout and show_output_on_error:
            log_error(f"Command stdout:\n{e.stdout}")
        if hasattr(e, 'stderr') and e.stderr and show_output_on_error:
            log_error(f"Command stderr:\n{e.stderr}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {cmd[0]}")
        raise


def run_sudo_command(cmd: list, check: bool = True, input_data: Optional[bytes] = None,
                     verbose: Optional[bool] = None) -> subprocess.CompletedProcess:
    """Run a command with sudo"""
    sudo_cmd = ['sudo'] + cmd
    if input_data:
        # Handle input data if needed
        result = subprocess.run(
            sudo_cmd,
            check=check,
            input=input_data,
            text=False if input_data else True
        )
        return result
    return run_command(sudo_cmd, check=check, verbose=verbose, env=None)
