"""
Shared constants and environment helpers for the hyprwhspr CLI
"""

import os
import sys
import json
import subprocess
import shutil
import re
from pathlib import Path

try:
    from ..paths import CONFIG_DIR
except ImportError:
    from paths import CONFIG_DIR

try:
    from ..backend_installer import MAX_COMPATIBLE_PYTHON, _get_python_version
except ImportError:
    from backend_installer import MAX_COMPATIBLE_PYTHON, _get_python_version

try:
    from ..output_control import log_error
except ImportError:
    from output_control import log_error


# Constants
HYPRWHSPR_ROOT = os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr')
SERVICE_NAME = 'hyprwhspr.service'
RESUME_SERVICE_NAME = 'hyprwhspr-resume.service'  # Deprecated, kept for cleanup in uninstall/status
YDOTOOL_UNIT = 'ydotool.service'
# Older hyprwhspr versions deployed a user-scope ydotool.service carrying one of
# these markers. hyprwhspr now runs a private ydotoold child instead, so the markers
# are used only to identify (and safely retire) a unit we authored — never a
# distro/admin/user-owned one.
_YDOTOOL_MARKERS = ('Managed by hyprwhspr', 'deployed by hyprwhspr')
USER_HOME = Path.home()
USER_CONFIG_DIR = CONFIG_DIR  # Use centralized path constant
USER_SYSTEMD_DIR = USER_HOME / '.config' / 'systemd' / 'user'


def _is_niri_session() -> bool:
    """Return true when the current process appears to be running inside Niri."""
    if os.environ.get('NIRI_SOCKET'):
        return True

    desktop_values = (
        os.environ.get('XDG_CURRENT_DESKTOP', ''),
        os.environ.get('XDG_SESSION_DESKTOP', ''),
        os.environ.get('DESKTOP_SESSION', ''),
    )
    for desktop_value in desktop_values:
        desktop_names = desktop_value.replace(';', ':').split(':')
        if 'niri' in [name.strip().lower() for name in desktop_names]:
            return True

    return False


def _is_gnome_or_mutter_session() -> bool:
    """Return true for GNOME/Mutter/Pop sessions using common desktop env vars."""
    desktop_values = (
        os.environ.get('XDG_CURRENT_DESKTOP', ''),
        os.environ.get('XDG_SESSION_DESKTOP', ''),
        os.environ.get('DESKTOP_SESSION', ''),
    )
    desktop = ':'.join(desktop_values).lower()
    desktop_tokens = set(filter(None, re.split(r'[^a-z0-9]+', desktop)))
    return bool(desktop_tokens & {'gnome', 'mutter', 'pop'})


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

    # Try rpm (openSUSE/Fedora/RHEL)
    if not version:
        try:
            result = subprocess.run(
                ['rpm', '-q', 'ydotool'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Output format: "ydotool-1.0.4-2.2.x86_64"
            match = re.search(r'ydotool-(\d+\.\d+\.?\d*)', result.stdout)
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
