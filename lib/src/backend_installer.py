"""
Backend installation module for hyprwhspr
Handles installation of pywhispercpp backends (CPU/NVIDIA/AMD)
"""

import os
import sys
import json
import subprocess
import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import output control system
try:
    from .output_control import (
        log_info, log_success, log_warning, log_error, log_debug, log_verbose,
        run_command, OutputController, VerbosityLevel
    )
except ImportError:
    from output_control import (
        log_info, log_success, log_warning, log_error, log_debug, log_verbose,
        run_command, OutputController, VerbosityLevel
    )

# Import prompt for user interaction
try:
    from rich.prompt import Confirm
except ImportError:
    # Fallback if rich is not available (shouldn't happen in normal usage)
    Confirm = None


def run_sudo_command(cmd: list, check: bool = True, input_data: Optional[bytes] = None,
                     verbose: Optional[bool] = None) -> subprocess.CompletedProcess:
    """Run a command with sudo"""
    sudo_cmd = ['sudo'] + cmd
    return run_command(sudo_cmd, check=check, verbose=verbose, env=None)


# Constants
HYPRWHSPR_ROOT = os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr')

# Runtime HYPRWHSPR_ROOT auto-correction for mise compatibility
# If running under mise Python but AUR installation exists, automatically use it
if '.local/share/mise' in sys.executable:
    aur_install_path = Path('/usr/lib/hyprwhspr')
    if aur_install_path.exists():
        # Verify it's a valid installation
        if (aur_install_path / 'bin' / 'hyprwhspr').exists() and (aur_install_path / 'lib' / 'main.py').exists():
            # Only override if HYPRWHSPR_ROOT wasn't explicitly set to a different value
            # (i.e., it's not in environment, or it's already set to the AUR path)
            current_root = os.environ.get('HYPRWHSPR_ROOT')
            if current_root is None or current_root == '/usr/lib/hyprwhspr':
                os.environ['HYPRWHSPR_ROOT'] = '/usr/lib/hyprwhspr'
                HYPRWHSPR_ROOT = '/usr/lib/hyprwhspr'

USER_BASE = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'hyprwhspr'
VENV_DIR = USER_BASE / 'venv'
PYWHISPERCPP_MODELS_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'pywhispercpp' / 'models'
STATE_DIR = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state')) / 'hyprwhspr'
STATE_FILE = STATE_DIR / 'install-state.json'
PYWHISPERCPP_SRC_DIR = USER_BASE / 'pywhispercpp-src'
PYWHISPERCPP_PINNED_COMMIT = "d8f202f4435cf3e5e0bf735723eaacbe84c6853c"
PARAKEET_VENV_DIR = USER_BASE / 'parakeet-venv'
PARAKEET_DIR = Path(HYPRWHSPR_ROOT) / 'lib' / 'backends' / 'parakeet'
PARAKEET_SCRIPT = PARAKEET_DIR / 'parakeet-tdt-0.6b-v3.py'
PARAKEET_REQUIREMENTS = PARAKEET_DIR / 'requirements.txt'

# Pre-built wheel configuration
WHEEL_BASE_URL = "https://github.com/goodroot/hyprwhspr/releases/download/wheels-v1"
WHEEL_CACHE_DIR = USER_BASE / 'wheel-cache'
PYWHISPERCPP_VERSION = "1.4.1"


def _safe_decode(output) -> str:
    """Safely decode output from run_command which may be string or bytes."""
    if isinstance(output, bytes):
        return output.decode('utf-8', errors='ignore')
    return output


# Maximum Python version compatible with ML packages (onnxruntime, etc.)
MAX_COMPATIBLE_PYTHON = (3, 14)


def _get_python_version(python_path: str) -> Optional[Tuple[int, int]]:
    """
    Get Python major.minor version from executable.

    Args:
        python_path: Path to Python executable

    Returns:
        Tuple of (major, minor) version, or None if detection failed
    """
    import re
    try:
        result = run_command(
            [python_path, '--version'],
            check=False,
            capture_output=True,
            verbose=False
        )
        if result.returncode == 0:
            # `python --version` output is not consistent across versions/builds:
            # some print to stderr, others to stdout. Prefer stderr, fall back to stdout.
            candidates = [
                getattr(result, 'stderr', None),
                getattr(result, 'stdout', None),
            ]
            for stream in candidates:
                if not stream:
                    continue
                output = _safe_decode(stream).strip()
                match = re.search(r'Python\s+(\d+)\.(\d+)', output)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass
    return None


def _python_compatibility_error(current_version: Optional[Tuple[int, int]]) -> None:
    """
    Print error message and exit when no compatible Python is found.

    Args:
        current_version: The detected system Python version, or None
    """
    version_str = f"{current_version[0]}.{current_version[1]}" if current_version else "unknown"
    max_str = f"{MAX_COMPATIBLE_PYTHON[0]}.{MAX_COMPATIBLE_PYTHON[1]}"

    log_error(f"System Python {version_str} is not compatible with ML packages (onnxruntime, etc.)")
    print(f"\nhyprwhspr requires Python {max_str} or earlier. No compatible Python found.", flush=True)
    print("\nInstall Python 3.14 or 3.13:", flush=True)
    print("", flush=True)
    print("  Fedora:     sudo dnf install python3.14", flush=True)
    print("  Arch:       yay -S python314  # or python313", flush=True)
    print("  Ubuntu/Deb: sudo apt install python3.13", flush=True)
    print("", flush=True)
    print("Then re-run: hyprwhspr setup", flush=True)
    print("", flush=True)
    print("Or specify Python explicitly:", flush=True)
    print("  hyprwhspr setup --python /path/to/python3.14", flush=True)
    print("", flush=True)
    print("Alternative: Use cloud transcription (no local Python requirement):", flush=True)
    print("  hyprwhspr setup  # Select 'REST API'", flush=True)
    sys.exit(1)


def _find_compatible_python(max_version: Tuple[int, int] = MAX_COMPATIBLE_PYTHON) -> Tuple[str, str]:
    """
    Find a compatible Python for venv creation.

    Fallback chain:
    1. System python3 if version <= max_version
    2. python3.14, python3.13, python3.12, python3.11 in /usr/bin, /usr/local/bin
    3. Error with actionable message

    Args:
        max_version: Maximum allowed Python version as (major, minor) tuple

    Returns:
        Tuple of (python_path, description)

    Raises:
        SystemExit if no compatible Python found
    """
    # Check if current Python is mise-managed
    current_is_mise = '.local/share/mise' in sys.executable
    mise_active = _check_mise_active()

    # 1. Check current/system Python
    if mise_active or current_is_mise:
        # Use system Python when MISE is active
        current_python = _get_system_python()
    else:
        current_python = sys.executable

    current_version = _get_python_version(current_python)
    if current_version and current_version <= max_version:
        return (current_python, f"Python {current_version[0]}.{current_version[1]}")

    # 2. Search for python3.14, python3.13, python3.12, python3.11 in common paths
    for minor in [14, 13, 12, 11]:
        # Skip versions that are too new
        if (3, minor) > max_version:
            continue

        for prefix in ['/usr/bin', '/usr/local/bin']:
            path = f"{prefix}/python3.{minor}"
            if os.path.isfile(path) and os.access(path, os.X_OK):
                # Verify it actually works
                test_version = _get_python_version(path)
                if test_version and test_version <= max_version:
                    return (path, f"python3.{minor}")

    # 3. Error with guidance
    _python_compatibility_error(current_version)
    # _python_compatibility_error calls sys.exit, but for type checker:
    raise SystemExit(1)


def _check_mise_active() -> bool:
    """
    Check if MISE (runtime version manager) is active in the current environment.

    Returns:
        True if MISE is active, False otherwise
    """
    # Check for MISE environment variables
    if os.environ.get('MISE_SHELL') or os.environ.get('__MISE_ACTIVATE'):
        return True

    # Check if Python is being managed by MISE
    python_path = shutil.which('python3') or shutil.which('python')
    if python_path and '.local/share/mise' in python_path:
        return True

    # Check if mise binary is managing this session
    if shutil.which('mise') and os.environ.get('MISE_DATA_DIR'):
        return True

    return False


def _create_mise_free_environment() -> dict:
    """
    Create environment with MISE deactivated for subprocesses.

    This prevents MISE from interfering with Python version detection
    during pip install operations and venv creation.

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
        
        # If all paths were filtered out, fall back to essential system paths
        # This prevents empty PATH which would break subprocess execution
        if not paths:
            essential_paths = ['/usr/bin', '/usr/local/bin', '/bin', '/usr/sbin', '/sbin']
            paths = [p for p in essential_paths if os.path.exists(p)]
            # If even essential paths don't exist (unlikely), at least set a minimal PATH
            if not paths:
                paths = ['/usr/bin', '/bin']
        
        env['PATH'] = ':'.join(paths)

    return env


def _get_system_python() -> str:
    """
    Get the system Python path, avoiding mise-managed Python.

    When mise is active, this function uses a mise-free environment
    to find the actual system Python (typically /usr/bin/python3).

    Returns:
        Path to system Python executable, or sys.executable as fallback
    """
    # Common system Python paths (Arch Linux)
    system_paths = ['/usr/bin/python3', '/usr/bin/python']
    
    # Check if current Python is mise-managed
    current_is_mise = '.local/share/mise' in sys.executable
    
    # If mise is active, use mise-free environment to find system Python
    if _check_mise_active() or current_is_mise:
        mise_free_env = _create_mise_free_environment()
        
        # Try system paths first
        for python_path in system_paths:
            if os.path.exists(python_path) and os.access(python_path, os.X_OK):
                # Verify it's actually Python
                try:
                    result = run_command(
                        [python_path, '--version'],
                        check=False,
                        capture_output=True,
                        env=mise_free_env
                    )
                    if result.returncode == 0:
                        return python_path
                except Exception:
                    continue
        
        # Fallback: check additional common system paths directly
        # (more robust than relying on 'which' command which may not be available)
        additional_paths = [
            '/usr/local/bin/python3',
            '/usr/local/bin/python',
            '/bin/python3',
            '/bin/python'
        ]
        for python_path in additional_paths:
            if os.path.exists(python_path) and os.access(python_path, os.X_OK):
                try:
                    result = run_command(
                        [python_path, '--version'],
                        check=False,
                        capture_output=True,
                        env=mise_free_env
                    )
                    if result.returncode == 0:
                        return python_path
                except Exception:
                    continue
        
        # Last resort: try 'which' command if available
        try:
            result = run_command(
                ['which', 'python3'],
                check=False,
                capture_output=True,
                env=mise_free_env
            )
            if result.returncode == 0 and result.stdout:
                found_path = result.stdout.strip()
                if found_path and os.path.exists(found_path) and '.local/share/mise' not in found_path:
                    return found_path
        except Exception:
            pass
        
        # If we couldn't find system Python but mise is active, warn about fallback
        if (_check_mise_active() or current_is_mise) and '.local/share/mise' in sys.executable:
            log_warning("Could not find system Python - falling back to current Python (may be mise-managed)")
    
    # Default: return current executable (may be mise-managed, but better than nothing)
    return sys.executable


# ==================== Pre-built Wheel Support ====================

def _detect_venv_python_version() -> str:
    """Detect Python version in the venv (e.g., '3.11')"""
    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        # Fallback to system Python version
        import re
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return version

    try:
        result = run_command(
            [str(venv_python), '--version'],
            check=False,
            capture_output=True
        )
        output = _safe_decode(result.stdout)
        # Parse "Python 3.11.5" -> "3.11"
        import re
        match = re.search(r'(\d+)\.(\d+)', output)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    except Exception:
        pass

    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _detect_cuda_version() -> Optional[str]:
    """Detect installed CUDA version from nvcc or nvidia-smi"""
    import re

    # Try nvcc first (more reliable for build compatibility)
    nvcc_path = shutil.which('nvcc')
    if not nvcc_path:
        nvcc_path = '/opt/cuda/bin/nvcc' if Path('/opt/cuda/bin/nvcc').exists() else None

    if nvcc_path:
        try:
            result = run_command([nvcc_path, '--version'], check=False, capture_output=True)
            output = _safe_decode(result.stdout)
            # Parse "release 12.2, V12.2.140" -> "12.2"
            match = re.search(r'release (\d+)\.(\d+)', output)
            if match:
                return f"{match.group(1)}.{match.group(2)}"
        except Exception:
            pass

    # Fallback to nvidia-smi
    if shutil.which('nvidia-smi'):
        try:
            result = run_command(['nvidia-smi'], check=False, capture_output=True)
            output = _safe_decode(result.stdout)
            # Parse "CUDA Version: 12.2" -> "12.2"
            match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', output)
            if match:
                return f"{match.group(1)}.{match.group(2)}"
        except Exception:
            pass

    return None


def _get_wheel_variant(cuda_version: Optional[str]) -> Optional[str]:
    """Get wheel variant suffix based on CUDA version.

    Returns None if no compatible pre-built wheel exists (triggers source build).
    """
    if not cuda_version:
        return "cpu"

    major, minor = cuda_version.split('.')[:2]
    major = int(major)

    # Map CUDA versions to our pre-built wheel variants
    # Only return variants we actually have pre-built wheels for
    if major == 11:
        return "cuda118"  # All CUDA 11.x uses 11.8 build
    elif major == 12:
        return "cuda122"  # All CUDA 12.x uses 12.2 build
    elif major >= 13:
        # CUDA 13+ not yet available in GitHub Actions - fall back to source build
        log_info(f"CUDA {cuda_version} detected - no pre-built wheel available, building from source")
        return None
    else:
        return "cpu"  # Very old CUDA, fallback to CPU


def _get_wheel_filename(python_version: str, variant: str, for_download: bool = True) -> str:
    """Construct wheel filename for given Python version and variant

    Args:
        python_version: e.g., '3.11'
        variant: 'cpu', 'cuda118', 'cuda122'
        for_download: If True, include variant suffix (for GitHub). If False, standard pip format.
    """
    # Python 3.11 -> cp311
    py_tag = f"cp{python_version.replace('.', '')}"
    base = f"pywhispercpp-{PYWHISPERCPP_VERSION}-{py_tag}-{py_tag}-linux_x86_64"
    if for_download:
        # GitHub release filename: pywhispercpp-<version>-cp311-cp311-linux_x86_64+cuda122.whl
        return f"{base}+{variant}.whl"
    else:
        # Standard pip-compatible filename: pywhispercpp-<version>-cp311-cp311-linux_x86_64.whl
        return f"{base}.whl"


def download_pywhispercpp_wheel(variant: Optional[str] = None) -> Optional[Path]:
    """
    Download pre-built pywhispercpp wheel if available.

    Args:
        variant: Optional variant override ('cpu', 'cuda118', 'cuda122').
                 If None, auto-detects based on system CUDA.

    Returns:
        Path to downloaded wheel file (with pip-compatible name), or None if unavailable/failed.
    """
    python_version = _detect_venv_python_version()

    if variant is None:
        cuda_version = _detect_cuda_version()
        variant = _get_wheel_variant(cuda_version)

    # If variant is still None, no compatible wheel exists
    if variant is None:
        return None

    # Filename on GitHub (with variant suffix)
    download_filename = _get_wheel_filename(python_version, variant, for_download=True)
    # Filename for pip (standard format without variant)
    install_filename = _get_wheel_filename(python_version, variant, for_download=False)

    wheel_url = f"{WHEEL_BASE_URL}/{download_filename}"

    # Create variant-specific cache directory to avoid collisions between cpu/cuda variants
    variant_cache_dir = WHEEL_CACHE_DIR / variant
    variant_cache_dir.mkdir(parents=True, exist_ok=True)
    download_path = variant_cache_dir / download_filename
    install_path = variant_cache_dir / install_filename

    # Check if already cached (check the pip-compatible filename in variant subdirectory)
    if install_path.exists() and install_path.stat().st_size > 10 * 1024 * 1024:  # >10MB
        log_info(f"Using cached wheel: {variant}/{install_filename}")
        return install_path

    log_info(f"Downloading pre-built wheel: {download_filename}")

    try:
        def show_progress(block_num, block_size, total_size):
            """Callback to show download progress"""
            if not OutputController.is_progress_enabled():
                return

            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size) if total_size > 0 else 0
            size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            downloaded_mb = downloaded / (1024 * 1024)

            progress_msg = f"\r[INFO] Downloading wheel: {downloaded_mb:.1f}/{size_mb:.1f} MB ({percent}%)"
            OutputController.write(progress_msg, VerbosityLevel.NORMAL, flush=True)

            if downloaded >= total_size and total_size > 0:
                OutputController.write("\n", VerbosityLevel.NORMAL, flush=True)

        urllib.request.urlretrieve(wheel_url, download_path, reporthook=show_progress)

        # Verify download
        if download_path.exists() and download_path.stat().st_size > 10 * 1024 * 1024:
            # Rename to pip-compatible filename (strip variant suffix)
            if download_path != install_path:
                if install_path.exists():
                    install_path.unlink()
                download_path.rename(install_path)
            log_success(f"Pre-built wheel downloaded: {install_filename}")
            return install_path
        else:
            log_warning("Downloaded wheel appears invalid (too small)")
            if download_path.exists():
                download_path.unlink()
            return None

    except urllib.error.HTTPError as e:
        if e.code == 404:
            log_debug(f"Pre-built wheel not available: {download_filename}")
        else:
            log_warning(f"Failed to download wheel: HTTP {e.code}")
        return None
    except Exception as e:
        log_warning(f"Failed to download wheel: {e}")
        if download_path.exists():
            download_path.unlink()
        return None


def install_pywhispercpp_from_wheel(pip_bin: Path, wheel_path: Path) -> bool:
    """Install pywhispercpp from a pre-built wheel file."""
    log_info(f"Installing from wheel: {wheel_path.name}")

    try:
        # Setup environment
        if _check_mise_active():
            env = _create_mise_free_environment()
        else:
            env = os.environ.copy()

        venv_bin = str(VENV_DIR / 'bin')
        env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"

        run_command(
            [str(pip_bin), 'install', '--force-reinstall', str(wheel_path)],
            check=True,
            env=env
        )
        log_success("pywhispercpp installed from pre-built wheel")
        return True
    except subprocess.CalledProcessError as e:
        log_warning(f"Wheel installation failed: {e}")
        return False


# ==================== State Management ====================

def init_state():
    """Initialize state directory and file"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text('{}')


def get_state(key: str) -> str:
    """Get a value from the state file"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(key, '')
        except (json.JSONDecodeError, IOError):
            return ''
    return ''


def set_state(key: str, value: str):
    """Set a value in the state file"""
    init_state()
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        data[key] = value
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        log_debug(f"Error writing state file: {e}")


def get_all_state() -> Dict:
    """Get all state data"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log_debug(f"Error reading state file: {e}")
            # Try to recover by creating a new state file
            try:
                STATE_FILE.unlink()
                init_state()
            except Exception:
                pass
            return {}
    return {}


def set_install_state(state: str, error: Optional[str] = None):
    """
    Set installation state with optional error message.
    
    Args:
        state: One of 'not_started', 'in_progress', 'completed', 'failed'
        error: Optional error message if state is 'failed'
    """
    init_state()
    data = get_all_state()
    data['install_state'] = state
    if error:
        data['last_error'] = error
        data['last_error_time'] = str(Path(__file__).stat().st_mtime)  # Simple timestamp
    elif state == 'completed':
        # Clear error on success
        data.pop('last_error', None)
        data.pop('last_error_time', None)
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        log_error(f"Failed to write state file: {e}")


def get_install_state() -> Tuple[str, Optional[str]]:
    """Get installation state and last error if any"""
    data = get_all_state()
    state = data.get('install_state', 'not_started')
    error = data.get('last_error')
    return state, error


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    if file_path.exists():
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    return ''


def check_model_validity(model_file: Path) -> bool:
    """Check if model file is valid"""
    if not model_file.exists():
        return False
    
    file_size = model_file.stat().st_size
    stored_hash = get_state("model_base_en_hash")
    current_hash = compute_file_hash(model_file)
    
    # If we have a stored hash and it matches, it's valid
    if stored_hash and current_hash == stored_hash:
        return True
    
    # If file is reasonable size (>100MB), it's probably valid
    if file_size > 100000000:
        return True
    
    return False


# ==================== Helper Functions ====================

def detect_cuda_host_compiler() -> Optional[str]:
    """Detect appropriate CUDA host compiler"""
    # Allow explicit override
    cuda_host = os.environ.get('HYPRWHSPR_CUDA_HOST')
    if cuda_host and Path(cuda_host).exists() and os.access(cuda_host, os.X_OK):
        return cuda_host
    
    # Check GCC version
    try:
        result = run_command(['gcc', '-dumpfullversion'], check=False, capture_output=True)
        if result.returncode == 0:
            gcc_version = _safe_decode(result.stdout).strip()
            gcc_major = int(gcc_version.split('.')[0])
            
            # If GCC >= 15, prefer gcc14 if present
            if gcc_major >= 15:
                if shutil.which('g++-14'):
                    return '/usr/bin/g++-14'
    except Exception:
        pass
    
    # Default to g++
    if shutil.which('g++'):
        return shutil.which('g++')
    
    return None


# ==================== System Dependencies ====================

def install_system_dependencies():
    """Install system dependencies needed for backend compilation.

    On Arch Linux, automatically installs missing packages via pacman.
    On other distributions, skips automatic installation and provides guidance.
    """
    log_info("Checking system dependencies...")

    # Check if we're on an Arch-based system
    if not shutil.which('pacman'):
        # Not Arch - check for essential build tools and provide guidance
        missing = []
        if not shutil.which('cmake'):
            missing.append('cmake')
        if not shutil.which('make'):
            missing.append('make')
        if not shutil.which('git'):
            missing.append('git')
        if not shutil.which('gcc') and not shutil.which('cc'):
            missing.append('gcc/build-essential')

        if missing:
            log_warning(f"Missing build tools: {', '.join(missing)}")
            log_info("Please install these using your distribution's package manager:")
            log_info("  Debian/Ubuntu: sudo apt install cmake make git build-essential python3-dev")
            log_info("  Fedora: sudo dnf install cmake make git gcc-c++ python3-devel")
            log_info("  openSUSE: sudo zypper install cmake make git gcc-c++ python3-devel")
        else:
            log_success("Build tools available")
        return

    # Arch Linux path - install via pacman
    pkgs = ['cmake', 'make', 'git', 'base-devel', 'python', 'curl']

    to_install = []
    for pkg in pkgs:
        result = run_command(['pacman', '-Q', pkg], check=False, capture_output=True)
        if result.returncode != 0:
            to_install.append(pkg)

    if to_install:
        log_info(f"Installing: {' '.join(to_install)}")
        run_sudo_command(['pacman', '-S', '--needed', '--noconfirm'] + to_install, check=False)

    log_success("Dependencies ready")


# ==================== GPU Support Setup ====================

def setup_nvidia_support() -> bool:
    """Setup NVIDIA/CUDA support. Returns True if CUDA is available."""
    log_info("GPU check…")

    # First check for actual NVIDIA hardware via lspci (like Omarchy does)
    # This prevents false positives when nvidia-utils is installed but no GPU exists
    try:
        result = run_command(['lspci'], capture_output=True, check=False, verbose=False)
        if result and result.returncode == 0:
            lspci_output = _safe_decode(result.stdout).lower()
            if 'nvidia' not in lspci_output:
                log_info("No NVIDIA hardware detected via lspci (CPU mode)")
                return False
    except Exception:
        # If lspci fails, continue to nvidia-smi check
        pass

    if not shutil.which('nvidia-smi'):
        log_info("No NVIDIA GPU detected (CPU mode)")
        return False

    # Test nvidia-smi
    try:
        result = run_command(['nvidia-smi', '-L'], check=False, capture_output=True, timeout=2)
        if result.returncode != 0:
            log_warning("nvidia-smi found but not responding (no GPU hardware or driver issue)")
            return False

        # Verify output actually lists a GPU
        # nvidia-smi -L outputs: "GPU 0: NVIDIA GeForce RTX 4070..."
        if result.stdout:
            output = _safe_decode(result.stdout).strip()
            output_lower = output.lower()
            # Look for "GPU N:" pattern which indicates an actual GPU listing
            if not ('gpu 0:' in output_lower or 'gpu 1:' in output_lower or 'gpu 2:' in output_lower or 'gpu 3:' in output_lower):
                log_info("nvidia-smi present but no NVIDIA GPU hardware detected (CPU mode)")
                return False
        else:
            log_warning("nvidia-smi returned no output (no GPU hardware)")
            return False
    except Exception:
        log_warning("nvidia-smi found but not responding")
        return False

    log_success("NVIDIA GPU detected")
    
    # Check for nvcc
    nvcc_path = None
    if Path('/opt/cuda/bin/nvcc').exists():
        nvcc_path = '/opt/cuda/bin/nvcc'
    elif shutil.which('nvcc'):
        nvcc_path = shutil.which('nvcc')
    
    if nvcc_path:
        # Set environment variables
        # Use the directory where nvcc was actually found, not hardcoded /opt/cuda/bin
        nvcc_dir = str(Path(nvcc_path).parent)
        os.environ['PATH'] = f'{nvcc_dir}:' + os.environ.get('PATH', '')
        os.environ['CUDACXX'] = nvcc_path
        log_success("CUDA toolkit present")
    else:
        log_warning("CUDA toolkit not found")
        # Try to install on Arch, provide guidance on other distros
        if shutil.which('pacman'):
            if Confirm is None:
                # Fallback if rich not available - just warn and skip
                log_warning("CUDA toolkit not found. Skipping CUDA installation.")
                log_info("You can install it manually later: sudo pacman -S cuda")
                return False
            
            log_warning("CUDA toolkit not found. CUDA is required for NVIDIA GPU acceleration.")
            log_info("CUDA installation can take 10-15 minutes and requires ~3GB of disk space.")
            if not Confirm.ask("Install CUDA toolkit now? (If no, will use CPU mode)", default=True):
                log_info("Skipping CUDA installation. Will use CPU mode instead.")
                return False
            
            log_info("Installing CUDA toolkit... This may take a while.")
            run_sudo_command(['pacman', '-S', '--needed', '--noconfirm', 'cuda'], check=False)
        else:
            log_info("Please install CUDA toolkit using your distribution's package manager:")
            log_info("  Debian/Ubuntu: sudo apt install nvidia-cuda-toolkit")
            log_info("  Fedora: sudo dnf install cuda")
            log_info("  Or download from: https://developer.nvidia.com/cuda-downloads")
            log_info("Without CUDA, the NVIDIA backend will fall back to CPU mode.")
            return False

        # Check for nvcc after installation attempt
        nvcc_path_after_install = None
        if Path('/opt/cuda/bin/nvcc').exists():
            nvcc_path_after_install = '/opt/cuda/bin/nvcc'
        elif Path('/usr/bin/nvcc').exists():
            nvcc_path_after_install = '/usr/bin/nvcc'
        elif shutil.which('nvcc'):
            nvcc_path_after_install = shutil.which('nvcc')
        
        if nvcc_path_after_install:
            # Use the directory where nvcc was actually found
            nvcc_dir = str(Path(nvcc_path_after_install).parent)
            os.environ['PATH'] = f'{nvcc_dir}:' + os.environ.get('PATH', '')
            os.environ['CUDACXX'] = nvcc_path_after_install
            log_success("CUDA installed")
        else:
            log_warning("nvcc still not visible; will build CPU-only")
            return False
    
    # Choose host compiler for NVCC
    host_compiler = detect_cuda_host_compiler()
    if host_compiler:
        os.environ['CUDAHOSTCXX'] = host_compiler
        log_info(f"CUDA host compiler: {host_compiler}")
        
        if host_compiler == '/usr/bin/g++':
            try:
                result = run_command(['gcc', '-dumpfullversion'], check=False, capture_output=True)
                if result.returncode == 0:
                    gcc_version = _safe_decode(result.stdout).strip()
                    gcc_major = int(gcc_version.split('.')[0])
                    if gcc_major >= 15:
                        log_warning(f"GCC {gcc_major} with NVCC can fail; consider:")
                        log_warning("  yay -S gcc14 gcc14-libs")
                        log_warning("  export HYPRWHSPR_CUDA_HOST=/usr/bin/g++-14")
            except Exception:
                pass
    else:
        log_warning("No suitable host compiler found; will build CPU-only")
        return False
    
    return True


def setup_amd_support() -> bool:
    """Setup AMD/ROCm support. Returns True if ROCm is available."""
    log_info("Checking for AMD GPU...")
    
    if not (shutil.which('rocm-smi') or Path('/opt/rocm').exists()):
        log_info("No AMD GPU detected")
        return False
    
    # Test rocm-smi
    try:
        result = run_command(['rocm-smi', '--showproductname'], check=False, capture_output=True, timeout=2)
        if result.returncode != 0:
            log_warning("rocm-smi found but not responding (no GPU hardware or driver issue)")
            return False
    except Exception:
        log_warning("rocm-smi found but not responding")
        return False
    
    log_success("AMD GPU with ROCm detected")
    
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    if Path(rocm_path).exists():
        os.environ['ROCM_PATH'] = rocm_path
        os.environ['PATH'] = f"{rocm_path}/bin:" + os.environ.get('PATH', '')
        log_success("ROCm toolkit present")
    else:
        log_warning("ROCm not found")
        # Try to install on Arch (requires AUR helper), provide guidance on other distros
        if shutil.which('yay'):
            log_info("Installing ROCm toolkit...")
            run_sudo_command(['yay', '-S', '--needed', '--noconfirm', 'rocm-hip-sdk', 'rocm-opencl-sdk'], check=False)
        elif shutil.which('pacman'):
            log_info("ROCm requires an AUR helper (yay) on Arch Linux")
            log_info("Install yay first, then re-run setup")
        else:
            log_info("Please install ROCm using your distribution's package manager:")
            log_info("  Ubuntu: Follow https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html")
            log_info("  Fedora: sudo dnf install rocm-hip rocm-opencl")

        if Path(rocm_path).exists():
            os.environ['ROCM_PATH'] = rocm_path
            os.environ['PATH'] = f"{rocm_path}/bin:" + os.environ.get('PATH', '')
            log_success("ROCm toolkit present")
        else:
            return False
    
    # Check for hipcc
    if not shutil.which('hipcc'):
        log_warning("ROCm detected but hipcc compiler missing")
        return False
    
    return True


def detect_gpu_type() -> str:
    """
    Auto-detect GPU type for automatic backend selection.

    Returns:
        'nvidia': NVIDIA discrete GPU detected (will use CUDA)
        'vulkan': Any other GPU detected (AMD discrete, AMD APU iGPU, Intel iGPU, etc.)
        'cpu': No GPU capability detected

    Detection strategy:
    1. Check for NVIDIA GPU via nvidia-smi
    2. Check for ANY GPU via vulkaninfo (AMD/Intel/ARM iGPUs and discrete GPUs)
    3. Fallback to CPU if no GPU detected
    """
    try:
        from .output_control import log_debug, log_info
    except ImportError:
        from output_control import log_debug, log_info

    # 1. Check for NVIDIA GPU
    # First check for actual NVIDIA hardware via lspci (like Omarchy does)
    # This prevents false positives when nvidia-utils is installed but no GPU exists
    try:
        result = run_command(['lspci'], capture_output=True, check=False, verbose=False)
        if result and result.returncode == 0:
            lspci_output = _safe_decode(result.stdout).lower()
            if 'nvidia' not in lspci_output:
                # Skip nvidia-smi check entirely - no hardware present
                pass
            else:
                nvidia_smi_path = shutil.which('nvidia-smi')
                if nvidia_smi_path:
                    # Hardware detected, now verify with nvidia-smi
                    try:
                        result = run_command(
                            ['nvidia-smi', '-L'],
                            capture_output=True,
                            check=False,
                            verbose=False,
                            timeout=2
                        )
                        if result:
                            log_info(f"[GPU Detection] nvidia-smi exit code: {result.returncode}")
                            if result.returncode == 0 and result.stdout:
                                # Verify output actually lists a GPU (not just error messages)
                                # nvidia-smi -L outputs: "GPU 0: NVIDIA GeForce RTX 4070..."
                                output = _safe_decode(result.stdout).strip()
                                output_lower = output.lower()
                                log_info(f"[GPU Detection] nvidia-smi output: {output[:100]}")
                                # Look for "GPU N:" pattern which indicates an actual GPU listing
                                # This won't match "No devices were found" or other error messages
                                if 'gpu 0:' in output_lower or 'gpu 1:' in output_lower or 'gpu 2:' in output_lower or 'gpu 3:' in output_lower:
                                    log_info(f"[GPU Detection] ✓ NVIDIA GPU confirmed: {output.splitlines()[0][:60]}")
                                    return 'nvidia'
                                else:
                                    log_info(f"[GPU Detection] nvidia-smi ran but no GPU pattern found")
                            else:
                                log_info(f"[GPU Detection] nvidia-smi failed or no output")
                                if result.stderr:
                                    stderr_str = _safe_decode(result.stderr)
                                    log_info(f"[GPU Detection] stderr: {stderr_str[:100]}")
                    except Exception as e:
                        log_info(f"[GPU Detection] nvidia-smi exception: {e}")
    except Exception as e:
        log_debug(f"lspci check failed: {e}")

    # 2. Check for ANY GPU via Vulkan
    # First check if vulkaninfo is installed
    if not shutil.which('vulkaninfo'):
        # Try to install Vulkan tools to check for GPU (Arch only, silent on other distros)
        if shutil.which('pacman'):
            try:
                log_debug("vulkaninfo not found, installing vulkan-tools for detection")
                run_sudo_command(['pacman', '-S', '--needed', '--noconfirm', 'vulkan-tools'], check=False, verbose=False)
            except Exception:
                pass

    if shutil.which('vulkaninfo'):
        try:
            result = run_command(
                ['vulkaninfo', '--summary'],
                capture_output=True,
                check=False,
                verbose=False,
                timeout=5
            )
            if result and result.returncode == 0 and result.stdout:
                # Check if output mentions GPU/device
                output = result.stdout.lower()
                if 'gpu' in output or 'device' in output:
                    # Further check: look for actual GPU device (not just software renderer)
                    if 'llvmpipe' not in output and 'software' not in output:
                        log_debug("GPU detected via vulkaninfo")
                        return 'vulkan'
        except Exception as e:
            log_debug(f"vulkaninfo check failed: {e}")

    # 3. Fallback to CPU
    log_debug("No GPU detected, falling back to CPU")
    return 'cpu'


def setup_vulkan_support() -> bool:
    """
    Setup Vulkan support for GPU acceleration.
    Works with AMD, Intel, and other non-NVIDIA GPUs (discrete and integrated).

    Returns:
        True if Vulkan is available and configured
        False if Vulkan setup failed
    """
    log_info("Setting up Vulkan support...")

    # 1. Install Vulkan dependencies (both runtime and development headers)
    if shutil.which('pacman'):
        log_info("Installing Vulkan dependencies...")
        vulkan_pkgs = ['vulkan-headers', 'vulkan-icd-loader', 'shaderc', 'vulkan-tools']
        try:
            result = run_sudo_command(
                ['pacman', '-S', '--needed', '--noconfirm'] + vulkan_pkgs,
                check=False
            )
            if not result or result.returncode != 0:
                log_warning("Failed to install some Vulkan packages")
                return False
        except Exception as e:
            log_error(f"Failed to install Vulkan dependencies: {e}")
            return False
    else:
        # Check if Vulkan development files are available
        log_info("Checking for Vulkan development files...")
        # Look for vulkan headers in common locations
        vulkan_header_paths = [
            '/usr/include/vulkan/vulkan.h',
            '/usr/local/include/vulkan/vulkan.h',
        ]
        has_vulkan_dev = any(Path(p).exists() for p in vulkan_header_paths)
        if not has_vulkan_dev:
            log_warning("Vulkan development headers not found")
            log_info("Please install Vulkan development packages:")
            log_info("  Debian/Ubuntu: sudo apt install libvulkan-dev vulkan-tools shaderc")
            log_info("  Fedora: sudo dnf install vulkan-headers vulkan-loader-devel shaderc")
            log_info("  openSUSE: sudo zypper install vulkan-devel shaderc")
            return False

    # 2. Verify Vulkan is now available
    if not shutil.which('vulkaninfo'):
        log_warning("vulkaninfo not available after installation")
        return False

    try:
        result = run_command(
            ['vulkaninfo', '--summary'],
            capture_output=True,
            check=False,
            verbose=False,
            timeout=5
        )
        if not result or result.returncode != 0:
            log_warning("Vulkan installed but vulkaninfo check failed")
            return False

        # Check for actual GPU (not software renderer)
        output = _safe_decode(result.stdout).lower()
        if 'llvmpipe' in output or 'software' in output:
            log_warning("Only software Vulkan renderer detected (no GPU)")
            return False

        log_success("Vulkan support configured successfully")
        return True

    except Exception as e:
        log_error(f"Vulkan verification failed: {e}")
        return False


# ==================== Python Environment ====================

def setup_python_venv(force_rebuild: bool = False, custom_python: Optional[str] = None) -> Path:
    """Create or update Python virtual environment. Returns path to pip binary.

    Args:
        force_rebuild: If True, delete and recreate venv even if it exists and Python version matches.
        custom_python: Optional path to Python executable to use for venv creation.
                       If None, auto-detects a compatible Python (3.14 or earlier).
    """
    log_info("Setting up Python virtual environment…")

    # Validate requirements.txt exists
    requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
    if not requirements_file.exists():
        log_error(f"requirements.txt not found at {requirements_file}")
        raise FileNotFoundError(f"requirements.txt not found at {requirements_file}")

    # Determine Python executable to use
    if custom_python:
        # User specified explicit Python path
        if not os.path.isfile(custom_python) or not os.access(custom_python, os.X_OK):
            log_error(f"Specified Python not found or not executable: {custom_python}")
            sys.exit(1)
        python_executable = custom_python
        version = _get_python_version(custom_python)
        version_str = f"{version[0]}.{version[1]}" if version else "unknown"
        # Validate version compatibility
        if version and version > MAX_COMPATIBLE_PYTHON:
            max_str = f"{MAX_COMPATIBLE_PYTHON[0]}.{MAX_COMPATIBLE_PYTHON[1]}"
            log_error(f"Specified Python {version_str} is not compatible (requires {max_str} or earlier)")
            log_error("ML packages like onnxruntime do not have wheels for this Python version yet.")
            log_error(f"Please specify a compatible Python, e.g.: --python /usr/bin/python3.14")
            sys.exit(1)
        log_info(f"Using specified Python: {custom_python} ({version_str})")
    else:
        # Auto-detect compatible Python
        python_executable, source = _find_compatible_python()
        log_info(f"Using {source}: {python_executable}")

    # Check if venv exists and if Python version matches
    venv_needs_recreation = force_rebuild
    if force_rebuild:
        log_info("Force rebuild requested - will recreate venv")
    if VENV_DIR.exists() and not force_rebuild:
        venv_python = VENV_DIR / 'bin' / 'python'
        if venv_python.exists():
            try:
                # Check Python version in venv
                result = run_command([str(venv_python), '--version'], check=False, capture_output=True)
                venv_version = result.stdout.strip() if result.returncode == 0 and result.stdout else ""
                
                # Get version of python_executable (system Python when mise is active, otherwise current Python)
                python_exec_version_result = run_command(
                    [python_executable, '--version'],
                    check=False,
                    capture_output=True
                )
                python_exec_version = python_exec_version_result.stdout.strip() if python_exec_version_result.returncode == 0 and python_exec_version_result.stdout else ""
                
                # Extract major.minor from both version strings
                import re
                venv_major_minor = ""
                if venv_version:
                    match = re.search(r'(\d+)\.(\d+)', venv_version)
                    if match:
                        venv_major_minor = f"{match.group(1)}.{match.group(2)}"
                
                python_exec_major_minor = ""
                if python_exec_version:
                    match = re.search(r'(\d+)\.(\d+)', python_exec_version)
                    if match:
                        python_exec_major_minor = f"{match.group(1)}.{match.group(2)}"
                
                # If we couldn't get python_exec version, handle based on whether it's the same as current Python
                if not python_exec_major_minor:
                    if python_executable == sys.executable:
                        # Same Python, safe to use sys.version_info as fallback
                        python_exec_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
                        python_exec_version = f"Python {python_exec_major_minor} (from sys.version_info)"
                    else:
                        # Different Python - can't verify version, be conservative and recreate venv
                        log_warning(f"Could not determine version of target Python ({python_executable})")
                        log_warning("Cannot verify venv Python version compatibility - will recreate venv to be safe")
                        venv_needs_recreation = True
                        # Skip version comparison since we don't have valid data
                        python_exec_major_minor = None
                
                # Check if versions match (major.minor) - only if we have valid version data
                if python_exec_major_minor and venv_major_minor and venv_major_minor != python_exec_major_minor:
                    log_warning(f"Venv Python version mismatch: venv has {venv_version}, target Python is {python_exec_version}")
                    log_info("Recreating venv to match target Python version...")
                    venv_needs_recreation = True
            except Exception:
                # If we can't check, assume it's fine
                pass
        else:
            venv_needs_recreation = True
    
    # Recreate venv if needed
    if venv_needs_recreation or not VENV_DIR.exists():
        if VENV_DIR.exists():
            log_info(f"Removing existing venv at {VENV_DIR}")
            import shutil
            shutil.rmtree(VENV_DIR)
        log_info(f"Creating venv at {VENV_DIR}")
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        # Use --system-site-packages to access system GTK/GLib bindings (python-gobject)
        run_command([python_executable, '-m', 'venv', '--system-site-packages', str(VENV_DIR)], check=True)
    else:
        log_info(f"Venv already exists at {VENV_DIR}")
    
    # Get pip binary
    pip_bin = VENV_DIR / 'bin' / 'pip'
    if not pip_bin.exists():
        log_error(f"pip not found in venv at {VENV_DIR}")
        raise FileNotFoundError(f"pip not found in venv")
    
    # Upgrade pip and wheel (mise-free env applied automatically via run_command)
    run_command([str(pip_bin), 'install', '--upgrade', 'pip', 'wheel'], check=True)
    
    return pip_bin


# ==================== pywhispercpp Installation ====================

def _should_skip_pygobject() -> bool:
    """Check if PyGObject should be skipped (already installed as system package)."""
    try:
        import gi
        # gi module exists - PyGObject is installed via system package
        log_info("PyGObject already available (system package), skipping pip install")
        return True
    except ImportError:
        return False


def _extract_package_name(requirement_line: str) -> str:
    """
    Extract the package name from a requirements.txt line.
    Handles version specifiers, extras, environment markers, and URL specs.
    Examples:
        'package>=1.0' -> 'package'
        'package[extra]>=1.0' -> 'package'
        'package>=1.0; python_version >= "3.8"' -> 'package'
        'package @ https://...' -> 'package'
    """
    import re
    line = requirement_line.strip().lower()
    # Match package name: everything before version specifiers, extras, markers, or URL
    match = re.match(r'^([a-z0-9][-a-z0-9_.]*)', line)
    return match.group(1) if match else ''


def _filter_requirements(requirements_file: Path, skip_packages: list) -> Path:
    """
    Create a filtered requirements file, skipping specified packages.
    Returns path to temp file (caller must clean up).
    """
    import tempfile
    skip_packages_lower = [pkg.lower() for pkg in skip_packages]
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    try:
        with open(requirements_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line_stripped = line.strip()
                # Skip empty lines/comments as-is
                if not line_stripped or line_stripped.startswith('#'):
                    temp_file.write(line)
                    continue
                # Extract package name and check for exact match
                pkg_name = _extract_package_name(line_stripped)
                if pkg_name not in skip_packages_lower:
                    temp_file.write(line)
        temp_file.close()
        return Path(temp_file.name)
    except Exception:
        temp_file.close()
        # Clean up the temp file on error
        try:
            Path(temp_file.name).unlink()
        except Exception:
            pass
        raise


def install_pywhispercpp_cpu(pip_bin: Path, requirements_file: Path) -> bool:
    """Install CPU-only pywhispercpp"""
    log_info("Installing pywhispercpp (CPU-only)...")

    # Track if wheel was successfully installed (to avoid overwriting with PyPI version)
    wheel_installed = False

    # Try pre-built wheel first (faster than pip resolving from PyPI)
    wheel_path = download_pywhispercpp_wheel(variant='cpu')
    if wheel_path:
        if install_pywhispercpp_from_wheel(pip_bin, wheel_path):
            wheel_installed = True
            # Still need to install other requirements
            skip_packages = ['pywhispercpp']
            if _should_skip_pygobject():
                skip_packages.append('PyGObject')
            temp_req_path = None
            try:
                temp_req_path = _filter_requirements(requirements_file, skip_packages)
                run_command([str(pip_bin), 'install', '-r', str(temp_req_path)], check=True)
                return True
            except subprocess.CalledProcessError as e:
                log_warning(f"Wheel installed but remaining deps failed: {e}")
                log_warning("Falling back to full pip install...")
            finally:
                if temp_req_path and temp_req_path.exists():
                    temp_req_path.unlink()
        else:
            log_warning("Pre-built wheel failed, falling back to pip install...")

    # Build skip list - always skip pywhispercpp if wheel was already installed
    skip_packages = []
    if wheel_installed:
        skip_packages.append('pywhispercpp')
    if _should_skip_pygobject():
        skip_packages.append('PyGObject')

    temp_req_path = None
    try:
        if skip_packages:
            temp_req_path = _filter_requirements(requirements_file, skip_packages)
            install_file = temp_req_path
        else:
            install_file = requirements_file

        run_command([str(pip_bin), 'install', '-r', str(install_file)], check=True)
        log_success("pywhispercpp installed (CPU-only mode)")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install pywhispercpp (CPU-only): {e}")
        return False
    finally:
        if temp_req_path and temp_req_path.exists():
            temp_req_path.unlink()


def install_pywhispercpp_cuda(pip_bin: Path) -> bool:
    """Install pywhispercpp with CUDA support"""
    log_info("Installing pywhispercpp with CUDA support...")

    # Try pre-built wheel first (much faster than source build)
    wheel_path = download_pywhispercpp_wheel()  # Auto-detects CUDA version
    if wheel_path:
        if install_pywhispercpp_from_wheel(pip_bin, wheel_path):
            return True
        log_warning("Pre-built wheel failed, falling back to source build...")

    log_info("Building from source (this may take several minutes)...")

    # Clean build artifacts if they exist (to avoid Python version mismatches)
    if PYWHISPERCPP_SRC_DIR.exists():
        log_info("Cleaning existing build artifacts...")
        import shutil
        # Remove common build directories
        build_dirs = [
            PYWHISPERCPP_SRC_DIR / 'build',
            PYWHISPERCPP_SRC_DIR / 'dist',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'build',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'ggml' / 'build',
        ]
        for build_dir in build_dirs:
            if build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)
        
        # Remove egg-info directories
        for egg_info in PYWHISPERCPP_SRC_DIR.glob('*.egg-info'):
            if egg_info.is_dir():
                shutil.rmtree(egg_info, ignore_errors=True)
        
        # Remove CMake cache files (these can cache Python version)
        for cmake_cache in PYWHISPERCPP_SRC_DIR.rglob('CMakeCache.txt'):
            cmake_cache.unlink(missing_ok=True)
        for cmake_files in PYWHISPERCPP_SRC_DIR.rglob('CMakeFiles'):
            if cmake_files.is_dir():
                shutil.rmtree(cmake_files, ignore_errors=True)
        
        # Clean __pycache__ directories
        for pycache in PYWHISPERCPP_SRC_DIR.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)
    
    # Clone or update pywhispercpp sources
    if not PYWHISPERCPP_SRC_DIR.exists() or not (PYWHISPERCPP_SRC_DIR / '.git').exists():
        log_info(f"Cloning pywhispercpp sources (v{PYWHISPERCPP_VERSION}) → {PYWHISPERCPP_SRC_DIR}")
        PYWHISPERCPP_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        run_command([
            'git', 'clone', '--recurse-submodules',
            'https://github.com/Absadiki/pywhispercpp.git',
            str(PYWHISPERCPP_SRC_DIR)
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'checkout', PYWHISPERCPP_PINNED_COMMIT
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'submodule', 'update', '--init', '--recursive'
        ], check=True, verbose=verbose)
    else:
        log_info(f"Updating pywhispercpp sources to v{PYWHISPERCPP_VERSION} in {PYWHISPERCPP_SRC_DIR}")
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'], 
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT], 
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'], 
                       check=False, verbose=verbose)
        except Exception as e:
            log_warning(f"Could not update pywhispercpp repository to v{PYWHISPERCPP_VERSION}: {e}")
    
    # Build with CUDA support
    log_info("Building pywhispercpp with CUDA (ggml CUDA) via pip - may take several minutes")
    # Start with mise-free environment if mise is active, otherwise use current environment
    if _check_mise_active():
        env = _create_mise_free_environment()
    else:
        env = os.environ.copy()
    env['GGML_CUDA'] = 'ON'
    
    # Force CMake to use venv's Python (critical for correct Python version detection)
    venv_python = VENV_DIR / 'bin' / 'python'
    env['CMAKE_ARGS'] = f"-DPython3_EXECUTABLE={venv_python}"
    env['PYTHON_EXECUTABLE'] = str(venv_python)
    
    # Also ensure venv's bin is first in PATH so CMake finds the right tools
    venv_bin = str(VENV_DIR / 'bin')
    env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
    
    try:
        # Only use -v flag if verbose mode is enabled
        verbosity = OutputController.get_verbosity()
        pip_args = [
            str(pip_bin), 'install',
            '-e', str(PYWHISPERCPP_SRC_DIR),
            '--no-cache-dir',
            '--force-reinstall'
        ]
        if verbosity.value >= VerbosityLevel.VERBOSE.value:
            pip_args.append('-v')
        
        run_command(pip_args, check=True, env=env, verbose=verbosity.value >= VerbosityLevel.VERBOSE.value)
        log_success("pywhispercpp installed with CUDA acceleration via pip")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"pip install of pywhispercpp with CUDA failed: {e}")
        return False


def install_pywhispercpp_rocm(pip_bin: Path) -> Tuple[bool, bool]:
    """Install pywhispercpp with ROCm support. Returns (success, should_fallback)."""
    log_info("Installing pywhispercpp with ROCm support...")
    
    # Clean build artifacts if they exist (to avoid Python version mismatches)
    if PYWHISPERCPP_SRC_DIR.exists():
        log_info("Cleaning existing build artifacts...")
        import shutil
        # Remove common build directories
        build_dirs = [
            PYWHISPERCPP_SRC_DIR / 'build',
            PYWHISPERCPP_SRC_DIR / 'dist',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'build',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'ggml' / 'build',
        ]
        for build_dir in build_dirs:
            if build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)
        
        # Remove egg-info directories
        for egg_info in PYWHISPERCPP_SRC_DIR.glob('*.egg-info'):
            if egg_info.is_dir():
                shutil.rmtree(egg_info, ignore_errors=True)
        
        # Remove CMake cache files (these can cache Python version)
        for cmake_cache in PYWHISPERCPP_SRC_DIR.rglob('CMakeCache.txt'):
            cmake_cache.unlink(missing_ok=True)
        for cmake_files in PYWHISPERCPP_SRC_DIR.rglob('CMakeFiles'):
            if cmake_files.is_dir():
                shutil.rmtree(cmake_files, ignore_errors=True)
        
        # Clean __pycache__ directories
        for pycache in PYWHISPERCPP_SRC_DIR.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)
    
    # Clone or update pywhispercpp sources
    if not PYWHISPERCPP_SRC_DIR.exists() or not (PYWHISPERCPP_SRC_DIR / '.git').exists():
        log_info(f"Cloning pywhispercpp sources (v{PYWHISPERCPP_VERSION}) → {PYWHISPERCPP_SRC_DIR}")
        PYWHISPERCPP_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        run_command([
            'git', 'clone', '--recurse-submodules',
            'https://github.com/Absadiki/pywhispercpp.git',
            str(PYWHISPERCPP_SRC_DIR)
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'checkout', PYWHISPERCPP_PINNED_COMMIT
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'submodule', 'update', '--init', '--recursive'
        ], check=True, verbose=verbose)
    else:
        log_info(f"Updating pywhispercpp sources to v{PYWHISPERCPP_VERSION} in {PYWHISPERCPP_SRC_DIR}")
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'], 
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT], 
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'], 
                       check=False, verbose=verbose)
        except Exception as e:
            log_warning(f"Could not update pywhispercpp repository to v{PYWHISPERCPP_VERSION}: {e}")
    
    # Set up ROCm environment
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    # Start with mise-free environment if mise is active, otherwise use current environment
    if _check_mise_active():
        env = _create_mise_free_environment()
    else:
        env = os.environ.copy()
    env['ROCM_PATH'] = rocm_path
    env['PATH'] = f"{rocm_path}/bin:" + env.get('PATH', '')
    env['GGML_HIPBLAS'] = 'ON'
    env['GGML_HIP'] = 'ON'
    env['GGML_ROCM'] = '1'
    env['CMAKE_PREFIX_PATH'] = rocm_path
    
    # Force CMake to use venv's Python (critical for correct Python version detection)
    venv_python = VENV_DIR / 'bin' / 'python'
    env['CMAKE_ARGS'] = f"-DPython3_EXECUTABLE={venv_python}"
    env['PYTHON_EXECUTABLE'] = str(venv_python)
    
    # Ensure venv's bin is first in PATH (after ROCm) so CMake finds the right tools
    venv_bin = str(VENV_DIR / 'bin')
    env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
    
    # Build with ROCm support
    log_info("Building pywhispercpp with ROCm (ggml HIPBLAS) via pip")
    try:
        # Only use -v flag if verbose mode is enabled
        verbosity = OutputController.get_verbosity()
        pip_args = [
            str(pip_bin), 'install',
            '-e', str(PYWHISPERCPP_SRC_DIR),
            '--no-cache-dir',
            '--force-reinstall'
        ]
        if verbosity.value >= VerbosityLevel.VERBOSE.value:
            pip_args.append('-v')
        
        run_command(pip_args, check=True, env=env, verbose=verbosity.value >= VerbosityLevel.VERBOSE.value)
        log_success("pywhispercpp installed with ROCm acceleration via pip")
        return True, False
    except subprocess.CalledProcessError:
        # Build failed - return should_fallback=True
        return False, True


def install_pywhispercpp_vulkan(pip_bin: Path) -> bool:
    """Install pywhispercpp with Vulkan support.

    Uses GGML_VULKAN=1 environment variable to enable Vulkan acceleration.
    Works with AMD/Intel/ARM GPUs (discrete and integrated).

    Returns:
        True if installation succeeded
        False if installation failed
    """
    log_info("Installing pywhispercpp with Vulkan support...")

    # Clean build artifacts if they exist (to avoid Python version mismatches)
    if PYWHISPERCPP_SRC_DIR.exists():
        log_info("Cleaning existing build artifacts...")
        import shutil
        # Remove common build directories
        build_dirs = [
            PYWHISPERCPP_SRC_DIR / 'build',
            PYWHISPERCPP_SRC_DIR / 'dist',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'build',
            PYWHISPERCPP_SRC_DIR / 'whisper.cpp' / 'ggml' / 'build',
        ]
        for build_dir in build_dirs:
            if build_dir.exists():
                shutil.rmtree(build_dir, ignore_errors=True)

        # Remove egg-info directories
        for egg_info in PYWHISPERCPP_SRC_DIR.glob('*.egg-info'):
            if egg_info.is_dir():
                shutil.rmtree(egg_info, ignore_errors=True)

        # Remove CMake cache files (these can cache Python version)
        for cmake_cache in PYWHISPERCPP_SRC_DIR.rglob('CMakeCache.txt'):
            cmake_cache.unlink(missing_ok=True)
        for cmake_files in PYWHISPERCPP_SRC_DIR.rglob('CMakeFiles'):
            if cmake_files.is_dir():
                shutil.rmtree(cmake_files, ignore_errors=True)

        # Clean __pycache__ directories
        for pycache in PYWHISPERCPP_SRC_DIR.rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache, ignore_errors=True)

    # Clone or update pywhispercpp sources
    if not PYWHISPERCPP_SRC_DIR.exists() or not (PYWHISPERCPP_SRC_DIR / '.git').exists():
        log_info(f"Cloning pywhispercpp sources (v{PYWHISPERCPP_VERSION}) → {PYWHISPERCPP_SRC_DIR}")
        PYWHISPERCPP_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        run_command([
            'git', 'clone', '--recurse-submodules',
            'https://github.com/Absadiki/pywhispercpp.git',
            str(PYWHISPERCPP_SRC_DIR)
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'checkout', PYWHISPERCPP_PINNED_COMMIT
        ], check=True, verbose=verbose)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'submodule', 'update', '--init', '--recursive'
        ], check=True, verbose=verbose)
    else:
        log_info(f"Updating pywhispercpp sources to v{PYWHISPERCPP_VERSION} in {PYWHISPERCPP_SRC_DIR}")
        verbosity = OutputController.get_verbosity()
        verbose = verbosity.value >= VerbosityLevel.VERBOSE.value
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'],
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT],
                       check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'],
                       check=False, verbose=verbose)
        except Exception as e:
            log_warning(f"Could not update pywhispercpp repository to v{PYWHISPERCPP_VERSION}: {e}")

    # Set up Vulkan environment
    # Start with mise-free environment if mise is active, otherwise use current environment
    if _check_mise_active():
        env = _create_mise_free_environment()
    else:
        env = os.environ.copy()
    env['GGML_VULKAN'] = '1'

    # Force CMake to use venv's Python (critical for correct Python version detection)
    venv_python = VENV_DIR / 'bin' / 'python'
    env['CMAKE_ARGS'] = f"-DPython3_EXECUTABLE={venv_python}"
    env['PYTHON_EXECUTABLE'] = str(venv_python)

    # Ensure venv's bin is first in PATH so CMake finds the right tools
    venv_bin = str(VENV_DIR / 'bin')
    env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"

    # Build with Vulkan support
    log_info("Building pywhispercpp with Vulkan via pip")
    try:
        # Only use -v flag if verbose mode is enabled
        verbosity = OutputController.get_verbosity()
        pip_args = [
            str(pip_bin), 'install',
            '-e', str(PYWHISPERCPP_SRC_DIR),
            '--no-cache-dir',
            '--force-reinstall'
        ]
        if verbosity.value >= VerbosityLevel.VERBOSE.value:
            pip_args.append('-v')

        run_command(pip_args, check=True, env=env, verbose=verbosity.value >= VerbosityLevel.VERBOSE.value)
        log_success("pywhispercpp installed with Vulkan acceleration via pip")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install pywhispercpp with Vulkan: {e}")
        return False


# ==================== Model Download ====================

def download_pywhispercpp_model(model_name: str = 'base') -> bool:
    """Download pywhispercpp model with progress feedback"""
    log_info(f"Downloading pywhispercpp model: {model_name}…")
    
    PYWHISPERCPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = PYWHISPERCPP_MODELS_DIR / f'ggml-{model_name}.bin'
    model_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    if check_model_validity(model_file):
        log_success("pywhispercpp base model present")
        return True
    
    if model_file.exists():
        log_warning("Existing base model appears invalid; re-downloading")
        model_file.unlink()
    
    log_info(f"Fetching {model_url}")
    try:
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
        
        # Store hash for future validation
        model_hash = compute_file_hash(model_file)
        set_state("model_base_en_hash", model_hash)
        
        log_success("pywhispercpp base model downloaded")
        return True
    except Exception as e:
        log_error(f"Failed to download pywhispercpp base model: {e}")
        return False


# ==================== Parakeet Installation ====================

def setup_parakeet_venv(force_rebuild: bool = False, custom_python: Optional[str] = None) -> Path:
    """Create or update Parakeet Python virtual environment. Returns path to pip binary.

    Args:
        force_rebuild: If True, delete and recreate venv even if it exists and Python version matches.
        custom_python: Optional path to Python executable to use for venv creation.
                       If None, auto-detects a compatible Python (3.14 or earlier).
    """
    log_info("Setting up Parakeet Python virtual environment…")

    # Validate requirements.txt exists
    if not PARAKEET_REQUIREMENTS.exists():
        log_error(f"Parakeet requirements.txt not found at {PARAKEET_REQUIREMENTS}")
        raise FileNotFoundError(f"Parakeet requirements.txt not found at {PARAKEET_REQUIREMENTS}")

    # Determine Python executable to use
    if custom_python:
        # User specified explicit Python path
        if not os.path.isfile(custom_python) or not os.access(custom_python, os.X_OK):
            log_error(f"Specified Python not found or not executable: {custom_python}")
            sys.exit(1)
        python_executable = custom_python
        version = _get_python_version(custom_python)
        version_str = f"{version[0]}.{version[1]}" if version else "unknown"
        # Validate version compatibility
        if version and version > MAX_COMPATIBLE_PYTHON:
            max_str = f"{MAX_COMPATIBLE_PYTHON[0]}.{MAX_COMPATIBLE_PYTHON[1]}"
            log_error(f"Specified Python {version_str} is not compatible (requires {max_str} or earlier)")
            log_error("ML packages like onnxruntime do not have wheels for this Python version yet.")
            log_error(f"Please specify a compatible Python, e.g.: --python /usr/bin/python3.14")
            sys.exit(1)
        log_info(f"Using specified Python: {custom_python} ({version_str})")
    else:
        # Auto-detect compatible Python
        python_executable, source = _find_compatible_python()
        log_info(f"Using {source}: {python_executable}")

    # Check if venv exists and if Python version matches
    venv_needs_recreation = force_rebuild
    if force_rebuild:
        log_info("Force rebuild requested - will recreate venv")
    if PARAKEET_VENV_DIR.exists() and not force_rebuild:
        venv_python = PARAKEET_VENV_DIR / 'bin' / 'python'
        if venv_python.exists():
            try:
                # Check Python version in venv
                result = run_command([str(venv_python), '--version'], check=False, capture_output=True)
                venv_version = result.stdout.strip() if result.returncode == 0 and result.stdout else ""
                
                # Get version of python_executable (system Python when mise is active, otherwise current Python)
                python_exec_version_result = run_command(
                    [python_executable, '--version'],
                    check=False,
                    capture_output=True
                )
                python_exec_version = python_exec_version_result.stdout.strip() if python_exec_version_result.returncode == 0 and python_exec_version_result.stdout else ""
                
                # Extract major.minor from both version strings
                import re
                venv_major_minor = ""
                if venv_version:
                    match = re.search(r'(\d+)\.(\d+)', venv_version)
                    if match:
                        venv_major_minor = f"{match.group(1)}.{match.group(2)}"
                
                python_exec_major_minor = ""
                if python_exec_version:
                    match = re.search(r'(\d+)\.(\d+)', python_exec_version)
                    if match:
                        python_exec_major_minor = f"{match.group(1)}.{match.group(2)}"
                
                # If we couldn't get python_exec version, handle based on whether it's the same as current Python
                if not python_exec_major_minor:
                    if python_executable == sys.executable:
                        # Same Python, safe to use sys.version_info as fallback
                        python_exec_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
                        python_exec_version = f"Python {python_exec_major_minor} (from sys.version_info)"
                    else:
                        # Different Python - can't verify version, be conservative and recreate venv
                        log_warning(f"Could not determine version of target Python ({python_executable})")
                        log_warning("Cannot verify venv Python version compatibility - will recreate venv to be safe")
                        venv_needs_recreation = True
                        # Skip version comparison since we don't have valid data
                        python_exec_major_minor = None
                
                # Check if versions match (major.minor) - only if we have valid version data
                if python_exec_major_minor and venv_major_minor and venv_major_minor != python_exec_major_minor:
                    log_warning(f"Parakeet venv Python version mismatch: venv has {venv_version}, target Python is {python_exec_version}")
                    log_info("Recreating venv to match target Python version...")
                    venv_needs_recreation = True
            except Exception:
                # If we can't check, assume it's fine
                pass
        else:
            venv_needs_recreation = True
    
    # Recreate venv if needed
    if venv_needs_recreation or not PARAKEET_VENV_DIR.exists():
        if PARAKEET_VENV_DIR.exists():
            log_info(f"Removing existing Parakeet venv at {PARAKEET_VENV_DIR}")
            import shutil
            shutil.rmtree(PARAKEET_VENV_DIR)
        log_info(f"Creating Parakeet venv at {PARAKEET_VENV_DIR}")
        PARAKEET_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_command([python_executable, '-m', 'venv', str(PARAKEET_VENV_DIR)], check=True)
    else:
        log_info(f"Parakeet venv already exists at {PARAKEET_VENV_DIR}")
    
    # Get pip binary
    pip_bin = PARAKEET_VENV_DIR / 'bin' / 'pip'
    if not pip_bin.exists():
        log_error(f"pip not found in Parakeet venv at {PARAKEET_VENV_DIR}")
        raise FileNotFoundError(f"pip not found in Parakeet venv")
    
    # Upgrade pip and wheel (mise-free env applied automatically via run_command)
    run_command([str(pip_bin), 'install', '--upgrade', 'pip', 'wheel'], check=True)
    
    return pip_bin


def install_parakeet_dependencies(pip_bin: Path) -> bool:
    """Install Parakeet backend dependencies"""
    log_info("Installing Parakeet dependencies...")
    
    # Check for CUDA availability
    enable_cuda = False
    if shutil.which('nvidia-smi'):
        try:
            result = run_command(['nvidia-smi', '-L'], check=False, capture_output=True, timeout=2)
            if result.returncode == 0:
                enable_cuda = True
                log_info("CUDA detected - will install PyTorch with CUDA support")
        except Exception:
            pass
    
    try:

        # Install ml_dtypes and numpy first with pinned versions to ensure compatibility
        # - ml_dtypes 0.5.4+ includes float4_e2m1fn required by onnx
        # - numpy must be <2.4 for numba compatibility (numba is a nemo_toolkit dep)
        log_info("Installing ml_dtypes and numpy (required for onnx/numba compatibility)...")
        run_command([str(pip_bin), 'install', '--upgrade', 'ml_dtypes>=0.5.4', 'numpy>=1.22,<2.4'], check=True)
       
        # Install base dependencies first (excluding torch)
        log_info("Installing base dependencies... May take a moment.")
        base_deps = [
            'nemo_toolkit[asr]',
            'fastapi',
            'uvicorn[standard]',
            'soundfile',
            'python-multipart',
        ]
        
        if enable_cuda:
            base_deps.append('cuda-python>=12.3')
        
        run_command([str(pip_bin), 'install'] + base_deps, check=True)

        # Re-pin ml_dtypes and numpy after nemo_toolkit installation
        # nemo_toolkit's dependency resolution can downgrade ml_dtypes to 0.4.x
        # which lacks float4_e2m1fn required by onnx
        log_info("Re-pinning ml_dtypes and numpy versions...")
        run_command([str(pip_bin), 'install', '--upgrade', 'ml_dtypes>=0.5.4', 'numpy>=1.22,<2.4'], check=True)

        # Install torch with appropriate CUDA support
        if enable_cuda:
            log_info("Installing PyTorch with CUDA 12.1 support... May take a moment.")
            # Use PyTorch CUDA index
            try:
                run_command([
                    str(pip_bin), 'install', 'torch',
                    '--index-url', 'https://download.pytorch.org/whl/cu121'
                ], check=True)
                log_success("PyTorch with CUDA support installed")
            except subprocess.CalledProcessError as e:
                log_warning(f"PyTorch CUDA installation failed: {e}")
                log_warning("Falling back to CPU-only PyTorch installation... GPU preferred but it works.")
                log_info("Installing PyTorch (CPU-only)...")
                run_command([str(pip_bin), 'install', 'torch'], check=True)
                log_success("PyTorch (CPU-only) installed as fallback")
        else:
            log_info("Installing PyTorch (CPU-only)...")
            run_command([str(pip_bin), 'install', 'torch'], check=True)
        
        log_success("Parakeet dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install Parakeet dependencies: {e}")
        return False


# ==================== ONNX-ASR Installation ====================

def install_onnx_asr(pip_bin: Path, enable_gpu: bool = False) -> bool:
    """
    Install onnx-asr into the main venv.

    onnx-asr is an ASR library using ONNX runtime, supporting both CPU and GPU.
    It provides significantly better performance than whisper.cpp.

    Args:
        pip_bin: Path to pip binary in the venv
        enable_gpu: If True, install GPU support (CUDA/TensorRT)

    Returns:
        True if installation succeeded, False otherwise
    """
    if enable_gpu:
        log_info("Installing onnx-asr with GPU support (CUDA/TensorRT)...")
        try:
            # Explicitly install onnxruntime-gpu first to ensure it's available
            log_info("Installing onnxruntime-gpu...")
            run_command([str(pip_bin), 'install', 'onnxruntime-gpu'], check=True)
            # Install onnx-asr with GPU backend and HuggingFace hub support
            # [cuda] = onnxruntime-gpu for CUDA/TensorRT (but we install it explicitly above)
            # [hub] = huggingface_hub for model downloads
            run_command([str(pip_bin), 'install', 'onnx-asr[cuda,hub]'], check=True)
            log_success("onnx-asr installed with GPU support")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to install onnx-asr with GPU support: {e}")
            log_warning("Falling back to CPU-only installation...")
            # Fall back to CPU installation
            enable_gpu = False
    
    if not enable_gpu:
        log_info("Installing onnx-asr (CPU-optimized)...")
        try:
            # Install onnx-asr with CPU backend and HuggingFace hub support
            # [cpu] = onnxruntime for CPU
            # [hub] = huggingface_hub for model downloads
            run_command([str(pip_bin), 'install', 'onnx-asr[cpu,hub]'], check=True)
            log_success("onnx-asr installed")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to install onnx-asr: {e}")
            return False


# ==================== faster-whisper Installation ====================

def install_faster_whisper(pip_bin: Path, enable_gpu: bool = False) -> bool:
    """Install faster-whisper into the main venv.

    Args:
        pip_bin: Path to pip executable in the target venv
        enable_gpu: If True, also install nvidia-cublas-cu12 and nvidia-cudnn-cu12
                    so CTranslate2 can load CUDA libs from the venv without requiring
                    system cuda/cudnn packages.
    """
    log_info("Installing faster-whisper...")
    try:
        run_command([str(pip_bin), 'install', 'faster-whisper'], check=True)
        if enable_gpu:
            log_info("Installing CUDA libraries for GPU support (nvidia-cublas-cu12, nvidia-cudnn-cu12)...")
            run_command([str(pip_bin), 'install', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12'], check=True)
            log_success("faster-whisper installed with CUDA support")
        else:
            log_success("faster-whisper installed (CPU mode)")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install faster-whisper: {e}")
        return False


# ==================== Parallel Installation Helpers ====================

def _parallel_setup_gpu_and_venv(backend_type: str, force_rebuild: bool = False,
                                 custom_python: Optional[str] = None) -> Tuple[Dict[str, bool], Optional[Path]]:
    """
    Run GPU detection and venv creation in parallel.

    This provides ~2-5 second speedup by overlapping independent operations.

    Args:
        backend_type: One of 'nvidia', 'amd', 'vulkan' (or 'cpu' for no GPU setup)
        force_rebuild: If True, recreate venv even if it exists
        custom_python: Optional path to Python executable for venv creation

    Returns:
        Tuple of (gpu_status dict, pip_bin Path or None if venv setup failed)
    """
    gpu_status = {'cuda': False, 'rocm': False, 'vulkan': False}
    pip_bin = None
    errors = []

    def setup_gpu():
        """Run GPU detection/setup based on backend type"""
        nonlocal gpu_status
        try:
            if backend_type == 'nvidia':
                gpu_status['cuda'] = setup_nvidia_support()
            elif backend_type == 'amd':
                gpu_status['rocm'] = setup_amd_support()
            elif backend_type == 'vulkan':
                gpu_status['vulkan'] = setup_vulkan_support()
        except Exception as e:
            errors.append(f"GPU setup error: {e}")

    def setup_venv():
        """Create/verify Python venv"""
        nonlocal pip_bin
        try:
            pip_bin = setup_python_venv(force_rebuild=force_rebuild, custom_python=custom_python)
        except Exception as e:
            errors.append(f"Venv setup error: {e}")

    # Run both in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        gpu_future = executor.submit(setup_gpu)
        venv_future = executor.submit(setup_venv)

        # Wait for both to complete
        for future in as_completed([gpu_future, venv_future]):
            try:
                future.result()
            except Exception as e:
                errors.append(str(e))

    if errors:
        for error in errors:
            log_warning(error)

    return gpu_status, pip_bin


def _parallel_deps_and_wheel(pip_bin: Path, requirements_file: Path, variant: str) -> Tuple[bool, Optional[Path]]:
    """
    Download wheel and install base dependencies in parallel.

    Args:
        pip_bin: Path to pip in venv
        requirements_file: Path to requirements.txt
        variant: Wheel variant ('cpu', 'cuda118', 'cuda122')

    Returns:
        Tuple of (deps_installed bool, wheel_path or None)
    """
    deps_ok = False
    wheel_path = None
    errors = []

    def install_deps():
        """Install base dependencies (excluding pywhispercpp)"""
        nonlocal deps_ok
        try:
            # Filter out pywhispercpp from requirements
            skip_packages = ['pywhispercpp']
            if _should_skip_pygobject():
                skip_packages.append('PyGObject')

            temp_req_path = None
            try:
                temp_req_path = _filter_requirements(requirements_file, skip_packages)
                run_command([str(pip_bin), 'install', '-r', str(temp_req_path)], check=True)
                deps_ok = True
            finally:
                if temp_req_path and temp_req_path.exists():
                    temp_req_path.unlink()
        except Exception as e:
            errors.append(f"Deps install error: {e}")

    def download_wheel():
        """Download pre-built wheel"""
        nonlocal wheel_path
        try:
            wheel_path = download_pywhispercpp_wheel(variant=variant)
        except Exception as e:
            errors.append(f"Wheel download error: {e}")

    # Run both in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        deps_future = executor.submit(install_deps)
        wheel_future = executor.submit(download_wheel)

        for future in as_completed([deps_future, wheel_future]):
            try:
                future.result()
            except Exception as e:
                errors.append(str(e))

    if errors:
        for error in errors:
            log_debug(error)

    return deps_ok, wheel_path


# ==================== Main Installation Function ====================

def install_backend(backend_type: str, cleanup_on_failure: bool = True, force_rebuild: bool = False,
                    custom_python: Optional[str] = None) -> bool:
    """
    Main function to install backend.

    Args:
        backend_type: One of 'cpu', 'nvidia', 'amd', 'vulkan', 'parakeet', 'onnx-asr'
        cleanup_on_failure: Whether to clean up partial installations on failure
        force_rebuild: If True, delete and recreate venv even if it exists
        custom_python: Optional path to Python executable to use for venv creation.
                       If None, auto-detects a compatible Python (3.14 or earlier).

    Returns:
        True if installation succeeded, False otherwise
    """
    init_state()
    set_install_state('in_progress')

    log_info(f"Installing {backend_type.upper()} backend...")

    # Check for MISE interference
    if _check_mise_active():
        log_warning("Warning! MISE is active. This may cause build errors.")
        log_warning("To fix: mise deactivate (or: mise unuse -g python)")

    # Validate backend type
    if backend_type not in ['cpu', 'nvidia', 'amd', 'vulkan', 'parakeet', 'onnx-asr', 'faster-whisper']:
        error_msg = f"Invalid backend type: {backend_type}"
        log_error(error_msg)
        set_install_state('failed', error_msg)
        return False
    
    # Track what we've created for cleanup
    created_items = {
        'venv_created': False,
        'venv_path': None,
        'git_clone_created': False,
        'git_clone_path': None,
        'packages_installed': []
    }
    
    try:
        # Install system dependencies
        install_system_dependencies()
        
        # Setup GPU support if needed
        enable_cuda = False
        enable_rocm = False
        enable_vulkan = False

        if backend_type == 'nvidia':
            enable_cuda = setup_nvidia_support()
            if not enable_cuda:
                log_warning("NVIDIA backend selected but CUDA not available, falling back to CPU")
                backend_type = 'cpu'
        elif backend_type == 'amd':
            enable_rocm = setup_amd_support()
            if not enable_rocm:
                log_warning("AMD backend selected but ROCm not available, falling back to CPU")
                backend_type = 'cpu'
        elif backend_type == 'vulkan':
            enable_vulkan = setup_vulkan_support()
            if not enable_vulkan:
                log_warning("Vulkan backend selected but Vulkan not available, falling back to CPU")
                backend_type = 'cpu'
        elif backend_type == 'parakeet':
            # Parakeet uses separate venv and REST API
            if not PARAKEET_SCRIPT.exists():
                error_msg = f"Parakeet script not found at {PARAKEET_SCRIPT}"
                log_error(error_msg)
                set_install_state('failed', error_msg)
                return False
            
            # Setup Parakeet venv
            parakeet_venv_existed = PARAKEET_VENV_DIR.exists()
            parakeet_pip_bin = setup_parakeet_venv(force_rebuild=force_rebuild, custom_python=custom_python)
            if (force_rebuild or not parakeet_venv_existed) and PARAKEET_VENV_DIR.exists():
                created_items['venv_created'] = True
                created_items['venv_path'] = str(PARAKEET_VENV_DIR)
            
            # Install Parakeet dependencies
            if not install_parakeet_dependencies(parakeet_pip_bin):
                error_msg = "Failed to install Parakeet dependencies"
                log_error(error_msg)
                if cleanup_on_failure:
                    log_info("Cleaning up partial installation...")
                    _cleanup_partial_installation(created_items, parakeet_pip_bin)
                set_install_state('failed', error_msg)
                return False
            
            # Installation successful for Parakeet
            set_install_state('completed')
            log_success("Parakeet backend installation completed!")
            return True
        elif backend_type == 'faster-whisper':
            # faster-whisper uses main venv with faster-whisper package
            venv_existed = VENV_DIR.exists()
            pip_bin = setup_python_venv(force_rebuild=force_rebuild, custom_python=custom_python)
            if (force_rebuild or not venv_existed) and VENV_DIR.exists():
                created_items['venv_created'] = True
                created_items['venv_path'] = str(VENV_DIR)

            # Detect NVIDIA GPU — if present, install CUDA pip libs so CTranslate2 can
            # find libcublas/libcudnn without requiring the system cuda/cudnn packages.
            # faster-whisper only supports NVIDIA CUDA (not AMD/Intel Vulkan/ROCm).
            enable_gpu = False
            if shutil.which('nvidia-smi'):
                try:
                    result = run_command(['nvidia-smi', '-L'], check=False, capture_output=True, timeout=2)
                    if result.returncode == 0 and result.stdout:
                        output = _safe_decode(result.stdout).strip()
                        output_lower = output.lower()
                        if 'gpu 0:' in output_lower or 'gpu 1:' in output_lower or 'gpu 2:' in output_lower or 'gpu 3:' in output_lower:
                            enable_gpu = True
                            log_info("NVIDIA GPU detected - will install faster-whisper with CUDA support")
                        else:
                            log_info("NVIDIA driver present but no GPU hardware detected")
                    else:
                        log_info("NVIDIA driver check failed - will use CPU mode")
                except Exception:
                    log_info("GPU detection failed - will use CPU mode")

            if not enable_gpu:
                log_info("Installing faster-whisper (CPU mode)")

            # Install base requirements first
            requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
            log_info("Installing base dependencies...")
            try:
                run_command([str(pip_bin), 'install', '-r', str(requirements_file)], check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install base dependencies: {e}"
                log_error(error_msg)
                if cleanup_on_failure:
                    log_info("Cleaning up partial installation...")
                    _cleanup_partial_installation(created_items, pip_bin)
                set_install_state('failed', error_msg)
                return False

            if not install_faster_whisper(pip_bin, enable_gpu=enable_gpu):
                error_msg = "Failed to install faster-whisper"
                log_error(error_msg)
                if cleanup_on_failure:
                    log_info("Cleaning up partial installation...")
                    _cleanup_partial_installation(created_items, pip_bin)
                set_install_state('failed', error_msg)
                return False

            cur_req_hash = compute_file_hash(requirements_file)
            set_state("requirements_hash", cur_req_hash)

            set_install_state('completed')
            log_success("faster-whisper backend installation completed!")
            return True

        elif backend_type == 'onnx-asr':
            # ONNX-ASR uses main venv with onnx-asr package
            # Setup main venv
            venv_existed = VENV_DIR.exists()
            pip_bin = setup_python_venv(force_rebuild=force_rebuild, custom_python=custom_python)
            if (force_rebuild or not venv_existed) and VENV_DIR.exists():
                created_items['venv_created'] = True
                created_items['venv_path'] = str(VENV_DIR)

            # Detect GPU availability for onnx-asr
            # Note: onnx-asr only needs NVIDIA drivers (nvidia-smi), not CUDA toolkit
            # Unlike pywhispercpp which needs nvcc to build, onnx-asr uses pre-built ONNX Runtime
            enable_gpu = False
            if shutil.which('nvidia-smi'):
                try:
                    result = run_command(['nvidia-smi', '-L'], check=False, capture_output=True, timeout=2)
                    if result.returncode == 0 and result.stdout:
                        output = _safe_decode(result.stdout).strip()
                        output_lower = output.lower()
                        # Check for "GPU N:" pattern which indicates an actual GPU listing
                        if 'gpu 0:' in output_lower or 'gpu 1:' in output_lower or 'gpu 2:' in output_lower or 'gpu 3:' in output_lower:
                            enable_gpu = True
                            log_info("NVIDIA GPU detected - will install onnx-asr with GPU support")
                        else:
                            log_info("NVIDIA driver present but no GPU hardware detected")
                    else:
                        log_info("NVIDIA driver check failed - will use CPU mode")
                except Exception:
                    log_info("GPU detection failed - will use CPU mode")
            
            if not enable_gpu:
                log_info("Installing onnx-asr (CPU-optimized)")

            # Install base requirements first
            requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
            log_info("Installing base dependencies...")
            try:
                run_command([str(pip_bin), 'install', '-r', str(requirements_file)], check=True)
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install base dependencies: {e}"
                log_error(error_msg)
                if cleanup_on_failure:
                    log_info("Cleaning up partial installation...")
                    _cleanup_partial_installation(created_items, pip_bin)
                set_install_state('failed', error_msg)
                return False

            # Install onnx-asr on top (with GPU support if available)
            if not install_onnx_asr(pip_bin, enable_gpu=enable_gpu):
                error_msg = "Failed to install onnx-asr"
                log_error(error_msg)
                if cleanup_on_failure:
                    log_info("Cleaning up partial installation...")
                    _cleanup_partial_installation(created_items, pip_bin)
                set_install_state('failed', error_msg)
                return False

            # Pre-download models so they're ready on first use
            log_info("Downloading ONNX-ASR model and VAD (this may take a moment)...")
            venv_python = VENV_DIR / 'bin' / 'python'
            try:
                # Download and cache the ASR model + Silero VAD
                # This mirrors what happens at runtime but ensures everything is ready
                download_script = '''
import onnx_asr
print("Downloading Parakeet TDT V3 model...", flush=True)
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
print("Downloading Silero VAD...", flush=True)
vad = onnx_asr.load_vad("silero")
print("Models cached successfully", flush=True)
'''
                run_command([str(venv_python), '-c', download_script], check=True)
                log_success("Models downloaded and cached")
            except subprocess.CalledProcessError as e:
                log_warning(f"Model download failed: {e}")
                log_warning("Models will be downloaded on first use instead")
                # Don't fail installation - models can still be downloaded on first use

            # Store requirements hash
            cur_req_hash = compute_file_hash(requirements_file)
            set_state("requirements_hash", cur_req_hash)

            # Installation successful for ONNX-ASR
            set_install_state('completed')
            log_success("ONNX-ASR backend installation completed!")
            return True

        # Setup Python venv (for cpu/nvidia/amd backends)
        venv_existed = VENV_DIR.exists()
        pip_bin = setup_python_venv(force_rebuild=force_rebuild, custom_python=custom_python)
        if (force_rebuild or not venv_existed) and VENV_DIR.exists():
            created_items['venv_created'] = True
            created_items['venv_path'] = str(VENV_DIR)
        
        # Check if dependencies are already installed
        requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
        cur_req_hash = compute_file_hash(requirements_file)
        stored_req_hash = get_state("requirements_hash")
        
        deps_installed = False
        try:
            python_bin = VENV_DIR / 'bin' / 'python'
            result = run_command([
                'timeout', '5s', str(python_bin), '-c',
                'import sounddevice, pywhispercpp'
            ], check=False, capture_output=True, show_output_on_error=False)
            deps_installed = result.returncode == 0
        except Exception:
            pass
        
        # Install pywhispercpp if needed
        if cur_req_hash != stored_req_hash or not stored_req_hash or not deps_installed:
            if not stored_req_hash:
                # First time setup - no stored hash means venv is new
                log_info("Installing Python dependencies...")
            elif cur_req_hash != stored_req_hash:
                # Requirements actually changed
                log_info("Installing Python dependencies (requirements.txt changed)...")
            else:
                # Dependencies missing but hash matches (shouldn't happen often)
                log_info("Installing Python dependencies (dependencies missing)...")

            if enable_cuda or enable_rocm or enable_vulkan:
                # GPU build path: install everything except pywhispercpp first
                log_info("Installing base Python dependencies (excluding pywhispercpp)...")

                # Determine packages to skip
                skip_pygobject = _should_skip_pygobject()

                # Use a writable temp directory instead of system directory
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_req:
                    temp_req_path = Path(temp_req.name)
                    try:
                        with open(requirements_file, 'r', encoding='utf-8') as f_in:
                            for line in f_in:
                                line_stripped = line.strip()
                                # Extract package name for exact matching
                                pkg_name = _extract_package_name(line_stripped)
                                # Skip pywhispercpp (built separately with GPU support)
                                if pkg_name == 'pywhispercpp':
                                    continue
                                # Skip PyGObject if already installed as system package
                                if skip_pygobject and pkg_name == 'pygobject':
                                    continue
                                temp_req.write(line)

                        temp_req.flush()

                        if temp_req_path.stat().st_size > 0:
                            run_command([str(pip_bin), 'install', '-r', str(temp_req_path)],
                                       check=True, verbose=OutputController.get_verbosity().value >= VerbosityLevel.VERBOSE.value)
                    except Exception as e:
                        error_msg = f"Failed to install base Python dependencies: {e}"
                        log_error(error_msg)
                        if cleanup_on_failure:
                            log_info("Cleaning up partial installation...")
                            # Uninstall any partially installed packages
                            try:
                                run_command([str(pip_bin), 'uninstall', '-y'] + created_items['packages_installed'], 
                                          check=False, capture_output=True)
                            except Exception:
                                pass
                        set_install_state('failed', error_msg)
                        return False
                    finally:
                        # Clean up temp file
                        if temp_req_path.exists():
                            temp_req_path.unlink()
                
                # Remove any pre-existing pywhispercpp
                run_command([str(pip_bin), 'uninstall', '-y', 'pywhispercpp'], check=False, capture_output=True)
                
                # Build pywhispercpp with GPU support
                if enable_cuda:
                    if not install_pywhispercpp_cuda(pip_bin):
                        error_msg = "Failed to install pywhispercpp with CUDA support"
                        log_error(error_msg)
                        if cleanup_on_failure:
                            log_info("Cleaning up partial installation...")
                            try:
                                run_command([str(pip_bin), 'uninstall', '-y', 'pywhispercpp'], 
                                          check=False, capture_output=True)
                            except Exception:
                                pass
                        set_install_state('failed', error_msg)
                        return False
                elif enable_rocm:
                    success, should_fallback = install_pywhispercpp_rocm(pip_bin)
                    if not success:
                        if should_fallback:
                            # ROCm build failed - fall back to CPU-only
                            log_warning("ROCm build failed - falling back to CPU-only installation")
                            log_warning("")
                            log_warning(f"ROCm 7.x has known compatibility issues with pywhispercpp v{PYWHISPERCPP_VERSION}")
                            log_warning("See: https://github.com/ggml-org/whisper.cpp/issues/3553")
                            log_warning("")
                            log_warning("Alternatives:")
                            log_warning("  • Use CPU mode (current fallback)")
                            log_warning("  • Use REST API transcription backend (see README)")
                            log_warning("")
                            log_info("Installing pywhispercpp with CPU-only support...")
                            if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                                error_msg = "Failed to install pywhispercpp (CPU-only fallback)"
                                log_error(error_msg)
                                set_install_state('failed', error_msg)
                                return False
                            log_success("pywhispercpp installed (CPU-only mode)")
                        else:
                            error_msg = "Failed to install pywhispercpp with ROCm support"
                            log_error(error_msg)
                            if cleanup_on_failure:
                                log_info("Cleaning up partial installation...")
                                try:
                                    run_command([str(pip_bin), 'uninstall', '-y', 'pywhispercpp'],
                                              check=False, capture_output=True)
                                except Exception:
                                    pass
                            set_install_state('failed', error_msg)
                            return False
                elif enable_vulkan:
                    if not install_pywhispercpp_vulkan(pip_bin):
                        # Vulkan build failed - fall back to CPU-only
                        log_warning("Vulkan build failed - falling back to CPU-only installation")
                        log_info("Installing pywhispercpp with CPU-only support...")
                        if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                            error_msg = "Failed to install pywhispercpp (CPU-only fallback)"
                            log_error(error_msg)
                            set_install_state('failed', error_msg)
                            return False
                        log_success("pywhispercpp installed (CPU-only mode)")
            else:
                # CPU-only path: install everything normally
                if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                    error_msg = "Failed to install pywhispercpp (CPU-only)"
                    log_error(error_msg)
                    set_install_state('failed', error_msg)
                    return False
            
            set_state("requirements_hash", cur_req_hash)
            log_success("Python dependencies installed")
        else:
            log_info("Python dependencies up to date (skipping pip install)")
        
        # Download base model
        if not download_pywhispercpp_model('base'):
            log_warning("Model download failed, but backend installation succeeded")
            # Don't fail the whole installation if model download fails
        
        # Installation successful
        set_install_state('completed')
        log_success(f"{backend_type.upper()} backend installation completed!")
        return True
        
    except KeyboardInterrupt:
        error_msg = "Installation interrupted by user"
        log_error(error_msg)
        set_install_state('failed', error_msg)
        if cleanup_on_failure:
            log_info("Cleaning up partial installation...")
            _cleanup_partial_installation(created_items, pip_bin if 'pip_bin' in locals() else None)
        raise
    except Exception as e:
        error_msg = f"Unexpected error during installation: {e}"
        log_error(error_msg)
        log_debug(f"Full error traceback: {sys.exc_info()}")
        set_install_state('failed', error_msg)
        if cleanup_on_failure:
            log_info("Cleaning up partial installation...")
            _cleanup_partial_installation(created_items, pip_bin if 'pip_bin' in locals() else None)
        return False


def _cleanup_partial_installation(created_items: dict, pip_bin: Optional[Path]):
    """Clean up partial installation on failure"""
    if created_items.get('venv_created') and created_items.get('venv_path'):
        log_info(f"Removing venv at {created_items['venv_path']}")
        try:
            venv_path = Path(created_items['venv_path'])
            if venv_path.exists():
                shutil.rmtree(venv_path, ignore_errors=True)
        except Exception:
            pass
    
    if created_items.get('git_clone_created') and created_items.get('git_clone_path'):
        log_info(f"Removing git clone at {created_items['git_clone_path']}")
        try:
            shutil.rmtree(Path(created_items['git_clone_path']), ignore_errors=True)
        except Exception:
            pass
    
    if pip_bin and created_items.get('packages_installed'):
        log_info("Uninstalling partially installed packages...")
        try:
            run_command([str(pip_bin), 'uninstall', '-y'] + created_items['packages_installed'],
                       check=False, capture_output=True)
        except Exception:
            pass
