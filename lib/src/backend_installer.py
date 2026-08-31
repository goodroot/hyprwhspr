"""
Backend installation module for hyprwhspr
Handles installation of pywhispercpp backends (CPU/NVIDIA/AMD)
"""

import os
import sys
import json
import subprocess
import tempfile
import hashlib
import shutil
import re
import urllib.request
import uuid
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

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

# Shared backend helpers
try:
    from .backend_utils import LOCAL_INSTALL_BACKENDS, vulkaninfo_has_hardware_gpu
except ImportError:
    from backend_utils import LOCAL_INSTALL_BACKENDS, vulkaninfo_has_hardware_gpu

try:
    from .nvidia_probe import responding_gpu_listing
except ImportError:
    from nvidia_probe import responding_gpu_listing

try:
    from . import visualizer_runtime
except ImportError:
    import visualizer_runtime


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
PYWHISPERCPP_PINNED_COMMIT = "f7bf62118c0a33a43cf8aabb58eef16cea5d16c4"

# Pre-built wheel configuration
WHEEL_BASE_URL = "https://github.com/goodroot/hyprwhspr/releases/download/wheels-v2"
WHEEL_CACHE_DIR = USER_BASE / 'wheel-cache'
PYWHISPERCPP_VERSION = "1.5.1"


class DependencyPlanError(RuntimeError):
    """The installed package is missing or has an invalid dependency manifest."""


try:
    from .dependency_manifest import (
        canonical_name as _canonical_package_name,
        fingerprint as _graph_fingerprint,
        option_argument as _graph_option_argument,
        package_name as _graph_package_name,
        parse_graph as _parse_manifest_graph,
        render_filtered as _render_filtered_manifest,
    )
except ImportError:
    from dependency_manifest import (
        canonical_name as _canonical_package_name,
        fingerprint as _graph_fingerprint,
        option_argument as _graph_option_argument,
        package_name as _graph_package_name,
        parse_graph as _parse_manifest_graph,
        render_filtered as _render_filtered_manifest,
    )

try:
    from .dependency_plan import DependencyPlan, PLAN_SPECS as _PLAN_SPECS, plan_key as _dependency_plan_key, resolve as _resolve_plan
except ImportError:
    from dependency_plan import DependencyPlan, PLAN_SPECS as _PLAN_SPECS, plan_key as _dependency_plan_key, resolve as _resolve_plan

def _plan_key(backend: str, provider: Optional[str], accelerated_variant: Optional[str]) -> str:
    return _dependency_plan_key(backend, provider, accelerated_variant, DependencyPlanError)


def _requirements_option_argument(raw: str, short_option: str,
                                  long_option: str) -> Optional[str]:
    """Return a path argument for a pip requirements-file option, if present."""
    return _graph_option_argument(raw, short_option, long_option, DependencyPlanError)


def _manifest_closure(manifest: Path) -> tuple[Path, ...]:
    """Resolve pip -r includes, rejecting missing/cyclic files before mutation."""
    return _parse_manifest_graph(manifest, DependencyPlanError).manifests


def resolve_dependency_plan(backend: str, provider: Optional[str] = None,
                            accelerated_variant: Optional[str] = None) -> DependencyPlan:
    """Fully resolve the authoritative dependency plan without changing the host."""
    return _resolve_plan(Path(HYPRWHSPR_ROOT), backend, provider,
                         accelerated_variant, DependencyPlanError)


def dependency_manifests(backend: str, provider: Optional[str] = None) -> list[Path]:
    """Compatibility wrapper returning the selected manifest include closure."""
    return list(resolve_dependency_plan(backend, provider).manifests)


def dependency_manifest_hash(manifests: list[Path]) -> str:
    """Hash manifest names and contents so selected dependency state is stable."""
    return _graph_fingerprint(manifests)


def _safe_decode(output) -> str:
    """Safely decode output from run_command which may be string or bytes."""
    if isinstance(output, bytes):
        return output.decode('utf-8', errors='ignore')
    return output


# Maximum Python version compatible with ML packages (onnxruntime, etc.)
MAX_COMPATIBLE_PYTHON = (3, 14)
SYSTEM_PYTHON_CANDIDATES = (
    '/usr/bin/python3',
    '/usr/bin/python',
    '/bin/python3',
    '/bin/python',
    '/usr/local/bin/python3',
    '/usr/local/bin/python',
)


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
    1. Ordered unversioned system candidates if compatible
    2. Versioned Python 3 executables from max_version down to 3.11
    3. The current interpreter, but only when it is not environment-managed
    4. Error with actionable guidance

    Args:
        max_version: Maximum allowed Python version as (major, minor) tuple

    Returns:
        Tuple of (python_path, description)

    Raises:
        SystemExit if no compatible Python found
    """
    for candidate in SYSTEM_PYTHON_CANDIDATES:
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            continue
        version = _get_python_version(candidate)
        if version and version <= max_version:
            return candidate, f"Python {version[0]}.{version[1]}"

    # Rolling distributions can advance /usr/bin/python3 before ML wheels catch
    # up while retaining a separately installed compatible interpreter.
    if max_version[0] == 3:
        for minor in range(max_version[1], 10, -1):
            for prefix in ('/usr/bin', '/usr/local/bin'):
                candidate = f'{prefix}/python3.{minor}'
                if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
                    continue
                version = _get_python_version(candidate)
                if version and version <= max_version:
                    return candidate, f"Python {version[0]}.{version[1]}"

    current_version = _get_python_version(sys.executable)
    if not _current_python_is_managed() and current_version and current_version <= max_version:
        return sys.executable, f"Python {current_version[0]}.{current_version[1]}"

    if _current_python_is_managed():
        log_error("Only an activated or version-manager Python is available.")
        log_error("Deactivate the environment or pass --python /path/to/system/python explicitly.")
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


def _current_python_is_managed() -> bool:
    """Return whether implicit use of the running interpreter would be unsafe."""
    executable = str(Path(sys.executable).resolve())
    user_home = str(Path.home().resolve()) + os.sep
    return bool(
        os.environ.get('VIRTUAL_ENV')
        or sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
        or _check_mise_active()
        or any(marker in executable for marker in ('/.local/share/mise/', '/.pyenv/', '/.asdf/'))
        or executable.startswith(user_home)
    )


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


# ==================== Pre-built Wheel Support ====================

def _detect_venv_python_version() -> str:
    """Detect Python version in the venv (e.g., '3.11')."""
    venv_python = VENV_DIR / 'bin' / 'python'
    if venv_python.exists():
        # _get_python_version handles interpreters that print --version to stderr.
        version = _get_python_version(str(venv_python))
        if version:
            return f"{version[0]}.{version[1]}"
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _detect_cuda_version() -> Optional[str]:
    """Detect installed CUDA version from nvcc or nvidia-smi"""
    # Try nvcc first (more reliable for build compatibility)
    nvcc_path = _cuda_nvcc_path()

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
    """CUDA wheel variant for the detected CUDA version, or None to source-build.

    We self-host CUDA wheels only; CPU installs come from PyPI.
    """
    if not cuda_version:
        return None

    major = int(cuda_version.split('.')[0])
    if major == 12:  # one 12.x wheel serves all CUDA 12
        return "cuda12"

    # CUDA 11 (EOL) and 13 (not yet CI-buildable) fall back to source.
    log_info(f"CUDA {cuda_version} detected - no pre-built wheel available, building from source")
    return None


def _get_wheel_filename(python_version: str, variant: str, for_download: bool = True) -> str:
    """Construct wheel filename for given Python version and variant

    Args:
        python_version: e.g., '3.11'
        variant: 'cuda12' (CUDA wheels are the only variant we self-host)
        for_download: If True, include variant suffix (for GitHub). If False, standard pip format.
    """
    # Python 3.11 -> cp311
    py_tag = f"cp{python_version.replace('.', '')}"
    base = f"pywhispercpp-{PYWHISPERCPP_VERSION}-{py_tag}-{py_tag}-linux_x86_64"
    if for_download:
        # GitHub release filename: pywhispercpp-<version>-cp311-cp311-linux_x86_64+cuda12.whl
        return f"{base}+{variant}.whl"
    else:
        # Standard pip-compatible filename: pywhispercpp-<version>-cp311-cp311-linux_x86_64.whl
        return f"{base}.whl"


def download_pywhispercpp_wheel(variant: Optional[str] = None) -> Optional[Path]:
    """
    Download pre-built pywhispercpp wheel if available.

    Args:
        variant: Optional variant override ('cuda12'). If None, auto-detects
                 based on system CUDA. CPU installs come from PyPI, not here.

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


def commit_dependency_state(plan: DependencyPlan):
    """Commit all dependency identity fields with one atomic replacement."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = get_all_state()
    data.update({
        'dependency_plan_fingerprint': plan.fingerprint,
        # Retained for migration compatibility with older releases.
        'dependency_manifest_hash': plan.fingerprint,
        'dependency_family': plan.family,
    })
    if plan.family == 'pywhispercpp' and plan.accelerated_variant:
        data['installed_backend'] = plan.accelerated_variant
    temp_path = STATE_DIR / f'.{STATE_FILE.name}.{uuid.uuid4().hex}.tmp'
    try:
        temp_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(temp_path, STATE_FILE)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


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

    stored_hash = get_state(f"model_hash_{model_file.name}")
    if stored_hash and compute_file_hash(model_file) == stored_hash:
        return True

    # No stored hash: size floor catches truncated downloads and HF error
    # pages (smallest real model, quantized tiny, is ~31MB)
    return model_file.stat().st_size > 20_000_000


# ==================== Helper Functions ====================

_MIN_CUDA_GCC_MAJOR = 6  # CUDA's host_config.h rejects GCC versions below 6.


def _cuda_nvcc_path() -> Optional[str]:
    """Return an existing nvcc path using the same lookup order as CUDA setup."""
    configured = os.environ.get('CUDACXX')
    if configured:
        configured_path = shutil.which(configured) if '/' not in configured else configured
        if configured_path and Path(configured_path).exists():
            return configured_path

    path_nvcc = shutil.which('nvcc')
    if path_nvcc:
        return path_nvcc

    for candidate in ('/opt/cuda/bin/nvcc', '/usr/bin/nvcc'):
        if Path(candidate).exists():
            return candidate
    return None


def _cuda_host_config_paths():
    """Yield likely host_config.h paths for the active CUDA toolkit."""
    nvcc_path = _cuda_nvcc_path()
    if not nvcc_path:
        return

    toolkit_root = Path(nvcc_path).resolve().parent.parent
    yield toolkit_root / 'include' / 'crt' / 'host_config.h'
    yield toolkit_root / 'targets' / 'x86_64-linux' / 'include' / 'crt' / 'host_config.h'
    yield from sorted(toolkit_root.glob('targets/*/include/crt/host_config.h'))


def _cuda_max_supported_gcc_version() -> Optional[Tuple[int, Optional[int]]]:
    """Read CUDA's GCC (major, optional minor) ceiling, or None if unknown."""
    for host_config in _cuda_host_config_paths():
        try:
            contents = host_config.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue

        # Only trust the preprocessor guard that emits NVCC's unsupported GNU
        # version error.  host_config.h also contains unrelated GCC feature
        # checks, which are not compatibility ceilings.
        guards = re.finditer(
            r'^\s*#\s*(?:if|elif)\s+(?P<condition>[^\n]*\b__GNUC__\s*>\s*\d+[^\n]*)$',
            contents,
            re.MULTILINE,
        )
        for guard in guards:
            body = re.split(
                r'^\s*#\s*(?:if|elif|else|endif)\b',
                contents[guard.end():],
                maxsplit=1,
                flags=re.MULTILINE,
            )[0]
            if 'unsupported GNU version' not in body:
                continue
            condition = guard.group('condition')
            major_match = re.search(r'__GNUC__\s*>\s*(\d+)', condition)
            if not major_match:
                continue
            major = int(major_match.group(1))
            if major < _MIN_CUDA_GCC_MAJOR:
                continue
            minor_match = re.search(
                r'__GNUC__\s*==\s*' + re.escape(major_match.group(1)) +
                r'.*?__GNUC_MINOR__\s*>\s*(\d+)',
                condition,
            )
            return major, int(minor_match.group(1)) if minor_match else None
    return None


def _compiler_version(compiler: str) -> Optional[Tuple[int, int]]:
    """Return a compiler's reported (major, minor) version, if available."""
    try:
        result = run_command([compiler, '-dumpfullversion'], check=False, capture_output=True)
        if result and result.returncode == 0:
            match = re.match(r'\s*(\d+)(?:\.(\d+))?', _safe_decode(result.stdout))
            if match:
                return int(match.group(1)), int(match.group(2) or 0)
    except Exception:
        pass
    return None


def _is_cuda_gcc_compatible(
    version: Optional[Tuple[int, int]], max_version: Tuple[int, Optional[int]]
) -> bool:
    """Whether a GCC version fits CUDA's parsed host-compiler ceiling."""
    if version is None or version[0] < _MIN_CUDA_GCC_MAJOR:
        return False
    max_major, max_minor = max_version
    return version[0] < max_major or (
        version[0] == max_major and (max_minor is None or version[1] <= max_minor)
    )


def detect_cuda_host_compiler() -> Optional[str]:
    """Select CUDA's newest supported GCC, returning None when none is installed.

    An unknown toolkit ceiling retains the system g++ optimistically; a known
    ceiling without a compatible compiler returns None so CUDA setup uses CPU.
    """
    # Allow explicit override
    cuda_host = os.environ.get('HYPRWHSPR_CUDA_HOST')
    if cuda_host and Path(cuda_host).exists() and os.access(cuda_host, os.X_OK):
        return cuda_host
    if cuda_host:
        log_warning(f"HYPRWHSPR_CUDA_HOST is not an executable file: {cuda_host}; auto-detecting")

    default_compiler = shutil.which('g++')
    max_version = _cuda_max_supported_gcc_version()
    if max_version is None:
        if default_compiler:
            log_warning(
                "Could not determine CUDA's supported GCC version; using system g++ "
                "(set HYPRWHSPR_CUDA_HOST to override)"
            )
        return default_compiler

    max_major, max_minor = max_version
    max_display = f"{max_major}.{max_minor}" if max_minor is not None else str(max_major)

    if default_compiler:
        default_version = _compiler_version(default_compiler)
        if _is_cuda_gcc_compatible(default_version, max_version):
            return default_compiler

    for major in range(max_major, _MIN_CUDA_GCC_MAJOR - 1, -1):
        candidate = shutil.which(f'g++-{major}')
        if not candidate:
            continue
        candidate_version = _compiler_version(candidate)
        if _is_cuda_gcc_compatible(candidate_version, max_version):
            return candidate

    log_warning(f"No installed g++ is compatible with CUDA's GCC <= {max_display} requirement")
    log_warning(f"Install a GCC {max_major} toolchain (on Arch: yay -S gcc{max_major} gcc{max_major}-libs)")
    log_warning("Or set HYPRWHSPR_CUDA_HOST to a compatible g++ executable")
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

def _detect_nvidia_gpu_listing() -> Optional[str]:
    """Return nvidia-smi's GPU listing when real NVIDIA hardware responds."""
    def runner(command, **kwargs):
        kwargs.pop('text', None)
        result = run_command(command, verbose=False, **kwargs)
        if result is not None:
            result.stdout = _safe_decode(result.stdout) if result.stdout else ''
        return result

    return responding_gpu_listing(runner=runner, which=shutil.which)

def setup_nvidia_support() -> bool:
    """Setup NVIDIA/CUDA support. Returns True if CUDA is available."""
    log_info("GPU check…")

    if not _detect_nvidia_gpu_listing():
        log_info("No responding NVIDIA GPU detected (CPU mode)")
        return False

    log_success("NVIDIA GPU detected")
    
    # Check for nvcc
    nvcc_path = _cuda_nvcc_path()
    
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
        nvcc_path_after_install = _cuda_nvcc_path()
        
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

    # 1. Check for NVIDIA GPU using the same hardware/driver probe as setup.
    nvidia_listing = _detect_nvidia_gpu_listing()
    if nvidia_listing:
        log_info(f"[GPU Detection] ✓ NVIDIA GPU confirmed: {nvidia_listing.splitlines()[0][:60]}")
        return 'nvidia'

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
                summary = _safe_decode(result.stdout)
                if vulkaninfo_has_hardware_gpu(summary):
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
                log_warning("Could not install Vulkan packages, checking for an existing Vulkan setup")
        except Exception as e:
            log_warning(f"Could not install Vulkan dependencies ({e}), checking for an existing Vulkan setup")
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
        summary = _safe_decode(result.stdout)
        if not vulkaninfo_has_hardware_gpu(summary):
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

    # Optional visualizer GUI deps — best-effort, never fatal (see helper docstring).
    install_visualizer_deps(pip_bin)

    return pip_bin


_IMPORT_PROBE = r'''
import contextlib, importlib.metadata, importlib.util, io, json, sys, traceback
name = sys.argv[1]
data = {"import_name": name}
try:
    spec = importlib.util.find_spec(name)
    data["module_origin"] = getattr(spec, "origin", None)
except BaseException:
    data["module_origin"] = None
    data["find_spec_traceback"] = traceback.format_exc()
try:
    spec = importlib.util.find_spec("numpy")
    data["numpy_origin"] = getattr(spec, "origin", None)
    import numpy
    data["numpy_version"] = numpy.__version__
except BaseException:
    data["numpy_traceback"] = traceback.format_exc()
try:
    data["distributions"] = sorted(set(
        importlib.metadata.packages_distributions().get(name, [])
    ))
except BaseException:
    data["distributions"] = []
stdout, stderr = io.StringIO(), io.StringIO()
if len(sys.argv) > 2 and sys.argv[2] == "snapshot":
    data["ok"] = data.get("module_origin") is not None
else:
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            __import__(name)
        data["ok"] = True
    except BaseException:
        data["ok"] = False
        data["traceback"] = traceback.format_exc()
data["import_stdout"] = stdout.getvalue()
data["import_stderr"] = stderr.getvalue()
data["prefix"] = sys.prefix
data["base_prefix"] = sys.base_prefix
print(json.dumps(data), flush=True)
sys.exit(0 if data["ok"] else 1)
'''

_NUMPY_ABI_PATTERNS = (
    re.compile(r'_ARRAY_API not found', re.I),
    re.compile(r'compiled using NumPy 1\.x cannot be run in NumPy 2', re.I),
    re.compile(r'numpy\.dtype size changed', re.I),
    re.compile(r'(?:compiled|built).*(?:NumPy|numpy).*(?:API|ABI).*(?:version|mismatch)', re.I | re.S),
    re.compile(r'(?:NumPy|numpy).*(?:API|ABI) version.*(?:mismatch|incompatible)', re.I | re.S),
    re.compile(r'module compiled against API version .*but this version of numpy is', re.I),
    re.compile(r'Numba needs NumPy .* or less', re.I),
)
_ABI_REPAIR_ALLOWLIST = frozenset({
    'numba', 'pandas', 'scikit-learn', 'scipy', 'sounddevice', 'soxr',
})


@dataclass
class ImportProbe:
    import_name: str
    ok: bool
    module_origin: Optional[str] = None
    stdout: str = ''
    stderr: str = ''
    traceback: str = ''
    numpy_version: Optional[str] = None
    numpy_origin: Optional[str] = None
    distributions: tuple[str, ...] = ()
    timed_out: bool = False

    def evidence(self) -> str:
        return '\n'.join(part for part in (
            self.stdout, self.stderr, self.traceback) if part)


@dataclass
class DependencyVerification:
    ok: bool
    combined_stdout: str = ''
    combined_stderr: str = ''
    failures: list[ImportProbe] = field(default_factory=list)
    combined_only_failure: bool = False
    timed_out: bool = False
    repair_error: str = ''


def _has_numpy_abi_signature(text: str) -> bool:
    return any(pattern.search(text or '') for pattern in _NUMPY_ABI_PATTERNS)


def _manifest_requirement_specs(plan: DependencyPlan) -> dict[str, Optional[str]]:
    """Return unambiguous package requirements exactly as declared by manifests."""
    requirements: dict[str, Optional[str]] = {}
    for manifest in plan.manifests:
        for raw in manifest.read_text(encoding='utf-8').splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line or line.startswith(('-', 'http:', 'https:')):
                continue
            name = _graph_package_name(line)
            if name:
                previous = requirements.get(name)
                if previous is not None and previous != line:
                    requirements[name] = None
                elif name not in requirements:
                    requirements[name] = line
    return requirements


def _manifest_package_names(plan: DependencyPlan) -> set[str]:
    return set(_manifest_requirement_specs(plan))


def _distribution_for_import(probe: ImportProbe, manifest_names: set[str]) -> Optional[str]:
    """Map one import to exactly one distribution selected by the manifest."""
    candidates = {
        _canonical_package_name(name) for name in probe.distributions if name
    } & manifest_names
    return next(iter(candidates)) if len(candidates) == 1 else None


def _parse_import_probe(name: str, result) -> ImportProbe:
    process_stdout = _safe_decode(getattr(result, 'stdout', '') or '')
    process_stderr = _safe_decode(getattr(result, 'stderr', '') or '')
    payload: Dict[str, Any] = {}
    payload_line = None
    stdout_lines = process_stdout.splitlines()
    for index in range(len(stdout_lines) - 1, -1, -1):
        line = stdout_lines[index]
        try:
            candidate = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict) and candidate.get('import_name') == name:
            payload = candidate
            payload_line = index
            break
    if payload_line is not None:
        del stdout_lines[payload_line]
    unredirected_stdout = '\n'.join(stdout_lines)
    tracebacks = '\n'.join(filter(None, (
        payload.get('find_spec_traceback', ''),
        payload.get('numpy_traceback', ''),
        payload.get('traceback', ''),
    )))
    return ImportProbe(
        import_name=name,
        ok=bool(payload.get('ok', result.returncode == 0)),
        module_origin=payload.get('module_origin'),
        stdout=(payload.get('import_stdout', '') + unredirected_stdout),
        stderr=(payload.get('import_stderr', '') + process_stderr),
        traceback=tracebacks,
        numpy_version=payload.get('numpy_version'),
        numpy_origin=payload.get('numpy_origin'),
        distributions=tuple(payload.get('distributions') or ()),
        timed_out=result.returncode == 124,
    )


def _probe_required_import(name: str, import_module: bool = True) -> ImportProbe:
    mode = [] if import_module else ['snapshot']
    result = run_command([
        'timeout', '180s', str(VENV_DIR / 'bin' / 'python'), '-c',
        _IMPORT_PROBE, name, *mode,
    ], check=False, capture_output=True, show_output_on_error=False)
    return _parse_import_probe(name, result)


def _verify_dependency_plan_detailed(plan: DependencyPlan) -> DependencyVerification:
    imports = ', '.join(plan.required_imports)
    combined = run_command([
        'timeout', '60s', str(VENV_DIR / 'bin' / 'python'), '-c', f'import {imports}'
    ], check=False, capture_output=True, show_output_on_error=False)
    combined_timed_out = combined.returncode == 124
    if combined_timed_out:
        combined = run_command([
            'timeout', '180s', str(VENV_DIR / 'bin' / 'python'), '-c', f'import {imports}'
        ], check=False, capture_output=True, show_output_on_error=False)
        combined_timed_out = combined.returncode == 124
    if combined.returncode == 0:
        return DependencyVerification(ok=True)

    probes = [_probe_required_import(name) for name in plan.required_imports]
    failures = [probe for probe in probes if not probe.ok]
    # A combined timeout is inconclusive. Successful isolated imports are enough
    # to retain the environment instead of discarding a large, valid installation.
    if combined_timed_out and not failures:
        return DependencyVerification(ok=True)
    return DependencyVerification(
        ok=False,
        combined_stdout=_safe_decode(getattr(combined, 'stdout', '') or ''),
        combined_stderr=_safe_decode(getattr(combined, 'stderr', '') or ''),
        failures=failures,
        combined_only_failure=not failures and not combined_timed_out,
        timed_out=combined_timed_out or any(probe.timed_out for probe in failures),
    )


def _format_dependency_diagnostic(plan: DependencyPlan,
                                  verification: DependencyVerification,
                                  snapshot: Optional[list[ImportProbe]] = None) -> str:
    lines = ['Dependency verification failed.',
             'Required imports: ' + ', '.join(plan.required_imports)]
    if verification.timed_out:
        lines.append('Dependency verification timed out after a 60-second probe and one 180-second retry.')
    elif verification.combined_only_failure:
        lines.append('Every isolated import succeeded; the combined probe failed (possible import-order interaction).')
    before_numpy = next((probe for probe in (snapshot or []) if probe.numpy_origin), None)
    after_numpy = next((probe for probe in verification.failures if probe.numpy_origin), None)
    if before_numpy:
        lines.append(
            f'NumPy before install: {before_numpy.numpy_version or "unknown"} '
            f'at {before_numpy.numpy_origin}'
        )
    if after_numpy:
        lines.append(
            f'NumPy after install: {after_numpy.numpy_version or "unknown"} '
            f'at {after_numpy.numpy_origin}'
        )
    if verification.combined_stdout:
        lines.append('Combined stdout:\n' + verification.combined_stdout.rstrip())
    if verification.combined_stderr:
        lines.append('Combined stderr:\n' + verification.combined_stderr.rstrip())
    for probe in verification.failures:
        lines.extend([
            f'Import: {probe.import_name}',
            f'Origin: {probe.module_origin or "unknown"}',
            f'Distribution candidates: {", ".join(probe.distributions) or "none"}',
            f'NumPy: {probe.numpy_version or "unknown"} ({probe.numpy_origin or "unknown"})',
        ])
        if probe.stdout:
            lines.append('stdout:\n' + probe.stdout.rstrip())
        if probe.stderr:
            lines.append('stderr:\n' + probe.stderr.rstrip())
        if probe.traceback:
            lines.append('traceback:\n' + probe.traceback.rstrip())
    if verification.repair_error:
        lines.append('Repair pip failure:\n' + verification.repair_error.rstrip())
    return '\n'.join(lines)


def _is_system_origin(origin: Optional[str]) -> bool:
    """Return whether an import is inherited from anywhere outside the venv."""
    if not origin:
        return False
    try:
        path = Path(origin).resolve()
        path.relative_to(VENV_DIR.resolve())
        return False
    except ValueError:
        return True
    except OSError:
        return False


def _has_missing_compiled_import(text: str) -> bool:
    return bool(re.search(
        r'(?:ModuleNotFoundError|ImportError):|No module named [\'\"]?_|'
        r'OSError:.*PortAudio|PortAudio.*not found',
        text or '', re.I | re.S,
    ))


def _numpy_needs_relocation(verification: DependencyVerification) -> bool:
    return any(
        _has_numpy_abi_signature(probe.evidence())
        and _is_system_origin(probe.numpy_origin)
        for probe in verification.failures
    )


def _repairable_distributions(plan: DependencyPlan,
                              verification: DependencyVerification) -> list[str]:
    # One repair round is intentional. If relocation exposes a second, opposite
    # ABI mismatch, the post-repair verification records it for manual resolution.
    requirement_specs = _manifest_requirement_specs(plan)
    manifest_names = set(requirement_specs)
    repairs = set()
    for probe in verification.failures:
        distribution = _distribution_for_import(probe, manifest_names)
        eligible_failure = (
            _has_numpy_abi_signature(probe.evidence())
            or _has_missing_compiled_import(probe.evidence())
        )
        if (distribution in _ABI_REPAIR_ALLOWLIST
                and requirement_specs.get(distribution)
                and eligible_failure
                and _is_system_origin(probe.module_origin)):
            repairs.add(distribution)
    ordered = sorted(repairs)
    if (_numpy_needs_relocation(verification)
            and requirement_specs.get('numpy')):
        ordered.insert(0, 'numpy')
    return ordered


def _repair_requirement_specs(plan: DependencyPlan,
                              distributions: list[str]) -> list[str]:
    specs = _manifest_requirement_specs(plan)
    return [specs[name] for name in distributions if specs.get(name)]


def _verify_and_repair_dependency_plan(plan: DependencyPlan, pip_bin: Path) -> DependencyVerification:
    """Verify required imports and perform at most one allowlisted repair round."""
    verification = _verify_dependency_plan_detailed(plan)
    repairs = _repairable_distributions(plan, verification)
    if not repairs:
        return verification

    log_warning('Relocating incompatible inherited packages into the venv: '
                + ', '.join(repairs))
    # --ignore-installed deliberately relocates the selected requirement and
    # its resolved dependency closure into the venv. Preserve manifest pins.
    # Conflicts with already-installed pins are intentionally caught by the
    # single post-repair verification rather than adding another repair round.
    repair_specs = _repair_requirement_specs(plan, repairs)
    try:
        repair_result = run_command(
            [str(pip_bin), 'install', '--ignore-installed', *repair_specs],
            check=False, capture_output=True, show_output_on_error=False)
        if repair_result.returncode == 0:
            return _verify_dependency_plan_detailed(plan)
        verification.repair_error = '\n'.join(filter(None, (
            _safe_decode(getattr(repair_result, 'stdout', '') or ''),
            _safe_decode(getattr(repair_result, 'stderr', '') or ''),
        ))) or f'pip exited with status {repair_result.returncode}'
    except Exception as exc:
        verification.repair_error = str(exc)
    return verification


def _snapshot_inherited_dependencies(plan: DependencyPlan) -> list[ImportProbe]:
    """Capture pre-install origins for diagnostics; never mutate dependencies."""
    snapshots = [
        _probe_required_import(name, import_module=False)
        for name in plan.required_imports
    ]
    for probe in snapshots:
        log_debug(
            f'Pre-install import snapshot: {probe.import_name}: '
            f'{probe.module_origin or "not found"}; NumPy '
            f'{probe.numpy_version or "unknown"} at {probe.numpy_origin or "unknown"}'
        )
    return snapshots


def _verify_dependency_plan(plan: DependencyPlan) -> bool:
    """Compatibility wrapper for callers that only need a boolean result."""
    return _verify_dependency_plan_detailed(plan).ok


class VenvTransaction:
    """Own replacement of one venv until verification commits it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.backup = self.path.with_name(
            f'{self.path.name}.rollback-{uuid.uuid4().hex}'
        )
        self.had_old = False
        self.active = False

    def begin(self):
        if self.active:
            return self
        self.had_old = self.path.exists()
        if self.had_old:
            self.path.rename(self.backup)
        self.active = True
        return self

    def commit(self):
        if not self.active:
            return
        if self.backup.exists():
            shutil.rmtree(self.backup)
        self.active = False

    def rollback(self):
        if not self.active:
            return
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        if self.had_old and self.backup.exists():
            self.backup.rename(self.path)
        self.active = False


def execute_dependency_plan(plan: DependencyPlan, custom_python: Optional[str] = None,
                            force_rebuild: bool = False) -> Path:
    """Install and verify a plan transactionally, restoring a usable old venv on failure."""
    # Re-resolve before any filesystem mutation; callers cannot pass a stale/malformed plan.
    checked = _manifest_closure(plan.manifest)
    if checked != plan.manifests or dependency_manifest_hash(list(checked)) != plan.fingerprint:
        raise DependencyPlanError('Dependency manifests changed after planning; retry setup')

    plan_fingerprint = get_state('dependency_plan_fingerprint')
    stored = plan_fingerprint or get_state('dependency_manifest_hash')
    if not force_rebuild and stored == plan.fingerprint and VENV_DIR.exists() and _verify_dependency_plan(plan):
        if not plan_fingerprint:
            try:
                commit_dependency_state(plan)
            except Exception as exc:
                log_warning(f'Could not migrate dependency state: {exc}')
        return VENV_DIR / 'bin' / 'pip'

    transaction = VenvTransaction(VENV_DIR).begin()
    try:
        pip_bin = setup_python_venv(custom_python=custom_python)
        snapshot = _snapshot_inherited_dependencies(plan)
        run_command([str(pip_bin), 'install', '-r', str(plan.manifest)], check=True)
        verification = _verify_and_repair_dependency_plan(plan, pip_bin)
        if not verification.ok:
            diagnostic = _format_dependency_diagnostic(plan, verification, snapshot)
            log_error(diagnostic)
            set_install_state('failed', diagnostic)
            raise RuntimeError(diagnostic)
    except BaseException:
        transaction.rollback()
        raise

    transaction.commit()
    try:
        commit_dependency_state(plan)
    except Exception as exc:
        # The verified environment is more valuable than bookkeeping. The next setup
        # verifies it again and migrates state to the new fingerprint.
        log_warning(f'Could not record dependency state: {exc}')
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


def install_visualizer_deps(pip_bin) -> bool:
    """Best-effort install of the optional mic-osd visualizer GUI deps.

    These GUI bindings (PyGObject, pycairo) build against system libraries and can
    fail on some distros — e.g. PyGObject >= 3.50 needs girepository-2.0, which is
    not packaged on Ubuntu 24.04. They live in requirements-visualizer.txt (not the
    core requirements.txt) and are installed here separately and non-fatally, so a
    GUI build failure only disables the animated overlay and never blocks the core
    dictation runtime. Skipped when PyGObject is already importable from system
    packages (the venv uses --system-site-packages).
    """
    bindings_ok = _should_skip_pygobject()
    if not bindings_ok:
        req = Path(HYPRWHSPR_ROOT) / 'requirements-visualizer.txt'
        if not req.exists():
            bindings_ok = True
        else:
            log_info("Installing optional mic-osd visualizer deps (best-effort)…")
            try:
                run_command([str(pip_bin), 'install', '-r', str(req)], check=True)
                bindings_ok = True
            except subprocess.CalledProcessError as e:
                log_warning(
                    "mic-osd visualizer deps could not be built — the animated overlay "
                    f"will be unavailable (core dictation is unaffected): {e}"
                )

    python_bin = Path(pip_bin).parent / 'python'
    if bindings_ok:
        install_gtk4_layer_shell_runtime(python_bin)
    return bindings_ok


_VISUALIZER_IMPORT_PROBE = (
    "import cairo, gi;"
    "gi.require_version('Gtk', '4.0');"
    "gi.require_version('Gtk4LayerShell', '1.0');"
    "from gi.repository import Gtk, Gtk4LayerShell"
)


def _visualizer_runtime_imports(python_bin: Path, env=None) -> bool:
    try:
        result = run_command(
            [str(python_bin), '-c', _VISUALIZER_IMPORT_PROBE],
            check=False, capture_output=True, env=env,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _download_visualizer_runtime(download_dir: Path) -> tuple[Path, str]:
    asset = download_dir / visualizer_runtime.GTK4_LAYER_SHELL_ASSET
    base = visualizer_runtime.GTK4_LAYER_SHELL_BASE_URL
    _download_bounded_file(
        f"{base}/{visualizer_runtime.GTK4_LAYER_SHELL_ASSET}", asset, 16 * 1024 * 1024
    )
    expected = visualizer_runtime.GTK4_LAYER_SHELL_SHA256.lower()
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise RuntimeError('pinned runtime checksum is invalid')
    actual = hashlib.sha256(asset.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('downloaded runtime does not match its pinned checksum')
    return asset, actual


def _download_bounded_file(url: str, destination: Path, maximum_bytes: int) -> None:
    """Download a small release asset with per-operation stall protection."""
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, destination.open('wb') as output:
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise RuntimeError(f'download exceeded the {maximum_bytes}-byte safety limit')
            output.write(block)


def _extract_visualizer_runtime(archive: Path, destination: Path, checksum: str) -> None:
    required = {
        'lib/libgtk4-layer-shell.so.0',
        'lib/girepository-1.0/Gtk4LayerShell-1.0.typelib',
        'LICENSE',
        'manifest.json',
    }
    with tarfile.open(archive, 'r:gz') as bundle:
        archive_members = bundle.getmembers()
        members = {member.name.lstrip('./'): member for member in archive_members}
        if (
            len(archive_members) != len(required)
            or set(members) != required
            or any(not member.isfile() or member.size > 16 * 1024 * 1024 for member in members.values())
        ):
            raise RuntimeError('runtime archive contains an unexpected or unsafe layout')
        for relative, member in members.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f'could not read {relative} from runtime archive')
            with source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    if (
        manifest.get('version') != visualizer_runtime.GTK4_LAYER_SHELL_VERSION
        or manifest.get('commit') != visualizer_runtime.GTK4_LAYER_SHELL_COMMIT
    ):
        raise RuntimeError('runtime manifest does not match the installer contract')
    (destination / '.sha256').write_text(checksum + '\n', encoding='utf-8')


def install_gtk4_layer_shell_runtime(python_bin: Path) -> bool:
    """Best-effort install of the Noble x86_64 app-private layer-shell runtime."""
    if _visualizer_runtime_imports(python_bin):
        return True
    if not visualizer_runtime.is_noble_x86_64():
        return False
    target = visualizer_runtime.runtime_dir()
    if visualizer_runtime.is_complete() and _visualizer_runtime_imports(
        python_bin, visualizer_runtime.bundled_environment()
    ):
        return True

    log_info('Installing optional bundled gtk4-layer-shell runtime…')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gtk4-layer-shell-', dir=target.parent))
    previous = target.with_name(f'.{target.name}.previous')
    try:
        with tempfile.TemporaryDirectory() as download_tmp:
            archive, checksum = _download_visualizer_runtime(Path(download_tmp))
            _extract_visualizer_runtime(archive, staging, checksum)
        if not _visualizer_runtime_imports(
            python_bin, visualizer_runtime.environment_for(staging)
        ):
            raise RuntimeError('the bundled runtime failed its import check')
        if previous.exists():
            shutil.rmtree(previous)
        if target.exists():
            target.replace(previous)
        staging.replace(target)
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)
        log_success('Bundled gtk4-layer-shell runtime installed')
        try:
            versions_root = visualizer_runtime.versions_dir()
            if target.parent.parent == versions_root:
                for old_version in versions_root.iterdir():
                    if old_version.is_dir() and old_version != target.parent:
                        shutil.rmtree(old_version, ignore_errors=True)
        except OSError as exc:
            log_debug(f'Could not remove an older visualizer runtime: {exc}')
        return True
    except Exception as exc:
        if previous.exists() and not target.exists():
            previous.replace(target)
        log_warning(
            'Bundled gtk4-layer-shell could not be installed; mic-osd will use '
            f'notifications instead (core dictation is unaffected): {exc}'
        )
        return False
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


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
    return _graph_package_name(requirement_line)


def _filter_requirements(requirements_file: Path, skip_packages: list) -> Path:
    """
    Create a filtered, self-contained requirements file. Requirement includes
    are expanded in place and relative constraint paths are made absolute, so
    pip can safely consume the result from a temporary directory.
    Returns path to temp file (caller must clean up).
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    try:
        _render_filtered_manifest(requirements_file, temp_file, skip_packages, DependencyPlanError)
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
    """Install CPU-only pywhispercpp from PyPI (PyPI ships CPU wheels for all
    supported Pythons; we self-host CUDA only)."""
    log_info("Installing pywhispercpp (CPU-only)...")

    skip_packages = []
    if _should_skip_pygobject():
        skip_packages.append('PyGObject')

    temp_req_path = None
    try:
        if skip_packages:
            temp_req_path = _filter_requirements(requirements_file, skip_packages)
            install_file = temp_req_path
        else:
            install_file = requirements_file

        # --only-binary for pywhispercpp: fail loudly if PyPI lacks a wheel for
        # this interpreter rather than silently attempting a source build.
        run_command([str(pip_bin), 'install', '--only-binary=pywhispercpp', '-r', str(install_file)], check=True)

        # pip can exit 0 without pywhispercpp present (e.g. filtered out of
        # requirements); confirm the backend is importable before claiming success.
        venv_python = pip_bin.parent / 'python'
        verify = run_command([str(venv_python), '-c', 'import pywhispercpp'], check=False, capture_output=True)
        if verify.returncode != 0:
            log_error("pip succeeded but 'import pywhispercpp' failed — backend not installed")
            return False

        log_success("pywhispercpp installed (CPU-only mode)")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install pywhispercpp (CPU-only): {e}")
        return False
    finally:
        if temp_req_path and temp_req_path.exists():
            temp_req_path.unlink()


def _prepare_pywhispercpp_sources() -> bool:
    """Clean stale build artifacts and put the pywhispercpp checkout on the pinned commit.

    Returns False if the tree does not end up on PYWHISPERCPP_PINNED_COMMIT: building from
    a stale checkout silently reintroduces whatever the pin bump was meant to fix.
    """
    verbosity = OutputController.get_verbosity()
    verbose = verbosity.value >= VerbosityLevel.VERBOSE.value

    # Clean build artifacts if they exist (to avoid Python version mismatches)
    if PYWHISPERCPP_SRC_DIR.exists():
        log_info("Cleaning existing build artifacts...")
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
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'],
                        check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT],
                        check=False, verbose=verbose)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'],
                        check=False, verbose=verbose)
        except Exception as e:
            log_warning(f"Could not update pywhispercpp repository to v{PYWHISPERCPP_VERSION}: {e}")

    # The fetch/checkout above are best-effort; confirm the pin actually landed rather
    # than building the previous commit and failing later with an unrelated-looking error.
    head = run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'rev-parse', 'HEAD'],
                       check=False, capture_output=True)
    current = (head.stdout or '').strip() if head.returncode == 0 else ''
    if current != PYWHISPERCPP_PINNED_COMMIT:
        log_error(
            f"pywhispercpp sources are at {current or 'an unknown commit'}, not the pinned "
            f"{PYWHISPERCPP_PINNED_COMMIT} (v{PYWHISPERCPP_VERSION})"
        )
        log_error(f"Remove {PYWHISPERCPP_SRC_DIR} and re-run to get a clean checkout")
        return False

    return True


def install_pywhispercpp_cuda(pip_bin: Path) -> bool:
    """Install pywhispercpp with CUDA support"""
    log_info("Installing pywhispercpp with CUDA support...")

    # Try pre-built wheel first (much faster than source build)
    wheel_path = download_pywhispercpp_wheel()  # Auto-detects CUDA version
    if wheel_path:
        if install_pywhispercpp_from_wheel(pip_bin, wheel_path):
            return True
        log_warning("Pre-built wheel failed, falling back to source build...")

    install_system_dependencies()
    log_info("Building from source (this may take several minutes)...")

    if not _prepare_pywhispercpp_sources():
        return False
    
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
    install_system_dependencies()
    
    if not _prepare_pywhispercpp_sources():
        return False, True
    
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
    install_system_dependencies()

    if not _prepare_pywhispercpp_sources():
        return False

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

VAD_MODEL_FILENAME = 'ggml-silero-v5.1.2.bin'
VAD_MODEL_URL = f"https://huggingface.co/ggml-org/whisper-vad/resolve/main/{VAD_MODEL_FILENAME}"


def download_vad_model() -> bool:
    """Download the Silero VAD model for whisper.cpp native VAD. Never raises."""
    try:
        PYWHISPERCPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        vad_file = PYWHISPERCPP_MODELS_DIR / VAD_MODEL_FILENAME

        # ~2MB model; anything tiny is a failed/partial download
        if vad_file.exists():
            if vad_file.stat().st_size > 500_000:
                return True
            log_warning("Existing VAD model appears invalid; re-downloading")
            vad_file.unlink()

        log_info(f"Fetching {VAD_MODEL_URL}")
        urllib.request.urlretrieve(VAD_MODEL_URL, vad_file)

        if vad_file.stat().st_size <= 500_000:
            log_error("Downloaded VAD model appears invalid")
            vad_file.unlink()
            return False

        log_success("Silero VAD model downloaded")
        return True
    except Exception as e:
        log_error(f"Failed to download Silero VAD model: {e}")
        return False


def download_pywhispercpp_model(model_name: str = 'base') -> bool:
    """Download pywhispercpp model with progress feedback"""
    log_info(f"Downloading pywhispercpp model: {model_name}…")
    
    PYWHISPERCPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_file = PYWHISPERCPP_MODELS_DIR / f'ggml-{model_name}.bin'
    model_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    if check_model_validity(model_file):
        log_success(f"pywhispercpp model present: {model_name}")
        return True

    if model_file.exists():
        log_warning(f"Existing {model_name} model appears invalid; re-downloading")
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
        set_state(f"model_hash_{model_file.name}", model_hash)

        log_success(f"pywhispercpp model downloaded: {model_name}")
        return True
    except Exception as e:
        log_error(f"Failed to download pywhispercpp model {model_name}: {e}")
        return False


_COHERE_DOWNLOAD_SCRIPT = r'''
import os, traceback
try:
    try:
        from huggingface_hub import enable_progress_bars
        enable_progress_bars()
    except ImportError:
        pass
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import torch, sys
    model_id = "CohereLabs/cohere-transcribe-03-2026"
    token = os.environ.get("HF_TOKEN") or None
    print("Downloading processor...", flush=True)
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    print("Downloading model weights (~4 GB)...", flush=True)
    sys.stdout.flush()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, trust_remote_code=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, token=token,
    )
    del model
    print("Model downloaded and cached successfully", flush=True)
except BaseException:
    diagnostic = traceback.format_exc()
    diagnostic_path = os.environ.get("HYPRWHSPR_DOWNLOAD_DIAGNOSTIC")
    if diagnostic_path:
        with open(diagnostic_path, "w", encoding="utf-8") as handle:
            handle.write(diagnostic)
    raise
'''


def download_cohere_transcribe_model(hf_token: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Download Cohere weights through the managed venv without a fixed timeout."""
    venv_python = VENV_DIR / 'bin' / 'python'
    if not venv_python.exists():
        return False, f'Cohere Transcribe venv not found at {venv_python}'
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    # Hugging Face retries/resumes files itself; this only bounds a stalled read,
    # not the total multi-gigabyte download duration.
    env.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    if hf_token:
        env['HF_TOKEN'] = hf_token
    diagnostic_file = tempfile.NamedTemporaryFile(
        prefix='hyprwhspr-cohere-download-', suffix='.log', delete=False)
    diagnostic_file.close()
    diagnostic_path = Path(diagnostic_file.name)
    env['HYPRWHSPR_DOWNLOAD_DIAGNOSTIC'] = str(diagnostic_path)
    try:
        run_command([str(venv_python), '-c', _COHERE_DOWNLOAD_SCRIPT], check=True,
                    env=env, verbose=True)
        return True, None
    except Exception as exc:
        try:
            detail = diagnostic_path.read_text(encoding='utf-8').strip()
        except OSError:
            detail = str(exc)
        detail = detail or str(exc)
        return False, f'Cohere Transcribe model download failed:\n{detail}'
    finally:
        try:
            diagnostic_path.unlink(missing_ok=True)
        except OSError:
            pass


def _complete_pywhispercpp_cpu_fallback(created_items: dict) -> str:
    """Clean a failed new source build and return the effective installed variant."""
    if not created_items.get('git_clone_created'):
        return 'cpu'
    source_path = created_items.get('git_clone_path')
    if source_path:
        shutil.rmtree(Path(source_path), ignore_errors=True)
    created_items['git_clone_created'] = False
    created_items['git_clone_path'] = None
    return 'cpu'


# ==================== Main Installation Function ====================

def install_backend(backend_type: str, cleanup_on_failure: bool = True, force_rebuild: bool = False,
                    custom_python: Optional[str] = None) -> bool:
    """
    Main function to install backend.

    Args:
        backend_type: One of 'cpu', 'nvidia', 'amd', 'vulkan', 'onnx-asr'
        cleanup_on_failure: Whether to clean up partial installations on failure
        force_rebuild: If True, delete and recreate venv even if it exists
        custom_python: Optional path to Python executable to use for venv creation.
                       If None, auto-detects a compatible Python (3.14 or earlier).

    Returns:
        True if installation succeeded, False otherwise
    """
    # Validate backend type
    if backend_type not in LOCAL_INSTALL_BACKENDS:
        error_msg = f"Invalid backend type: {backend_type}"
        log_error(error_msg)
        set_install_state('failed', error_msg)
        return False

    # Packaging errors must be reported before state, GPU, venv, pip or cache mutation.
    try:
        initial_plan = resolve_dependency_plan(backend_type)
        # Hardware detection chooses between these later. Validate every possible
        # packaged selection now so a broken payload cannot mutate the host first.
        if backend_type == 'faster-whisper':
            resolve_dependency_plan(backend_type, accelerated_variant='cuda')
        elif backend_type == 'onnx-asr':
            resolve_dependency_plan(backend_type, accelerated_variant='gpu')
    except DependencyPlanError as exc:
        log_error(str(exc))
        return False

    init_state()
    set_install_state('in_progress')

    log_info(f"Installing {backend_type.upper()} backend...")

    # Check for MISE interference
    if _check_mise_active():
        log_warning("Warning! MISE is active. This may cause build errors.")
        log_warning("To fix: mise deactivate (or: mise unuse -g python)")

    dependency_family = initial_plan.family
    previous_family = get_state("dependency_family")
    if previous_family and previous_family != dependency_family:
        log_info(f"Backend dependency family changed ({previous_family} → {dependency_family}); recreating virtual environment...")
        force_rebuild = True
    
    # Track what we've created for cleanup
    created_items = {
        'venv_created': False,
        'venv_path': None,
        'git_clone_created': False,
        'git_clone_path': None,
        'packages_installed': []
    }
    
    try:
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
        elif backend_type == 'faster-whisper':
            # Detect NVIDIA GPU — if present, install CUDA pip libs so CTranslate2 can
            # find libcublas/libcudnn without requiring the system cuda/cudnn packages.
            # faster-whisper only supports NVIDIA CUDA (not AMD/Intel Vulkan/ROCm).
            enable_gpu = False
            if _detect_nvidia_gpu_listing():
                enable_gpu = True
                log_info("NVIDIA GPU detected - will install faster-whisper with CUDA support")

            if not enable_gpu:
                log_info("Installing faster-whisper (CPU mode)")

            plan = resolve_dependency_plan(
                'faster-whisper', accelerated_variant='cuda' if enable_gpu else None)
            try:
                execute_dependency_plan(plan, custom_python=custom_python,
                                        force_rebuild=force_rebuild)
            except Exception as exc:
                error_msg = f"Failed to install faster-whisper: {exc}"
                log_error(error_msg)
                set_install_state('failed', error_msg)
                return False

            set_install_state('completed')
            log_success("faster-whisper backend installation completed!")
            return True

        elif backend_type == 'cohere-transcribe':
            plan = resolve_dependency_plan('cohere-transcribe')
            try:
                execute_dependency_plan(plan, custom_python=custom_python,
                                        force_rebuild=force_rebuild)
            except Exception as exc:
                error_msg = f"Failed to install Cohere Transcribe dependencies: {exc}"
                log_error(error_msg)
                set_install_state('failed', error_msg)
                return False

            log_info("Downloading Cohere Transcribe model from HuggingFace (~4 GB)...")
            log_info("This may take several minutes depending on your connection speed.")
            hf_token = None
            try:
                from credential_manager import get_credential
                hf_token = get_credential('huggingface')
            except Exception:
                pass

            downloaded, diagnostic = download_cohere_transcribe_model(hf_token)
            if downloaded:
                log_success("Cohere Transcribe model downloaded and cached")
            else:
                error_msg = diagnostic or 'Cohere Transcribe model download failed'
                log_error(error_msg)
                log_info("Re-running setup resumes the cached Hugging Face download.")
                log_info("If Cohere is already configured, run: hyprwhspr model download")
                set_install_state('failed', error_msg)
                return False

            set_install_state('completed')
            log_success("Cohere Transcribe backend installation completed!")
            return True

        elif backend_type == 'onnx-asr':
            # Detect GPU availability for onnx-asr
            # Note: onnx-asr only needs NVIDIA drivers (nvidia-smi), not CUDA toolkit
            # Unlike pywhispercpp which needs nvcc to build, onnx-asr uses pre-built ONNX Runtime
            enable_gpu = False
            if _detect_nvidia_gpu_listing():
                enable_gpu = True
                log_info("NVIDIA GPU detected - will install onnx-asr with GPU support")
            
            if not enable_gpu:
                log_info("Installing onnx-asr (CPU-optimized)")

            plan = resolve_dependency_plan(
                'onnx-asr', accelerated_variant='gpu' if enable_gpu else None)
            try:
                execute_dependency_plan(plan, custom_python=custom_python,
                                        force_rebuild=force_rebuild)
            except Exception as exc:
                if not enable_gpu:
                    error_msg = f"Failed to install onnx-asr: {exc}"
                    log_error(error_msg)
                    set_install_state('failed', error_msg)
                    return False
                log_warning(f"ONNX GPU dependencies failed: {exc}")
                log_warning("Falling back to CPU-only ONNX installation")
                plan = resolve_dependency_plan('onnx-asr')
                try:
                    execute_dependency_plan(plan, custom_python=custom_python,
                                            force_rebuild=force_rebuild)
                except Exception as cpu_exc:
                    error_msg = f"Failed to install onnx-asr CPU fallback: {cpu_exc}"
                    log_error(error_msg)
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
            except Exception as e:
                log_warning(f"Model download failed: {e}")
                log_warning("Models will be downloaded on first use instead")
                # Don't fail installation - models can still be downloaded on first use

            # Installation successful for ONNX-ASR
            set_install_state('completed')
            log_success("ONNX-ASR backend installation completed!")
            return True

        # Plan the specialized pywhispercpp transaction after GPU selection.
        manifests = dependency_manifests(backend_type)
        requirements_file = manifests[-1]
        cur_req_hash = dependency_manifest_hash(manifests)
        stored_req_hash = get_state("dependency_manifest_hash")

        # Which pywhispercpp build variant is being requested *right now*?
        # (Differs from backend_type only inside this function: GPU setup may
        # have already downgraded backend_type='vulkan' to 'cpu' if Vulkan
        # wasn't usable.)
        if enable_cuda:
            requested_variant = 'nvidia'
        elif enable_rocm:
            requested_variant = 'rocm'
        elif enable_vulkan:
            requested_variant = 'vulkan'
        else:
            requested_variant = 'cpu'

        planned_dependency_state = resolve_dependency_plan(
            backend_type, accelerated_variant=requested_variant)
        deps_installed = False
        if VENV_DIR.exists():
            try:
                # A failed required import—including inherited NumPy/soxr—forces
                # a fresh transaction instead of repairing the live venv in place.
                deps_installed = _verify_dependency_plan_detailed(
                    planned_dependency_state).ok
            except Exception:
                pass

        stored_installed_backend = get_state("installed_backend")
        backend_mismatch = bool(stored_installed_backend) and stored_installed_backend != requested_variant
        needs_install = bool(
            force_rebuild or cur_req_hash != stored_req_hash or not stored_req_hash
            or not deps_installed or backend_mismatch
        )

        transaction = VenvTransaction(VENV_DIR)
        if needs_install:
            transaction.begin()
            created_items['venv_transaction'] = transaction
            if transaction.had_old:
                created_items['venv_backup_path'] = str(transaction.backup)
        pip_bin = setup_python_venv(custom_python=custom_python)
        if needs_install:
            created_items['venv_created'] = True
            created_items['venv_path'] = str(VENV_DIR)
            snapshot_plan = resolve_dependency_plan(
                backend_type, accelerated_variant=requested_variant)
            snapshot = _snapshot_inherited_dependencies(snapshot_plan)
        else:
            snapshot = []

        # Install pywhispercpp if needed
        if needs_install:
            if not stored_req_hash:
                # First time setup - no stored hash means venv is new
                log_info("Installing Python dependencies...")
            elif cur_req_hash != stored_req_hash:
                # Requirements actually changed
                log_info("Installing Python dependencies (requirements.txt changed)...")
            elif backend_mismatch:
                log_info(f"Installing Python dependencies (backend variant changed: {stored_installed_backend or 'unknown'} → {requested_variant})...")
            else:
                # Dependencies missing but hash matches (shouldn't happen often)
                log_info("Installing Python dependencies (dependencies missing)...")

            if enable_cuda or enable_rocm or enable_vulkan:
                # GPU build path: install everything except pywhispercpp first
                log_info("Installing base Python dependencies (excluding pywhispercpp)...")

                # If source fallback creates this tree, remove it when this package
                # transaction fails or is interrupted. Existing user caches are kept.
                if not PYWHISPERCPP_SRC_DIR.exists():
                    created_items['git_clone_created'] = True
                    created_items['git_clone_path'] = str(PYWHISPERCPP_SRC_DIR)

                # Determine packages to skip
                skip_pygobject = _should_skip_pygobject()
                skip_packages = ['pywhispercpp'] + (['PyGObject'] if skip_pygobject else [])

                # The filtered file expands requirements includes in place and
                # anchors constraints, so it is safe to pass to pip from /tmp.
                temp_req_path = None
                try:
                    temp_req_path = _filter_requirements(requirements_file, skip_packages)

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
                    _cleanup_partial_installation(created_items, pip_bin)
                    return False
                finally:
                    # Clean up temp file
                    if temp_req_path is not None and temp_req_path.exists():
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
                        _cleanup_partial_installation(created_items, pip_bin)
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
                                _cleanup_partial_installation(created_items, pip_bin)
                                return False
                            log_success("pywhispercpp installed (CPU-only mode)")
                            requested_variant = _complete_pywhispercpp_cpu_fallback(created_items)
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
                            _cleanup_partial_installation(created_items, pip_bin)
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
                            _cleanup_partial_installation(created_items, pip_bin)
                            return False
                        log_success("pywhispercpp installed (CPU-only mode)")
                        requested_variant = _complete_pywhispercpp_cpu_fallback(created_items)
            else:
                # CPU-only path: install everything normally
                if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                    error_msg = "Failed to install pywhispercpp (CPU-only)"
                    log_error(error_msg)
                    set_install_state('failed', error_msg)
                    _cleanup_partial_installation(created_items, pip_bin)
                    return False

            log_success("Python dependencies installed")
        else:
            log_info("Python dependencies up to date (skipping pip install)")
        
        plan = resolve_dependency_plan(backend_type, accelerated_variant=requested_variant)
        if needs_install:
            verification = _verify_and_repair_dependency_plan(plan, pip_bin)
        else:
            # The precheck above proved this live venv healthy. Do not mutate it
            # outside a VenvTransaction if a later/racing verification now fails.
            verification = _verify_dependency_plan_detailed(plan)
        if not verification.ok:
            diagnostic = _format_dependency_diagnostic(plan, verification, snapshot)
            log_error(diagnostic)
            set_install_state('failed', diagnostic)
            _cleanup_partial_installation(created_items, pip_bin)
            return False

        if needs_install:
            transaction.commit()
        created_items['venv_backup_path'] = None
        created_items['venv_transaction'] = None
        created_items['venv_created'] = False
        created_items['git_clone_created'] = False
        created_items['git_clone_path'] = None
        try:
            commit_dependency_state(plan)
        except Exception as exc:
            log_warning(f"Could not record dependency state: {exc}")

        # Download base model only after the verified environment commits.
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
        if (cleanup_on_failure or created_items.get('venv_created')
                or created_items.get('venv_backup_path')):
            log_info("Cleaning up partial installation...")
            _cleanup_partial_installation(created_items, pip_bin if 'pip_bin' in locals() else None)
        raise
    except Exception as e:
        error_msg = f"Unexpected error during installation: {e}"
        log_error(error_msg)
        log_debug(f"Full error traceback: {sys.exc_info()}")
        set_install_state('failed', error_msg)
        if (cleanup_on_failure or created_items.get('venv_created')
                or created_items.get('venv_backup_path')):
            log_info("Cleaning up partial installation...")
            _cleanup_partial_installation(created_items, pip_bin if 'pip_bin' in locals() else None)
        return False
    finally:
        transaction = created_items.get('venv_transaction')
        if transaction is not None and transaction.active:
            transaction.rollback()


def _cleanup_partial_installation(created_items: dict, pip_bin: Optional[Path]):
    """Clean up partial installation on failure"""
    transaction = created_items.get('venv_transaction')
    if transaction is not None:
        transaction.rollback()
        created_items['venv_transaction'] = None
        created_items['venv_created'] = False
        created_items['venv_backup_path'] = None

    if transaction is None and created_items.get('venv_created') and created_items.get('venv_path'):
        log_info(f"Removing venv at {created_items['venv_path']}")
        try:
            venv_path = Path(created_items['venv_path'])
            if venv_path.exists():
                shutil.rmtree(venv_path, ignore_errors=True)
        except Exception:
            pass

    backup_path = created_items.get('venv_backup_path')
    if backup_path:
        backup = Path(backup_path)
        if backup.exists():
            if VENV_DIR.exists():
                shutil.rmtree(VENV_DIR, ignore_errors=True)
            backup.rename(VENV_DIR)
    
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
