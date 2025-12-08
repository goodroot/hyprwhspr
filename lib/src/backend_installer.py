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
from typing import Optional, Tuple

# Define logging functions locally to avoid circular imports
def log_info(msg: str):
    """Print info message"""
    print(f"[INFO] {msg}")


def log_success(msg: str):
    """Print success message"""
    print(f"[SUCCESS] {msg}")


def log_warning(msg: str):
    """Print warning message"""
    print(f"[WARNING] {msg}")


def log_error(msg: str):
    """Print error message"""
    print(f"[ERROR] {msg}", file=sys.stderr)


def run_command(cmd: list, check: bool = True, capture_output: bool = False, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a shell command"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
            env=env
        )
        return result
    except subprocess.CalledProcessError:
        log_error(f"Command failed: {' '.join(cmd)}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {cmd[0]}")
        raise


def run_sudo_command(cmd: list, check: bool = True, input_data: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """Run a command with sudo"""
    sudo_cmd = ['sudo'] + cmd
    try:
        result = subprocess.run(
            sudo_cmd,
            check=check,
            input=input_data,
            text=False if input_data else True
        )
        return result
    except subprocess.CalledProcessError:
        log_error(f"Command failed: {' '.join(sudo_cmd)}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {sudo_cmd[0]}")
        raise


# Constants
HYPRWHSPR_ROOT = os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr')
USER_BASE = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'hyprwhspr'
VENV_DIR = USER_BASE / 'venv'
PYWHISPERCPP_MODELS_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')) / 'pywhispercpp' / 'models'
STATE_DIR = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local' / 'state')) / 'hyprwhspr'
STATE_FILE = STATE_DIR / 'install-state.json'
PYWHISPERCPP_SRC_DIR = USER_BASE / 'pywhispercpp-src'
PYWHISPERCPP_PINNED_COMMIT = "4ab96165f84e8eb579077dfc3d0476fa5606affe"


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
    except (json.JSONDecodeError, IOError):
        pass


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
            gcc_version = result.stdout.strip().decode()
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
    """Install system dependencies needed for backend compilation"""
    log_info("Ensuring system dependencies...")
    
    # Only install build dependencies, not waybar (CLI handles that)
    pkgs = ['cmake', 'make', 'git', 'base-devel', 'python', 'curl']
    
    to_install = []
    for pkg in pkgs:
        try:
            run_command(['pacman', '-Q', pkg], check=False, capture_output=True)
        except subprocess.CalledProcessError:
            to_install.append(pkg)
    
    if to_install:
        log_info(f"Installing: {' '.join(to_install)}")
        run_sudo_command(['pacman', '-S', '--needed', '--noconfirm'] + to_install, check=False)
    
    log_success("Dependencies ready")


# ==================== GPU Support Setup ====================

def setup_nvidia_support() -> bool:
    """Setup NVIDIA/CUDA support. Returns True if CUDA is available."""
    log_info("GPU check…")
    
    if not shutil.which('nvidia-smi'):
        log_info("No NVIDIA GPU detected (CPU mode)")
        return False
    
    # Test nvidia-smi
    try:
        result = run_command(['timeout', '2s', 'nvidia-smi', '-L'], check=False, capture_output=True)
        if result.returncode != 0:
            log_warning("nvidia-smi found but not responding (no GPU hardware or driver issue)")
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
        os.environ['PATH'] = '/opt/cuda/bin:' + os.environ.get('PATH', '')
        os.environ['CUDACXX'] = nvcc_path
        log_success("CUDA toolkit present")
    else:
        log_warning("CUDA toolkit not found; installing…")
        run_sudo_command(['pacman', '-S', '--needed', '--noconfirm', 'cuda'], check=False)
        
        if Path('/opt/cuda/bin/nvcc').exists():
            os.environ['PATH'] = '/opt/cuda/bin:' + os.environ.get('PATH', '')
            os.environ['CUDACXX'] = '/opt/cuda/bin/nvcc'
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
                    gcc_version = result.stdout.strip().decode()
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
        result = run_command(['timeout', '2s', 'rocm-smi', '--showproductname'], check=False, capture_output=True)
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
        log_warning("ROCm not found; installing...")
        run_sudo_command(['yay', '-S', '--needed', '--noconfirm', 'rocm-hip-sdk', 'rocm-opencl-sdk'], check=False)
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


# ==================== Python Environment ====================

def setup_python_venv() -> Path:
    """Create or update Python virtual environment. Returns path to pip binary."""
    log_info("Setting up Python virtual environment…")
    
    # Validate requirements.txt exists
    requirements_file = Path(HYPRWHSPR_ROOT) / 'requirements.txt'
    if not requirements_file.exists():
        log_error(f"requirements.txt not found at {requirements_file}")
        raise FileNotFoundError(f"requirements.txt not found at {requirements_file}")
    
    # Check if venv exists and if Python version matches
    venv_needs_recreation = False
    if VENV_DIR.exists():
        venv_python = VENV_DIR / 'bin' / 'python'
        if venv_python.exists():
            try:
                # Check Python version in venv
                result = run_command([str(venv_python), '--version'], check=False, capture_output=True)
                venv_version = result.stdout.strip() if result.returncode == 0 else ""
                current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                
                # Extract major.minor from venv version string (e.g., "Python 3.11.5" -> "3.11")
                venv_major_minor = ""
                if venv_version:
                    import re
                    match = re.search(r'(\d+)\.(\d+)', venv_version)
                    if match:
                        venv_major_minor = f"{match.group(1)}.{match.group(2)}"
                
                # Check if versions match (major.minor)
                if venv_major_minor and venv_major_minor != current_version:
                    log_warning(f"Venv Python version mismatch: venv has {venv_version}, current is Python {current_version}")
                    log_info("Recreating venv to match current Python version...")
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
        run_command([sys.executable, '-m', 'venv', str(VENV_DIR)], check=True)
    else:
        log_info(f"Venv already exists at {VENV_DIR}")
    
    # Get pip binary
    pip_bin = VENV_DIR / 'bin' / 'pip'
    if not pip_bin.exists():
        log_error(f"pip not found in venv at {VENV_DIR}")
        raise FileNotFoundError(f"pip not found in venv")
    
    # Upgrade pip and wheel
    run_command([str(pip_bin), 'install', '--upgrade', 'pip', 'wheel'], check=True)
    
    return pip_bin


# ==================== pywhispercpp Installation ====================

def install_pywhispercpp_cpu(pip_bin: Path, requirements_file: Path) -> bool:
    """Install CPU-only pywhispercpp"""
    log_info("Installing pywhispercpp (CPU-only)...")
    try:
        run_command([str(pip_bin), 'install', '-r', str(requirements_file)], check=True)
        log_success("pywhispercpp installed (CPU-only mode)")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to install pywhispercpp (CPU-only): {e}")
        return False


def install_pywhispercpp_cuda(pip_bin: Path) -> bool:
    """Install pywhispercpp with CUDA support"""
    log_info("Installing pywhispercpp with CUDA support...")
    
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
        log_info(f"Cloning pywhispercpp sources (v1.4.0) → {PYWHISPERCPP_SRC_DIR}")
        PYWHISPERCPP_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_command([
            'git', 'clone', '--recurse-submodules',
            'https://github.com/Absadiki/pywhispercpp.git',
            str(PYWHISPERCPP_SRC_DIR)
        ], check=True)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'checkout', PYWHISPERCPP_PINNED_COMMIT
        ], check=True)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'submodule', 'update', '--init', '--recursive'
        ], check=True)
    else:
        log_info(f"Updating pywhispercpp sources to v1.4.0 in {PYWHISPERCPP_SRC_DIR}")
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'], check=False)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT], check=False)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'], check=False)
        except Exception:
            log_warning("Could not update pywhispercpp repository to v1.4.0")
    
    # Build with CUDA support
    log_info("Building pywhispercpp with CUDA (ggml CUDA) via pip")
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
        run_command([
            str(pip_bin), 'install',
            '-e', str(PYWHISPERCPP_SRC_DIR),
            '--no-cache-dir',
            '--force-reinstall',
            '-v'
        ], check=True, env=env)
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
        log_info(f"Cloning pywhispercpp sources (v1.4.0) → {PYWHISPERCPP_SRC_DIR}")
        PYWHISPERCPP_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_command([
            'git', 'clone', '--recurse-submodules',
            'https://github.com/Absadiki/pywhispercpp.git',
            str(PYWHISPERCPP_SRC_DIR)
        ], check=True)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'checkout', PYWHISPERCPP_PINNED_COMMIT
        ], check=True)
        run_command([
            'git', '-C', str(PYWHISPERCPP_SRC_DIR),
            'submodule', 'update', '--init', '--recursive'
        ], check=True)
    else:
        log_info(f"Updating pywhispercpp sources to v1.4.0 in {PYWHISPERCPP_SRC_DIR}")
        try:
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'fetch', '--tags'], check=False)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'checkout', PYWHISPERCPP_PINNED_COMMIT], check=False)
            run_command(['git', '-C', str(PYWHISPERCPP_SRC_DIR), 'submodule', 'update', '--init', '--recursive'], check=False)
        except Exception:
            log_warning("Could not update pywhispercpp repository to v1.4.0")
    
    # Set up ROCm environment
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
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
        run_command([
            str(pip_bin), 'install',
            '-e', str(PYWHISPERCPP_SRC_DIR),
            '--no-cache-dir',
            '--force-reinstall',
            '-v'
        ], check=True, env=env)
        log_success("pywhispercpp installed with ROCm acceleration via pip")
        return True, False
    except subprocess.CalledProcessError:
        # Build failed - return should_fallback=True
        return False, True


# ==================== Model Download ====================

def download_pywhispercpp_model(model_name: str = 'base.en') -> bool:
    """Download pywhispercpp model"""
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
        urllib.request.urlretrieve(model_url, model_file)
        
        # Store hash for future validation
        model_hash = compute_file_hash(model_file)
        set_state("model_base_en_hash", model_hash)
        
        log_success("pywhispercpp base model downloaded")
        return True
    except Exception as e:
        log_error(f"Failed to download pywhispercpp base model: {e}")
        return False


# ==================== Main Installation Function ====================

def install_backend(backend_type: str) -> bool:
    """
    Main function to install backend.
    
    Args:
        backend_type: One of 'cpu', 'nvidia', 'amd'
    
    Returns:
        True if installation succeeded, False otherwise
    """
    log_info(f"Installing {backend_type.upper()} backend...")
    
    # Validate backend type
    if backend_type not in ['cpu', 'nvidia', 'amd']:
        log_error(f"Invalid backend type: {backend_type}")
        return False
    
    # Install system dependencies
    install_system_dependencies()
    
    # Setup GPU support if needed
    enable_cuda = False
    enable_rocm = False
    
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
    
    # Setup Python venv
    try:
        pip_bin = setup_python_venv()
    except Exception as e:
        log_error(f"Failed to setup Python venv: {e}")
        return False
    
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
        ], check=False, capture_output=True)
        deps_installed = result.returncode == 0
    except Exception:
        pass
    
    # Install pywhispercpp if needed
    if cur_req_hash != stored_req_hash or not stored_req_hash or not deps_installed:
        log_info("Installing Python dependencies (requirements.txt changed or deps missing)")
        
        if enable_cuda or enable_rocm:
            # GPU build path: install everything except pywhispercpp first
            log_info("Installing base Python dependencies (excluding pywhispercpp)...")
            # Use a writable temp directory instead of system directory
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_req:
                temp_req_path = Path(temp_req.name)
                try:
                    with open(requirements_file, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            if not line.strip().startswith('pywhispercpp'):
                                temp_req.write(line)
                    
                    temp_req.flush()
                    
                    if temp_req_path.stat().st_size > 0:
                        run_command([str(pip_bin), 'install', '-r', str(temp_req_path)], check=True)
                except Exception as e:
                    log_error(f"Failed to install base Python dependencies: {e}")
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
                    log_error("Failed to install pywhispercpp with CUDA support")
                    return False
            elif enable_rocm:
                success, should_fallback = install_pywhispercpp_rocm(pip_bin)
                if not success:
                    if should_fallback:
                        # ROCm build failed - fall back to CPU-only
                        log_warning("ROCm build failed - falling back to CPU-only installation")
                        log_warning("")
                        log_warning("ROCm 7.x has known compatibility issues with pywhispercpp v1.4.0")
                        log_warning("See: https://github.com/ggml-org/whisper.cpp/issues/3553")
                        log_warning("")
                        log_warning("Alternatives:")
                        log_warning("  • Use CPU mode (current fallback)")
                        log_warning("  • Use remote transcription backend (see README)")
                        log_warning("")
                        log_info("Installing pywhispercpp with CPU-only support...")
                        if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                            log_error("Failed to install pywhispercpp (CPU-only fallback)")
                            return False
                        log_success("pywhispercpp installed (CPU-only mode)")
                    else:
                        log_error("Failed to install pywhispercpp with ROCm support")
                        return False
        else:
            # CPU-only path: install everything normally
            if not install_pywhispercpp_cpu(pip_bin, requirements_file):
                return False
        
        set_state("requirements_hash", cur_req_hash)
        log_success("Python dependencies installed")
    else:
        log_info("Python dependencies up to date (skipping pip install)")
    
    # Download base model
    if not download_pywhispercpp_model('base.en'):
        log_warning("Model download failed, but backend installation succeeded")
        # Don't fail the whole installation if model download fails
    
    log_success(f"{backend_type.upper()} backend installation completed!")
    return True

