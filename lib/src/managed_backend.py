"""Build accelerated whisper wheels using the existing authoritative contract."""
import os
from pathlib import Path
import sys
import backend_installer as backend

backend.VENV_DIR = Path(os.environ['HYPRWHSPR_BUILD_ENV'])
backend.WHEEL_CACHE_DIR = Path(os.environ['HYPRWHSPR_BUILD_WORK']) / 'wheels'
backend.PYWHISPERCPP_SRC_DIR = Path(os.environ['HYPRWHSPR_BUILD_WORK']) / 'sources'
pip = backend.VENV_DIR / 'bin/pip'
variant = sys.argv[1]
# Remove the CPU distribution before selecting an accelerated build.
backend.run_command([str(pip), 'uninstall', '-y', 'pywhispercpp'], check=True)
if variant == 'amd':
    success, _ = backend.install_pywhispercpp_rocm(pip)
else:
    success = {'nvidia': backend.install_pywhispercpp_cuda,
               'vulkan': backend.install_pywhispercpp_vulkan}[variant](pip)
if not success:
    raise SystemExit('Accelerated backend build failed')
