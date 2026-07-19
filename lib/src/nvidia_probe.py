"""Small, dependency-free NVIDIA hardware and driver probe."""

import re
import shutil
import subprocess
from typing import Callable, Optional


GPU_LISTING = re.compile(r"(?m)^GPU\s+\d+:", re.IGNORECASE)


def responding_gpu_listing(
        runner: Callable = subprocess.run,
        which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """Return ``nvidia-smi -L`` output only when hardware and driver respond."""
    try:
        pci = runner(
            ["lspci"], capture_output=True, check=False, timeout=2,
            text=True,
        )
        if pci.returncode == 0 and "nvidia" not in (pci.stdout or "").lower():
            return None
    except (OSError, subprocess.SubprocessError):
        # lspci is an optimization against false positives, not a requirement.
        pass

    if not which("nvidia-smi"):
        return None
    try:
        result = runner(
            ["nvidia-smi", "-L"], capture_output=True, check=False,
            timeout=2, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or "").strip()
    if result.returncode != 0 or not GPU_LISTING.search(output):
        return None
    return output
