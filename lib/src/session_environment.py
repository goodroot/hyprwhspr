"""
Session environment fallbacks for service startup.
"""

import os
import stat
from pathlib import Path


def ensure_wayland_display():
    """
    Populate WAYLAND_DISPLAY from XDG_RUNTIME_DIR when systemd has not imported it.

    This is a best-effort fallback for compositor startup races. It intentionally
    does not override an existing WAYLAND_DISPLAY value.
    """
    if os.environ.get("WAYLAND_DISPLAY"):
        return

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        print("[WARN] WAYLAND_DISPLAY unset and XDG_RUNTIME_DIR missing", flush=True)
        return

    runtime_path = Path(runtime_dir)
    if not runtime_path.is_dir():
        print("[WARN] WAYLAND_DISPLAY unset and XDG_RUNTIME_DIR is not a directory", flush=True)
        return

    candidates = []
    for path in runtime_path.glob("wayland-*"):
        try:
            path_stat = path.stat()
        except OSError:
            continue
        if stat.S_ISSOCK(path_stat.st_mode):
            candidates.append((path_stat.st_mtime, path.name, path))

    if not candidates:
        return

    # A newly bound compositor socket is the most likely active display after a
    # startup race leaves WAYLAND_DISPLAY out of the systemd user environment.
    _mtime, display_name, _path = max(candidates)
    os.environ["WAYLAND_DISPLAY"] = display_name
    print(f"[INIT] WAYLAND_DISPLAY was unset; using {display_name}", flush=True)
