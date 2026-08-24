"""Contract and path helpers for the optional gtk4-layer-shell runtime."""

import json
import os
import platform
from pathlib import Path


GTK4_LAYER_SHELL_VERSION = "1.3.0"
GTK4_LAYER_SHELL_COMMIT = "1c963c51514581c41b9bdae08cdf69171265cdda"
GTK4_LAYER_SHELL_RELEASE_TAG = "gtk4-layer-shell-runtime-v1.3.0-1"
GTK4_LAYER_SHELL_ASSET = "gtk4-layer-shell-1.3.0-ubuntu24.04-x86_64.tar.gz"
GTK4_LAYER_SHELL_SHA256 = "5a35be4870a3e3e83502ef8d556d3a2989d013276c2201a3f47066e05af3ebe1"
GTK4_LAYER_SHELL_BASE_URL = (
    "https://github.com/goodroot/hyprwhspr/releases/download/"
    f"{GTK4_LAYER_SHELL_RELEASE_TAG}"
)


def user_base() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "hyprwhspr"


def versions_dir() -> Path:
    return user_base() / "runtime" / "gtk4-layer-shell"


def runtime_dir() -> Path:
    return versions_dir() / GTK4_LAYER_SHELL_VERSION / "x86_64"


def library_path() -> Path:
    return runtime_dir() / "lib" / "libgtk4-layer-shell.so.0"


def typelib_dir() -> Path:
    return runtime_dir() / "lib" / "girepository-1.0"


def manifest_path() -> Path:
    return runtime_dir() / "manifest.json"


def is_complete() -> bool:
    try:
        manifest = json.loads(manifest_path().read_text(encoding="utf-8"))
        checksum = (runtime_dir() / ".sha256").read_text(encoding="utf-8").strip()
        return (
            library_path().is_file()
            and (typelib_dir() / "Gtk4LayerShell-1.0.typelib").is_file()
            and manifest.get("version") == GTK4_LAYER_SHELL_VERSION
            and manifest.get("commit") == GTK4_LAYER_SHELL_COMMIT
            and checksum == GTK4_LAYER_SHELL_SHA256
        )
    except (OSError, ValueError, TypeError):
        return False


def prepend_env_path(env: dict, name: str, value: Path) -> None:
    current = env.get(name, "")
    env[name] = f"{value}{os.pathsep}{current}" if current else str(value)


def environment_for(root: Path, base=None) -> dict:
    """Return a child environment that activates the app-private runtime."""
    env = dict(os.environ if base is None else base)
    lib_dir = root / "lib"
    library = lib_dir / "libgtk4-layer-shell.so.0"
    prepend_env_path(env, "GI_TYPELIB_PATH", lib_dir / "girepository-1.0")
    prepend_env_path(env, "LD_LIBRARY_PATH", lib_dir)
    preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = f"{library} {preload}".strip()
    return env


def bundled_environment(base=None) -> dict:
    return environment_for(runtime_dir(), base)


def is_noble_x86_64(os_release_path=Path("/etc/os-release")) -> bool:
    """Return whether the host matches the runtime's supported ABI baseline."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    try:
        values = {}
        for raw in os_release_path.read_text(encoding="utf-8").splitlines():
            if "=" not in raw or raw.lstrip().startswith("#"):
                continue
            key, value = raw.split("=", 1)
            values[key] = value.strip().strip('"\'').lower()
        lineage = " ".join((values.get("ID", ""), values.get("ID_LIKE", "")))
        codename = values.get("UBUNTU_CODENAME") or values.get("VERSION_CODENAME", "")
        return "ubuntu" in lineage and codename == "noble"
    except OSError:
        return False
