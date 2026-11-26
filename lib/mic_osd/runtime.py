"""Mic-OSD runtime discovery. GTK loads only in probe subprocesses: a process binds one GTK version."""

import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    from ..src import visualizer_runtime
except ImportError:
    from src import visualizer_runtime

PROBE_TIMEOUT = 5

RUNTIME_X11 = "X11 GTK3"
RUNTIME_SYSTEM = "system"
RUNTIME_BUNDLED = f"bundled {visualizer_runtime.GTK4_LAYER_SHELL_VERSION}"
RUNTIME_NONE = "unavailable"

X11_REASON = "X11 overlay requires GTK3 introspection bindings and PyCairo available to the service Python."

# Prints the first missing system piece, or 'ok'.
_SYSTEM_PROBE = """
try:
    import cairo
    import gi
except ImportError as e:
    print('gtk4' if getattr(e, 'name', '') == 'gi' else 'cairo')
    raise SystemExit
for namespace, version, missing in (('Gtk', '4.0', 'gtk4'), ('Gtk4LayerShell', '1.0', 'layer-shell')):
    try:
        gi.require_version(namespace, version)
    except ValueError:
        print(missing)
        raise SystemExit
print('ok')
"""

_BUNDLED_PROBE = (
    "import cairo, gi;"
    "gi.require_version('Gtk', '4.0');"
    "gi.require_version('Gtk4LayerShell', '1.0');"
    "from gi.repository import Gtk, Gtk4LayerShell"
)

_GTK3_PROBE = (
    "import cairo, gi;"
    "gi.require_version('Gtk', '3.0');"
    "from gi.repository import Gtk, Gdk;"
    "print('1')"
)

_LIBRARY_PROBE = (
    "import gi; gi.require_version('Gtk4LayerShell', '1.0');"
    "from gi.repository import Gtk4LayerShell;"
    "Gtk4LayerShell.get_major_version();"
    "print(*{line.split(None, 5)[5].strip() for line in open('/proc/self/maps')"
    " if 'libgtk4-layer-shell' in line}, sep='\\n')"
)

_SUPPORT_PROBE = (
    "import gi; gi.require_version('Gtk', '4.0');"
    "gi.require_version('Gtk4LayerShell', '1.0');"
    "from gi.repository import Gtk, Gtk4LayerShell;"
    "Gtk.init();"
    "print('1' if Gtk4LayerShell.is_supported() else '0')"
)


class LayerShellResolutionError(RuntimeError):
    pass


def _python() -> str:
    return sys.executable or 'python3'


def _base(environ) -> dict:
    return dict(os.environ if environ is None else environ)


def system_status(environ=None) -> tuple:
    """(status, detail): status is 'ok', the first missing piece ('cairo', 'gtk4',
    'layer-shell'), or None when the probe itself failed, explained by detail."""
    try:
        result = subprocess.run(
            [_python(), '-c', _SYSTEM_PROBE],
            env=_base(environ), capture_output=True, text=True, errors='replace',
            timeout=PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e) or type(e).__name__
    status = result.stdout.strip()
    if result.returncode != 0 or status not in ('ok', 'cairo', 'gtk4', 'layer-shell'):
        return None, (result.stderr.strip().splitlines() or [f'exit {result.returncode}'])[-1]
    return status, ""


def bundled_available(environ=None) -> bool:
    """The private gtk4-layer-shell runtime is installed and imports."""
    if not visualizer_runtime.is_complete():
        return False
    try:
        result = subprocess.run(
            [_python(), '-c', _BUNDLED_PROBE],
            env=visualizer_runtime.bundled_environment(_base(environ)),
            capture_output=True, timeout=PROBE_TIMEOUT,
        )
        return result.returncode == 0
    except Exception:
        return False


def gtk3_available(environ=None) -> bool:
    """GTK3 and PyCairo import for the X11 overlay."""
    try:
        result = subprocess.run(
            [_python(), '-c', _GTK3_PROBE],
            env=_base(environ), capture_output=True, timeout=PROBE_TIMEOUT,
        )
        return result.stdout.decode('utf-8', 'ignore').strip() == '1'
    except Exception:
        return False


def without_layer_shell_preload(value: str) -> str:
    """Drop gtk4-layer-shell entries from an LD_PRELOAD value."""
    entries = value.replace(':', ' ').split()
    return ' '.join(e for e in entries if 'libgtk4-layer-shell' not in os.path.basename(e))


def layer_shell_path_search() -> str:
    """Fallback: first gtk4-layer-shell .so in common library paths."""
    for pattern in [
        '/usr/lib64/libgtk4-layer-shell.so*',
        '/usr/lib/libgtk4-layer-shell.so*',
        '/usr/lib/*/libgtk4-layer-shell.so*',
        '/usr/local/lib64/libgtk4-layer-shell.so*',
        '/usr/local/lib/libgtk4-layer-shell.so*',
    ]:
        for candidate in sorted(glob.glob(pattern)):
            resolved = os.path.realpath(candidate)
            if os.path.isfile(resolved):
                return resolved
    return ""


def mapped_layer_shell_library(environ=None) -> str:
    """The gtk4-layer-shell library that the Gtk4LayerShell typelib loads, or '' if the probe failed.

    The preload must be that exact file. A second copy elsewhere on the
    system (Ghostty's package, for one, installs /usr/lib/libgtk4-layer-shell.so)
    keeps its own state, so the typelib's copy never hooks libwayland,
    init_for_window() fails, and the OSD becomes an ordinary toplevel that
    takes keyboard focus from the window dictation should paste into.

    The library is loaded lazily, so the probe calls into it before reading
    its own mappings. Inherited layer-shell preloads are dropped so the probe
    sees only the typelib's copy.
    """
    env = _base(environ)
    env['LD_PRELOAD'] = without_layer_shell_preload(env.get('LD_PRELOAD', ''))
    try:
        result = subprocess.run(
            [_python(), '-c', _LIBRARY_PROBE],
            env=env, capture_output=True, text=True, errors='replace', timeout=PROBE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[MIC-OSD] layer-shell library probe failed: {e}", flush=True)
        return ""
    libraries = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not libraries:
        detail = (result.stderr.strip().splitlines() or ['no library mapped'])[-1]
        print(f"[MIC-OSD] layer-shell library probe failed: {detail}", flush=True)
        return ""
    if len(libraries) > 1:
        raise LayerShellResolutionError(
            f"expected one mapped gtk4-layer-shell library, found {libraries}"
        )
    return libraries[0]


def resolve_layer_shell_library(environ=None) -> str:
    """The typelib's library; a failed probe falls back to the path search rather than disabling the overlay."""
    return mapped_layer_shell_library(environ) or layer_shell_path_search()


def gtk4_environment(runtime: str, library: str = "", environ=None) -> dict:
    """A fresh child environment for the GTK4 overlay on `runtime`."""
    env = _base(environ)
    if runtime == RUNTIME_SYSTEM:
        if library:
            current = without_layer_shell_preload(env.get('LD_PRELOAD', ''))
            env['LD_PRELOAD'] = f"{library} {current}".strip()
    elif runtime == RUNTIME_BUNDLED:
        env = visualizer_runtime.bundled_environment(env)
    return env


def layer_shell_supported(env: dict) -> bool:
    """The compositor offers layer-shell. False on any uncertainty."""
    try:
        result = subprocess.run(
            [_python(), '-c', _SUPPORT_PROBE],
            env=env, capture_output=True, timeout=PROBE_TIMEOUT,
        )
        return result.stdout.decode('utf-8', 'ignore').strip() == '1'
    except Exception:
        return False


def distro_packages() -> tuple:
    """Return (gtk_pkg, layer_shell_pkg) package names for current distro."""
    try:
        if Path('/etc/debian_version').exists():
            return ('python3-gi python3-cairo gir1.2-gtk-4.0', 'gir1.2-gtk4layershell-1.0')
        elif Path('/etc/arch-release').exists():
            return ('python-gobject python-cairo gtk4', 'gtk4-layer-shell')
        elif Path('/etc/fedora-release').exists():
            return ('python3-gobject python3-cairo gtk4', 'gtk4-layer-shell')
        elif Path('/etc/os-release').exists():
            content = Path('/etc/os-release').read_text(encoding='utf-8').lower()
            if 'debian' in content or 'ubuntu' in content:
                return ('python3-gi python3-cairo gir1.2-gtk-4.0', 'gir1.2-gtk4layershell-1.0')
            elif 'fedora' in content or 'rhel' in content:
                return ('python3-gobject python3-cairo gtk4', 'gtk4-layer-shell')
            elif 'suse' in content:
                return ('python3-gobject python3-pycairo typelib-1_0-Gtk-4_0', 'gtk4-layer-shell')
    except Exception:
        pass
    return ('python-gobject python-cairo gtk4', 'gtk4-layer-shell')


def gtk4_reason(status, bundled: bool, detail: str = "", packages=None) -> str:
    """Why the GTK4 overlay can't run, or '' if it can."""
    if status == 'ok' or (status == 'layer-shell' and bundled):
        return ""
    gtk_pkg, layer_pkg = packages or distro_packages()
    if status == 'cairo':
        return f"PyCairo bindings not installed. Install: {gtk_pkg}"
    if status == 'gtk4':
        return f"GTK4 bindings not installed. Install: {gtk_pkg}"
    if status == 'layer-shell':
        return f"gtk4-layer-shell not installed. Install: {layer_pkg}"
    return f"GTK4 dependency check failed: {detail or 'unknown error'}"
