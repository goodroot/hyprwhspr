"""Write the dictation shortcut into compositors other than Hyprland.

A compositor binding needs no keyboard access, so no input group and no logout.
Every writer validates a candidate before the live file changes; on any doubt
the file is left alone and the line is printed for the user to add.
"""

import ast
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    from ..installation import tray_command as stable_tray_command
except ImportError:
    from installation import tray_command as stable_tray_command

try:
    from ..output_control import log_info, log_success, log_warning
except ImportError:
    from output_control import log_info, log_success, log_warning

from ._shared import (HYPRWHSPR_ROOT, USER_HOME, _is_gnome_or_mutter_session,
                      _is_kde_plasma_session, _is_niri_session,
                      _is_xfce_session)

MARKER = 'hyprwhspr (added by hyprwhspr setup)'
LABELS = {'niri': 'niri', 'sway': 'Sway', 'gnome': 'GNOME', 'kde': 'KDE Plasma',
          'xfce': 'XFCE'}

GNOME_SCHEMA = 'org.gnome.settings-daemon.plugins.media-keys'
GNOME_PATH = '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/hyprwhspr/'
GNOME_BINDING = '<Super><Alt>d'

XFCE_CHANNEL = 'xfce4-keyboard-shortcuts'
XFCE_PROPERTY = '/commands/custom/<Super><Alt>d'
XFCE_BINDING = '<Super><Alt>d'

# A line that already starts or toggles recording, whoever wrote it.
_RECORD_BINDING = re.compile(r'hyprwhspr.*\brecord\b|recording_control|--managed-tray')


def _config_home() -> Path:
    return Path(os.environ.get('XDG_CONFIG_HOME') or USER_HOME / '.config')


def detect_compositor() -> Optional[str]:
    """The session whose shortcut setup can configure, Hyprland aside."""
    if _is_niri_session():
        return 'niri'
    try:
        from sway_focus import is_sway_session
    except ImportError:
        from ..sway_focus import is_sway_session
    if is_sway_session():
        return 'sway'
    if _is_gnome_or_mutter_session():
        return 'gnome'
    if _is_kde_plasma_session():
        return 'kde'
    if _is_xfce_session():
        return 'xfce'
    return None


def _manual(kind: str, where: str, line: str) -> bool:
    log_warning(f"Could not set the {LABELS[kind]} shortcut automatically.")
    print(f"  Add this to {where}:")
    print(f"  {line}")
    print("  Then set \"external_bindings\": true with: hyprwhspr config edit")
    return False


def _has_record_binding(text: str, comment: str) -> bool:
    return any(_RECORD_BINDING.search(line)
               for line in text.splitlines()
               if not line.lstrip().startswith(comment))


def _write_validated(path: Path, original: str, updated: str, validate) -> bool:
    """Replace path only after a sibling candidate passes validation.

    The candidate sits beside the real file so relative includes resolve, and
    the live file is swapped in one step: a compositor that hot-reloads never
    sees a config that failed to validate.
    """
    target = path.resolve()
    handle, name = tempfile.mkstemp(prefix=f'.{target.name}.hyprwhspr-', dir=target.parent)
    candidate = Path(name)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(updated)
        shutil.copymode(target, candidate)
        if not validate(candidate):
            return False
        # Someone else edited it while we validated: do not clobber their change.
        if target.read_text(encoding='utf-8') != original:
            return False
        os.replace(candidate, target)
        return True
    finally:
        candidate.unlink(missing_ok=True)


def _passes(command) -> bool:
    try:
        return subprocess.run(command, capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _owned_edit(path: Path, edit) -> bool:
    """Run edit(); under a managed install, record the insertion for uninstall."""
    if not os.environ.get('HYPRWHSPR_GENERATION'):
        return edit()
    from managed_integrations import edit_files
    with edit_files([path], [path]) as proceed:
        return edit() if proceed else False


def _kdl_string(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _setup_niri(command: str) -> bool:
    path = _config_home() / 'niri' / 'config.kdl'
    spawn = ' '.join(_kdl_string(part) for part in shlex.split(command) + ['record'])
    line = f'Mod+Alt+D {{ spawn {spawn}; }}'
    where = f'the binds block in {path}'
    try:
        original = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return _manual('niri', where, line)
    if _has_record_binding(original, '//'):
        log_info(f"niri already has a dictation binding in {path}, skipping")
        return True
    # niri rejects a second binds block, so the line goes inside the first.
    opening = re.search(r'^binds\s*\{[^\n]*\n', original, re.MULTILINE)
    if not opening or not shutil.which('niri'):
        return _manual('niri', where, line)
    updated = (original[:opening.end()] + f'    // {MARKER}\n    {line}\n'
               + original[opening.end():])

    def edit():
        return _write_validated(path, original, updated,
                                lambda candidate: _passes(['niri', 'validate', '-c', str(candidate)]))
    if not _owned_edit(path, edit):
        return _manual('niri', where, line)
    log_success(f"Added Mod+Alt+D to {path}")
    return True


def _sway_config() -> Optional[Path]:
    for path in (_config_home() / 'sway' / 'config', USER_HOME / '.sway' / 'config'):
        if path.is_file():
            return path
    return None


def _sway_chord_taken(text: str) -> bool:
    """Sway silently lets a later bindsym win, so look for the chord ourselves."""
    wanted = {'mod4', 'mod1', 'd'}
    for raw in text.splitlines():
        words = raw.split()
        if len(words) < 2 or words[0] != 'bindsym':
            continue
        chord = next((word for word in words[1:] if not word.startswith('--')), '')
        keys = {{'$mod': 'mod4', 'alt': 'mod1'}.get(key, key) for key in chord.lower().split('+')}
        if keys == wanted:
            return True
    return False


def _setup_sway(command: str) -> bool:
    line = f'bindsym Mod4+Mod1+d exec {command} record'
    path = _sway_config()
    if path is None:
        return _manual('sway', '~/.config/sway/config', line)
    try:
        original = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return _manual('sway', str(path), line)
    if _has_record_binding(original, '#'):
        log_info(f"Sway already has a dictation binding in {path}, skipping")
        return True
    if _sway_chord_taken(original) or not shutil.which('sway'):
        return _manual('sway', str(path), line)
    updated = original + ('' if original.endswith('\n') else '\n') + f'\n# {MARKER}\n{line}\n'

    def edit():
        return _write_validated(path, original, updated,
                                lambda candidate: _passes(['sway', '-C', '-c', str(candidate)]))
    if not _owned_edit(path, edit):
        return _manual('sway', str(path), line)
    log_success(f"Added Mod4+Mod1+d to {path}")
    if os.environ.get('SWAYSOCK') and shutil.which('swaymsg'):
        _passes(['swaymsg', 'reload'])
    else:
        log_info("Reload Sway to apply the binding")
    return True


def _gsettings(*args) -> Optional[str]:
    try:
        result = subprocess.run(['gsettings', *args], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _gnome_paths() -> Optional[list]:
    raw = _gsettings('get', GNOME_SCHEMA, 'custom-keybindings')
    if raw is None:
        return None
    try:
        paths = ast.literal_eval(raw.removeprefix('@as').strip())
    except (ValueError, SyntaxError):
        return None
    return paths if isinstance(paths, list) and all(isinstance(p, str) for p in paths) else None


def _setup_gnome(command: str) -> bool:
    full = f'{command} record'
    where = 'Settings → Keyboard → View and Customize Shortcuts → Custom Shortcuts'
    manual = f'Super+Alt+D  →  {full}'
    paths = _gnome_paths() if shutil.which('gsettings') else None
    if paths is None:
        return _manual('gnome', where, manual)
    entry = f'{GNOME_SCHEMA}.custom-keybinding'
    if GNOME_PATH in paths:
        log_info("GNOME already has the hyprwhspr shortcut, skipping")
        return True
    for other in paths:
        if (_gsettings('get', f'{entry}:{other}', 'binding') or '').strip("'").lower() == GNOME_BINDING.lower():
            log_warning("Super+Alt+D is already a GNOME custom shortcut.")
            return _manual('gnome', where, manual)
    for key, value in (('name', 'hyprwhspr dictation'), ('command', full), ('binding', GNOME_BINDING)):
        if _gsettings('set', f'{entry}:{GNOME_PATH}', key, value) is None:
            return _manual('gnome', where, manual)
    # The list goes last: GNOME only looks at an entry once the list names it.
    if _gsettings('set', GNOME_SCHEMA, 'custom-keybindings', str(paths + [GNOME_PATH])) is None \
            or GNOME_PATH not in (_gnome_paths() or []):
        return _manual('gnome', where, manual)
    log_success("Added Super+Alt+D to GNOME custom shortcuts")
    return True


def _xfconf(*args) -> Optional[str]:
    try:
        result = subprocess.run(['xfconf-query', '-c', XFCE_CHANNEL, *args],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _setup_xfce(command: str) -> bool:
    full = f'{command} record'
    where = 'Settings → Keyboard → Application Shortcuts'
    manual = f'{XFCE_BINDING}  →  {full}'
    if not shutil.which('xfconf-query'):
        return _manual('xfce', where, manual)
    existing = _xfconf('-p', XFCE_PROPERTY)
    if existing is None:
        # A missing property is the normal first-install case; xfconf-query -n
        # creates it. A channel that is not running fails here and stays manual.
        if _xfconf('-n', '-t', 'string', '-s', full, '-p', XFCE_PROPERTY) is None:
            return _manual('xfce', where, manual)
        if _xfconf('-p', XFCE_PROPERTY) != full:
            return _manual('xfce', where, manual)
        log_success(f"Added {XFCE_BINDING} to XFCE application shortcuts")
        return True
    if _RECORD_BINDING.search(existing):
        log_info("XFCE already has the hyprwhspr shortcut, skipping")
        return True
    log_warning(f"{XFCE_BINDING} is already an XFCE shortcut: {existing}")
    return _manual('xfce', where, manual)


def _setup_kde(command: str) -> bool:
    print("  KDE: System Settings → Keyboard → Shortcuts → Add New → Command or Script")
    print(f"  Command: {command} record")
    print("  Shortcut: Meta+Alt+D")
    print("  Then set \"external_bindings\": true with: hyprwhspr config edit")
    return False


def setup_compositor_binding(kind: str) -> bool:
    """True once the compositor owns the shortcut; False leaves evdev in charge."""
    command = stable_tray_command(HYPRWHSPR_ROOT)
    if command is None:
        return False
    writer = {'niri': _setup_niri, 'sway': _setup_sway,
              'gnome': _setup_gnome, 'kde': _setup_kde, 'xfce': _setup_xfce}.get(kind)
    return writer(command) if writer else False
