"""
Text injector for hyprwhspr
Handles injecting transcribed text into other applications using paste strategy
"""

import os
import re
import sys
import shutil
import subprocess
import time
import threading
import json
import ast
from typing import Optional, Dict, Any, List, Tuple

try:
    from .ydotoold_session import YdotooldSession
except ImportError:
    from ydotoold_session import YdotooldSession

class _LazyPyperclip:
    """Load the optional fallback only when native clipboard tools are absent."""

    _module = None

    def _load(self):
        if self._module is not None:
            return self._module
        try:
            import pyperclip as module
        except ImportError as exc:
            raise RuntimeError(
                "clipboard fallback unavailable; install pyperclip and xclip (or xsel)"
            ) from exc

        # Avoid pyperclip's GTK fallback: mic_osd may already have selected GTK4,
        # while pyperclip requests GTK3. Prefer subprocess-only X11 backends.
        try:
            if shutil.which("xclip"):
                module.set_clipboard("xclip")
            elif shutil.which("xsel"):
                module.set_clipboard("xsel")
        except Exception:
            # Preserve pyperclip's own error for the eventual copy/paste call.
            pass
        self._module = module
        return module

    def __getattr__(self, name):
        return getattr(self._load(), name)


pyperclip = _LazyPyperclip()

DEFAULT_PASTE_KEYCODE = 47  # Linux evdev KEY_V on QWERTY
NON_XKB_INPUT_METHOD_LAYOUT = '__non_xkb_input_method__'

PASTE_MODE_CHORDS = {
    'ctrl_shift': 'ctrl+shift+v',
    'ctrl': 'ctrl+v',
    'super': 'super+v',
    'alt': 'alt+v',
}

WTYPE_MODIFIERS = {
    'ctrl': 'ctrl',
    'shift': 'shift',
    'super': 'logo',
    'alt': 'alt',
}

# Accepted modifier spellings -> canonical token (a key of WTYPE_MODIFIERS).
MODIFIER_ALIASES = {
    'ctrl': 'ctrl', 'control': 'ctrl',
    'shift': 'shift',
    'super': 'super', 'cmd': 'super', 'meta': 'super', 'logo': 'super',
    'alt': 'alt',
}

# wtype's `-k` resolves keys through libxkbcommon, whose keysym names are
# case-sensitive and differ from our normalized lowercase tokens (e.g. our
# "enter"/"backspace"/"pageup" are xkb "Return"/"BackSpace"/"Page_Up").
# Map the named keys; single letters and digits already match xkb keysym names.
# Function keys (f1..f24 -> F1..F24) are handled in _wtype_key_name().
WTYPE_KEY_NAMES = {
    'space': 'space',
    'tab': 'Tab',
    'enter': 'Return',
    'return': 'Return',
    'esc': 'Escape',
    'escape': 'Escape',
    'backspace': 'BackSpace',
    'delete': 'Delete',
    'home': 'Home',
    'end': 'End',
    'pageup': 'Page_Up',
    'pagedown': 'Page_Down',
    'up': 'Up',
    'down': 'Down',
    'left': 'Left',
    'right': 'Right',
    'minus': 'minus',
    'equal': 'equal',
    'comma': 'comma',
    'dot': 'period',
    'period': 'period',
    'slash': 'slash',
    'semicolon': 'semicolon',
    'apostrophe': 'apostrophe',
    'grave': 'grave',
    'leftbrace': 'bracketleft',
    'rightbrace': 'bracketright',
    'backslash': 'backslash',
}

YDOTOOL_MODIFIERS = {
    'ctrl': 29,
    'shift': 42,
    'super': 125,
    'alt': 56,
}

YDOTOOL_KEYCODES = {
    **{chr(ord('a') + i): code for i, code in enumerate([
        30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50,
        49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44,
    ])},
    **{str(i): code for i, code in enumerate([11, 2, 3, 4, 5, 6, 7, 8, 9, 10])},
    'space': 57,
    'tab': 15,
    'enter': 28,
    'return': 28,
    'esc': 1,
    'escape': 1,
    'backspace': 14,
    'delete': 111,
    'home': 102,
    'end': 107,
    'pageup': 104,
    'pagedown': 109,
    'up': 103,
    'down': 108,
    'left': 105,
    'right': 106,
    'minus': 12,
    'equal': 13,
    'comma': 51,
    'dot': 52,
    'period': 52,
    'slash': 53,
    'semicolon': 39,
    'apostrophe': 40,
    'grave': 41,
    'leftbrace': 26,
    'rightbrace': 27,
    'backslash': 43,
}

FKEY_CODES = {
    **{f'f{i}': 58 + i for i in range(1, 11)},
    'f11': 87,
    'f12': 88,
    **{f'f{i}': 183 + (i - 13) for i in range(13, 25)},
}

class TextInjector:
    """Handles injecting text into focused applications"""

    _LAYOUT_CACHE_TTL_S = 1.0

    def __init__(self, config_manager=None):
        # Configuration
        self.config_manager = config_manager

        # Detect available injectors once for the active display protocol.
        self.session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
        self.ydotool_available = self._check_ydotool()
        self.wtype_available = not self._is_x11_session() and shutil.which('wtype') is not None
        self.xdotool_available = self._is_x11_session() and shutil.which('xdotool') is not None
        self._hyprland_shortcut_syntax = None

        # Private ydotoold instance (lazily started on first uinput-fallback use, so
        # wtype-only sessions never spawn it). Replaces the old shared/managed
        # ydotool.service: hyprwhspr owns this daemon on its own socket.
        self._ydotoold = YdotooldSession()
        self._atspi_unavailable = False

        if (
            not self.ydotool_available
            and not self.wtype_available
            and not self.xdotool_available
            and not self._is_hyprland_session()
        ):
            print("⚠️  No injection backend found. Install wtype or ydotool on Wayland, or xdotool on X11.")
        elif not self._is_x11_session() and not self.wtype_available and self.ydotool_available:
            print("ℹ️  wtype not found. Falling back to ydotool for paste hotkey injection.")

    def _check_ydotool(self) -> bool:
        """Check if ydotool is usable (both the client and the ydotoold daemon)."""
        return YdotooldSession.is_available()

    def _is_x11_session(self) -> bool:
        """Use the display protocol captured when this injector was initialized."""
        session_type = getattr(self, 'session_type', os.environ.get('XDG_SESSION_TYPE', '').lower())
        return session_type == 'x11'

    def _is_hyprland_session(self) -> bool:
        """Return whether this is an active Hyprland Wayland session with IPC."""
        if self._is_x11_session() or shutil.which('hyprctl') is None:
            return False
        if os.environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
            return True
        desktop_values = (
            os.environ.get('XDG_CURRENT_DESKTOP', ''),
            os.environ.get('XDG_SESSION_DESKTOP', ''),
            os.environ.get('DESKTOP_SESSION', ''),
        )
        return any(
            token.strip().lower() == 'hyprland'
            for value in desktop_values
            for token in re.split(r'[:;,]', value)
        )

    def _run_ydotool(self, args, timeout):
        """Run a ydotool client command against our private ydotoold daemon.

        Ensures the daemon is running and points the client at our private socket
        via YDOTOOL_SOCKET. Returns the CompletedProcess, or None if the daemon
        could not be started (callers degrade gracefully).
        """
        if not self._ydotoold.ensure_running():
            return None
        return subprocess.run(
            ['ydotool', *args],
            capture_output=True,
            timeout=timeout,
            env=self._ydotoold.socket_env(),
        )

    def close(self):
        """Tear down the private ydotoold daemon. Idempotent; safe to call on exit."""
        ydotoold = getattr(self, '_ydotoold', None)
        if ydotoold is not None:
            ydotoold.close()

    def _get_paste_keycode(self) -> int:
        """
        Get the Linux evdev keycode used for the 'V' part of paste chords.

        ydotool's `key` command sends raw keycodes (physical keys). On non-QWERTY
        layouts, KEY_V (47) may not map to a keysym 'v', so Ctrl+KEY_V won't paste.
        Users can set either:
        - `paste_keycode_wev`: the Wayland/XKB keycode printed by `wev` (we subtract 8)
        - `paste_keycode`: the Linux evdev keycode directly (advanced)
        """
        keycode = DEFAULT_PASTE_KEYCODE
        if self.config_manager:
            wev_keycode = self.config_manager.get_setting('paste_keycode_wev', None)
            if wev_keycode is not None:
                try:
                    # wev reports Wayland/XKB keycodes, which are typically evdev+8
                    wev_keycode_int = int(wev_keycode)
                    converted = wev_keycode_int - 8
                    return converted if converted > 0 else DEFAULT_PASTE_KEYCODE
                except Exception:
                    # If parsing fails, fall back to evdev keycode setting
                    pass

            keycode = self.config_manager.get_setting('paste_keycode', DEFAULT_PASTE_KEYCODE)

        try:
            keycode_int = int(keycode)
            return keycode_int if keycode_int > 0 else DEFAULT_PASTE_KEYCODE
        except Exception:
            return DEFAULT_PASTE_KEYCODE

    def _has_custom_paste_keycode(self) -> bool:
        """Return True when config indicates a non-QWERTY paste key workaround."""
        if not self.config_manager:
            return False

        wev_keycode = self.config_manager.get_setting('paste_keycode_wev', None)
        if wev_keycode is not None:
            try:
                return int(wev_keycode) - 8 != DEFAULT_PASTE_KEYCODE
            except Exception:
                return True

        paste_keycode = self.config_manager.get_setting('paste_keycode', DEFAULT_PASTE_KEYCODE)
        try:
            return int(paste_keycode) != DEFAULT_PASTE_KEYCODE
        except Exception:
            return True

    def _read_active_layout(self) -> str:
        """Best-effort active XKB keyboard layout, lowercased (e.g. 'us', 'de').

        Returns '' when undetectable. Returns NON_XKB_INPUT_METHOD_LAYOUT when
        GNOME's active source is an input method rather than an XKB layout.
        Tries, in order: GNOME input-sources
        (the most-recently-used source is the active one), `localectl` (system
        X11 layout), then the XKB_DEFAULT_LAYOUT environment variable.
        """
        try:
            out = subprocess.run(
                ['gsettings', 'get', 'org.gnome.desktop.input-sources', 'mru-sources'],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                sources = []
                try:
                    parsed = ast.literal_eval(out.stdout.strip())
                    if isinstance(parsed, list):
                        sources = [
                            item for item in parsed
                            if isinstance(item, tuple) and len(item) >= 2
                        ]
                except Exception:
                    sources = re.findall(r"\('([^']+)',\s*'([^']+)'\)", out.stdout)
                if sources:
                    source_type, source_id = sources[0][0], sources[0][1]
                    if source_type == 'xkb':
                        return source_id.split('+')[0].lower()
                    return NON_XKB_INPUT_METHOD_LAYOUT
        except Exception:
            pass
        try:
            out = subprocess.run(['localectl', 'status'], capture_output=True, text=True, timeout=2)
            if out.returncode == 0:
                m = re.search(r'X11 Layout:\s*([^\s,]+)', out.stdout)
                if m:
                    layout = m.group(1).lower()
                    if layout not in {'(unset)', 'unset', 'n/a', 'none'}:
                        return layout
        except Exception:
            pass
        env_layout = os.environ.get('XKB_DEFAULT_LAYOUT', '')
        if env_layout:
            return env_layout.split(',')[0].strip().lower()
        return ''

    def _detect_active_layout(self) -> str:
        """Return the active layout using a short cache to avoid repeated stalls."""
        now = time.monotonic()
        cache_time = getattr(self, '_layout_cache_time', None)
        if cache_time is not None and now - cache_time < self._LAYOUT_CACHE_TTL_S:
            return getattr(self, '_layout_cache_value', '')

        layout = self._read_active_layout()
        self._layout_cache_value = layout
        self._layout_cache_time = now
        return layout

    def _layout_is_type_safe(self) -> bool:
        """True unless we positively detect a non-US keyboard layout.

        `ydotool type` assumes US/QWERTY keycodes and can only emit ASCII, so on
        non-US layouts (de, fr, ...) it mangles output (z<->y, ?-> _, dropped
        umlauts). There we must use layout-independent clipboard paste instead.
        Conservative toward the status quo: an unknown layout keeps direct typing.
        """
        layout = self._detect_active_layout()
        return layout == '' or layout.startswith('us')

    def _force_clipboard_paste(self) -> bool:
        """User override (config `prefer_clipboard_paste`): always use verbatim
        clipboard paste, never direct typing."""
        if not self.config_manager:
            return False
        return bool(self.config_manager.get_setting('prefer_clipboard_paste', False))

    def _get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """Get active window info, trying multiple compositor APIs."""
        # Niri
        try:
            result = subprocess.run(
                ['niri', 'msg', '--json', 'focused-window'],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode == 0:
                window = json.loads(result.stdout)
                app_id = window.get('app_id')
                if app_id:
                    return {
                        'class': app_id,
                        'title': window.get('title', ''),
                        'source': 'niri',
                    }
        except Exception:
            pass

        # Hyprland
        try:
            result = subprocess.run(
                ['hyprctl', 'activewindow', '-j'],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode == 0:
                window = json.loads(result.stdout)
                if isinstance(window, dict):
                    window.setdefault('source', 'hyprland')
                return window
        except Exception:
            pass

        # X11 / XWayland fallback (works on GNOME, KDE, etc. when XWayland is running)
        if shutil.which('xdotool') and shutil.which('xprop'):
            try:
                id_result = subprocess.run(
                    ['xdotool', 'getactivewindow'],
                    capture_output=True, text=True, timeout=0.5
                )
                if id_result.returncode == 0:
                    window_id = id_result.stdout.strip()
                    prop_result = subprocess.run(
                        ['xprop', '-id', window_id, 'WM_CLASS'],
                        capture_output=True, text=True, timeout=0.5
                    )
                    if prop_result.returncode == 0:
                        # WM_CLASS(STRING) = "ptyxis", "io.gitlab.ptyxis.Ptyxis"
                        # Use the second (instance) class which is more specific
                        matches = re.findall(r'"([^"]+)"', prop_result.stdout)
                        if matches:
                            wm_class = matches[-1] if len(matches) >= 2 else matches[0]
                            return {'class': wm_class, 'source': 'xwayland'}
            except Exception:
                pass

        # Do not probe AT-SPI outside a graphical session.
        if not os.environ.get('WAYLAND_DISPLAY') and not os.environ.get('DISPLAY'):
            return None

        # AT-SPI fallback for native Wayland compositors (GNOME, KDE, etc.)
        # Run this in a child process: dbind/AT-SPI can abort the interpreter when
        # the accessibility bus is unavailable, bypassing Python exception handling.
        if getattr(self, '_atspi_unavailable', False):
            return None

        try:
            code = r'''
import json
import sys

try:
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi
    Atspi.init()
    desktop = Atspi.get_desktop(0)
    for i in range(desktop.get_child_count()):
        app = desktop.get_child_at_index(i)
        if app is None:
            continue
        for j in range(app.get_child_count()):
            window = app.get_child_at_index(j)
            if window is None:
                continue
            if window.get_state_set().contains(Atspi.StateType.ACTIVE):
                name = None
                try:
                    pid = app.get_process_id()
                    if pid > 0:
                        with open(f'/proc/{pid}/comm') as f:
                            name = f.read().strip().lower()
                except Exception:
                    pass
                if not name:
                    name = (app.get_name() or '').lower()
                if name:
                    print(json.dumps({'class': name, 'source': 'at-spi'}))
                    sys.exit(0)
    sys.exit(2)
except Exception:
    sys.exit(1)
'''
            result = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True, text=True, timeout=0.7,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            if result.returncode != 2:
                self._atspi_unavailable = True
        except Exception:
            self._atspi_unavailable = True
            pass

        return None

    @staticmethod
    def _normalize_window_identifier(value: Any) -> str:
        """Normalize app/window identifiers for config matching."""
        text = str(value or '').strip().lower()
        if text.endswith('.desktop'):
            text = text[:-len('.desktop')]
        text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
        return text

    @classmethod
    def focused_window_identifiers(cls, window_info: Optional[Dict[str, Any]]) -> List[str]:
        """Return stable-ish identifiers that application config rules can match."""
        if not window_info:
            return []

        raw_values = []
        for key in ('class', 'app_id', 'initialClass', 'initialTitle'):
            value = window_info.get(key)
            if value:
                raw_values.append(value)
                if isinstance(value, str):
                    stable_value = value[:-len('.desktop')] if value.endswith('.desktop') else value
                    if '.' in stable_value:
                        raw_values.append(stable_value.rsplit('.', 1)[-1])

        title = window_info.get('title')
        if title:
            raw_values.append(title)
            for part in re.split(r'\s+[-–—]\s+', str(title)):
                raw_values.append(part)

        identifiers = []
        seen = set()
        for value in raw_values:
            ident = cls._normalize_window_identifier(value)
            if ident and ident not in seen:
                identifiers.append(ident)
                seen.add(ident)
        return identifiers

    def _get_application_rule(self, window_info: Optional[Dict[str, Any]]):
        """Return (identifier, rule) for the first matching application rule."""
        if not self.config_manager:
            return None, None
        applications = self.config_manager.get_setting('applications', {})
        if not isinstance(applications, dict) or not applications:
            return None, None

        normalized_rules = {}
        for key, rule in applications.items():
            ident = self._normalize_window_identifier(key)
            if ident:
                normalized_rules[ident] = rule

        for ident in self.focused_window_identifiers(window_info):
            if ident in normalized_rules:
                return ident, normalized_rules[ident]
        return None, None

    def _is_terminal(self, window_info: Optional[Dict[str, Any]] = None) -> bool:
        """Check if focused window is a terminal emulator."""
        if window_info is None:
            window_info = self._get_active_window_info()
        if not window_info:
            return False
        window_class = window_info.get('class', '').lower()
        window_identifiers = {window_class}
        if window_class.endswith('.desktop'):
            window_identifiers.add(window_class[:-len('.desktop')])
        if '.' in window_class:
            window_identifiers.add(window_class.rsplit('.', 1)[-1])

        terminals = {
            'ghostty', 'com.mitchellh.ghostty',
            'kitty',
            'wezterm', 'org.wezfurlong.wezterm',
            'alacritty', 'org.alacritty.alacritty',
            'foot',
            'konsole', 'org.kde.konsole',
            'gnome-terminal', 'org.gnome.terminal',
            'ptyxis', 'org.gnome.ptyxis', 'io.gitlab.ptyxis.ptyxis',
            'xfce4-terminal',
            'terminator',
            'tilix',
            'urxvt',
            'xterm',
            'st-256color',
            'sakura',
            'guake',
            'yakuake',
            'terminology',
            'cool-retro-term',
            'contour',
            'rio',
            'warp',
            'tabby',
            'hyper',
        }
        return bool(window_identifiers & terminals)

    def _detect_paste_mode(self, window_info: Optional[Dict[str, Any]] = None) -> str:
        """Auto-detect paste key combo. Terminals → Ctrl+Shift+V, else → Ctrl+V."""
        if self._is_terminal(window_info):
            return 'ctrl_shift'
        return 'ctrl'

    def _active_window_lookup_needed(self) -> bool:
        """True when injection actually consumes the focused-window identity.

        The window class feeds (a) per-application paste rules and (b) paste-mode
        auto-detection (terminal → Ctrl+Shift+V). When neither applies, the lookup
        result is unused — skipping it avoids spawning the AT-SPI probe subprocess
        on GNOME/KDE native-Wayland windows (no cost on Hyprland, which never
        reaches that branch).
        """
        if not self.config_manager:
            return True
        applications = self.config_manager.get_setting('applications', {})
        if isinstance(applications, dict) and applications:
            return True
        if self.config_manager.get_setting('paste_mode', None):
            return False  # explicit paste_mode — no auto-detect, no class needed
        if self.config_manager.get_setting('shift_paste', None) is not None:
            return False  # legacy explicit override — no auto-detect
        return True  # auto-detect path needs the window class

    def _chord_is_usable(self, chord: str) -> bool:
        """True if `chord` resolves to keys for at least one available backend."""
        parsed = self._parse_key_chord(PASTE_MODE_CHORDS.get(chord, chord))
        if not parsed:
            return False
        _modifiers, key = parsed
        if getattr(self, 'xdotool_available', False):
            return True
        if self._is_hyprland_session() and self._wtype_key_name(key) is not None:
            return True
        if self.wtype_available and self._wtype_key_name(key) is not None:
            return True
        if self.ydotool_available and self._keycode_for_chord_key(key) is not None:
            return True
        # No injection backend at all → can't paste regardless; a parseable chord
        # shouldn't trigger a spurious misconfiguration warning.
        return not self.wtype_available and not self.ydotool_available and not getattr(self, 'xdotool_available', False)

    def _resolve_paste_chord(self, window_info: Optional[Dict[str, Any]] = None):
        """Resolve application rule / explicit paste_mode / legacy / auto into a chord.

        Returns:
            (False, matched_identifier) when injection is disabled for the app.
            (chord_string, matched_identifier_or_None) otherwise.
        """
        matched_identifier, rule = self._get_application_rule(window_info)
        if isinstance(rule, dict) and 'auto_paste' in rule:
            auto_paste = rule.get('auto_paste')
            if auto_paste is False:
                return False, matched_identifier
            if isinstance(auto_paste, str) and auto_paste.strip():
                chord = auto_paste.strip()
                if not self._chord_is_usable(chord):
                    print(f"⚠️  Ignoring unusable auto_paste chord {chord!r} for app "
                          f"'{matched_identifier}'. Use a form like 'ctrl+v' or 'ctrl+y'; "
                          f"text will be left on the clipboard.", flush=True)
                return chord, matched_identifier

        paste_mode = None
        if self.config_manager:
            paste_mode = self.config_manager.get_setting('paste_mode', None)
        if not paste_mode:
            shift_paste = self.config_manager.get_setting('shift_paste', None) if self.config_manager else None
            if shift_paste is not None:
                paste_mode = 'ctrl_shift' if shift_paste else 'ctrl'
            else:
                paste_mode = self._detect_paste_mode(window_info)
        chord = PASTE_MODE_CHORDS.get(paste_mode, paste_mode)
        # Auto-detected modes ('ctrl'/'ctrl_shift') always resolve; only an
        # explicitly-configured paste_mode can be unusable here.
        if not self._chord_is_usable(chord):
            print(f"⚠️  Configured paste_mode {paste_mode!r} is not a usable chord; "
                  f"text will be left on the clipboard.", flush=True)
        return chord, matched_identifier

    def _clear_stuck_modifiers(self):
        """
        Clear any stuck modifier keys via ydotool uinput.
        Required after wtype paste: wtype sends Wayland modifier events, but
        ydotool's uinput layer may still consider those modifiers held, causing
        subsequent physical keypresses to behave incorrectly.

        Only runs when our ydotoold is *already* alive — there is nothing to clear
        before the daemon's first use, and we must not spawn it on wtype-only
        (wlroots) sessions just to release modifiers.
        """
        if not self.ydotool_available or not self._ydotoold.is_running():
            return

        try:
            # Release common modifier keys that might be stuck:
            # 125 = LeftMeta/Super,  126 = RightMeta/Super
            # 56  = LeftAlt,         100 = RightAlt
            # 29  = LeftCtrl,        97  = RightCtrl
            # 42  = LeftShift,       54  = RightShift
            modifiers_to_clear = ['125:0', '126:0', '56:0', '100:0', '29:0', '97:0', '42:0', '54:0']
            self._run_ydotool(['key'] + modifiers_to_clear, timeout=1)
        except Exception as e:
            print(f"Warning: Could not clear stuck modifiers: {e}")

    def _parse_key_chord(self, chord: str) -> Optional[Tuple[List[str], str]]:
        """Parse a single-key chord such as ctrl+shift+v.

        Modifiers are peeled off the front (separated by '+' or '-'); whatever
        remains is the key and is never split. This keeps the convenient
        'ctrl-shift-v' spelling working while still allowing a chord whose key
        is the literal '-' (e.g. 'super+-', normalized to 'minus').
        """
        if not isinstance(chord, str):
            return None
        remainder = chord.strip().lower()
        if not remainder:
            return None

        modifiers: List[str] = []
        while True:
            m = re.match(r'([a-z]+)\s*[+-]\s*(.+)$', remainder)
            if not m:
                break
            canonical = MODIFIER_ALIASES.get(m.group(1))
            if canonical is None:
                break  # leading token isn't a modifier — the rest is the key
            if canonical not in modifiers:
                modifiers.append(canonical)
            remainder = m.group(2).strip()

        key = 'minus' if remainder == '-' else remainder
        if not key:
            return None
        # A chord must terminate in a key, not a dangling modifier (e.g. 'ctrl+shift').
        if MODIFIER_ALIASES.get(key) in WTYPE_MODIFIERS:
            return None
        return modifiers, key

    def _send_paste_keys_wtype(self, paste_chord: str) -> bool:
        """Send paste hotkey via wtype's Wayland virtual-keyboard protocol."""
        paste_chord = PASTE_MODE_CHORDS.get(paste_chord, paste_chord)
        parsed = self._parse_key_chord(paste_chord)
        if not parsed:
            return False
        modifiers, key = parsed
        key_name = self._wtype_key_name(key)
        if key_name is None:
            return False
        args = []
        for modifier in modifiers:
            args.extend(['-M', WTYPE_MODIFIERS[modifier]])
        args.extend(['-k', key_name])
        for modifier in reversed(modifiers):
            args.extend(['-m', WTYPE_MODIFIERS[modifier]])
        try:
            result = subprocess.run(['wtype'] + args, capture_output=True, timeout=5)
            if result.returncode != 0:
                stderr = (result.stderr or b'').decode('utf-8', 'ignore')
                print(f"  wtype paste failed: {stderr}")
                return False
            return True
        except Exception as e:
            print(f"wtype paste failed: {e}")
            return False

    def _send_shortcut_hyprland(self, shortcut: str) -> bool:
        """Send a shortcut to Hyprland's active window without a virtual keyboard."""
        shortcut = PASTE_MODE_CHORDS.get(shortcut, shortcut)
        parsed = self._parse_key_chord(shortcut)
        if not parsed:
            return False
        modifiers, key = parsed
        key_name = self._wtype_key_name(key)
        if key_name is None:
            return False

        hypr_modifiers = ' + '.join(modifier.upper() for modifier in modifiers)
        lua = (
            'hl.dsp.send_shortcut({ '
            f'mods = "{hypr_modifiers}", key = "{key_name}", '
            'window = "activewindow" })'
        )
        legacy = f'{" ".join(modifier.upper() for modifier in modifiers)}, {key_name}, activewindow'
        commands = {
            'lua': ['hyprctl', 'dispatch', lua],
            'legacy': ['hyprctl', 'dispatch', 'sendshortcut', legacy],
        }
        cached = getattr(self, '_hyprland_shortcut_syntax', None)
        syntaxes = [cached] if cached in commands else []
        syntaxes.extend(syntax for syntax in ('lua', 'legacy') if syntax != cached)

        for syntax in syntaxes:
            try:
                result = subprocess.run(commands[syntax], capture_output=True, timeout=1)
            except Exception:
                result = None
            output = b'' if result is None else (
                (getattr(result, 'stdout', None) or b'')
                + (getattr(result, 'stderr', None) or b'')
            )
            if (
                result is not None
                and result.returncode == 0
                and b'invalid dispatcher' not in output.lower()
            ):
                self._hyprland_shortcut_syntax = syntax
                return True
            if syntax == cached:
                self._hyprland_shortcut_syntax = None
        return False

    def _send_paste_keys_xdotool(self, paste_chord: str) -> bool:
        """Send a symbolic paste chord through the native X11 input path."""
        paste_chord = PASTE_MODE_CHORDS.get(paste_chord, paste_chord)
        parsed = self._parse_key_chord(paste_chord)
        if not parsed:
            return False
        modifiers, key = parsed
        chord = '+'.join([*modifiers, key])
        try:
            result = subprocess.run(
                ['xdotool', 'key', '--clearmodifiers', chord],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return True
            stderr = (result.stderr or b'').decode('utf-8', 'ignore')
            print(f"  xdotool paste failed: {stderr}")
        except Exception as e:
            print(f"xdotool paste failed: {e}")
        return False

    @staticmethod
    def _wtype_key_name(key: str) -> Optional[str]:
        """Translate a normalized chord key into the xkb keysym name wtype expects."""
        if key in WTYPE_KEY_NAMES:
            return WTYPE_KEY_NAMES[key]
        if key in FKEY_CODES:  # f1..f24 -> F1..F24
            return 'F' + key[1:]
        if len(key) == 1 and (key.isalpha() or key.isdigit()):
            return key  # single letters/digits match xkb keysym names directly
        return None

    def _keycode_for_chord_key(self, key: str) -> Optional[int]:
        if key == 'v':
            return self._get_paste_keycode()
        if key in FKEY_CODES:
            return FKEY_CODES[key]
        return YDOTOOL_KEYCODES.get(key)

    def _send_paste_keys_slow(self, paste_chord: str) -> bool:
        """
        Send paste keystroke with delays between events via ydotool.
        Used as fallback when wtype is unavailable.
        """
        paste_chord = PASTE_MODE_CHORDS.get(paste_chord, paste_chord)
        parsed = self._parse_key_chord(paste_chord)
        if not parsed:
            return False
        modifiers, key = parsed
        keycode = self._keycode_for_chord_key(key)
        if keycode is None:
            return False
        press_args = [f'{YDOTOOL_MODIFIERS[modifier]}:1' for modifier in modifiers]
        release_args = [f'{YDOTOOL_MODIFIERS[modifier]}:0' for modifier in reversed(modifiers)]

        def _key(*args):
            result = self._run_ydotool(['key'] + list(args), timeout=1)
            if result is None:
                raise RuntimeError("ydotoold unavailable")
            if result.returncode != 0:
                stderr = (result.stderr or b'').decode('utf-8', 'ignore')
                raise RuntimeError(f"ydotool key {' '.join(args)} failed: {stderr}")

        try:
            if press_args:
                _key(*press_args)
            time.sleep(0.015)
            _key(f'{keycode}:1', f'{keycode}:0')
            time.sleep(0.010)
            if release_args:
                _key(*release_args)
            return True

        except Exception as e:
            print(f"Slow paste key injection failed: {e}")
            return False

    def _is_gnome_wayland_session(self) -> bool:
        """Return True for Mutter/GNOME Wayland sessions where uinput chords are unreliable."""
        session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
        if session_type and session_type != 'wayland':
            return False
        if not session_type and not os.environ.get('WAYLAND_DISPLAY'):
            return False

        desktop_values = [
            os.environ.get('XDG_CURRENT_DESKTOP', ''),
            os.environ.get('XDG_SESSION_DESKTOP', ''),
            os.environ.get('DESKTOP_SESSION', ''),
        ]
        desktop = ':'.join(desktop_values).lower()
        desktop_tokens = set(filter(None, re.split(r'[^a-z0-9]+', desktop)))
        return bool(desktop_tokens & {'gnome', 'mutter', 'pop'})

    def _type_text_ydotool(self, text: str) -> bool:
        """
        Type text directly with ydotool.

        This is slower and less layout-aware than clipboard paste on compositors
        where paste chords work, but it avoids Mutter dropping synthetic modifier
        combos from uinput devices.
        """
        if not self.ydotool_available:
            return False

        try:
            result = self._run_ydotool(
                ['type', '--key-delay', '5', '--key-hold', '5', '--', text],
                timeout=10,
            )
            if result is None or result.returncode != 0:
                stderr = (result.stderr or b'').decode('utf-8', 'ignore') if result else 'ydotoold unavailable'
                print(f"  ydotool type failed: {stderr}")
                return False
            return True
        except Exception as e:
            print(f"ydotool type injection failed: {e}")
            return False

    def _gnome_force_latin_layout(self):
        """Temporarily switch the GNOME input source to a Latin (ASCII) layout so
        the raw-keycode paste chord (Ctrl+V) resolves to keysym 'v'.

        The clipboard chord is sent as physical keycodes via ydotool, and Mutter
        translates them through the *active* XKB layout. On a non-Latin layout
        (Thai, Russian, Arabic, Greek, Hebrew, …) KEY_V has no 'v' keysym at all,
        so Ctrl+KEY_V never triggers paste and the dictated text silently fails to
        land. (This is distinct from the `paste_keycode` workaround, which only
        helps Latin layouts like Dvorak where 'v' merely moves to another key.)

        Returns the previous input-source index to restore, or None when no switch
        is needed/possible (not GNOME, no Latin source configured, already Latin).
        """
        if not self._is_gnome_wayland_session():
            return None
        try:
            import ast
            schema = 'org.gnome.desktop.input-sources'
            srcs_raw = subprocess.run(['gsettings', 'get', schema, 'sources'],
                                      capture_output=True, text=True, timeout=1).stdout.strip()
            cur_raw = subprocess.run(['gsettings', 'get', schema, 'current'],
                                     capture_output=True, text=True, timeout=1).stdout.strip()
            cur_idx = int(cur_raw.split()[-1])
            sources = ast.literal_eval(srcs_raw)  # e.g. [('xkb', 'us'), ('xkb', 'th')]
            latin = {'us', 'gb', 'dvorak', 'colemak', 'workman', 'de', 'fr', 'es',
                     'it', 'pt', 'latam', 'no', 'se', 'dk', 'fi'}
            latin_idx = None
            for i, (stype, name) in enumerate(sources):  # prefer plain 'us'
                if stype == 'xkb' and name.split('+')[0] == 'us':
                    latin_idx = i
                    break
            if latin_idx is None:
                for i, (stype, name) in enumerate(sources):
                    if stype == 'xkb' and name.split('+')[0] in latin:
                        latin_idx = i
                        break
            if latin_idx is None or latin_idx == cur_idx:
                return None
            subprocess.run(['gsettings', 'set', schema, 'current', f'uint32 {latin_idx}'], timeout=1)
            time.sleep(0.12)  # let gnome-shell apply the layout before the chord
            return cur_idx
        except Exception as e:
            print(f"  paste layout switch skipped: {e}")
            return None

    def _gnome_restore_layout(self, prev_idx):
        """Restore the input source saved by _gnome_force_latin_layout(). No-op
        when prev_idx is None."""
        if prev_idx is None:
            return
        try:
            subprocess.run(['gsettings', 'set', 'org.gnome.desktop.input-sources',
                            'current', f'uint32 {prev_idx}'], timeout=1)
        except Exception:
            pass

    def _save_clipboard(self) -> Optional[bytes]:
        """Save current clipboard contents. Returns raw bytes or None."""
        if not self._is_x11_session() and shutil.which("wl-paste"):
            try:
                result = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, timeout=2)
                if result.returncode == 0:
                    return result.stdout
            except Exception:
                pass
        # Fallback: pyperclip (X11 or non-standard Wayland setups)
        try:
            text = pyperclip.paste()
            if text:
                return text.encode("utf-8")
        except Exception:
            pass
        return None

    def _try_wl_copy(self, data: bytes) -> Tuple[bool, Optional[str]]:
        """Try the Wayland clipboard writer and retain a useful failure detail."""
        if self._is_x11_session() or not shutil.which("wl-copy"):
            return False, None
        try:
            result = subprocess.run(["wl-copy"], input=data, timeout=2)
            if result.returncode == 0:
                return True, None
            return False, f"wl-copy exited with status {result.returncode}"
        except Exception as exc:
            return False, f"wl-copy failed: {exc}"

    def _copy_text_to_clipboard(self, text: str) -> bool:
        """Copy text to the clipboard without triggering paste."""
        copied, wayland_error = self._try_wl_copy(text.encode("utf-8"))
        if copied:
            return True
        # Fallback: pyperclip (X11, or wl-copy present but no compositor to reach)
        try:
            pyperclip.copy(text)
            return True
        except Exception as e:
            detail = f" after {wayland_error}" if wayland_error else ""
            print(f"ERROR: Clipboard copy failed{detail}: {e}")
            return False

    def _restore_clipboard(self, saved: Optional[bytes], injected: Optional[bytes] = None, delay: float = 5.0):
        """Restore clipboard to saved contents after a delay (background thread).

        If `injected` is provided, the restore is skipped if the clipboard no longer
        contains the injected text — meaning the user has copied something else.
        """
        if saved is None:
            return

        def _restore():
            time.sleep(delay)
            try:
                # Guard: if the user copied something else during the delay, don't clobber it.
                if injected is not None:
                    current = self._save_clipboard()
                    if current != injected:
                        return

                restored, wayland_error = self._try_wl_copy(saved)
                if not restored:
                    # Fallback: pyperclip (X11, or wl-copy present but no compositor to
                    # reach). It's text-only; only restore if the saved bytes are valid
                    # UTF-8 text. Binary clipboard data (images, etc.) cannot be
                    # round-tripped through pyperclip without corruption.
                    try:
                        pyperclip.copy(saved.decode("utf-8"))
                    except UnicodeDecodeError:
                        if wayland_error:
                            print(f"Warning: Could not restore binary clipboard: {wayland_error}")
                        # Binary data cannot safely pass through the text fallback.
                    except Exception as exc:
                        detail = f" after {wayland_error}" if wayland_error else ""
                        raise RuntimeError(f"clipboard fallback failed{detail}: {exc}") from exc
            except Exception as e:
                print(f"Warning: Could not restore clipboard: {e}")

        threading.Thread(target=_restore, daemon=True).start()

    def _send_enter_if_auto_submit(self):
        """Send Enter key if auto_submit is enabled"""
        if not (self.config_manager and self.config_manager.get_setting('auto_submit', False)):
            return
        try:
            if self._is_x11_session() and getattr(self, 'xdotool_available', False):
                enter_result = subprocess.run(
                    ['xdotool', 'key', '--clearmodifiers', 'Return'],
                    capture_output=True, timeout=1,
                )
                if enter_result.returncode != 0:
                    stderr = (enter_result.stderr or b'').decode('utf-8', 'ignore')
                    print(f"  xdotool Enter key failed: {stderr}")
            elif self._is_hyprland_session() and self._send_shortcut_hyprland('enter'):
                return
            elif self.ydotool_available:
                enter_result = self._run_ydotool(['key', '28:1', '28:0'], timeout=1)  # 28 = Enter
                if enter_result is None or enter_result.returncode != 0:
                    stderr = (enter_result.stderr or b"").decode("utf-8", "ignore") if enter_result else "ydotoold unavailable"
                    print(f"  ydotool Enter key failed: {stderr}")
            elif self.wtype_available:
                enter_result = subprocess.run(
                    ['wtype', '-k', 'Return'],
                    capture_output=True, timeout=1
                )
                if enter_result.returncode != 0:
                    stderr = (enter_result.stderr or b"").decode("utf-8", "ignore")
                    print(f"  wtype Enter key failed: {stderr}")
            else:
                print("  auto_submit enabled but no key-injection tool available")
        except Exception as e:
            print(f"  auto_submit Enter key failed: {e}")

    # ------------------------ Public API ------------------------

    def inject_text(self, text: str) -> bool:
        """
        Inject text into the currently focused application

        Args:
            text: Text to inject

        Returns:
            True if successful, False otherwise
        """
        if not text or text.strip() == "":
            print("No text to inject (empty or whitespace)")
            return True

        # Preprocess; also trim trailing newlines (avoid unwanted Enter)
        processed_text = self._preprocess_text(text).rstrip("\r\n")
        processed_text = self._run_post_transcription_hook(processed_text) + ' '

        try:
            inject_mode = None
            if self.config_manager:
                inject_mode = self.config_manager.get_setting('inject_mode', None)

            if inject_mode in ('wtype', 'ydotool_type'):
                print(f"⚠️  inject_mode='{inject_mode}' is deprecated: direct typing drops characters at speed. "
                      f"Using clipboard+paste instead.")

            return self._inject_via_clipboard_and_hotkey(processed_text)

        except Exception as e:
            print(f"Primary injection method failed: {e}")
            return False

    # ------------------------ Helpers ------------------------

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to handle common speech-to-text corrections and remove unwanted line breaks
        """
        # Normalize line breaks to spaces to avoid unintended "Enter"
        processed = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')

        # Apply user-defined overrides first
        processed = self._apply_word_overrides(processed)

        # Filter filler words if enabled
        processed = self._filter_filler_words(processed)

        # Built-in speech-to-text replacements (can be disabled via config)
        symbol_replacements_enabled = True
        if self.config_manager:
            symbol_replacements_enabled = self.config_manager.get_setting('symbol_replacements', True)

        if not symbol_replacements_enabled:
            # Collapse runs of whitespace (newlines already normalized to spaces on line 243)
            processed = re.sub(r'[ \t]+', ' ', processed)
            return processed.strip()

        replacements = {
            r'\bperiod\b': '.',
            r'\bcomma\b': ',',
            r'\bquestion mark\b': '?',
            r'\bexclamation mark\b': '!',
            r'\bcolon\b': ':',
            r'\bsemicolon\b': ';',
            r'\bnew line\b': '\n',
            r'\btab\b': '\t',
            r'\bdash\b': '-',
            r'\bunderscore\b': '_',
            r'\bopen paren\b': '(',
            r'\bclose paren\b': ')',
            r'\bopen bracket\b': '[',
            r'\bclose bracket\b': ']',
            r'\bopen brace\b': '{',
            r'\bclose brace\b': '}',
            r'\bat symbol\b': '@',
            r'\bhash\b': '#',
            r'\bdollar sign\b': '$',
            r'\bpercent\b': '%',
            r'\bcaret\b': '^',
            r'\bampersand\b': '&',
            r'\basterisk\b': '*',
            r'\bplus\b': '+',
            r'\bequals\b': '=',
            r'\bless than\b': '<',
            r'\bgreater than\b': '>',
            r'\bslash\b': '/',
            r'\bbackslash\b': r'\\',
            r'\bpipe\b': '|',
            r'\btilde\b': '~',
            r'\bgrave\b': '`',
            r'\bquote\b': '"',
            r'\bapostrophe\b': "'",
        }

        for pattern, replacement in replacements.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

        # Collapse runs of whitespace, preserve intentional newlines
        processed = re.sub(r'[ \t]+', ' ', processed)
        processed = re.sub(r' *\n *', '\n', processed)
        processed = processed.strip()

        return processed

    def _run_post_transcription_hook(self, text: str) -> str:
        """Pipe text through the user's post_transcription_hook shell command.

        Stdin = text; non-empty stdout replaces it (empty stdout = observer-only).
        Env: HYPRWHSPR_MODEL, HYPRWHSPR_BACKEND. 5s timeout. Any error
        preserves the original text — a broken hook must never eat a dictation.
        """
        if not (self.config_manager and text):
            return text
        cmd = self.config_manager.get_setting('post_transcription_hook', None)
        if not (isinstance(cmd, str) and cmd.strip()):
            return text

        env = os.environ.copy()
        env['HYPRWHSPR_MODEL'] = str(self.config_manager.get_setting('model', '') or '')
        env['HYPRWHSPR_BACKEND'] = str(self.config_manager.get_setting('transcription_backend', '') or '')

        # shell=True is deliberate: the command is user-authored config, same trust
        # level as the rest of config.json, and users rely on pipes/redirects/chains.
        try:
            result = subprocess.run(
                cmd, shell=True, input=text, capture_output=True,
                text=True, timeout=5.0, env=env,
            )
        except Exception as e:
            print(f"post_transcription_hook failed: {e}", flush=True)
            return text
        if result.returncode != 0:
            stderr = (result.stderr or '').strip()
            print(f"post_transcription_hook exited {result.returncode}: {stderr}", flush=True)
            return text
        out = result.stdout.rstrip("\r\n")
        return out if out else text

    def _apply_word_overrides(self, text: str) -> str:
        """Apply user-defined word overrides to the text"""
        if not self.config_manager:
            return text

        word_overrides = self.config_manager.get_word_overrides()
        if not word_overrides:
            return text

        processed = text
        for original, replacement in word_overrides.items():
            # Only require original to be non-empty; replacement can be empty string to delete words
            if original:
                if len(original) == 1:
                    # Single characters can't use \b word boundaries (e.g. ß mid-word in Straße)
                    processed = re.sub(re.escape(original), replacement, processed, flags=re.IGNORECASE)
                else:
                    pattern = r'\b' + re.escape(original) + r'\b'
                    processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

        # Clean up extra spaces left by word deletions (multiple spaces -> single space)
        processed = re.sub(r' +', ' ', processed)
        processed = processed.strip()

        return processed

    def _filter_filler_words(self, text: str) -> str:
        """Remove filler words like uh, um, er if enabled in config"""
        if not self.config_manager:
            return text

        if not self.config_manager.get_filter_filler_words():
            return text

        filler_words = self.config_manager.get_filler_words()
        if not filler_words:
            return text

        processed = text
        for word in filler_words:
            if word:
                pattern = r'\b' + re.escape(word) + r'\b'
                processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)

        # Clean up extra spaces left by word deletions
        processed = re.sub(r' +', ' ', processed)
        processed = processed.strip()

        return processed

    # ------------------------ Paste injection (primary method) ------------------------

    def _inject_via_clipboard_and_hotkey(self, text: str) -> bool:
        """Copy text to clipboard, then trigger the compositor-native paste path."""
        try:
            window_info = (
                self._get_active_window_info()
                if self._active_window_lookup_needed()
                else None
            )
            gnome_wayland_session = self._is_gnome_wayland_session()
            paste_chord, app_match = self._resolve_paste_chord(window_info)
            if paste_chord is False:
                # Injection is explicitly disabled for this app. Do nothing at all —
                # no paste, no clipboard write. "Disabled" means hands off, which also
                # avoids leaking dictated text (e.g. into a password field) onto the
                # clipboard where other apps could read it.
                print(f"Injection disabled for focused app ({app_match}); leaving it untouched.")
                return True

            # On GNOME/Mutter the layer-shell overlay is unavailable, so we can
            # type directly with `ydotool type` to avoid touching the clipboard.
            # But that is ONLY correct for pure-ASCII text on a US layout:
            # ydotool type assumes US keycodes and can't emit non-ASCII, so on a
            # non-US layout (z<->y, ?-> _) or with umlauts/typographic characters
            # it mangles the output. In those cases — or with a custom paste
            # keycode / the prefer_clipboard_paste override — fall through to the
            # layout-independent verbatim clipboard paste below.
            if (
                gnome_wayland_session
                and self.ydotool_available
                and app_match is None
                and not self._has_custom_paste_keycode()
                and not self._force_clipboard_paste()
                and self._layout_is_type_safe()
                and text.isascii()
            ):
                self._clear_stuck_modifiers()
                time.sleep(0.05)
                typed = self._type_text_ydotool(text)
                if typed:
                    self._send_enter_if_auto_submit()
                    return True

            saved_clipboard = self._save_clipboard()

            # Copy text to clipboard
            if not self._copy_text_to_clipboard(text):
                return False
            time.sleep(0.15)

            # Send paste hotkey through the session-native path first: xdotool on
            # X11, Hyprland's dispatcher there, or wtype on other Wayland sessions.
            # Fall back to wtype and then ydotool if native Hyprland dispatch fails.
            # ydotool key chords DO reach Mutter (uinput
            # is seen as a real device, unlike wtype's virtual-keyboard protocol
            # which Mutter blocks), so we use them on GNOME too — this is the path
            # taken when direct typing was skipped for a non-US layout / non-ASCII text.
            pasted = False
            if self._is_x11_session() and getattr(self, 'xdotool_available', False):
                pasted = self._send_paste_keys_xdotool(paste_chord)
            elif self._is_hyprland_session():
                pasted = self._send_shortcut_hyprland(paste_chord)
                if not pasted and self.wtype_available:
                    pasted = self._send_paste_keys_wtype(paste_chord)
                    if pasted:
                        self._clear_stuck_modifiers()
            elif self.wtype_available:
                pasted = self._send_paste_keys_wtype(paste_chord)
                if pasted:
                    # wtype sends Wayland modifier events; clear ydotool's uinput modifier
                    # state so subsequent physical keypresses are not affected.
                    self._clear_stuck_modifiers()

            if not pasted and self.ydotool_available:
                self._clear_stuck_modifiers()
                time.sleep(0.02)
                # Non-Latin layouts (Thai, Russian, …) remap KEY_V, so the raw-keycode
                # Ctrl+V chord lands as the wrong keysym and paste silently fails. Force
                # a Latin layout just for the chord, then restore — the clipboard text is
                # Unicode and pastes correctly regardless of the active layout.
                _prev_layout = self._gnome_force_latin_layout()
                try:
                    pasted = self._send_paste_keys_slow(paste_chord)
                finally:
                    self._gnome_restore_layout(_prev_layout)

            if (
                not pasted
                and not self.wtype_available
                and not self.ydotool_available
                and not getattr(self, 'xdotool_available', False)
            ):
                print("No key-injection tool available; text is on the clipboard.")
                # Text is clipboard-only: don't restore old clipboard (would erase it)
                # and don't auto-submit (nothing was pasted into the field).
                return True

            # Only restore clipboard after successful injection — if injection failed,
            # leave dictated text on clipboard so the user can paste manually. GNOME is
            # the exception: the ydotool chord is an automatic fallback after direct
            # typing was skipped, and a failed chord should not clobber the user's
            # previous clipboard.
            if pasted:
                restore_delay = 5.0
                if self.config_manager:
                    restore_delay = float(self.config_manager.get_setting('clipboard_clear_delay', 5.0))
                self._restore_clipboard(saved_clipboard, injected=text.encode("utf-8"), delay=restore_delay)
                self._send_enter_if_auto_submit()
            elif gnome_wayland_session:
                self._restore_clipboard(saved_clipboard, injected=text.encode("utf-8"), delay=0)

            return pasted

        except Exception as e:
            print(f"Clipboard+hotkey injection failed: {e}")
            return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Fallback: copy text to clipboard when no paste tool is available."""
        if self._copy_text_to_clipboard(text):
            print("Text copied to clipboard (no paste tool available)")
            return True
        return False
