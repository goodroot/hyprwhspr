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
from typing import Optional, Dict, Any

try:
    from .dependencies import require_package
except ImportError:
    from dependencies import require_package

pyperclip = require_package('pyperclip')

DEFAULT_PASTE_KEYCODE = 47  # Linux evdev KEY_V on QWERTY

# AT-SPI is not thread-safe — only one thread may access it at a time.
# _atspi_available: None = untested, True = working, False = unavailable.
# The module reference is cached after successful init so init runs only once.
_atspi_lock: threading.Lock = threading.Lock()
_atspi_module = None
_atspi_available = None


class TextInjector:
    """Handles injecting text into focused applications"""

    def __init__(self, config_manager=None):
        # Configuration
        self.config_manager = config_manager

        # Detect available injectors
        self.ydotool_available = self._check_ydotool()
        self.wtype_available = shutil.which('wtype') is not None

        if not self.ydotool_available and not self.wtype_available:
            print("⚠️  No injection backend found (wtype or ydotool). hyprwhspr requires wtype or ydotool for paste injection.")
        elif not self.wtype_available and self.ydotool_available:
            print("ℹ️  wtype not found. Falling back to ydotool for paste hotkey injection.")

    def _check_ydotool(self) -> bool:
        """Check if ydotool is available on the system"""
        try:
            result = subprocess.run(['which', 'ydotool'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

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

    def _get_active_window_info(self) -> Optional[Dict[str, Any]]:
        """Get active window info, trying multiple compositor APIs."""
        # Hyprland
        try:
            result = subprocess.run(
                ['hyprctl', 'activewindow', '-j'],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
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
                            return {'class': wm_class}
            except Exception:
                pass

        # AT-SPI fallback for native Wayland compositors (GNOME, KDE, etc.)
        # gi.repository ships with python3-gi, part of the GNOME/GTK stack —
        # no additional packages needed on systems where this problem exists.
        global _atspi_module, _atspi_available

        # Acquire the lock before reading _atspi_available so concurrent first-callers
        # cannot each pass the None check and spawn parallel Atspi.init() calls.
        if not _atspi_lock.acquire(timeout=0.5):
            return None
        try:
            if _atspi_available is None:
                # First call: probe in a thread with a timeout to guard against a
                # missing or slow AT-SPI bus. Caches the result for all future calls.
                _probe_result: list = [None]

                def _probe():
                    try:
                        import gi
                        gi.require_version('Atspi', '2.0')
                        from gi.repository import Atspi
                        Atspi.init()
                        _probe_result[0] = Atspi
                    except Exception:
                        pass

                t = threading.Thread(target=_probe, daemon=True)
                t.start()
                t.join(timeout=0.5)

                if _probe_result[0] is not None:
                    _atspi_module = _probe_result[0]
                    _atspi_available = True
                else:
                    _atspi_available = False

            if not _atspi_available:
                return None

            # Query under the same lock — AT-SPI is not thread-safe.
            Atspi = _atspi_module
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
                        # Prefer the process name from /proc (matches WM_CLASS-style
                        # identifiers like "gnome-terminal", "ptyxis", "kitty").
                        # Fall back to the AT-SPI app display name if unavailable.
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
                            return {'class': name}
        except Exception:
            pass
        finally:
            _atspi_lock.release()

        return None

    def _is_terminal(self, window_info: Optional[Dict[str, Any]] = None) -> bool:
        """Check if focused window is a terminal emulator."""
        if window_info is None:
            window_info = self._get_active_window_info()
        if not window_info:
            return False
        window_class = window_info.get('class', '').lower()
        terminals = {
            'ghostty', 'com.mitchellh.ghostty',
            'kitty',
            'wezterm', 'org.wezfurlong.wezterm',
            'alacritty',
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
        return window_class in terminals

    def _detect_paste_mode(self, window_info: Optional[Dict[str, Any]] = None) -> str:
        """Auto-detect paste key combo. Terminals → Ctrl+Shift+V, else → Ctrl+V."""
        if self._is_terminal(window_info):
            return 'ctrl_shift'
        return 'ctrl'

    def _clear_stuck_modifiers(self):
        """
        Clear any stuck modifier keys via ydotool uinput.
        Required after wtype paste: wtype sends Wayland modifier events, but
        ydotool's uinput layer may still consider those modifiers held, causing
        subsequent physical keypresses to behave incorrectly.
        """
        if not self.ydotool_available:
            return

        try:
            # Release common modifier keys that might be stuck:
            # 125 = LeftMeta/Super,  126 = RightMeta/Super
            # 56  = LeftAlt,         100 = RightAlt
            # 29  = LeftCtrl,        97  = RightCtrl
            # 42  = LeftShift,       54  = RightShift
            modifiers_to_clear = ['125:0', '126:0', '56:0', '100:0', '29:0', '97:0', '42:0', '54:0']
            subprocess.run(
                ['ydotool', 'key'] + modifiers_to_clear,
                capture_output=True,
                timeout=1
            )
        except Exception as e:
            print(f"Warning: Could not clear stuck modifiers: {e}")

    def _send_paste_keys_wtype(self, paste_mode: str) -> bool:
        """Send paste hotkey via wtype's Wayland virtual-keyboard protocol."""
        mode_map = {
            'ctrl_shift': ['-M', 'ctrl', '-M', 'shift', '-k', 'v', '-m', 'shift', '-m', 'ctrl'],
            'ctrl':       ['-M', 'ctrl', '-k', 'v', '-m', 'ctrl'],
            'super':      ['-M', 'logo', '-k', 'v', '-m', 'logo'],
            'alt':        ['-M', 'alt', '-k', 'v', '-m', 'alt'],
        }
        args = mode_map.get(paste_mode)
        if not args:
            return False
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

    # ydotool evdev keycodes for modifier press/release per paste mode.
    # Keys within each list are sent as a single ydotool command (simultaneous).
    # Release order is reversed so chord unwinds cleanly.
    _YDOTOOL_MOD_PRESS = {
        'ctrl_shift': ['29:1', '42:1'],  # Ctrl + Shift
        'ctrl':       ['29:1'],
        'super':      ['125:1'],
        'alt':        ['56:1'],
    }
    _YDOTOOL_MOD_RELEASE = {
        'ctrl_shift': ['42:0', '29:0'],  # reverse order
        'ctrl':       ['29:0'],
        'super':      ['125:0'],
        'alt':        ['56:0'],
    }

    def _send_paste_keys_slow(self, paste_mode: str) -> bool:
        """
        Send paste keystroke with delays between events via ydotool.
        Used as fallback when wtype is unavailable.
        """
        press_args = self._YDOTOOL_MOD_PRESS.get(paste_mode)
        release_args = self._YDOTOOL_MOD_RELEASE.get(paste_mode)
        if press_args is None:
            return False

        def _key(*args):
            result = subprocess.run(['ydotool', 'key'] + list(args), capture_output=True, timeout=1)
            if result.returncode != 0:
                stderr = (result.stderr or b'').decode('utf-8', 'ignore')
                raise RuntimeError(f"ydotool key {' '.join(args)} failed: {stderr}")

        try:
            paste_keycode = self._get_paste_keycode()
            _key(*press_args)
            time.sleep(0.015)
            _key(f'{paste_keycode}:1', f'{paste_keycode}:0')
            time.sleep(0.010)
            _key(*release_args)
            return True

        except Exception as e:
            print(f"Slow paste key injection failed: {e}")
            return False

    def _save_clipboard(self) -> Optional[bytes]:
        """Save current clipboard contents. Returns raw bytes or None."""
        if shutil.which("wl-paste"):
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

    def _restore_clipboard(self, saved: Optional[bytes], injected: Optional[bytes] = None, delay: float = 0.5):
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

                if shutil.which("wl-copy"):
                    subprocess.run(["wl-copy"], input=saved, check=True, timeout=2)
                else:
                    # pyperclip is text-only; only restore if the saved bytes are
                    # valid UTF-8 text. Binary clipboard data (images, etc.) cannot
                    # be round-tripped through pyperclip without corruption.
                    try:
                        pyperclip.copy(saved.decode("utf-8"))
                    except UnicodeDecodeError:
                        pass  # Binary data — skip rather than corrupt
            except Exception as e:
                print(f"Warning: Could not restore clipboard: {e}")

        threading.Thread(target=_restore, daemon=True).start()

    def _send_enter_if_auto_submit(self):
        """Send Enter key if auto_submit is enabled"""
        if not (self.config_manager and self.config_manager.get_setting('auto_submit', False)):
            return
        try:
            if self.ydotool_available:
                enter_result = subprocess.run(
                    ['ydotool', 'key', '28:1', '28:0'],  # 28 = Enter key
                    capture_output=True, timeout=1
                )
                if enter_result.returncode != 0:
                    stderr = (enter_result.stderr or b"").decode("utf-8", "ignore")
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
                print("  auto_submit enabled but no key-injection tool available (ydotool or wtype required)")
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
            print(f"post_transcription_hook failed: {e}")
            return text
        if result.returncode != 0:
            stderr = (result.stderr or '').strip()
            print(f"post_transcription_hook exited {result.returncode}: {stderr}")
            return text
        out = result.stdout.rstrip('\n')
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
        """Copy text to clipboard, then trigger paste via wtype (or ydotool fallback)."""
        try:
            window_info = self._get_active_window_info()
            saved_clipboard = self._save_clipboard()

            # Copy text to clipboard
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True, timeout=2)
            else:
                pyperclip.copy(text)
            time.sleep(0.15)

            # Resolve paste mode: explicit config override → shift_paste back-compat → auto-detect
            paste_mode = None
            if self.config_manager:
                paste_mode = self.config_manager.get_setting('paste_mode', None)
            if not paste_mode:
                # Back-compat: honour shift_paste boolean if set in config
                shift_paste = self.config_manager.get_setting('shift_paste', None) if self.config_manager else None
                if shift_paste is not None:
                    paste_mode = 'ctrl_shift' if shift_paste else 'ctrl'
                else:
                    paste_mode = self._detect_paste_mode(window_info)

            # Send paste hotkey: prefer wtype (Wayland virtual-keyboard), fall back to ydotool
            pasted = False
            if self.wtype_available:
                pasted = self._send_paste_keys_wtype(paste_mode)
                if pasted:
                    # wtype sends Wayland modifier events; clear ydotool's uinput modifier
                    # state so subsequent physical keypresses are not affected.
                    self._clear_stuck_modifiers()

            if not pasted and self.ydotool_available:
                self._clear_stuck_modifiers()
                time.sleep(0.02)
                pasted = self._send_paste_keys_slow(paste_mode)

            if not pasted and not self.wtype_available and not self.ydotool_available:
                print("No key-injection tool available; text is on the clipboard.")
                # Text is clipboard-only: don't restore old clipboard (would erase it)
                # and don't auto-submit (nothing was pasted into the field).
                return True

            # Only restore clipboard after a successful hotkey paste — if paste failed,
            # leave dictated text on clipboard so the user can paste manually.
            if pasted:
                restore_delay = 0.5
                if self.config_manager:
                    restore_delay = float(self.config_manager.get_setting('clipboard_clear_delay', 0.5))
                self._restore_clipboard(saved_clipboard, injected=text.encode("utf-8"), delay=restore_delay)
                self._send_enter_if_auto_submit()

            return pasted

        except Exception as e:
            print(f"Clipboard+hotkey injection failed: {e}")
            return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Fallback: copy text to clipboard when no paste tool is available."""
        try:
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True, timeout=2)
            else:
                pyperclip.copy(text)

            print("Text copied to clipboard (no paste tool available)")
            return True
        except Exception as e:
            print(f"ERROR: Clipboard fallback failed: {e}")
            return False
