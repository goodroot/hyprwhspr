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


class TextInjector:
    """Handles injecting text into focused applications"""

    def __init__(self, config_manager=None):
        # Configuration
        self.config_manager = config_manager

        # Detect available injectors
        self.ydotool_available = self._check_ydotool()

        if not self.ydotool_available:
            print("⚠️  No typing backend found (ydotool). hyprwhspr requires ydotool for paste injection.")

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
        """Get active window info from Hyprland (if available)"""
        try:
            result = subprocess.run(
                ['hyprctl', 'activewindow', '-j'],
                capture_output=True, text=True, timeout=0.5
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return None

    def _is_kitty_protocol_terminal(self, window_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if focused window is a terminal that uses Kitty keyboard protocol.
        These terminals need special handling to avoid escape sequence artifacts.
        """
        if window_info is None:
            window_info = self._get_active_window_info()

        if not window_info:
            return False

        # Known terminals that use Kitty keyboard protocol
        kitty_terminals = {
            'ghostty',
            'kitty',
            'wezterm',
            'org.wezfurlong.wezterm'
        }

        window_class = window_info.get('class', '').lower()
        return any(term in window_class for term in kitty_terminals)

    def _clear_stuck_modifiers(self):
        """
        Clear any stuck modifier keys that might interfere with paste.
        This is especially important for Kitty-protocol terminals that can
        misinterpret synthetic key events when modifiers are stuck.
        """
        if not self.ydotool_available:
            return

        try:
            # Release common modifier keys that might be stuck:
            # 125 = LeftMeta/Super
            # 126 = RightMeta/Super
            # 56 = LeftAlt
            # 100 = RightAlt
            # 29 = LeftCtrl
            # 97 = RightCtrl
            # 42 = LeftShift
            # 54 = RightShift

            modifiers_to_clear = ['125:0', '126:0', '56:0', '100:0', '29:0', '97:0', '42:0', '54:0']

            subprocess.run(
                ['ydotool', 'key'] + modifiers_to_clear,
                capture_output=True,
                timeout=1
            )
        except Exception as e:
            # Non-fatal, just log
            print(f"Warning: Could not clear stuck modifiers: {e}")

    def _send_paste_keys_slow(self, paste_mode: str) -> bool:
        """
        Send paste keystroke with delays between events.
        This prevents Kitty-protocol terminals from misinterpreting
        the key sequence when modifiers arrive too quickly.
        """
        try:
            paste_keycode = self._get_paste_keycode()
            paste_keycode_pressed = f'{paste_keycode}:1'
            paste_keycode_released = f'{paste_keycode}:0'

            if paste_mode == 'super':
                # Super+V with delays: Super down, delay, V down, V up, Super up
                subprocess.run(['ydotool', 'key', '125:1'], capture_output=True, timeout=1)
                time.sleep(0.015)
                subprocess.run(['ydotool', 'key', paste_keycode_pressed, paste_keycode_released], capture_output=True, timeout=1)
                time.sleep(0.010)
                subprocess.run(['ydotool', 'key', '125:0'], capture_output=True, timeout=1)

            elif paste_mode == 'ctrl_shift':
                # Ctrl+Shift+V with delays: mods down, delay, V, delay, mods up
                subprocess.run(['ydotool', 'key', '29:1', '42:1'], capture_output=True, timeout=1)
                time.sleep(0.015)
                subprocess.run(['ydotool', 'key', paste_keycode_pressed, paste_keycode_released], capture_output=True, timeout=1)
                time.sleep(0.010)
                subprocess.run(['ydotool', 'key', '42:0', '29:0'], capture_output=True, timeout=1)

            elif paste_mode == 'ctrl':
                # Ctrl+V with delays
                subprocess.run(['ydotool', 'key', '29:1'], capture_output=True, timeout=1)
                time.sleep(0.015)
                subprocess.run(['ydotool', 'key', paste_keycode_pressed, paste_keycode_released], capture_output=True, timeout=1)
                time.sleep(0.010)
                subprocess.run(['ydotool', 'key', '29:0'], capture_output=True, timeout=1)

            elif paste_mode == 'alt':
                # Alt+V with delays
                subprocess.run(['ydotool', 'key', '56:1'], capture_output=True, timeout=1)
                time.sleep(0.015)
                subprocess.run(['ydotool', 'key', paste_keycode_pressed, paste_keycode_released], capture_output=True, timeout=1)
                time.sleep(0.010)
                subprocess.run(['ydotool', 'key', '56:0'], capture_output=True, timeout=1)

            else:
                return False

            return True

        except Exception as e:
            print(f"Slow paste key injection failed: {e}")
            return False

    def _send_enter_if_auto_submit(self):
        """Send Enter key if auto_submit is enabled"""
        if self.config_manager and self.config_manager.get_setting('auto_submit', False):
            try:
                enter_result = subprocess.run(
                    ['ydotool', 'key', '28:1', '28:0'],  # 28 = Enter key
                    capture_output=True, timeout=1
                )
                if enter_result.returncode != 0:
                    stderr = (enter_result.stderr or b"").decode("utf-8", "ignore")
                    print(f"  ydotool Enter key failed: {stderr}")
            except Exception as e:
                print(f"  auto_submit Enter key failed: {e}")

    def _clear_clipboard(self):
        """Clear the clipboard by setting it to empty content"""
        try:
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=b"", check=True)
            else:
                pyperclip.copy("")
        except Exception as e:
            print(f"Warning: Could not clear clipboard: {e}")

    def _schedule_clipboard_clear(self, delay: float):
        """Schedule clipboard clearing after the specified delay"""
        def clear_after_delay():
            time.sleep(delay)
            self._clear_clipboard()
            print(f"📋 Clipboard cleared after {delay}s delay")
        
        # Run in a separate thread to avoid blocking
        clear_thread = threading.Thread(target=clear_after_delay, daemon=True)
        clear_thread.start()

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
        processed_text = self._preprocess_text(text).rstrip("\r\n") + ' '

        try:
            # Use strategy-based injection
            success = False
            if self.ydotool_available:
                success = self._inject_via_clipboard_and_hotkey(processed_text)
            else:
                success = self._inject_via_clipboard(processed_text)

            # Check if clipboard clearing is enabled
            if success and self.config_manager:
                clipboard_behavior = self.config_manager.get_setting('clipboard_behavior', False)
                if clipboard_behavior:
                    clear_delay = self.config_manager.get_setting('clipboard_clear_delay', 5.0)
                    self._schedule_clipboard_clear(clear_delay)

            return success

        except Exception as e:
            print(f"Primary injection method failed: {e}")

            # No fallback needed - paste strategy is always reliable
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
        """Fast path: copy to clipboard, then press Ctrl+V via ydotool."""
        try:
            # Get active window info once for all checks
            window_info = self._get_active_window_info()
            is_kitty_terminal = self._is_kitty_protocol_terminal(window_info)

            # 1) Set clipboard (prefer wl-copy on Wayland)
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
            else:
                pyperclip.copy(text)

            # Use longer delay for Kitty-protocol terminals to ensure clipboard sync
            clipboard_delay = 0.25 if is_kitty_terminal else 0.12
            time.sleep(clipboard_delay)

            # 2) Press paste key combination based on config
            if self.ydotool_available:
                # For Kitty-protocol terminals, clear stuck modifiers first
                # (especially Super, which can interfere with paste recognition)
                if is_kitty_terminal:
                    self._clear_stuck_modifiers()
                    time.sleep(0.02)  # Brief settle after clearing modifiers

                paste_keycode = self._get_paste_keycode()
                paste_keycode_pressed = f'{paste_keycode}:1'
                paste_keycode_released = f'{paste_keycode}:0'

                # Paste chords are sent as modifiers + a configurable keycode (default KEY_V=47).
                paste_mode = None
                if self.config_manager:
                    paste_mode = self.config_manager.get_setting('paste_mode', None)

                # Use spaced-out key events for Kitty-protocol terminals to prevent
                # escape sequence artifacts (8;8u fragments from misinterpreted modifiers)
                if is_kitty_terminal:
                    if paste_mode in ['super', 'ctrl_shift', 'ctrl', 'alt']:
                        success = self._send_paste_keys_slow(paste_mode)
                        if not success:
                            print(f"  Slow paste failed for mode {paste_mode}")
                            return False
                        self._send_enter_if_auto_submit()
                        return True
                    elif paste_mode is None:
                        # Back-compat: use shift_paste setting
                        shift_paste = True
                        if self.config_manager:
                            shift_paste = self.config_manager.get_setting('shift_paste', True)

                        mode = 'ctrl_shift' if shift_paste else 'ctrl'
                        success = self._send_paste_keys_slow(mode)
                        if not success:
                            print(f"  Slow paste failed for back-compat mode {mode}")
                            return False
                        self._send_enter_if_auto_submit()
                        return True

                # Fast path for non-Kitty terminals (original behavior)
                if paste_mode == 'super':
                    # LeftMeta (Super) = 125
                    result = subprocess.run(
                        ['ydotool', 'key', '125:1', paste_keycode_pressed, paste_keycode_released, '125:0'],
                        capture_output=True, timeout=5
                    )
                elif paste_mode == 'ctrl_shift':
                    result = subprocess.run(
                        ['ydotool', 'key', '29:1', '42:1', paste_keycode_pressed, paste_keycode_released, '42:0', '29:0'],
                        capture_output=True, timeout=5
                    )
                elif paste_mode == 'ctrl':
                    result = subprocess.run(
                        ['ydotool', 'key', '29:1', paste_keycode_pressed, paste_keycode_released, '29:0'],
                        capture_output=True, timeout=5
                    )
                elif paste_mode == 'alt':
                    # LeftAlt = 56
                    result = subprocess.run(
                        ['ydotool', 'key', '56:1', paste_keycode_pressed, paste_keycode_released, '56:0'],
                        capture_output=True, timeout=5
                    )
                else:
                    # Back-compat path: fall back to legacy shift_paste boolean
                    shift_paste = True
                    if self.config_manager:
                        shift_paste = self.config_manager.get_setting('shift_paste', True)
                    if shift_paste:
                        result = subprocess.run(
                            ['ydotool', 'key', '29:1', '42:1', paste_keycode_pressed, paste_keycode_released, '42:0', '29:0'],
                            capture_output=True, timeout=5
                        )
                    else:
                        result = subprocess.run(
                            ['ydotool', 'key', '29:1', paste_keycode_pressed, paste_keycode_released, '29:0'],
                            capture_output=True, timeout=5
                        )
                
                if result.returncode != 0:
                    stderr = (result.stderr or b"").decode("utf-8", "ignore")
                    print(f"  ydotool paste command failed: {stderr}")
                    return False

                self._send_enter_if_auto_submit()
                return True

            print("No key-injection tool available; text is on the clipboard.")
            return True

        except Exception as e:
            print(f"Clipboard+hotkey injection failed: {e}")
            return False

    def _inject_via_clipboard(self, text: str) -> bool:
        """Fallback: copy text to clipboard if ydotool is not available."""
        try:
            if shutil.which("wl-copy"):
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)
            else:
                pyperclip.copy(text)
            
            print("Text copied to clipboard (ydotool not available for paste)")
            return True
        except Exception as e:
            print(f"ERROR: Clipboard fallback failed: {e}")
            return False

