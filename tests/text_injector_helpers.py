"""Shared isolated text-injector fixtures; no desktop discovery or devices."""

import sys
import threading
import types
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))
sys.modules.setdefault("pyperclip", types.SimpleNamespace(copy=lambda text: None, paste=lambda: ""))
from text_injector import TextInjector


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get_setting(self, name, default=None):
        return self.settings.get(name, default)

    def get_word_overrides(self):
        return self.settings.get("word_overrides", {})

    def get_filter_filler_words(self):
        return self.settings.get("filter_filler_words", False)

    def get_filler_words(self):
        return self.settings.get("filler_words", [])


def make_injector():
    injector = TextInjector.__new__(TextInjector)
    injector.config_manager = ConfigStub()
    injector._delivery_lock = threading.RLock()
    injector._clipboard_lock = threading.RLock()
    injector._last_text_lock = threading.Lock()
    injector._last_text = None
    injector._clipboard_generation = 0
    injector.session_type = "wayland"
    injector.ydotool_available = True
    injector.wtype_available = False
    injector.xdotool_available = False
    injector._hyprland_shortcut_syntax = None
    # Private ydotoold daemon manager: not running by default (so
    # _clear_stuck_modifiers is a no-op), but ensure_running() succeeds when a
    # ydotool command is actually issued.
    injector._ydotoold = mock.Mock()
    injector._ydotoold.is_running.return_value = False
    injector._ydotoold.ensure_running.return_value = True
    injector._ydotoold.socket_env.return_value = {"YDOTOOL_SOCKET": "/run/x.sock"}
    return injector

