import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("pyperclip", types.SimpleNamespace(copy=lambda text: None, paste=lambda: ""))

from text_injector import TextInjector


class ConfigStub:
    def get_setting(self, name, default=None):
        return default


class TextInjectorInjectionTests(unittest.TestCase):
    def _injector(self):
        injector = TextInjector.__new__(TextInjector)
        injector.config_manager = ConfigStub()
        injector.ydotool_available = True
        injector.wtype_available = False
        return injector

    def test_gnome_wayland_uses_ydotool_type_instead_of_paste_chord(self):
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_restore_clipboard") as restore_clipboard,
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "pop:GNOME",
                    "XDG_SESSION_DESKTOP": "pop",
                    "DESKTOP_SESSION": "pop",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        paste_chord.assert_not_called()
        direct_type.assert_called_once_with("hello")
        restore_clipboard.assert_called_once_with(
            b"old clipboard",
            injected=b"hello",
            delay=0.5,
        )

    def test_non_gnome_leaves_clipboard_on_failed_paste_chord(self):
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=None),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=False) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "sway",
                    "XDG_SESSION_DESKTOP": "sway",
                    "DESKTOP_SESSION": "sway",
                    "WAYLAND_DISPLAY": "wayland-1",
                },
                clear=True,
            ),
        ):
            self.assertFalse(injector._inject_via_clipboard_and_hotkey("hello"))

        paste_chord.assert_called_once_with("ctrl")
        direct_type.assert_not_called()

    def test_ydotool_type_command_uses_small_delay_and_argument_separator(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        with mock.patch("text_injector.subprocess.run", return_value=completed) as run:
            self.assertTrue(injector._type_text_ydotool("hello -- world"))

        run.assert_called_once_with(
            ['ydotool', 'type', '--key-delay', '5', '--key-hold', '5', '--', 'hello -- world'],
            capture_output=True,
            timeout=10,
        )

    def test_gnome_detection_does_not_match_pop_substrings(self):
        injector = self._injector()

        with mock.patch.dict(
            "text_injector.os.environ",
            {
                "XDG_SESSION_TYPE": "wayland",
                "XDG_CURRENT_DESKTOP": "popup",
                "XDG_SESSION_DESKTOP": "popterm",
                "DESKTOP_SESSION": "custom",
                "WAYLAND_DISPLAY": "wayland-1",
            },
            clear=True,
        ):
            self.assertFalse(injector._is_gnome_wayland_session())


if __name__ == "__main__":
    unittest.main()
