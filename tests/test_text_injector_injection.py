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
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get_setting(self, name, default=None):
        return self.settings.get(name, default)


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
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard") as save_clipboard,
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_layout_is_type_safe", return_value=True),
            mock.patch.object(injector, "_restore_clipboard") as restore_clipboard,
            mock.patch.object(injector, "_send_enter_if_auto_submit") as auto_submit,
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
        auto_submit.assert_called_once_with()
        save_clipboard.assert_not_called()
        copy.assert_not_called()
        restore_clipboard.assert_not_called()

    def test_gnome_wayland_with_custom_paste_keycode_uses_ydotool_chord(self):
        # A custom paste keycode means direct typing can't honour the layout, so
        # we use verbatim clipboard paste. The ydotool key chord DOES reach Mutter,
        # so paste now succeeds on GNOME instead of being clipboard-only.
        injector = self._injector()
        injector.config_manager = ConfigStub({"paste_keycode": 54})

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_restore_clipboard") as restore_clipboard,
            mock.patch.object(injector, "_send_enter_if_auto_submit") as auto_submit,
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "GNOME",
                    "XDG_SESSION_DESKTOP": "gnome",
                    "DESKTOP_SESSION": "gnome",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        copy.assert_called_once_with("hello")
        paste_chord.assert_called_once_with("ctrl")
        direct_type.assert_not_called()
        restore_clipboard.assert_called_once()
        auto_submit.assert_called_once_with()

    def test_gnome_non_us_layout_uses_clipboard_paste_not_direct_typing(self):
        # German (or any non-US) layout: ydotool type would mangle keys, so we
        # must use verbatim clipboard paste even though it's a GNOME session.
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_layout_is_type_safe", return_value=False),
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit") as auto_submit,
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "GNOME",
                    "XDG_SESSION_DESKTOP": "gnome",
                    "DESKTOP_SESSION": "gnome",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        direct_type.assert_not_called()
        copy.assert_called_once_with("hello")
        paste_chord.assert_called_once_with("ctrl")
        auto_submit.assert_called_once_with()

    def test_gnome_non_ascii_text_uses_clipboard_paste_not_direct_typing(self):
        # Umlauts/typographic characters can't be produced by ydotool type
        # (ASCII-only), so non-ASCII text always goes through clipboard paste.
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_layout_is_type_safe", return_value=True),
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "GNOME",
                    "XDG_SESSION_DESKTOP": "gnome",
                    "DESKTOP_SESSION": "gnome",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("Größe"))

        direct_type.assert_not_called()
        copy.assert_called_once_with("Größe")
        paste_chord.assert_called_once_with("ctrl")

    def test_prefer_clipboard_paste_override_skips_direct_typing_on_gnome(self):
        # Explicit user override: never type directly, even for ASCII on a US layout.
        injector = self._injector()
        injector.config_manager = ConfigStub({"prefer_clipboard_paste": True})

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_layout_is_type_safe", return_value=True),
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "GNOME",
                    "XDG_SESSION_DESKTOP": "gnome",
                    "DESKTOP_SESSION": "gnome",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        direct_type.assert_not_called()
        paste_chord.assert_called_once_with("ctrl")

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
