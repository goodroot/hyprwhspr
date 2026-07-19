import io
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("pyperclip", types.SimpleNamespace(copy=lambda text: None, paste=lambda: ""))

from text_injector import TextInjector, _LazyPyperclip


class ConfigStub:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def get_setting(self, name, default=None):
        return self.settings.get(name, default)


class TextInjectorInjectionTests(unittest.TestCase):
    def _injector(self):
        injector = TextInjector.__new__(TextInjector)
        injector.config_manager = ConfigStub()
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

    def test_wayland_clipboard_success_does_not_use_fallback(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0)
        with (
            mock.patch("text_injector.shutil.which", return_value="/usr/bin/wl-copy"),
            mock.patch("text_injector.subprocess.run", return_value=completed) as run,
            mock.patch("text_injector.pyperclip.copy") as fallback,
        ):
            self.assertTrue(injector._copy_text_to_clipboard("hello"))

        run.assert_called_once()
        fallback.assert_not_called()

    def test_failed_wayland_clipboard_uses_x11_fallback(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=1)
        with (
            mock.patch("text_injector.shutil.which", return_value="/usr/bin/wl-copy"),
            mock.patch("text_injector.subprocess.run", return_value=completed),
            mock.patch("text_injector.pyperclip.copy") as fallback,
        ):
            self.assertTrue(injector._copy_text_to_clipboard("hello"))

        fallback.assert_called_once_with("hello")

    def test_clipboard_save_falls_back_after_failed_wayland_read(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=1, stdout=b"")
        with (
            mock.patch("text_injector.shutil.which", return_value="/usr/bin/wl-paste"),
            mock.patch("text_injector.subprocess.run", return_value=completed),
            mock.patch("text_injector.pyperclip.paste", return_value="saved") as fallback,
        ):
            self.assertEqual(injector._save_clipboard(), b"saved")

        fallback.assert_called_once_with()

    def test_restore_fallback_restores_utf8_but_skips_binary(self):
        injector = self._injector()

        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target

            def start(self):
                self.target()

        with (
            mock.patch("text_injector.time.sleep"),
            mock.patch("text_injector.threading.Thread", ImmediateThread),
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as fallback,
        ):
            injector._restore_clipboard("café".encode(), delay=0)
            injector._restore_clipboard(b"\xff\x00", delay=0)

        fallback.assert_called_once_with("café")

    def test_x11_skips_wayland_clipboard_tools_and_uses_xdotool_chord(self):
        injector = self._injector()
        injector.session_type = "x11"
        injector.ydotool_available = False
        injector.wtype_available = False
        injector.xdotool_available = True
        clipboard = types.SimpleNamespace(copy=mock.Mock(), paste=mock.Mock(return_value="old"))
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        with (
            mock.patch("text_injector.pyperclip", clipboard),
            mock.patch("text_injector.shutil.which") as which,
            mock.patch("text_injector.subprocess.run", return_value=completed) as run,
        ):
            self.assertEqual(injector._save_clipboard(), b"old")
            self.assertTrue(injector._copy_text_to_clipboard("hello"))
            self.assertTrue(injector._send_paste_keys_xdotool("ctrl_shift"))

        which.assert_not_called()
        clipboard.copy.assert_called_once_with("hello")
        run.assert_called_once_with(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            capture_output=True,
            timeout=5,
        )

    def test_x11_injection_prefers_xdotool_without_wtype_or_ydotool(self):
        injector = self._injector()
        injector.session_type = "x11"
        injector.ydotool_available = False
        injector.wtype_available = False
        injector.xdotool_available = True
        with (
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old"),
            mock.patch.object(injector, "_copy_text_to_clipboard", return_value=True),
            mock.patch.object(injector, "_send_paste_keys_xdotool", return_value=True) as xdotool,
            mock.patch.object(injector, "_send_paste_keys_wtype") as wtype,
            mock.patch.object(injector, "_send_paste_keys_slow") as ydotool,
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        xdotool.assert_called_once_with("ctrl+v")
        wtype.assert_not_called()
        ydotool.assert_not_called()

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

    def test_application_rule_uses_custom_chord_for_emacs(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({
            "applications": {
                "emacs": {
                    "auto_paste": "ctrl+y",
                },
            },
        })

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value={"class": "Emacs", "title": "main.py - GNU Emacs"}),
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
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        copy.assert_called_once_with("hello")
        paste_chord.assert_called_once_with("ctrl+y")
        direct_type.assert_not_called()

    def test_application_rule_can_disable_injection(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({
            "applications": {
                "other-app": {
                    "auto_paste": False,
                },
            },
        })

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy") as copy,
            mock.patch.object(injector, "_get_active_window_info", return_value={"class": "Other App"}),
            mock.patch.object(injector, "_save_clipboard") as save_clipboard,
            mock.patch.object(injector, "_send_paste_keys_slow") as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool") as direct_type,
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        # "Disabled" is a true no-op: no paste, and crucially no clipboard write
        # (so dictated text can't leak onto the clipboard for a disabled app).
        copy.assert_not_called()
        save_clipboard.assert_not_called()
        paste_chord.assert_not_called()
        direct_type.assert_not_called()

    def test_invalid_application_rule_still_suppresses_gnome_direct_typing(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({
            "applications": {
                "emacs": {
                    "auto_paste": True,
                },
            },
        })

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info", return_value={"class": "emacs"}),
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
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        direct_type.assert_not_called()
        paste_chord.assert_called_once_with("ctrl+v")

    def test_focused_window_identifiers_include_title_segments(self):
        identifiers = TextInjector.focused_window_identifiers({
            "class": "org.gnu.Emacs.desktop",
            "title": "main.py - GNU Emacs",
        })

        self.assertIn("org-gnu-emacs", identifiers)
        self.assertIn("emacs", identifiers)
        self.assertIn("gnu-emacs", identifiers)

    def test_wtype_sends_arbitrary_single_key_chord(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        with mock.patch("text_injector.subprocess.run", return_value=completed) as run:
            self.assertTrue(injector._send_paste_keys_wtype("ctrl+y"))

        run.assert_called_once_with(
            ["wtype", "-M", "ctrl", "-k", "y", "-m", "ctrl"],
            capture_output=True,
            timeout=5,
        )

    def test_wtype_translates_named_keys_to_xkb_keysym_names(self):
        # wtype's -k resolves through libxkbcommon: case-sensitive names that
        # differ from our lowercase tokens (enter -> Return, f11 -> F11).
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        cases = {
            "ctrl+enter": "Return",
            "ctrl+f11": "F11",
            "super+pageup": "Page_Up",
        }
        for chord, expected_key in cases.items():
            with mock.patch("text_injector.subprocess.run", return_value=completed) as run:
                self.assertTrue(injector._send_paste_keys_wtype(chord))
            sent = run.call_args.args[0]
            self.assertEqual(sent[sent.index("-k") + 1], expected_key)

    def test_hyprland_session_detection_requires_exact_desktop_or_signature(self):
        injector = self._injector()
        cases = (
            ({"HYPRLAND_INSTANCE_SIGNATURE": "abc"}, True),
            ({"XDG_CURRENT_DESKTOP": "Hyprland:uwsm"}, True),
            ({"XDG_SESSION_DESKTOP": "hyprland"}, True),
            ({"XDG_CURRENT_DESKTOP": "NotHyprland"}, False),
            ({"XDG_CURRENT_DESKTOP": "GNOME"}, False),
        )
        for environment, expected in cases:
            with (
                self.subTest(environment=environment),
                mock.patch.dict("text_injector.os.environ", environment, clear=True),
                mock.patch("text_injector.shutil.which", return_value="/usr/bin/hyprctl"),
            ):
                self.assertEqual(injector._is_hyprland_session(), expected)
        with (
            mock.patch.dict("text_injector.os.environ", {"HYPRLAND_INSTANCE_SIGNATURE": "abc"}, clear=True),
            mock.patch("text_injector.shutil.which", return_value=None),
        ):
            self.assertFalse(injector._is_hyprland_session())

        injector.session_type = "x11"
        with (
            mock.patch.dict(
                "text_injector.os.environ",
                {"HYPRLAND_INSTANCE_SIGNATURE": "stale", "XDG_CURRENT_DESKTOP": "Hyprland"},
                clear=True,
            ),
            mock.patch("text_injector.shutil.which", return_value="/usr/bin/hyprctl"),
        ):
            self.assertFalse(injector._is_hyprland_session())

    def test_hyprland_shortcut_constructs_lua_for_named_and_function_keys(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")
        cases = {
            "ctrl+shift+enter": 'mods = "CTRL + SHIFT", key = "Return"',
            "super+pageup": 'mods = "SUPER", key = "Page_Up"',
            "alt+f11": 'mods = "ALT", key = "F11"',
            "ctrl+y": 'mods = "CTRL", key = "y"',
        }
        for chord, fragment in cases.items():
            injector._hyprland_shortcut_syntax = None
            with mock.patch("text_injector.subprocess.run", return_value=completed) as run:
                self.assertTrue(injector._send_shortcut_hyprland(chord))
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["hyprctl", "dispatch"])
            self.assertIn(fragment, command[2])
            self.assertIn('window = "activewindow"', command[2])

    def test_hyprland_shortcut_negotiates_caches_and_renegotiates(self):
        injector = self._injector()
        failed = types.SimpleNamespace(returncode=1, stderr=b"bad syntax")
        completed = types.SimpleNamespace(returncode=0, stderr=b"")
        with mock.patch(
            "text_injector.subprocess.run",
            side_effect=[failed, completed, completed, failed, completed],
        ) as run:
            self.assertTrue(injector._send_shortcut_hyprland("ctrl+v"))
            self.assertEqual(injector._hyprland_shortcut_syntax, "legacy")
            self.assertTrue(injector._send_shortcut_hyprland("ctrl+v"))
            self.assertTrue(injector._send_shortcut_hyprland("ctrl+v"))
            self.assertEqual(injector._hyprland_shortcut_syntax, "lua")

        self.assertEqual(run.call_args_list[1].args[0], [
            "hyprctl", "dispatch", "sendshortcut", "CTRL, v, activewindow",
        ])
        self.assertEqual(run.call_args_list[2].args[0][2], "sendshortcut")
        self.assertTrue(run.call_args_list[4].args[0][2].startswith("hl.dsp.send_shortcut"))

    def test_hyprland_shortcut_rejects_invalid_chords_and_handles_timeouts(self):
        injector = self._injector()
        with mock.patch("text_injector.subprocess.run") as run:
            self.assertFalse(injector._send_shortcut_hyprland("ctrl+shift"))
            self.assertFalse(injector._send_shortcut_hyprland("ctrl+unknown-key"))
        run.assert_not_called()

        with mock.patch(
            "text_injector.subprocess.run", side_effect=subprocess.TimeoutExpired("hyprctl", 1)
        ) as run:
            self.assertFalse(injector._send_shortcut_hyprland("ctrl+v"))
        self.assertEqual(run.call_count, 2)
        self.assertIsNone(injector._hyprland_shortcut_syntax)

    def test_hyprland_native_paste_avoids_wtype(self):
        injector = self._injector()
        with (
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old"),
            mock.patch.object(injector, "_copy_text_to_clipboard", return_value=True),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=True) as native,
            mock.patch.object(injector, "_send_paste_keys_wtype") as wtype,
            mock.patch.object(injector, "_send_paste_keys_slow") as ydotool,
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))
        native.assert_called_once_with("ctrl+v")
        wtype.assert_not_called()
        ydotool.assert_not_called()

    def test_hyprland_failure_falls_back_to_wtype_then_ydotool(self):
        injector = self._injector()
        injector.wtype_available = True
        with (
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old"),
            mock.patch.object(injector, "_copy_text_to_clipboard", return_value=True),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=False),
            mock.patch.object(injector, "_send_paste_keys_wtype", return_value=False) as wtype,
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as ydotool,
            mock.patch.object(injector, "_clear_stuck_modifiers"),
            mock.patch.object(injector, "_gnome_force_latin_layout", return_value=None),
            mock.patch.object(injector, "_gnome_restore_layout"),
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))
        wtype.assert_called_once_with("ctrl+v")
        ydotool.assert_called_once_with("ctrl+v")

    def test_hyprland_failure_stops_at_successful_wtype_fallback(self):
        injector = self._injector()
        injector.wtype_available = True
        with (
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old"),
            mock.patch.object(injector, "_copy_text_to_clipboard", return_value=True),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=False),
            mock.patch.object(injector, "_send_paste_keys_wtype", return_value=True) as wtype,
            mock.patch.object(injector, "_send_paste_keys_slow") as ydotool,
            mock.patch.object(injector, "_clear_stuck_modifiers") as clear_modifiers,
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))
        wtype.assert_called_once_with("ctrl+v")
        clear_modifiers.assert_called_once_with()
        ydotool.assert_not_called()

    def test_hyprland_auto_submit_uses_native_enter(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({"auto_submit": True})
        with (
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=True) as native,
            mock.patch.object(injector, "_run_ydotool") as ydotool,
            mock.patch("text_injector.subprocess.run") as run,
        ):
            injector._send_enter_if_auto_submit()
        native.assert_called_once_with("enter")
        ydotool.assert_not_called()
        run.assert_not_called()

    def test_hyprland_auto_submit_native_failure_falls_back_to_ydotool(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({"auto_submit": True})
        completed = types.SimpleNamespace(returncode=0, stderr=b"")
        with (
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=False) as native,
            mock.patch.object(injector, "_run_ydotool", return_value=completed) as ydotool,
            mock.patch("text_injector.subprocess.run") as run,
        ):
            injector._send_enter_if_auto_submit()
        native.assert_called_once_with("enter")
        ydotool.assert_called_once_with(["key", "28:1", "28:0"], timeout=1)
        run.assert_not_called()

    def test_ydotool_sends_arbitrary_single_key_chord(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        with mock.patch.object(injector, "_run_ydotool", return_value=completed) as run:
            self.assertTrue(injector._send_paste_keys_slow("ctrl+y"))

        run.assert_has_calls([
            mock.call(["key", "29:1"], timeout=1),
            mock.call(["key", "21:1", "21:0"], timeout=1),
            mock.call(["key", "29:0"], timeout=1),
        ])

    def test_ydotool_function_key_chords_use_non_contiguous_evdev_codes(self):
        injector = self._injector()

        self.assertEqual(injector._keycode_for_chord_key("f10"), 68)
        self.assertEqual(injector._keycode_for_chord_key("f11"), 87)
        self.assertEqual(injector._keycode_for_chord_key("f12"), 88)
        self.assertEqual(injector._keycode_for_chord_key("f13"), 183)
        self.assertEqual(injector._keycode_for_chord_key("f24"), 194)

    def test_failed_atspi_probe_is_negative_cached(self):
        injector = self._injector()
        failed = types.SimpleNamespace(returncode=1, stdout="", stderr="")

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.subprocess.run", return_value=failed) as run,
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertIsNone(injector._get_active_window_info())
            self.assertIsNone(injector._get_active_window_info())

        # First call probes niri, hyprctl, and AT-SPI. Second call skips AT-SPI.
        self.assertEqual(run.call_count, 5)

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
        paste_chord.assert_called_once_with("ctrl+v")
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
        paste_chord.assert_called_once_with("ctrl+v")
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
        paste_chord.assert_called_once_with("ctrl+v")

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
        paste_chord.assert_called_once_with("ctrl+v")

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

        paste_chord.assert_called_once_with("ctrl+v")
        direct_type.assert_not_called()

    def test_gnome_failed_paste_restores_previous_clipboard(self):
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=False),
            mock.patch.object(injector, "_type_text_ydotool", return_value=True),
            mock.patch.object(injector, "_layout_is_type_safe", return_value=False),
            mock.patch.object(injector, "_restore_clipboard") as restore_clipboard,
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "GNOME",
                    "WAYLAND_DISPLAY": "wayland-0",
                },
                clear=True,
            ),
        ):
            self.assertFalse(injector._inject_via_clipboard_and_hotkey("hello"))

        restore_clipboard.assert_called_once_with(
            b"old clipboard", injected=b"hello", delay=0
        )

    def test_ydotool_type_command_uses_small_delay_and_argument_separator(self):
        injector = self._injector()
        completed = types.SimpleNamespace(returncode=0, stderr=b"")

        with mock.patch("text_injector.subprocess.run", return_value=completed) as run:
            self.assertTrue(injector._type_text_ydotool("hello -- world"))

        # Now routed through _run_ydotool: same argv, plus the private-socket env.
        injector._ydotoold.ensure_running.assert_called_once_with()
        run.assert_called_once_with(
            ['ydotool', 'type', '--key-delay', '5', '--key-hold', '5', '--', 'hello -- world'],
            capture_output=True,
            timeout=10,
            env={"YDOTOOL_SOCKET": "/run/x.sock"},
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

    def test_mutter_desktop_is_gnome_wayland_session(self):
        injector = self._injector()

        with mock.patch.dict(
            "text_injector.os.environ",
            {
                "XDG_SESSION_TYPE": "wayland",
                "XDG_CURRENT_DESKTOP": "Mutter",
                "WAYLAND_DISPLAY": "wayland-1",
            },
            clear=True,
        ):
            self.assertTrue(injector._is_gnome_wayland_session())

    def test_layout_type_safe_redetects_each_call(self):
        injector = self._injector()

        with mock.patch.object(
            injector, "_detect_active_layout", side_effect=["us", "de"]
        ) as detect:
            self.assertTrue(injector._layout_is_type_safe())
            self.assertFalse(injector._layout_is_type_safe())

        self.assertEqual(detect.call_count, 2)

    def test_clipboard_restore_default_delay_is_config_default(self):
        injector = self._injector()

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info", return_value=None),
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True),
            mock.patch.object(injector, "_type_text_ydotool", return_value=True),
            mock.patch.object(injector, "_restore_clipboard") as restore_clipboard,
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "sway",
                    "WAYLAND_DISPLAY": "wayland-1",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        restore_clipboard.assert_called_once_with(
            b"old clipboard", injected=b"hello", delay=5.0
        )

    def test_localectl_unset_layout_is_unknown_not_non_us(self):
        injector = self._injector()
        calls = [
            types.SimpleNamespace(returncode=1, stdout=""),
            types.SimpleNamespace(returncode=0, stdout="System Locale: LANG=en_US.UTF-8\n   X11 Layout: (unset)\n"),
        ]

        with mock.patch("text_injector.subprocess.run", side_effect=calls):
            self.assertEqual(injector._detect_active_layout(), "")

    def test_gnome_non_xkb_mru_source_is_not_type_safe(self):
        injector = self._injector()
        completed = types.SimpleNamespace(
            returncode=0,
            stdout="[('ibus', 'mozc-jp'), ('xkb', 'us')]",
        )

        with mock.patch("text_injector.subprocess.run", return_value=completed):
            self.assertFalse(injector._layout_is_type_safe())

    def test_layout_detection_uses_short_ttl_cache(self):
        injector = self._injector()
        calls = [
            types.SimpleNamespace(returncode=0, stdout="[('xkb', 'us')]"),
            types.SimpleNamespace(returncode=0, stdout="[('xkb', 'de')]"),
        ]

        with (
            mock.patch("text_injector.subprocess.run", side_effect=calls) as run,
            mock.patch("text_injector.time.monotonic", side_effect=[10.0, 10.5, 11.2]),
        ):
            self.assertEqual(injector._detect_active_layout(), "us")
            self.assertEqual(injector._detect_active_layout(), "us")
            self.assertEqual(injector._detect_active_layout(), "de")

        self.assertEqual(run.call_count, 2)


    # ---- Fix 1: skip the active-window lookup when its result is unused ----

    def test_active_window_lookup_needed_predicate(self):
        injector = self._injector()

        # Default (auto-detect): paste_mode/shift_paste unset → window class is
        # needed for terminal detection.
        injector.config_manager = ConfigStub()
        self.assertTrue(injector._active_window_lookup_needed())

        # Explicit paste_mode, no app rules → lookup result is unused.
        injector.config_manager = ConfigStub({"paste_mode": "ctrl"})
        self.assertFalse(injector._active_window_lookup_needed())

        # Legacy explicit override → also no auto-detect needed.
        injector.config_manager = ConfigStub({"shift_paste": True})
        self.assertFalse(injector._active_window_lookup_needed())

        # App rules present → window identity is needed even with explicit paste_mode.
        injector.config_manager = ConfigStub(
            {"paste_mode": "ctrl", "applications": {"emacs": {"auto_paste": "ctrl+y"}}}
        )
        self.assertTrue(injector._active_window_lookup_needed())

    def test_inject_skips_window_lookup_with_explicit_paste_mode(self):
        injector = self._injector()
        injector.config_manager = ConfigStub({"paste_mode": "ctrl_shift"})

        with (
            mock.patch("text_injector.shutil.which", return_value=None),
            mock.patch("text_injector.pyperclip.copy"),
            mock.patch.object(injector, "_get_active_window_info") as get_window,
            mock.patch.object(injector, "_save_clipboard", return_value=b"old clipboard"),
            mock.patch.object(injector, "_send_paste_keys_slow", return_value=True) as paste_chord,
            mock.patch.object(injector, "_type_text_ydotool", return_value=True) as direct_type,
            mock.patch.object(injector, "_restore_clipboard"),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.dict(
                "text_injector.os.environ",
                {
                    "XDG_SESSION_TYPE": "wayland",
                    "XDG_CURRENT_DESKTOP": "sway",
                    "WAYLAND_DISPLAY": "wayland-1",
                },
                clear=True,
            ),
        ):
            self.assertTrue(injector._inject_via_clipboard_and_hotkey("hello"))

        # The expensive lookup (AT-SPI subprocess on GNOME/KDE) is never reached
        # because nothing consumes the window identity here.
        get_window.assert_not_called()
        paste_chord.assert_called_once_with("ctrl+shift+v")
        direct_type.assert_not_called()

    # ---- Fix 2: chord parsing handles both separators and symbol keys ----

    def test_parse_key_chord_accepts_separators_and_symbol_keys(self):
        injector = self._injector()

        self.assertEqual(injector._parse_key_chord("ctrl+shift+v"), (["ctrl", "shift"], "v"))
        self.assertEqual(injector._parse_key_chord("ctrl-shift-v"), (["ctrl", "shift"], "v"))
        self.assertEqual(injector._parse_key_chord("super+minus"), (["super"], "minus"))
        # The literal dash key survives instead of being mangled into a separator.
        self.assertEqual(injector._parse_key_chord("super+-"), (["super"], "minus"))
        self.assertEqual(injector._parse_key_chord("ctrl+y"), (["ctrl"], "y"))
        self.assertEqual(injector._parse_key_chord("f5"), ([], "f5"))
        self.assertEqual(injector._parse_key_chord("cmd+v"), (["super"], "v"))

    def test_parse_key_chord_rejects_empty_and_modifier_only(self):
        injector = self._injector()

        self.assertIsNone(injector._parse_key_chord(""))
        self.assertIsNone(injector._parse_key_chord("ctrl+shift"))
        self.assertIsNone(injector._parse_key_chord("ctrl"))
        self.assertIsNone(injector._parse_key_chord(None))

    # ---- Fix 3: an unusable configured chord warns instead of failing silently ----

    def test_unusable_application_chord_warns_and_returns_chord(self):
        injector = self._injector()
        injector.config_manager = ConfigStub(
            {"applications": {"emacs": {"auto_paste": "ctrl+nope"}}}
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            chord, app = injector._resolve_paste_chord({"class": "emacs"})

        # Chord returned unchanged (text still lands on clipboard for manual paste),
        # but the user is told why their config did nothing.
        self.assertEqual(chord, "ctrl+nope")
        self.assertEqual(app, "emacs")
        output = buf.getvalue()
        self.assertIn("ctrl+nope", output)
        self.assertIn("emacs", output)

    def test_valid_application_chord_does_not_warn(self):
        injector = self._injector()
        injector.config_manager = ConfigStub(
            {"applications": {"emacs": {"auto_paste": "ctrl+y"}}}
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            chord, app = injector._resolve_paste_chord({"class": "emacs"})

        self.assertEqual(chord, "ctrl+y")
        self.assertEqual(buf.getvalue(), "")


class LazyPyperclipTests(unittest.TestCase):
    def test_selects_xclip_then_xsel_on_first_use(self):
        for available, expected in (("xclip", "xclip"), ("xsel", "xsel")):
            module = types.SimpleNamespace(
                set_clipboard=mock.Mock(), copy=mock.Mock(), paste=mock.Mock(return_value="")
            )
            lazy = _LazyPyperclip()
            with (
                mock.patch.dict(sys.modules, {"pyperclip": module}),
                mock.patch(
                    "text_injector.shutil.which",
                    side_effect=lambda name, selected=available: f"/usr/bin/{name}" if name == selected else None,
                ),
            ):
                lazy.copy("text")

            module.set_clipboard.assert_called_once_with(expected)
            module.copy.assert_called_once_with("text")

    def test_construction_does_not_import_optional_dependency(self):
        lazy = _LazyPyperclip()
        self.assertIsNone(lazy._module)


if __name__ == "__main__":
    unittest.main()
