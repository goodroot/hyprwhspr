import importlib
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported.

    cli_commands transitively imports global_shortcuts -> evdev (a hard import)
    and rich (prompt/console/table). On a dev machine these are installed and we
    must NOT shadow them; in a bare CI we fall back to a minimal stub so the
    import still works.
    """
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing("rich")
_stub_if_missing("rich.prompt", Prompt=object, Confirm=object)
_stub_if_missing("rich.console", Console=object)
_stub_if_missing("rich.table", Table=object)
_stub_if_missing(
    "evdev",
    InputDevice=object,
    categorize=lambda *a, **k: None,
    ecodes=types.SimpleNamespace(),
    UInput=object,
)

import cli_commands  # noqa: E402


def _kb(name):
    return {"name": name, "is_keyboard": True, "is_mouse": False,
            "is_virtual": False}


class KeyboardSelectionTests(unittest.TestCase):
    """Cover the recommend-&-confirm keyboard-allowlist selection flow."""

    def _run(self, *, confirm, prompt_inputs=(),
             candidates=("KB1", "KB2")):
        cands = [_kb(n) for n in candidates]
        with mock.patch.object(cli_commands, "_gather_keyboard_candidates",
                               return_value=cands), \
                mock.patch.object(cli_commands, "Table"), \
                mock.patch.object(cli_commands, "Console"), \
                mock.patch.object(cli_commands, "Confirm") as confirm_mock, \
                mock.patch.object(cli_commands, "Prompt") as prompt_mock, \
                mock.patch.object(cli_commands.sys.stdin, "isatty",
                                  return_value=True), \
                mock.patch.object(cli_commands.sys.stdout, "isatty",
                                  return_value=True), \
                mock.patch.dict(cli_commands.os.environ, {}, clear=False):
            cli_commands.os.environ.pop("HYPRWHSPR_NONINTERACTIVE", None)
            confirm_mock.ask.return_value = confirm
            prompt_mock.ask.side_effect = list(prompt_inputs)
            return cli_commands._run_keyboard_selection()

    def test_confirm_accepts_recommendation(self):
        # Two pure keyboards are recommended; a single Y accepts them.
        result = self._run(confirm=True)
        self.assertEqual(result, ["KB1", "KB2"])

    def test_decline_then_numeric_edit(self):
        # Declining drops into the editor; entering "2" keeps only KB2.
        result = self._run(confirm=False, prompt_inputs=["2"])
        self.assertEqual(result, ["KB2"])

    def test_zero_selects_none(self):
        # "0" in the editor means listen to none (auto-detect / legacy).
        result = self._run(confirm=False, prompt_inputs=["0"])
        self.assertEqual(result, [])

    def test_enter_with_empty_recommendation_selects_none(self):
        candidates = [{
            "name": "Mouse Keyboard Thing",
            "is_keyboard": True,
            "is_mouse": True,
            "is_virtual": False,
        }]
        with mock.patch.object(cli_commands, "Prompt") as prompt_mock:
            prompt_mock.ask.return_value = ""
            result = cli_commands._edit_keyboard_selection(candidates, set())

        self.assertEqual(result, [])

    def test_detect_flushes_input_buffer(self):
        # Regression: physical keystrokes during evdev detection are echoed into
        # the tty line buffer; detect_keyboard must drain them so they don't leak
        # into the shell afterwards.
        fake_cfg = types.SimpleNamespace(
            get_setting=lambda *a, **k: "Super+Alt+D")
        with mock.patch.object(cli_commands, "ConfigManager",
                               return_value=fake_cfg), \
                mock.patch.object(cli_commands, "_gather_keyboard_candidates",
                                  return_value=[_kb("KB1")]), \
                mock.patch.object(cli_commands, "_detect_pressed_keyboard",
                                  return_value="KB1"), \
                mock.patch.object(cli_commands, "_flush_input_buffer") as flush:
            cli_commands.detect_keyboard()
        self.assertTrue(
            flush.called,
            "input buffer must be flushed after keypress detection",
        )


if __name__ == "__main__":
    unittest.main()
