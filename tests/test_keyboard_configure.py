import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported.

    cli_commands transitively imports global_shortcuts -> evdev and rich.prompt.
    On a dev machine both are installed and we must NOT shadow them; in a bare
    CI we fall back to a minimal stub so importing cli_commands still works.
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


def _mk(name, kbd, mouse, virt, **extra):
    d = {'name': name, 'is_keyboard': kbd, 'is_mouse': mouse, 'is_virtual': virt}
    d.update(extra)
    return d


class KeyboardPreselectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            _mk('AT Translated Set 2 keyboard', True, False, False),
            _mk('Logitech USB Receiver', True, False, False),
            _mk('Logitech MX Vertical', True, True, False),    # dual-role (mouse)
            _mk('solaar-keyboard', True, True, False),         # dual-role
            _mk('ydotoold virtual device', True, False, True),  # virtual
        ]

    def test_no_allowlist_preselects_pure_keyboards_only(self):
        sel = cli_commands._keyboard_preselection(self.candidates, [])
        self.assertEqual(sel, {'AT Translated Set 2 keyboard', 'Logitech USB Receiver'})

    def test_dual_role_and_virtual_not_preselected(self):
        sel = cli_commands._keyboard_preselection(self.candidates, [])
        self.assertNotIn('Logitech MX Vertical', sel)
        self.assertNotIn('solaar-keyboard', sel)
        self.assertNotIn('ydotoold virtual device', sel)

    def test_existing_allowlist_preselected_exactly(self):
        sel = cli_commands._keyboard_preselection(self.candidates, ['Logitech USB Receiver'])
        self.assertEqual(sel, {'Logitech USB Receiver'})

    def test_existing_allowlist_is_case_insensitive(self):
        sel = cli_commands._keyboard_preselection(self.candidates, ['logitech usb receiver'])
        self.assertEqual(sel, {'Logitech USB Receiver'})

    def test_degraded_classification_preselects_all_non_virtual(self):
        unclassified = [
            _mk('Kbd A', False, False, False),
            _mk('Kbd B', False, False, False),
            _mk('ydotoold virtual device', False, False, True),
        ]
        sel = cli_commands._keyboard_preselection(unclassified, [])
        self.assertEqual(sel, {'Kbd A', 'Kbd B'})


class GatherCandidatesTests(unittest.TestCase):
    def test_dedup_classify_sort_and_preselect(self):
        raw = [
            {'name': 'Logitech USB Receiver', 'path': '/dev/input/event11', 'display_name': ''},
            {'name': 'Logitech USB Receiver', 'path': '/dev/input/event15', 'display_name': ''},
            {'name': 'Logitech MX Vertical', 'path': '/dev/input/event10', 'display_name': ''},
            {'name': 'ydotoold virtual device', 'path': '/dev/input/event24', 'display_name': ''},
        ]
        classification = {
            '/dev/input/event11': {'is_keyboard': True, 'is_mouse': False},
            '/dev/input/event15': {'is_keyboard': True, 'is_mouse': False},
            '/dev/input/event10': {'is_keyboard': True, 'is_mouse': True},
            '/dev/input/event24': {'is_keyboard': True, 'is_mouse': False},
        }
        orig_get = cli_commands.get_available_keyboards
        orig_cls = cli_commands._classify_input_devices
        try:
            cli_commands.get_available_keyboards = lambda shortcut: list(raw)
            cli_commands._classify_input_devices = lambda: dict(classification)
            cands = cli_commands._gather_keyboard_candidates('SUPER+ALT+D')
        finally:
            cli_commands.get_available_keyboards = orig_get
            cli_commands._classify_input_devices = orig_cls

        names = [c['name'] for c in cands]
        # Deduped by name: receiver appears once despite two event nodes.
        self.assertEqual(names.count('Logitech USB Receiver'), 1)
        # Virtual flagged via the ydotoold token.
        self.assertTrue(next(c for c in cands if c['name'] == 'ydotoold virtual device')['is_virtual'])
        # MX Vertical is dual-role (keyboard + mouse) from udev.
        mx = next(c for c in cands if c['name'] == 'Logitech MX Vertical')
        self.assertTrue(mx['is_mouse'] and mx['is_keyboard'])
        # Sort order: pure keyboard < dual-role < virtual.
        self.assertLess(names.index('Logitech USB Receiver'), names.index('Logitech MX Vertical'))
        self.assertLess(names.index('Logitech MX Vertical'), names.index('ydotoold virtual device'))
        # Preselection (no allowlist) keeps the receiver, drops mouse + virtual.
        sel = cli_commands._keyboard_preselection(cands, [])
        self.assertEqual(sel, {'Logitech USB Receiver'})


class SetupDefaultTests(unittest.TestCase):
    def test_choice_default_hit(self):
        self.assertEqual(
            cli_commands._choice_default('small', ['tiny', 'base', 'small'], '2'), '3')

    def test_choice_default_miss_falls_back(self):
        self.assertEqual(
            cli_commands._choice_default('nope', ['tiny', 'base'], '2'), '2')

    def test_choice_default_none_falls_back(self):
        self.assertEqual(cli_commands._choice_default(None, ['tiny', 'base'], '2'), '2')

    def test_bool_default_present(self):
        self.assertFalse(cli_commands._bool_default({'audio_ducking': False}, 'audio_ducking', True))
        self.assertTrue(cli_commands._bool_default({'audio_ducking': True}, 'audio_ducking', False))

    def test_bool_default_missing_uses_fallback(self):
        self.assertTrue(cli_commands._bool_default({}, 'audio_ducking', True))

    def test_bool_default_non_bool_uses_fallback(self):
        self.assertTrue(cli_commands._bool_default({'audio_ducking': 'yes'}, 'audio_ducking', True))

    def test_backend_choice_map_matches_menu(self):
        # The reported symptom: configured 'vulkan' must default to choice '5'.
        self.assertEqual(cli_commands._BACKEND_CHOICE['vulkan'], '5')
        # Legacy 'amd' normalizes to vulkan -> still '5'.
        self.assertEqual(
            cli_commands._BACKEND_CHOICE.get(cli_commands.normalize_backend('amd'), '1'), '5')
        self.assertEqual(cli_commands._BACKEND_CHOICE['onnx-asr'], '1')
        self.assertEqual(cli_commands._BACKEND_CHOICE['pywhispercpp'], '2')


if __name__ == '__main__':
    unittest.main()
