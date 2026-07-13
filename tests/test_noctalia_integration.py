import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported."""
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


def _write_settings(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".toml", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _no_noctalia_binary():
    """Skip the `noctalia config validate` round-trip in tests."""
    return mock.patch.object(cli_commands.shutil, "which", return_value=None)


class MigrateLegacySettingsTests(unittest.TestCase):
    def _migrate(self, content: str) -> str:
        settings = _write_settings(content)
        self.addCleanup(settings.unlink)
        dst = cli_commands._noctalia_paths()
        with _no_noctalia_binary():
            self.assertTrue(
                cli_commands._noctalia_migrate_legacy_settings(settings, dst))
        return settings.read_text(encoding="utf-8")

    def test_renames_plugin_and_template_ids(self):
        out = self._migrate(
            '[plugins]\nenabled = [ "goodroot/hyprwhspr" ]\n\n'
            '[widget.status]\ntype = "goodroot/hyprwhspr:status"\n\n'
            '[theme.templates.user.hyprwhspr_mic_osd]\n'
        )
        self.assertNotIn("goodroot/hyprwhspr", out.replace("goodroot/noctwhspr", ""))
        self.assertIn('enabled = [ "goodroot/noctwhspr" ]', out)
        self.assertIn('type = "goodroot/noctwhspr:status"', out)
        self.assertIn("[theme.templates.user.noctwhspr_mic_osd]", out)

    def test_superstring_ids_are_untouched(self):
        # Regression: unanchored replace corrupted ids that merely contain the
        # legacy id as a prefix.
        out = self._migrate(
            'enabled = [ "goodroot/hyprwhspr", "goodroot/hyprwhspr-extras" ]\n')
        self.assertIn('"goodroot/noctwhspr"', out)
        self.assertIn('"goodroot/hyprwhspr-extras"', out)

    def test_template_path_migrates_regardless_of_serialization(self):
        # Regression: only two literal path spellings were replaced, so a
        # tilde-form input_path kept pointing at the deleted legacy file.
        out = self._migrate(
            '[theme.templates.user.hyprwhspr_mic_osd]\n'
            'input_path = "~/.config/noctalia/templates/hyprwhspr-mic-osd.css"\n'
        )
        self.assertIn('"~/.config/noctalia/templates/noctwhspr-mic-osd.css"', out)
        out = self._migrate(
            'input_path = "$XDG_CONFIG_HOME/noctalia/templates/hyprwhspr-mic-osd.css"\n')
        self.assertIn(
            '"$XDG_CONFIG_HOME/noctalia/templates/noctwhspr-mic-osd.css"', out)

    def test_noop_when_nothing_legacy(self):
        content = '[plugins]\nenabled = [ "goodroot/noctwhspr" ]\n'
        self.assertEqual(self._migrate(content), content)


class RegisterTemplateTests(unittest.TestCase):
    def test_registered_section_with_stale_input_path_is_reported(self):
        # Regression: the marker-only early return masked a section whose
        # input_path no longer referenced the current template file.
        dst = cli_commands._noctalia_paths()
        settings = _write_settings(
            '[theme.templates.user.noctwhspr_mic_osd]\n'
            'input_path = "$XDG_CONFIG_HOME/noctalia/templates/somewhere-else.css"\n'
        )
        self.addCleanup(settings.unlink)
        with _no_noctalia_binary():
            self.assertFalse(cli_commands._noctalia_register_template(
                settings, dst['template_input'], dst['template_output']))

    def test_registered_section_pointing_at_current_file_passes(self):
        dst = cli_commands._noctalia_paths()
        settings = _write_settings(
            '[theme.templates.user.noctwhspr_mic_osd]\n'
            f'input_path = "$XDG_CONFIG_HOME/noctalia/templates/{dst["template_input"].name}"\n'
        )
        self.addCleanup(settings.unlink)
        with _no_noctalia_binary():
            self.assertTrue(cli_commands._noctalia_register_template(
                settings, dst['template_input'], dst['template_output']))


if __name__ == "__main__":
    unittest.main()
