import importlib
import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
from config_manager import ConfigManager  # noqa: E402


class SetupCommandScopeTests(unittest.TestCase):
    def test_run_command_is_not_function_local(self):
        # Regression: a conditional local re-import of run_command made it a
        # function-local throughout setup_command, so non-cloud GNOME setups hit
        # UnboundLocalError at the gsettings (toolkit-accessibility) call. It must
        # resolve to the module-level helper instead.
        self.assertNotIn(
            "run_command", cli_commands.setup_command.__code__.co_varnames
        )


class ConfigDefaultTests(unittest.TestCase):
    def test_defaults_match_schema_for_new_settings(self):
        with mock.patch.object(ConfigManager, "_ensure_config_dir"), \
                mock.patch.object(ConfigManager, "_load_config"):
            defaults = ConfigManager().get_all_settings()

        self.assertEqual(defaults["audio_volume"], 0.5)
        self.assertEqual(defaults["error_sound_volume"], 0.5)
        self.assertEqual(defaults["task"], "transcribe")
        self.assertEqual(defaults["sampling_strategy"], "beam_search")
        self.assertEqual(defaults["beam_size"], 5)
        self.assertFalse(defaults["prefer_clipboard_paste"])


class UninstallYdotoolOwnershipTests(unittest.TestCase):
    def _make_ydotool_unit(self, unit_text):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        user_systemd = root / "systemd"
        user_systemd.mkdir()
        ydotool_unit = user_systemd / cli_commands.YDOTOOL_UNIT
        ydotool_unit.write_text(unit_text, encoding="utf-8")
        return root, user_systemd, ydotool_unit

    def _run_uninstall_with_ydotool_unit(self, unit_text):
        root, user_systemd, ydotool_unit = self._make_ydotool_unit(unit_text)

        calls = []

        def fake_run_command(cmd, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        patches = [
            mock.patch.object(cli_commands, "USER_SYSTEMD_DIR", user_systemd),
            mock.patch.object(cli_commands, "USER_HOME", root / "missing-home"),
            mock.patch.object(cli_commands, "USER_CONFIG_DIR", root / "missing-config"),
            mock.patch.object(cli_commands, "VENV_DIR", root / "missing-venv"),
            mock.patch.object(cli_commands, "PARAKEET_VENV_DIR", root / "missing-parakeet"),
            mock.patch.object(cli_commands, "PYWHISPERCPP_SRC_DIR", root / "missing-src"),
            mock.patch.object(cli_commands, "PYWHISPERCPP_MODELS_DIR", root / "missing-models"),
            mock.patch.object(cli_commands, "STATE_DIR", root / "missing-state"),
            mock.patch.object(cli_commands, "CREDENTIALS_FILE", root / "missing-creds"),
            mock.patch.object(cli_commands, "USER_BASE", root / "missing-user-base"),
            mock.patch.object(cli_commands, "setup_waybar"),
            mock.patch.object(cli_commands, "_detect_current_backend", return_value=None),
            mock.patch.object(cli_commands, "run_command", side_effect=fake_run_command),
        ]
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            cli_commands.uninstall_command(skip_permissions=True, yes=True)

        return ydotool_unit, calls

    def _run_migration_with_ydotool_unit(self, unit_text):
        _root, user_systemd, ydotool_unit = self._make_ydotool_unit(unit_text)
        calls = []

        def fake_run_command(cmd, **kwargs):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(cli_commands, "USER_SYSTEMD_DIR", user_systemd), \
                mock.patch.object(cli_commands, "run_command", side_effect=fake_run_command):
            cli_commands._migrate_remove_managed_ydotool_unit()

        return ydotool_unit, calls

    def test_migration_removes_hyprwhspr_managed_ydotool_unit(self):
        unit, calls = self._run_migration_with_ydotool_unit(
            "[Unit]\nDescription=ydotoold\n# deployed by hyprwhspr\n"
        )

        self.assertFalse(unit.exists())
        self.assertIn(
            ["systemctl", "--user", "disable", "--now", cli_commands.YDOTOOL_UNIT],
            calls,
        )
        self.assertIn(["systemctl", "--user", "daemon-reload"], calls)

    def test_migration_leaves_foreign_ydotool_unit_untouched(self):
        unit, calls = self._run_migration_with_ydotool_unit(
            "[Unit]\nDescription=user-owned ydotoold\n"
        )

        self.assertTrue(unit.exists())
        self.assertEqual(calls, [])

    def test_uninstall_removes_hyprwhspr_managed_ydotool_unit(self):
        unit, calls = self._run_uninstall_with_ydotool_unit(
            "[Unit]\nDescription=ydotoold\n# Managed by hyprwhspr\n"
        )

        self.assertFalse(unit.exists())
        self.assertIn(["systemctl", "--user", "stop", cli_commands.YDOTOOL_UNIT], calls)
        self.assertIn(["systemctl", "--user", "disable", cli_commands.YDOTOOL_UNIT], calls)

    def test_uninstall_leaves_foreign_ydotool_unit_untouched(self):
        unit, calls = self._run_uninstall_with_ydotool_unit(
            "[Unit]\nDescription=user-owned ydotoold\n"
        )

        self.assertTrue(unit.exists())
        self.assertNotIn(["systemctl", "--user", "stop", cli_commands.YDOTOOL_UNIT], calls)
        self.assertNotIn(["systemctl", "--user", "disable", cli_commands.YDOTOOL_UNIT], calls)


if __name__ == "__main__":
    unittest.main()
