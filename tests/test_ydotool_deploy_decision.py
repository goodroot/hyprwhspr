import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported.

    cli_commands transitively imports global_shortcuts -> evdev (a hard import)
    and rich.prompt. On a dev machine both are installed and we must NOT shadow
    them (importing the real module leaves it in sys.modules); in a bare CI we
    fall back to a minimal stub so importing cli_commands still works.
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
_stub_if_missing(
    "evdev",
    InputDevice=object,
    categorize=lambda *a, **k: None,
    ecodes=types.SimpleNamespace(),
    UInput=object,
)

import cli_commands  # noqa: E402

TARGET = "/home/u/.config/systemd/user/ydotool.service"
TEMPLATE = (
    "# Managed by hyprwhspr.\n"
    "[Unit]\n"
    "Description=ydotool user daemon (deployed by hyprwhspr)\n"
    "[Service]\n"
    "ExecStart=/usr/bin/ydotoold\n"
)


class YdotoolDeployDecisionTests(unittest.TestCase):
    def decide(self, fragment_path, target_text):
        return cli_commands._ydotool_deploy_decision(
            fragment_path, TARGET, target_text, TEMPLATE
        )

    def test_distro_user_unit_is_respected(self):
        action, _ = self.decide("/usr/lib/systemd/user/ydotool.service", None)
        self.assertEqual(action, "skip")

    def test_admin_unit_is_respected(self):
        action, _ = self.decide("/etc/systemd/user/ydotool.service", None)
        self.assertEqual(action, "skip")

    def test_fresh_system_writes(self):
        action, _ = self.decide("", None)
        self.assertEqual(action, "write")

    def test_our_file_up_to_date_skips(self):
        action, _ = self.decide(TARGET, TEMPLATE)
        self.assertEqual(action, "skip")

    def test_our_file_drift_updates(self):
        action, _ = self.decide(TARGET, TEMPLATE + "# user tweak\n")
        self.assertEqual(action, "update")

    def test_legacy_marker_updates(self):
        legacy = (
            "[Unit]\n"
            "Description=ydotool user daemon (deployed by hyprwhspr)\n"
            "[Service]\nExecStart=/usr/bin/ydotoold\n"
        )
        action, _ = self.decide(TARGET, legacy)
        self.assertEqual(action, "update")

    def test_foreign_file_at_target_is_respected(self):
        foreign = "[Unit]\nDescription=my own ydotoold\n[Service]\nExecStart=/usr/bin/ydotoold\n"
        action, _ = self.decide(TARGET, foreign)
        self.assertEqual(action, "skip")

    def test_empty_fragment_with_our_file_falls_through(self):
        # FragmentPath can be empty if the manager hasn't reloaded; must still
        # treat an existing marked file as ours.
        action, _ = self.decide("", TEMPLATE + "# drift\n")
        self.assertEqual(action, "update")


if __name__ == "__main__":
    unittest.main()
