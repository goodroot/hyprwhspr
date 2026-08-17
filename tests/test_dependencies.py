import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import dependencies


class DependencyErrorTests(unittest.TestCase):
    def require_missing(self, install_hint=None):
        stderr = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=ImportError("missing test module")):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                dependencies.require_package(
                    "sounddevice",
                    install_hint=install_hint,
                )
        self.assertEqual(raised.exception.code, 1)
        return stderr.getvalue()

    def test_default_error_recommends_setup_without_invalid_distro_package(self):
        output = self.require_missing()

        self.assertIn("python-sounddevice is not available", output)
        self.assertIn("ImportError: missing test module", output)
        self.assertIn("hyprwhspr setup", output)
        self.assertIn("./bin/hyprwhspr setup", output)
        self.assertNotIn("apt install python3-sounddevice", output)

    def test_explicit_install_hint_is_preserved(self):
        output = self.require_missing("Use the custom repair command")

        self.assertIn("Use the custom repair command", output)
        self.assertNotIn("hyprwhspr setup", output)


if __name__ == "__main__":
    unittest.main()
