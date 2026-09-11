import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "bin" / "hyprwhspr"


class LauncherStartupTests(unittest.TestCase):
    def run_launcher(self, data_home, *args):
        env = os.environ.copy()
        env["XDG_DATA_HOME"] = str(data_home)
        return subprocess.run(
            [str(LAUNCHER), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_launch_without_venv_reports_incomplete_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_launcher(Path(tmp))

        self.assertEqual(result.returncode, 1)
        self.assertIn("hyprwhspr setup is incomplete", result.stderr)
        self.assertIn("bin/hyprwhspr setup", result.stderr)
        self.assertNotIn("python3-sounddevice", result.stderr)

    def test_backend_test_without_venv_reports_incomplete_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_launcher(Path(tmp), "test")

        self.assertEqual(result.returncode, 1)
        self.assertIn("hyprwhspr setup is incomplete", result.stderr)

    def test_default_launch_uses_venv_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_home = Path(tmp)
            venv_python = data_home / "hyprwhspr" / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\"\n",
                encoding="utf-8",
            )
            venv_python.chmod(venv_python.stat().st_mode | stat.S_IXUSR)

            result = self.run_launcher(data_home)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(ROOT / "lib" / "main.py"), result.stdout)

    def test_legacy_launch_preserves_user_site_and_inherited_search_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / 'hyprwhspr/venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "${PYTHONNOUSERSITE:-enabled}" "$PYTHONPATH"\n')
            interpreter.chmod(0o755)
            env = os.environ.copy()
            env.pop('PYTHONNOUSERSITE', None)
            env.update(XDG_DATA_HOME=tmp, PYTHONPATH='/legacy/dependencies')
            result = subprocess.run([str(LAUNCHER)], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], 'enabled')
        self.assertIn('/legacy/dependencies', result.stdout)

    def test_version_remains_available_without_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_launcher(Path(tmp), "--version")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("setup is incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
