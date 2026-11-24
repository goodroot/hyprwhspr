import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "bin" / "hyprwhspr"


class LauncherStartupTests(unittest.TestCase):
    def launcher_env(self, data_home):
        env = os.environ.copy()
        for key in ('VIRTUAL_ENV', 'HYPRWHSPR_INSTALL_METHOD',
                    'HYPRWHSPR_BACKEND_ENV', 'HYPRWHSPR_NIX_ROOT'):
            env.pop(key, None)
        env["XDG_DATA_HOME"] = str(data_home)
        return env

    def run_launcher(self, data_home, *args, launcher=LAUNCHER, **overrides):
        env = self.launcher_env(data_home)
        env.update(overrides)
        return subprocess.run(
            [str(launcher), *args],
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
            env = self.launcher_env(Path(tmp))
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

    def make_python(self, path, supported=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\n'
                        f'if [ "$1" = -c ]; then exit {0 if supported else 1}; fi\n'
                        'printf "%s\\n" "$0" "$@"\n')
        path.chmod(0o755)
        return path

    def make_package(self, root, git='directory', installed=False, release=False):
        package = root / 'package'
        library = package / ('lib/hyprwhspr' if installed else 'lib')
        library.mkdir(parents=True)
        (library / 'cli.py').write_text('print("system cli")\n')
        (package / 'bin').mkdir()
        launcher = package / 'bin/hyprwhspr'
        shutil.copy2(LAUNCHER, launcher)
        if git == 'directory':
            (package / '.git').mkdir()
        elif git == 'file':
            (package / '.git').write_text('gitdir: /unused/worktree\n')
        if release:
            (package / 'release.json').write_text('{}')
        return launcher, library

    def test_activated_checkout_uses_environment_for_all_commands(self):
        routes = (((), 'main.py', ()), (('test',), 'cli.py', ('test',)),
                  (('transcribe', 'a b.wav'), 'cli.py', ('transcribe', 'a b.wav')),
                  (('config', 'show'), 'cli.py', ('config', 'show')),
                  (('--help',), 'cli.py', ('--help',)),
                  (('help',), 'cli.py', ('--help',)),
                  (('--version',), 'cli.py', ('--version',)))
        for git in ('directory', 'file'):
            with self.subTest(git=git), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                launcher, library = self.make_package(root, git=git)
                active = self.make_python(root / 'package/active env/bin/python')
                self.make_python(root / 'data/hyprwhspr/venv/bin/python')
                for args, script, forwarded in routes:
                    with self.subTest(args=args):
                        result = self.run_launcher(root / 'data', *args, launcher=launcher,
                                                   VIRTUAL_ENV=str(active.parent.parent))
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout.splitlines(),
                                         [str(active), '-B', str(library / script), *forwarded])

    def test_unusable_activation_preserves_existing_selection(self):
        for activation in (None, '', 'missing', 'not executable', 'unsupported'):
            with self.subTest(activation=activation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                launcher, library = self.make_package(root)
                fallback = self.make_python(root / 'data/hyprwhspr/venv/bin/python')
                overrides = {}
                if activation is not None:
                    overrides['VIRTUAL_ENV'] = str(root / 'package' / activation) if activation else ''
                if activation == 'not executable':
                    active = self.make_python(root / 'package' / activation / 'bin/python')
                    active.chmod(0o644)
                if activation == 'unsupported':
                    self.make_python(root / 'package' / activation / 'bin/python', supported=False)
                result = self.run_launcher(root / 'data', launcher=launcher, **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 [str(fallback), '-B', str(library / 'main.py')])
                result = self.run_launcher(root / 'data', 'config', 'show',
                                           launcher=launcher, **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'system cli')

    def test_git_install_ignores_external_environment(self):
        for location in ('external', 'package-other', 'symlink'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                launcher, library = self.make_package(root)
                active = self.make_python(root / location / 'bin/python')
                environment = active.parent.parent
                if location == 'symlink':
                    link = root / 'package/.venv'
                    link.symlink_to(environment, target_is_directory=True)
                    environment = link
                fallback = self.make_python(root / 'data/hyprwhspr/venv/bin/python')
                # Legacy setup exposes the checkout launcher through a symlink.
                command = root / 'hyprwhspr'
                command.symlink_to(launcher)
                for args in ((), ('test',), ('transcribe', 'sample.wav')):
                    result = self.run_launcher(root / 'data', *args, launcher=command,
                                               VIRTUAL_ENV=str(environment))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    script = 'cli.py' if args else 'main.py'
                    self.assertEqual(result.stdout.splitlines(),
                                     [str(fallback), '-B', str(library / script), *args])
                for args in (('status',), ('config', 'show')):
                    result = self.run_launcher(root / 'data', *args, launcher=command,
                                               VIRTUAL_ENV=str(environment))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), 'system cli')

    def test_installed_packages_ignore_activation(self):
        for options in ({'git': None}, {'installed': True}, {'release': True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                launcher, library = self.make_package(root, **options)
                active = self.make_python(root / 'package/active/bin/python')
                fallback = self.make_python(root / 'data/hyprwhspr/venv/bin/python')
                result = self.run_launcher(root / 'data', launcher=launcher,
                                           VIRTUAL_ENV=str(active.parent.parent))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(),
                                 [str(fallback), '-B', str(library / 'main.py')])

    def test_managed_and_nix_launchers_keep_precedence(self):
        for mode in ('managed', 'nix'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                launcher, library = self.make_package(root, release=(mode == 'managed'))
                active = self.make_python(root / 'package/active/bin/python')
                managed = self.make_python(root / 'data/hyprwhspr/launcher')
                overrides = {'VIRTUAL_ENV': str(active.parent.parent)}
                expected = [str(managed)]
                if mode == 'nix':
                    nix = self.make_python(root / 'nix/bin/python')
                    overrides.update(HYPRWHSPR_INSTALL_METHOD='nix',
                                     HYPRWHSPR_NIX_ROOT=str(launcher.parent.parent),
                                     HYPRWHSPR_BACKEND_ENV=str(nix.parent.parent))
                    expected = [str(nix), '-B', str(library / 'main.py')]
                result = self.run_launcher(root / 'data', launcher=launcher, **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
