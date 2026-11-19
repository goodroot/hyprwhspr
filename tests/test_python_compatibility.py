"""Keep independently bootstrappable Python gates and release coverage aligned."""
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/src'))
import backend_installer
import managed_install
from cli import _shared


class PythonCompatibilityTests(unittest.TestCase):
    def test_error_hints_follow_current_and_future_ceiling(self):
        for minor in (15, 16):
            maximum, previous = f'3.{minor}', f'3.{minor - 1}'
            with self.subTest(minor=minor), \
                    mock.patch.object(backend_installer, 'MAX_COMPATIBLE_PYTHON', (3, minor)), \
                    mock.patch('sys.stdout', new_callable=io.StringIO) as output:
                with self.assertRaises(SystemExit):
                    backend_installer._python_compatibility_error((3, minor + 1))
                text = output.getvalue()
                self.assertIn(f'Install Python {maximum} or {previous}', text)
                self.assertIn(f'dnf install python{maximum}', text)
                self.assertIn(f'python3{minor}  # or python3{minor - 1}', text)
                self.assertIn(f'/path/to/python{maximum}', text)
            with mock.patch.object(_shared, 'MAX_COMPATIBLE_PYTHON', (3, minor)), \
                    mock.patch.object(_shared, '_get_python_version', return_value=(3, minor + 1)), \
                    mock.patch.object(_shared.shutil, 'which', return_value='/fixture/python'), \
                    mock.patch.object(_shared.os.path, 'isfile', return_value=False):
                accepted, _, text = _shared._check_python_compatibility()
                self.assertFalse(accepted)
                self.assertIn(f'Install Python {maximum} or {previous}', text)
                self.assertIn(f'dnf install python{maximum}', text)
                self.assertIn(f'/path/to/python{maximum}', text)
            with mock.patch.object(backend_installer, 'MAX_COMPATIBLE_PYTHON', (3, minor)), \
                    mock.patch.object(backend_installer, 'HYPRWHSPR_ROOT', str(ROOT)), \
                    mock.patch.object(backend_installer.os.path, 'isfile', return_value=True), \
                    mock.patch.object(backend_installer.os, 'access', return_value=True), \
                    mock.patch.object(backend_installer, '_get_python_version', return_value=(3, minor + 1)), \
                    mock.patch.object(backend_installer, 'log_error') as error:
                with self.assertRaises(SystemExit):
                    backend_installer.setup_python_venv(custom_python='/fixture/python')
                self.assertIn(f'--python /usr/bin/python{maximum}', error.call_args.args[0])

    def test_shell_package_examples_follow_ceiling(self):
        source = (ROOT / 'scripts/install-deps.sh').read_text()
        snippet = re.search(r'echo "  1\) Install Python.*?\n        esac', source, re.S).group(0)
        for minor in (15, 16):
            for manager in ('dnf', 'apt', 'zypper'):
                with self.subTest(minor=minor, manager=manager):
                    command = (f'source scripts/install-deps.sh; MAX_PYTHON_MINOR={minor}; '
                               f'PKG_MANAGER={manager}; ' + snippet)
                    result = subprocess.run(['bash', '-c', command], cwd=ROOT,
                                            capture_output=True, text=True, check=True)
                    self.assertIn(f'Install Python 3.{minor} or 3.{minor - 1}', result.stdout)
                    package = f'python3{minor}' if manager == 'zypper' else f'python3.{minor}'
                    self.assertIn(f'install {package}', result.stdout)

    def test_launchers_and_managed_probe_accept_only_supported_boundaries(self):
        ceiling = backend_installer.MAX_COMPATIBLE_PYTHON
        cases = [((3, 10), False), ((3, 11), True), (ceiling, True),
                 ((ceiling[0], ceiling[1] + 1), False)]
        for path in ('bin/hyprwhspr', 'scripts/managed-launcher.sh'):
            source = (ROOT / path).read_text()
            probe = re.search(r"-c '([^']*sys\.version_info[^']*)'", source).group(1)
            for version, accepted in cases:
                with self.subTest(path=path, version=version), mock.patch.object(sys, 'version_info', version):
                    with self.assertRaises(SystemExit) as result:
                        exec(probe, {})
                    self.assertEqual(result.exception.code, 0 if accepted else 1)
        for version, accepted in cases:
            identity = {'path': '/fixture/python', 'version': [*version, 0], 'identity': str(version)}
            with self.subTest(managed=version), mock.patch.object(
                    managed_install, 'run', return_value=mock.Mock(stdout=json.dumps(identity))):
                if accepted:
                    self.assertEqual(managed_install.interpreter('/fixture/python'), identity)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'No compatible Python'):
                        managed_install.interpreter('/fixture/python')

    def test_shell_dependency_bounds_match_runtime_contract(self):
        command = ('source scripts/install-deps.sh; '
                   'printf "%s %s %s %s" "$MIN_PYTHON_MAJOR" "$MIN_PYTHON_MINOR" '
                   '"$MAX_PYTHON_MAJOR" "$MAX_PYTHON_MINOR"')
        result = subprocess.run(['bash', '-c', command], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(tuple(map(int, result.stdout.split())),
                         (3, 11, *backend_installer.MAX_COMPATIBLE_PYTHON))

    def test_ci_covers_floor_ceiling_and_every_wheel_minor(self):
        maximum = backend_installer.MAX_COMPATIBLE_PYTHON
        self.assertEqual(maximum[0], 3)
        versions = [f'3.{minor}' for minor in range(11, maximum[1] + 1)]
        workflow = (ROOT / '.github/workflows/build-wheels.yml').read_text()
        declared = re.search(r"PYTHON_VERSIONS: '([^']+)'", workflow).group(1).split()
        self.assertEqual(declared, versions)
        matrices = re.findall(r'python-version: \[([^\]]+)\]', workflow)
        self.assertEqual(len(matrices), 4)
        for matrix in matrices:
            self.assertEqual(re.findall(r"'([^']+)'", matrix), versions)
        tests = (ROOT / '.github/workflows/tests.yml').read_text()
        matrix = re.search(r'python-version: \[([^\]]+)\]', tests).group(1)
        self.assertEqual(re.findall(r"'([^']+)'", matrix), [versions[0], versions[-1]])

    def test_documented_ranges_match_enforced_range(self):
        maximum = '.'.join(map(str, backend_installer.MAX_COMPATIBLE_PYTHON))
        for path in ('README.md', 'docs/QUICK_START.md', 'docs/MANAGED_INSTALLATION.md',
                     'scripts/install.sh', 'bin/hyprwhspr', 'lib/src/managed_install.py'):
            with self.subTest(path=path):
                ranges = re.findall(r'3\.11[-–](3\.\d+)', (ROOT / path).read_text())
                self.assertTrue(ranges)
                self.assertEqual(set(ranges), {maximum})

    def test_cli_fallback_tracks_ceiling_and_stops_at_floor(self):
        # Also simulate the next bump so an enumerated list cannot silently lag.
        for ceiling in ((3, 15), (3, 16)):
            for available_minor in (ceiling[1], 11, None):
                candidate = f'/usr/bin/python3.{available_minor}' if available_minor else None
                def version(path):
                    return (3, ceiling[1] + 1) if path == '/fixture/default' else (3, available_minor)
                with self.subTest(ceiling=ceiling, available=available_minor), \
                        mock.patch.object(_shared, 'MAX_COMPATIBLE_PYTHON', ceiling), \
                        mock.patch.object(_shared.shutil, 'which', return_value='/fixture/default'), \
                        mock.patch.object(_shared, '_get_python_version', side_effect=version), \
                        mock.patch.object(_shared.os.path, 'isfile', side_effect=lambda p: p == candidate) as exists, \
                        mock.patch.object(_shared.os, 'access', return_value=True):
                    accepted, _, guidance = _shared._check_python_compatibility()
                self.assertEqual(accepted, candidate is not None)
                if candidate:
                    self.assertIn(candidate, guidance)
                else:
                    self.assertEqual([call.args[0] for call in exists.call_args_list],
                                     [f'{prefix}/python3.{minor}'
                                      for minor in range(ceiling[1], 10, -1)
                                      for prefix in ('/usr/bin', '/usr/local/bin')])


if __name__ == '__main__':
    unittest.main()
