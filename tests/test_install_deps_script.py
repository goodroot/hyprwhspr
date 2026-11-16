import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'scripts/install-deps.sh').read_text(encoding='utf-8')


def body(name):
    return SCRIPT.split(name + '() {', 1)[1].split('\n}\n', 1)[0]


def run(snippet, **env):
    return subprocess.run(['bash', '-c', 'source scripts/install-deps.sh; ' + snippet], cwd=ROOT,
                          capture_output=True, text=True, timeout=30, env={**os.environ, **env})


class InstallDepsScriptTests(unittest.TestCase):
    def test_wtype_is_installed_by_every_package_manager(self):
        # The docs name wtype as the primary Wayland typing path.
        for name in ('install_deps_apt', 'install_deps_dnf', 'install_deps_zypper'):
            self.assertRegex(body(name), r'\bwtype\b', name)

    def test_ydotool_package_is_fetched_over_https_and_checked_before_dpkg(self):
        self.assertNotIn('http://', SCRIPT)
        self.assertRegex(SCRIPT, r'YDOTOOL_DEB_URL="https://deb\.debian\.org/\S+_amd64\.deb"')
        self.assertRegex(SCRIPT, r'YDOTOOL_DEB_SHA256="[0-9a-f]{64}"')
        backports = body('install_ydotool_backports')
        self.assertLess(backports.index('sha256sum -c'), backports.index('sudo dpkg -i'))
        self.assertLess(backports.index('dpkg --print-architecture'), backports.index('wget'))

    def test_checksum_gate_rejects_a_different_file(self):
        with tempfile.TemporaryDirectory() as temp:
            deb = Path(temp) / 'ydotool.deb'
            deb.write_bytes(b'not the package')
            result = run(f'echo "$YDOTOOL_DEB_SHA256  {deb}" | sha256sum -c --status')
        self.assertNotEqual(result.returncode, 0)

    def test_missing_ydotool_outside_debian_warns_and_continues(self):
        result = run('PKG_MANAGER=dnf; PATH=/nonexistent; handle_ydotool_other')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('sudo dnf install ydotool', result.stdout)

    def test_failed_backport_does_not_abort_the_run(self):
        for call in re.findall(r'^\s*install_ydotool_backports.*$', body('handle_ydotool_debian'), re.MULTILINE):
            self.assertIn('||', call)

    def test_a_failing_package_step_is_reported_not_silent(self):
        main = body('main')
        self.assertIn('set -E', main)
        self.assertLess(main.index('trap '), main.index('read -p'))
        with tempfile.TemporaryDirectory() as temp:
            sudo = Path(temp) / 'sudo'
            sudo.write_text('#!/bin/sh\nexit 3\n')
            sudo.chmod(0o755)
            trap = re.search(r"^\s*(trap '.*' ERR)$", main, re.MULTILINE).group(1)
            result = run(f'set -eE; {trap}; install_deps_dnf', PATH=f"{temp}:{os.environ['PATH']}")
        self.assertEqual(result.returncode, 3)
        self.assertIn('Dependency installation stopped', result.stdout)

    def test_closing_message_never_tells_a_bootstrap_user_to_clone(self):
        self.assertNotIn('git clone', SCRIPT)
        self.assertNotIn('Log out', SCRIPT)
        installer = (ROOT / 'scripts/install.sh').read_text(encoding='utf-8')
        self.assertIn('HYPRWHSPR_BOOTSTRAP=1 bash "$CLONE_DIR/scripts/install-deps.sh"', installer)
        self.assertIn('HYPRWHSPR_BOOTSTRAP', body('main'))

    def test_validate_hint_uses_https(self):
        self.assertNotIn('http://deb.debian.org',
                         (ROOT / 'lib/src/cli/maintenance.py').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
