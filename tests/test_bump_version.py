"""Exercise release orchestration with no real git, network, or AUR writes."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BumpVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        shutil.copy2(ROOT / 'bump-version.sh', self.repo)
        for name in ('lib/cli.py', 'scripts/install.sh', 'scripts/install-deps.sh'):
            path = self.repo / name
            path.parent.mkdir(exist_ok=True)
            path.touch()
        self.aur = self.root / 'aur-hyprwhspr'
        self.aur.mkdir()
        self.recipe = "pkgver=1.0.0\npkgrel=2\nsha256sums=('old')\n"
        (self.aur / 'PKGBUILD').write_text(self.recipe)
        self.archive = self.root / 'fixture.tar.gz'
        with tarfile.open(self.archive, 'w:gz') as bundle:
            info = tarfile.TarInfo('source/README')
            info.size = 2
            bundle.addfile(info, io.BytesIO(b'ok'))
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        stub = '''import os, pathlib, shutil, sys
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (root / 'calls').open('a') as log:
    log.write(name + ' ' + ' '.join(args) + '\\n')
if name == 'git':
    if args[:2] == ['rev-parse', '--verify']:
        sys.exit(0 if os.environ.get('EXISTING_TAG') else 1)
    if args == ['rev-parse', 'HEAD']:
        print('a' * 40)
    elif args[:1] == ['rev-parse'] and args[-1].endswith('^{commit}'):
        print(('b' if os.environ.get('WRONG_TAG') else 'a') * 40)
    elif args[:1] == ['status'] and os.environ.get('DIRTY'):
        print(' M candidate.py')
    if args[:1] == ['push']:
        sys.exit(int(os.environ.get('PUSH_EXIT', '0')))
if name == 'gh':
    sys.exit(int(os.environ.get('AUTH_EXIT', '0')))
if name == 'python' and args[:1] == ['scripts/release-checks.py']:
    sys.exit(int(os.environ.get('PREFLIGHT_EXIT' if args[1] == 'preflight' else 'WAIT_EXIT', '0')))
if name == 'curl':
    counter = root / 'attempts'
    attempt = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempt))
    output = pathlib.Path(args[args.index('--output') + 1])
    mode = os.environ.get('DOWNLOAD_MODE', '')
    if mode == 'fail' or (mode == 'retry' and attempt == 1):
        output.write_text('HTTP error page')
        sys.exit(22)
    if mode == 'html':
        output.write_text('not an archive')
    else:
        shutil.copyfile(root / 'fixture.tar.gz', output)
'''
        for name in ('git', 'gh', 'curl', 'python', 'sleep', 'makepkg'):
            path = self.bin / name
            path.write_text(f'#!{sys.executable}\n' + stub)
            path.chmod(0o755)

    def run_bump(self, version='1.1.0', **settings):
        env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}',
                   FIXTURE_ROOT=str(self.root), **settings)
        return subprocess.run(['bash', str(self.repo / 'bump-version.sh'), version],
                              env=env, capture_output=True, text=True, timeout=20)

    def calls(self):
        path = self.root / 'calls'
        return path.read_text() if path.exists() else ''

    def test_invalid_versions_fail_before_external_commands(self):
        for version in ('v1.2.3', '1.2', '1.2.3-rc1', '../1.2.3', '1.2.3\n'):
            with self.subTest(version=version):
                self.assertNotEqual(self.run_bump(version).returncode, 0)
                self.assertEqual(self.calls(), '')

    def test_download_failure_retries_and_hashes_only_valid_archive(self):
        result = self.run_bump(DOWNLOAD_MODE='retry')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / 'attempts').read_text(), '2')
        checksum = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.assertIn(checksum, (self.aur / 'PKGBUILD').read_text())
        self.assertIn('curl --fail', self.calls())

    def test_failed_or_invalid_download_leaves_recipe_untouched(self):
        for mode in ('fail', 'html'):
            with self.subTest(mode=mode):
                (self.root / 'attempts').write_text('0')
                result = self.run_bump(DOWNLOAD_MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((self.root / 'attempts').read_text(), '10')
                self.assertEqual((self.aur / 'PKGBUILD').read_text(), self.recipe)
                self.assertFalse((self.aur / '.SRCINFO').exists())

    def test_existing_tag_is_pushed_without_recreation(self):
        result = self.run_bump(EXISTING_TAG='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('git push origin refs/tags/v1.1.0', self.calls())
        self.assertNotIn('git tag ', self.calls())

    def test_rejected_push_stops_before_download_or_recipe_changes(self):
        result = self.run_bump(EXISTING_TAG='1', PUSH_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('curl ', self.calls())
        self.assertEqual((self.aur / 'PKGBUILD').read_text(), self.recipe)

    def test_preflight_failures_never_create_or_push_tag(self):
        for settings in ({'DIRTY': '1'}, {'AUTH_EXIT': '1'}, {'PREFLIGHT_EXIT': '1'},
                         {'EXISTING_TAG': '1', 'WRONG_TAG': '1'}):
            with self.subTest(settings=settings):
                (self.root / 'calls').write_text('')
                result = self.run_bump(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('git tag ', self.calls())
                self.assertNotIn('git push ', self.calls())
                self.assertEqual((self.aur / 'PKGBUILD').read_text(), self.recipe)

    def test_missing_tool_stops_before_github_or_tag_changes(self):
        isolated = self.root / 'isolated-bin'
        isolated.mkdir()
        for name in ('dirname', 'git'):
            (isolated / name).symlink_to(self.bin / name if name == 'git' else shutil.which(name))
        result = subprocess.run(['/bin/bash', str(self.repo / 'bump-version.sh'), '1.1.0'],
                                env=dict(os.environ, PATH=str(isolated), FIXTURE_ROOT=str(self.root)),
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Required command missing: gh', result.stderr)
        self.assertEqual(self.calls(), '')

    def test_workflow_failure_stops_before_archive_or_aur_changes(self):
        result = self.run_bump(WAIT_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('git push origin refs/tags/v1.1.0', self.calls())
        self.assertNotIn('curl ', self.calls())
        self.assertEqual((self.aur / 'PKGBUILD').read_text(), self.recipe)
        self.assertFalse((self.aur / '.SRCINFO').exists())

    def test_success_orders_checks_before_release_and_aur_updates(self):
        result = self.run_bump()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertLess(calls.index('release-checks.py preflight'), calls.index('git tag '))
        self.assertLess(calls.index('git push '), calls.index('release-checks.py wait'))
        self.assertIn('--tag v1.1.0 --commit ' + 'a' * 40, calls)
        self.assertLess(calls.index('release-checks.py wait'), calls.index('curl '))
        self.assertLess(calls.index('curl '), calls.index('makepkg --printsrcinfo'))
        self.assertIn('publish it manually', result.stdout)
