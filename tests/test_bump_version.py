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
    if args[:1] == ['push']:
        sys.exit(int(os.environ.get('PUSH_EXIT', '0')))
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
        for name in ('git', 'curl', 'python', 'sleep', 'makepkg'):
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
