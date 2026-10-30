"""Run the workflow's draft staging shell against a fake GitHub CLI."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/release-application.yml'


class ApplicationReleaseWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.workflow = WORKFLOW.read_text()
        staging = self.workflow.split('      - name: Prepare draft application release\n', 1)[1]
        self.script = textwrap.dedent(staging.split('        run: |\n', 1)[1])
        release = self.root / 'release'
        release.mkdir()
        self.assets = ['hyprwhspr-v1.2.3.tar.gz', 'managed_install.py',
                       'hyprwhspr-release.json', 'SHA256SUMS']
        for name in self.assets[:-1]:
            (release / name).write_bytes(b'fixture')
        checksum = hashlib.sha256(b'fixture').hexdigest()
        (release / 'SHA256SUMS').write_text(''.join(
            f'{checksum}  {name}\n' for name in self.assets[:-1]))
        fake_bin = self.root / 'bin'
        fake_bin.mkdir()
        gh = fake_bin / 'gh'
        gh.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
root = pathlib.Path(os.environ['FIXTURE_ROOT'])
args = sys.argv[1:]
with (root / 'calls').open('a') as log:
    log.write(json.dumps(args) + '\\n')
operation = args[0] if args[0] == 'api' else args[1]
if operation == os.environ.get('FAIL_OPERATION'):
    sys.exit(1)
if operation == 'api':
    print('v0.9.0\\tpublished')
    state = os.environ.get('RELEASE_STATE', '')
    if state:
        print(os.environ['GITHUB_REF_NAME'] + '\\t' + state)
elif operation == 'view':
    print('https://github.com/example/repo/releases/tag/v1.2.3')
''')
        gh.chmod(0o755)
        self.env = dict(os.environ, PATH=f'{fake_bin}:{os.environ["PATH"]}',
                        FIXTURE_ROOT=str(self.root), GITHUB_REPOSITORY='example/repo',
                        GITHUB_REF_NAME='v1.2.3', GITHUB_STEP_SUMMARY=str(self.root / 'summary'))

    def run_stage(self, state='', failure=''):
        result = subprocess.run(['bash', '-c', self.script], cwd=self.root,
                                env=dict(self.env, RELEASE_STATE=state, FAIL_OPERATION=failure),
                                capture_output=True, text=True, timeout=10)
        calls = self.root / 'calls'
        self.calls = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result

    def test_missing_release_creates_draft_with_all_assets(self):
        result = self.run_stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        create = next(call for call in self.calls if call[:2] == ['release', 'create'])
        for flag in ('--draft', '--verify-tag', '--generate-notes'):
            self.assertIn(flag, create)
        for asset in self.assets:
            self.assertIn('./' + asset, create)
        summary = (self.root / 'summary').read_text()
        self.assertIn('https://github.com/example/repo/releases/tag/v1.2.3', summary)
        self.assertIn('publish the draft manually', summary)

    def test_existing_draft_only_uploads_assets_and_reads_url(self):
        result = self.run_stage('draft')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[:2] for call in self.calls[1:]],
                         [['release', 'upload'], ['release', 'view']])
        self.assertIn('--clobber', self.calls[1])
        for asset in self.assets:
            self.assertIn('./' + asset, self.calls[1])

    def test_published_release_is_untouched(self):
        result = self.run_stage('published')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls), 1)
        self.assertIn('already published', (self.root / 'summary').read_text())

    def test_lookup_and_mutation_errors_fail_without_success_summary(self):
        for state, operation in (('', 'api'), ('', 'create'), ('draft', 'upload')):
            with self.subTest(operation=operation):
                result = self.run_stage(state, operation)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'summary').exists())
                if operation == 'api':
                    self.assertEqual(len(self.calls), 1)

    def test_bad_checksum_prevents_github_calls(self):
        (self.root / 'release' / self.assets[0]).write_bytes(b'corrupt')
        self.assertNotEqual(self.run_stage().returncode, 0)
        self.assertEqual(self.calls, [])

    def test_staging_keeps_build_and_smoke_gates(self):
        staging = self.workflow.split('  prepare-draft:\n', 1)[1]
        self.assertIn('needs: [payload, smoke]', staging)
        self.assertIn('actions/download-artifact@', staging)
        self.assertIn('group: application-release-${{ github.ref }}', self.workflow)
        self.assertIn('cancel-in-progress: false', self.workflow)
        self.assertIn("tags: ['v[0-9]*']", self.workflow)
        self.assertNotIn('gh release edit', self.workflow)
        self.assertNotIn('--draft=false', self.workflow)
