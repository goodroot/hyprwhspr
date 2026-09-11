"""Exercise recovery state readers consistently, including rollback queue merges."""
import io
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
from unittest import mock

from tests import test_managed_install as fixtures
from tests.test_install_second_review import ROOT
import managed_install as managed


class SeventeenthReviewTests(fixtures.ManagedFixture):
    def test_all_auxiliary_recovery_reads_tolerate_corruption_and_io_errors(self):
        files = ('deferred-cleanup.json', 'deferred-restoration.json',
                 'unverified-generations.json', 'deferred-legacy-owned.json')
        for name in files:
            for error in (ValueError('truncated JSON'), PermissionError('denied'),
                          IsADirectoryError('directory'), UnicodeDecodeError('utf-8', b'\x80', 0, 1, 'bad byte')):
                for activated in (False, True):
                    with self.subTest(name=name, error=type(error).__name__, activated=activated):
                        base = self.root / f'case-{name}-{type(error).__name__}-{activated}'
                        installation = managed.Installation(base / 'data', base / 'state')
                        stage = installation.data / 'environments/staged'
                        stage.mkdir(parents=True)
                        bad = installation.state / name
                        managed.atomic_json(bad, {'invalid': 'evidence'})
                        if activated:
                            config = base / 'bindings.conf'
                            config.write_text('new config')
                            managed.atomic_json(installation.current, {'root': 'new'})
                            managed.atomic_json(installation.journal, {'phase': 'activated', 'old': {'root': 'old'},
                                'created': [str(stage)], 'integrations': {str(config): {'content': 'old config', 'mode': 0o644}}})
                        original = managed.read_json
                        raised = []
                        def read(path, *args):
                            if Path(path) == bad and not raised:
                                raised.append(True)
                                raise error
                            return original(path, *args)
                        with mock.patch.object(managed, 'read_json', read), mock.patch.object(installation, 'service_state', return_value={}), mock.patch.object(installation, 'daemon_running', return_value=False):
                            for _ in range(3):
                                installation.recover()
                        self.assertTrue(raised, 'The failing read must actually execute')
                        self.assertFalse(installation.journal.exists())
                        self.assertTrue(stage.exists())
                        self.assertTrue(list(installation.state.glob(name + '.unreadable-*')))
                        if activated:
                            self.assertEqual(managed.read_json(installation.current), {'root': 'old'})

    def test_tolerant_state_shapes_do_not_weaken_strict_json(self):
        path = self.root / 'state'
        for content in ('{truncated', 'null', '{"not":"a path list"}', '[null]'):
            path.write_text(content)
            self.assertEqual(managed.read_state(path, []), [])
        path.write_text('{truncated')
        with self.assertRaises(ValueError):
            managed.read_json(path, {})

    def test_missing_precondition_retires_after_transient_read_failure_without_overwriting(self):
        path = self.root / 'bindings.conf'
        path.write_text('user config')
        stage = self.install.data / 'environments/cuda'
        stage.mkdir(parents=True)
        queue = self.install.state / 'deferred-restoration.json'
        managed.atomic_json(queue, {'files': {str(path): {'snapshot': None}}, 'protected': [str(stage)]})
        managed.atomic_json(self.install.state / 'deferred-cleanup.json', [str(stage)])
        with mock.patch.object(self.install, 'integration_fingerprint', side_effect=PermissionError('transient')):
            self.install.recover()
        self.assertTrue(stage.exists())
        self.install.recover()
        self.assertEqual(path.read_text(), 'user config')
        self.assertFalse(queue.exists())
        self.assertFalse(stage.exists())
        self.assertTrue(list(self.install.state.glob('restoration-review-*.json')))

    def test_bootstrap_http_failures_are_actionable_without_traceback(self):
        source = (ROOT / 'scripts/install.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        for code in (403, 404):
            with mock.patch.object(sys, 'argv', ['bootstrap', '', '/python', '0', '']), \
                    mock.patch.object(urllib.request, 'urlopen', side_effect=urllib.error.HTTPError('https://example', code, 'failed', {}, None)):
                with self.assertRaises(SystemExit) as error:
                    exec(compile(source, 'bootstrap', 'exec'), {})
            self.assertIn(f'HTTP {code}', str(error.exception))
            self.assertIn('Release download failed', str(error.exception))

    def test_bootstrap_repair_with_corrupt_current_selects_latest_with_warning(self):
        current = self.root / 'xdg-data/hyprwhspr/current.json'
        current.parent.mkdir(parents=True)
        current.write_text('{truncated')
        source = (ROOT / 'scripts/install.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with mock.patch.object(sys, 'argv', ['bootstrap', '', '/python', '1', '']), \
                mock.patch.object(urllib.request, 'urlopen', return_value=io.BytesIO(b'[]')) as request, \
                mock.patch('sys.stderr', new_callable=io.StringIO) as output:
            with self.assertRaisesRegex(SystemExit, 'No compatible'):
                exec(compile(source, 'bootstrap', 'exec'), {})
        self.assertIn(str(current), output.getvalue())
        self.assertIn('latest compatible release', output.getvalue())
        self.assertIn('?per_page=100', request.call_args.args[0].full_url)
