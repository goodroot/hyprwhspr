"""Convergence after migration and retryable permission removal."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests import test_install_sixth_review as sixth_review
from tests.test_install_second_review import ROOT
import managed_install as managed


class TenthReviewTests(ManagedFixture):
    def test_migrated_binding_with_optional_sources_allows_legacy_cleanup(self):
        legacy = self.install.data / 'src'
        legacy.mkdir(parents=True)
        path = self.root / 'config/hypr/hyprland.conf'
        path.parent.mkdir(parents=True)
        path.write_text(f'source = optional/*.conf\nsource = $configs/optional.conf\n# hyprwhspr setup\nbind = SUPER, D, exec, {shlex.quote(str(self.install.data / "launcher"))} --managed-tray record\n')
        for _ in range(2):
            self.assertEqual(self.install.binding_migrations(), {})
            self.assertEqual(self.install.legacy_binding_references(), [])
        with mock.patch.object(managed, 'run', side_effect=[mock.Mock(stdout='https://github.com/goodroot/hyprwhspr.git'), mock.Mock(stdout=''), mock.Mock(stdout='')]):
            self.install.clean_legacy()
        self.assertFalse(legacy.exists())

    def test_actual_legacy_source_path_remains_a_reference(self):
        path = self.root / 'config/hypr/hyprland.conf'
        path.parent.mkdir(parents=True)
        path.write_text(f'source = {self.install.data}/src/config/hyprland/other.conf\n')
        self.assertEqual(self.install.legacy_binding_references(), [path])

    def test_second_widget_migration_is_silent_noop(self):
        sixth_review.SixthReviewTests.seed_bars(self)
        with mock.patch.object(self.install, 'bar_processes', return_value={}):
            for path, content in self.install.bar_migrations().items():
                managed.atomic_text(path, content)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(self.install.bar_migrations(), {})
        self.assertNotIn('template changed', output.getvalue())

    def test_git_probe_failures_preserve_checkout_without_backend_adoption(self):
        legacy = self.install.data / 'src/.git'
        legacy.mkdir(parents=True)
        for error in (FileNotFoundError('git missing'), subprocess.CalledProcessError(2, ['git'])):
            with mock.patch.object(managed, 'run', side_effect=error), redirect_stdout(io.StringIO()) as output:
                self.assertIsNone(self.install.legacy_selection())
            self.assertIn('continuing without adopting', output.getvalue())
            self.assertTrue(legacy.exists())

    def test_launcher_reads_utf8_json_even_with_ascii_default_encoding(self):
        base = self.root / 'xdg-data/hyprwhspr'
        base.mkdir(parents=True)
        payload = self.root / 'payload/lib/src'
        payload.mkdir(parents=True)
        (payload / 'managed_install.py').write_text('print("reached")\n')
        (base / 'current.json').write_text(json.dumps({'root': str(payload.parents[1]), 'notes': 'é —'}, ensure_ascii=False), encoding='utf-8')
        wrapper = self.root / 'python-ascii'
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -X utf8=0 "$@"\n')
        wrapper.chmod(0o755)
        (base / 'interpreter').write_text(str(wrapper) + '\n')
        result = subprocess.run(['bash', str(ROOT / 'scripts/managed-launcher.sh')], env=dict(os.environ, LC_ALL='C', LANG='C'), capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'reached', result.stdout)

    def permission_fixture(self):
        rule = self.root / 'rule'
        rule.write_text('owned')
        entry = {'kind': 'rule', 'path': str(rule), 'added': True, 'sha256': managed.digest(rule)}
        managed.atomic_json(self.install.receipt, {'permissions': [entry]})
        return rule

    def call_uninstall(self):
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False):
            return managed.uninstall(['--yes', '--remove-permissions'])

    def test_modified_rule_keeps_receipt_until_restored_and_removed(self):
        rule = self.permission_fixture()
        rule.write_text('user modification')
        with mock.patch.object(managed, 'run') as run, redirect_stdout(io.StringIO()) as output:
            self.call_uninstall()
        run.assert_not_called()
        self.assertIn('Preserved modified permission rule', output.getvalue())
        self.assertTrue(managed.read_json(self.install.receipt)['permissions'][0]['added'])
        rule.write_text('owned')
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            if command[1] == 'rm':
                rule.unlink()
        with mock.patch.object(managed, 'run', side_effect=run):
            self.call_uninstall()
        self.assertEqual(calls, [['sudo', 'rm', str(rule)], ['sudo', 'udevadm', 'control', '--reload-rules']])
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])

    def test_failed_rule_reload_retains_receipt_for_retry_without_second_unlink(self):
        rule = self.permission_fixture()
        def run(command, **kwargs):
            if command[1] == 'rm':
                rule.unlink()
            else:
                raise subprocess.CalledProcessError(1, command)
        with mock.patch.object(managed, 'run', side_effect=run):
            with self.assertRaisesRegex(RuntimeError, 'Uninstall incomplete'):
                self.call_uninstall()
        self.assertTrue(managed.read_json(self.install.receipt)['permissions'][0]['added'])
        with mock.patch.object(managed, 'run') as run:
            self.call_uninstall()
        run.assert_called_once_with(['sudo', 'udevadm', 'control', '--reload-rules'])
        self.assertFalse(managed.read_json(self.install.receipt)['permissions'][0]['added'])
