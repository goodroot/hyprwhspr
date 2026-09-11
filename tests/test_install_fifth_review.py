"""Deferred garbage must not prevent independent lifecycle work."""
from pathlib import Path
import json
import os
from unittest import mock

from tests.test_managed_install import ManagedFixture
import managed_install as managed
import managed_integrations as integrations


class FifthReviewTests(ManagedFixture):
    def seed(self, garbage):
        managed.atomic_json(self.install.journal, {'phase': 'committed', 'garbage': [str(garbage)]})

    def test_rejected_symlink_is_retained_without_replaying_transaction(self):
        external = self.root / 'external'
        external.mkdir()
        garbage = self.install.data / 'environments/backend-deadbeef'
        garbage.parent.mkdir(parents=True)
        garbage.symlink_to(external, target_is_directory=True)
        self.seed(garbage)
        for _ in range(3):
            self.install.recover()
            self.assertFalse(self.install.journal.exists())
            self.assertEqual(managed.read_json(self.install.state / 'deferred-cleanup.json'), [str(garbage)])
        self.assertTrue(garbage.is_symlink())
        self.assertTrue(external.exists())

    def test_permission_failure_is_queued_on_first_attempt_and_retried(self):
        garbage = self.install.data / 'environments/old'
        garbage.mkdir(parents=True)
        self.seed(garbage)
        with mock.patch.object(managed.shutil, 'rmtree', side_effect=PermissionError('root-owned file')):
            with self.assertRaises(managed.CleanupError):
                self.install.recover(strict=True)
            self.install.recover()
        self.assertFalse(self.install.journal.exists())
        self.assertIn(str(garbage), managed.read_json(self.install.state / 'deferred-cleanup.json'))
        with mock.patch.object(self.install, 'daemon_running', return_value=False):
            self.install.recover()
        self.assertFalse(garbage.exists())
        self.assertEqual(managed.read_json(self.install.state / 'deferred-cleanup.json'), [])

    def test_queue_write_failure_keeps_transaction_for_recovery(self):
        garbage = self.install.data / 'environments/old'
        garbage.mkdir(parents=True)
        self.seed(garbage)
        with mock.patch.object(managed.shutil, 'rmtree', side_effect=PermissionError('denied')), mock.patch.object(managed, 'atomic_json', side_effect=OSError('state disk full')):
            with self.assertRaises(OSError):
                self.install.recover()
        self.assertTrue(self.install.journal.exists())

    def test_legacy_cleanup_authorization_survives_journal_retirement(self):
        legacy = self.install.data / 'src'
        legacy.mkdir(parents=True)
        managed.atomic_json(self.install.journal, {'phase': 'committed', 'garbage': [str(legacy)], 'legacy_owned': [str(legacy)]})
        with mock.patch.object(managed.shutil, 'rmtree', side_effect=PermissionError('denied')):
            self.install.recover()
        self.assertFalse(self.install.journal.exists())
        with mock.patch.object(self.install, 'daemon_running', return_value=False):
            self.install.recover()
        self.assertFalse(legacy.exists())

    def test_config_and_uninstall_run_despite_rejected_old_garbage(self):
        garbage = self.root / 'foreign-directory'
        garbage.mkdir()
        self.seed(garbage)
        root = self.install.data / 'releases/current'
        cli = self.install.data / 'environments/current-cli'
        root.mkdir(parents=True)
        cli.mkdir(parents=True)
        managed.atomic_json(self.install.current, {'root': str(root), 'cli': {'path': str(cli)}})
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(managed.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run:
            self.assertEqual(managed.launch(['config', 'show']), 0)
            self.assertEqual(managed.launch(['config', 'show']), 0)
            self.assertEqual(run.call_count, 2)
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'check_integrations'), mock.patch.object(self.install, 'service_state', return_value={}), mock.patch.object(self.install, 'daemon_running', return_value=False):
            self.assertEqual(managed.uninstall(['--yes']), 0)
        self.assertFalse(root.exists())
        self.assertFalse(cli.exists())
        self.assertFalse(self.install.current.exists())
        self.assertFalse(self.install.journal.exists())
        self.assertTrue(garbage.exists())

    def test_record_file_receipt_can_transition_to_reversible_editor_receipt(self):
        path = self.root / 'integration.json'
        path.write_text('{"before":true}')
        with mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(integrations, 'Installation', return_value=self.install):
            managed.record_file(path)
            with integrations.edit_files([path], [path]):
                path.write_text('{"after":true}')
            entry = managed.read_json(self.install.receipt)['files'][str(path)]
            self.assertEqual(entry['restore_sha256'], managed.digest(path))
            self.assertTrue(integrations.remove_entry(path, entry))
        self.assertEqual(path.read_text(), '{"before":true}')
