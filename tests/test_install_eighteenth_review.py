"""Ownership recovery must preserve uncertain files and keep lifecycle usable."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest import mock

from tests import test_managed_install as fixtures
import managed_install as managed
import managed_integrations as integrations


class EighteenthReviewTests(fixtures.ManagedFixture):
    def seed_core(self):
        root = self.install.data / 'releases/active'
        files = {'bin/hyprwhspr': 'launcher', 'lib/cli.py': '', 'lib/main.py': '',
                 'lib/src/managed_install.py': '', 'requirements-cli.txt': '', 'requirements.txt': '',
                 'share/config.schema.json': '{}', 'scripts/managed-launcher.sh': '#!/bin/sh\n',
                 'config/systemd/hyprwhspr.service': '[Service]\nExecStart=/usr/lib/hyprwhspr/bin/hyprwhspr\n'}
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        managed.atomic_json(root / 'release.json', {'format': 1, 'version': 'v1.0.0',
            'files': {name: managed.digest(root / name) for name in files}})
        env = self.install.data / 'environments/cli'
        env.mkdir(parents=True)
        generation = {'root': str(root), 'version': 'v1.0.0', 'python': {'path': '/recorded/python'},
                      'cli': {'path': str(env)}, 'backend': None, 'selection': None}
        managed.atomic_json(self.install.current, generation)
        launcher = self.install.data / 'launcher'
        launcher.write_text(files['scripts/managed-launcher.sh'])
        (self.install.data / 'interpreter').write_text('/recorded/python\n')
        command = self.install.command_path()
        command.parent.mkdir(parents=True)
        command.symlink_to(launcher)
        unit = self.root / 'config/systemd/user/hyprwhspr.service'
        unit.parent.mkdir(parents=True)
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            unit.write_text(managed.service_content(root, launcher))
        self.install.state.mkdir(parents=True, exist_ok=True)
        self.install.receipt.write_text('{truncated ownership')
        return root, generation, launcher, unit

    def test_corrupt_receipt_reconstructs_verified_core_but_not_personal_ownership(self):
        root, generation, launcher, unit = self.seed_core()
        personal = self.root / 'personal'
        personal.write_text('settings')
        with mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}):
            self.install.check_integrations()
            receipt = self.install.read_receipt()
        self.assertIn(str(unit), receipt['files'])
        self.assertIn(str(launcher), receipt['files'])
        self.assertIn(str(self.install.command_path()), receipt['files'])
        self.assertNotIn(str(personal), receipt['files'])
        self.assertEqual(receipt.get('permissions', []), [])
        archives = list(self.install.state.glob('ownership.json.unreadable-*'))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), '{truncated ownership')

    def test_custom_unit_and_tampered_payload_are_not_adopted(self):
        root, _, launcher, unit = self.seed_core()
        unit.write_text('custom unit')
        with mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}):
            with self.assertRaisesRegex(RuntimeError, 'Customized or unowned'):
                self.install.check_integrations()
            self.assertNotIn(str(unit), self.install.read_receipt()['files'])
        (root / 'scripts/managed-launcher.sh').write_text('tampered')
        self.install.receipt.write_text('{broken again')
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            self.assertNotIn(str(launcher), self.install.read_receipt().get('files', {}))
        self.assertEqual(unit.read_text(), 'custom unit')

    def test_update_repair_and_uninstall_survive_corrupt_receipt(self):
        root, generation, launcher, unit = self.seed_core()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(managed, 'Installation', return_value=self.install))
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=generation['python']))
            stack.enter_context(mock.patch.object(managed, 'run'))
            stack.enter_context(mock.patch.object(self.install, 'service'))
            stack.enter_context(mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}))
            for name, result in [('build', {**generation['cli'], 'key': 'repaired'}),
                    ('build_backend', (None, None)), ('validate_config', None), ('daemon_running', False),
                    ('binding_migrations', {}), ('bar_migrations', {}), ('clean_legacy', None), ('legacy_selection', None)]:
                stack.enter_context(mock.patch.object(self.install, name, return_value=result))
            updated = self.install.update(local_payload=True, repair=True)
            self.assertEqual(managed.read_json(self.install.current), updated)
            self.install.receipt.write_text('{broken before uninstall')
            self.assertEqual(managed.uninstall(['--yes']), 0)
        self.assertFalse(launcher.exists())
        self.assertFalse(unit.exists())
        self.assertFalse(root.exists())
        self.assertFalse(self.install.current.exists())

    def test_receipt_shapes_and_editor_use_the_same_recovery_reader(self):
        for value in ('{broken', '[]', '{"files":null}', '{"permissions":[null]}'):
            with self.subTest(value=value):
                self.install.state.mkdir(parents=True, exist_ok=True)
                self.install.receipt.write_text(value)
                path = self.root / 'shared.conf'
                path.write_text('user\n')
                with mock.patch.object(integrations, 'Installation', return_value=self.install):
                    with integrations.edit_files([path], [path]) as proceed:
                        self.assertTrue(proceed)
                        path.write_text('user\nowned\n')
                self.assertIn(str(path), self.install.read_receipt()['files'])
                self.assertTrue(list(self.install.state.glob('ownership.json.unreadable-*')))

    def test_status_lists_snapshot_evidence_without_reading_contents_or_mutating(self):
        review = self.install.state / 'restoration-review-example.json'
        archived = self.install.state / 'ownership.json.unreadable-example'
        review.parent.mkdir(parents=True)
        review.write_text('private snapshot')
        archived.write_text('private receipt')
        self.install.receipt.write_text('{broken')
        before = {p: p.read_bytes() for p in self.install.state.iterdir()}
        result = self.install.status()
        self.assertEqual(result['restoration_reviews'], [str(review)])
        self.assertEqual(result['preserved_ownership_receipts'], [str(archived)])
        self.assertNotIn('private snapshot', json.dumps(result))
        self.assertIn(str(self.install.receipt), result['ownership']['error'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.install.state.iterdir()})
