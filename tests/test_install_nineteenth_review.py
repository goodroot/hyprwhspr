"""Nineteenth review: ownership of created files, unit refresh, redirect and state reads."""
import json
import os
from pathlib import Path
import io
import shlex
import sys
import urllib.error
import urllib.request
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_install_second_review import ROOT
import managed_install as managed
import managed_integrations as integrations


class ReviewHelpers:
    def waybar_config(self):
        config = Path(os.environ['XDG_CONFIG_HOME']) / 'waybar'
        config.mkdir(parents=True, exist_ok=True)
        return config / 'config.jsonc'

    def record(self, path, text):
        """Drive a real edit_files pass so the receipt is built the way setup builds it."""
        with integrations.edit_files([path], [path]) as proceed:
            self.assertTrue(proceed)
            managed.atomic_text(path, text, 0o644)

    def entry(self, path):
        # edit_files builds its own environment-rooted Installation.
        return managed.Installation().read_receipt()['files'][str(path)]

    def unit(self, content):
        unit = Path(os.environ['XDG_CONFIG_HOME']) / 'systemd/user/hyprwhspr.service'
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(content, encoding='utf-8')
        with self.install.receipts() as receipt:
            receipt.setdefault('files', {})[str(unit)] = {'sha256': managed.digest(unit), 'kind': 'integration'}
        return unit


class NineteenthReviewTests(ReviewHelpers, ManagedFixture):
    def test_user_edited_created_file_is_preserved_on_uninstall(self):
        target = self.waybar_config()
        self.record(target, '{"created": true}\n')
        # The user edits the file hyprwhspr created, then a later setup pass
        # reformats it without producing insert-only opcodes.
        target.write_text('{"created": true, "custom/foo": 1}\n', encoding='utf-8')
        self.record(target, '{"created": true, "custom/foo": 1, "reformatted": true}\n')
        entry = self.entry(target)
        self.assertIsNone(entry['original'])
        self.assertNotEqual(entry['sha256'], entry['restore_sha256'])
        self.assertFalse(integrations.remove_entry(target, entry))
        self.assertTrue(target.exists())
        self.assertIn('custom/foo', target.read_text(encoding='utf-8'))

    def test_multiple_rewrites_never_reclaim_intervening_user_edits(self):
        for existed in (False, True):
            with self.subTest(preexisting=existed):
                target = self.waybar_config().with_name(f'shared-{existed}.json')
                if existed:
                    target.write_text('{"baseline": true}\n', encoding='utf-8')
                self.record(target, '{"installed": true}\n')
                target.write_text('{"installed": true, "custom/foo": 1}\n', encoding='utf-8')
                for revision in (2, 3, 4):
                    self.record(target, json.dumps({'installed': True, 'custom/foo': 1,
                                                   'revision': revision}) + '\n')
                    entry = self.entry(target)
                    self.assertNotEqual(entry['sha256'], entry['restore_sha256'])
                    self.assertFalse(integrations.remove_entry(target, entry))
                    self.assertEqual(json.loads(target.read_text(encoding='utf-8'))['custom/foo'], 1)

    def test_managed_tray_receives_pinned_config_root(self):
        generation = {'root': str(ROOT), 'cli': {'path': '/unused'},
                      'xdg': {'config': str(self.root / 'pinned')}}
        with mock.patch.dict(os.environ, {'HYPRWHSPR_RESOLVED_GENERATION': json.dumps(generation),
                                         'XDG_CONFIG_HOME': str(self.root / 'ambient')}), \
             mock.patch.object(managed.os, 'execve', side_effect=SystemExit) as execute:
            with self.assertRaises(SystemExit):
                managed.launch(['--managed-tray', 'status'])
        self.assertEqual(execute.call_args.args[2]['XDG_CONFIG_HOME'], generation['xdg']['config'])

    def test_tray_model_probe_uses_xdg_config_and_home_fallback(self):
        # Execute only the function definition, avoiding the tray's desktop actions.
        import subprocess
        source = (ROOT / 'config/hyprland/hyprwhspr-tray.sh').read_text(encoding='utf-8')
        function = source[source.index('model_exists() {'):source.index('# Microphone detection functions')]
        pinned = self.root / 'pinned config'
        for directory, backend in ((pinned, 'rest-api'), (self.root / '.config', 'pywhispercpp')):
            (directory / 'hyprwhspr').mkdir(parents=True)
            (directory / 'hyprwhspr/config.json').write_text(
                json.dumps({'transcription_backend': backend, 'model': str(self.root / 'missing.bin')}),
                encoding='utf-8')
        for config, expected in ((str(pinned), 0), ('', 1)):
            env = dict(os.environ, XDG_CONFIG_HOME=config, SYSTEM_PYTHON=sys.executable)
            result = subprocess.run(['bash', '-c', function + '\nmodel_exists'], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)

    def test_untouched_created_file_is_still_removed(self):
        target = self.waybar_config()
        self.record(target, '{"created": true}\n')
        self.assertTrue(integrations.remove_entry(target, self.entry(target)))
        self.assertFalse(target.exists())

    def test_reinstall_without_user_edits_still_removes_created_file(self):
        target = self.waybar_config()
        self.record(target, '{"created": true}\n')
        self.record(target, '{"created": true, "second": "pass"}\n')
        entry = self.entry(target)
        self.assertEqual(entry['sha256'], entry['restore_sha256'])
        self.assertTrue(integrations.remove_entry(target, entry))
        self.assertFalse(target.exists())

    def test_legacy_entry_without_restore_digest_is_still_removable(self):
        target = self.waybar_config()
        target.write_text('{"legacy": true}\n', encoding='utf-8')
        entry = {'sha256': managed.digest(target), 'kind': 'integration'}
        self.assertTrue(integrations.remove_entry(target, entry))
        self.assertFalse(target.exists())

    def test_owned_unit_is_refreshed_when_the_release_template_changes(self):
        unit = self.unit('[Service]\nExecStart="/stale/launcher"\n')
        with mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}), \
             mock.patch.object(self.install, 'command_path', return_value=self.root / 'absent'), \
             mock.patch.object(managed, 'service_content', return_value='[Service]\nExecStart="/new/launcher"\n'):
            self.assertEqual(self.install.check_integrations(ROOT), str(unit))

    def test_owned_unit_is_left_alone_when_it_already_matches(self):
        content = '[Service]\nExecStart="/current/launcher"\n'
        unit = self.unit(content)
        with mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}), \
             mock.patch.object(self.install, 'command_path', return_value=self.root / 'absent'), \
             mock.patch.object(managed, 'service_content', return_value=content):
            self.assertIsNone(self.install.check_integrations(ROOT))

    def test_unreadable_owned_unit_does_not_abort_the_update(self):
        unit = self.unit('[Service]\n')
        with mock.patch.object(self.install, 'service_state', return_value={'FragmentPath': str(unit)}), \
             mock.patch.object(self.install, 'command_path', return_value=self.root / 'absent'), \
             mock.patch.object(managed, 'service_content', side_effect=OSError('template gone')):
            self.assertIsNone(self.install.check_integrations(ROOT))

    def test_download_refuses_a_plaintext_redirect(self):
        handler = managed.HTTPSRedirectHandler()
        request = urllib.request.Request('https://example.invalid/a')
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            handler.redirect_request(request, None, 302, 'Found', {}, 'http://example.invalid/b')

    def test_download_allows_an_https_redirect(self):
        handler = managed.HTTPSRedirectHandler()
        request = urllib.request.Request('https://example.invalid/a')
        self.assertIsNotNone(
            handler.redirect_request(request, None, 302, 'Found', {}, 'https://example.invalid/b'))

    def test_corrupt_bar_process_marker_does_not_fail_cleanup(self):
        marker = self.install.state / 'legacy-bar-processes.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{truncated', encoding='utf-8')
        legacy = self.install.data / 'src'
        legacy.mkdir(parents=True)
        with mock.patch.object(self.install, 'legacy_bar_references', return_value=[]), \
             mock.patch.object(self.install, 'legacy_binding_references', return_value=[]), \
             mock.patch.object(managed, 'run', side_effect=OSError('no git')):
            self.install.clean_legacy()
        self.assertTrue(legacy.exists())

    def test_managed_widget_keeps_the_packaged_root_fallbacks(self):
        template = (ROOT / 'config/noctalia/plugin/widget.luau').read_text(encoding='utf-8')
        patched = integrations.managed_widget_content(template, self.root / 'shim.sh')
        self.assertIn(integrations.MANAGED_WIDGET_MARKER, patched)
        self.assertIn(str(self.root / 'shim.sh'), patched)
        # TRAY_REL and the root candidates must survive so a missing shim still resolves.
        self.assertIn('local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"', patched)
        self.assertIn('table.insert(candidates, "/usr/lib/hyprwhspr")', patched)
        self.assertNotIn('local TRAY_REL = ""', patched)

    def test_patched_widget_is_not_rescanned_as_a_legacy_reference(self):
        template = (ROOT / 'config/noctalia/plugin/widget.luau').read_text(encoding='utf-8')
        patched = integrations.managed_widget_content(template, self.root / 'shim.sh')
        self.assertTrue(self.install.unmanaged_widget(template))
        self.assertFalse(self.install.unmanaged_widget(patched))

    def test_migration_without_a_backend_offers_setup(self):
        install = managed.Installation()  # main() resolves its own paths from the environment
        managed.atomic_json(install.current, {'version': 'v1.0.0', 'root': str(self.root)})
        (install.data / 'src').mkdir(parents=True)
        calls = []
        with mock.patch.object(managed.Installation, 'update',
                               return_value={'version': 'v1.0.0', 'root': str(self.root)}), \
             mock.patch.object(managed.subprocess, 'call', side_effect=lambda *a, **k: calls.append(a) or 0), \
             mock.patch.object(managed, 'open', mock.mock_open(), create=True):
            self.assertEqual(managed.main(['update']), 0)
        self.assertTrue(calls, 'setup was never offered for a backend-less generation')
        self.assertIn('setup', calls[0][0])


class NineteenthReviewFollowupTests(ReviewHelpers, ManagedFixture):
    """Second pass: defects found in the nineteenth-review fixes themselves."""

    def test_created_file_the_user_deleted_is_still_removable(self):
        target = self.waybar_config()
        self.record(target, '{"created": true}\n')
        target.unlink()
        # A later release regenerates it with different content; no user edit ever
        # intervened, so the file stays ours to remove.
        self.record(target, '{"created": true, "regenerated": true}\n')
        entry = self.entry(target)
        self.assertEqual(entry['sha256'], entry['restore_sha256'])
        self.assertTrue(integrations.remove_entry(target, entry))
        self.assertFalse(target.exists())

    def test_deleted_shared_file_does_not_become_ours_to_remove(self):
        target = self.waybar_config()
        target.write_text('{"user": true}\n', encoding='utf-8')
        self.record(target, '{"user": true, "ours": 1}\n')
        target.unlink()
        self.record(target, '{"ours": 2}\n')
        entry = self.entry(target)
        self.assertNotEqual(entry['sha256'], entry['restore_sha256'])
        self.assertFalse(integrations.remove_entry(target, entry))

    def test_corrupt_bar_marker_does_not_protect_every_generation(self):
        marker = self.install.state / 'legacy-bar-processes.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{truncated', encoding='utf-8')
        (self.install.data / 'releases/old').mkdir(parents=True)
        legacy = self.install.data / 'src'
        legacy.mkdir(parents=True)
        with mock.patch.object(self.install, 'legacy_bar_references', return_value=[]), \
             mock.patch.object(self.install, 'legacy_binding_references', return_value=[]), \
             mock.patch.object(managed, 'run', side_effect=OSError('no git')):
            self.install.clean_legacy()
        self.assertFalse((self.install.state / 'unverified-generations.json').exists())

    def test_registry_widget_does_not_pin_the_legacy_checkout(self):
        widget = self.install.legacy_bar_paths()[2]
        widget.parent.mkdir(parents=True)
        # An upstream noctwhspr copy: mentions the tray path, never the checkout.
        widget.write_text((ROOT / 'config/noctalia/plugin/widget.luau').read_text(encoding='utf-8'),
                          encoding='utf-8')
        self.assertEqual(self.install.legacy_bar_references(), [])

    def test_widget_referencing_the_checkout_still_pins_it(self):
        widget = self.install.legacy_bar_paths()[2]
        widget.parent.mkdir(parents=True)
        widget.write_text(f'local tray = "{self.install.data / "src"}/config/hyprland/hyprwhspr-tray.sh"\n',
                          encoding='utf-8')
        self.assertEqual(self.install.legacy_bar_references(), [widget])

    def test_historical_widget_branch_marks_itself_managed(self):
        historical = ('local root = noctalia.getenv("HYPRWHSPR_ROOT")\n'
                      'local tray = root .. "/config/hyprland/hyprwhspr-tray.sh"\n'
                      '-- see /config/hyprland/hyprwhspr-tray.sh for details\n')
        patched = integrations.managed_widget_content(historical, self.root / 'shim.sh')
        self.assertIn(integrations.MANAGED_WIDGET_MARKER, patched)
        # A stray mention of the path must not keep it looking unmigrated.
        self.assertFalse(self.install.unmanaged_widget(patched))

    def test_declined_setup_after_migration_does_not_fail_the_update(self):
        install = managed.Installation()
        managed.atomic_json(install.current, {'version': 'v1.0.0', 'root': str(self.root)})
        (install.data / 'src').mkdir(parents=True)
        with mock.patch.object(managed.Installation, 'update',
                               return_value={'version': 'v1.0.0', 'root': str(self.root)}), \
             mock.patch.object(managed.subprocess, 'call', return_value=130), \
             mock.patch.object(managed, 'open', mock.mock_open(), create=True):
            self.assertEqual(managed.main(['update']), 0)

    def test_bootstrap_refuses_a_plaintext_redirect(self):
        source = (ROOT / 'scripts/install.sh').read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        namespace = {}
        with mock.patch.object(sys, 'argv', ['bootstrap', '', '/python', '0', '']), \
                mock.patch.object(urllib.request, 'urlopen', return_value=io.BytesIO(b'[]')):
            with self.assertRaises(SystemExit):
                exec(compile(source, 'bootstrap', 'exec'), namespace)
        handler = namespace['HTTPSOnly']()
        request = urllib.request.Request('https://example.invalid/a')
        with self.assertRaises(SystemExit) as error:
            handler.redirect_request(request, None, 302, 'Found', {}, 'http://example.invalid/b')
        self.assertIn('HTTPS', str(error.exception))


class TwentiethReviewTests(ReviewHelpers, ManagedFixture):
    """Defects found in the nineteenth-review follow-up fixes."""

    def test_declined_setup_on_a_fresh_install_does_not_fail_the_install(self):
        install = managed.Installation()
        with mock.patch.object(managed.Installation, 'update',
                               return_value={'version': 'v1.0.0', 'root': str(self.root),
                                             'backend': {'path': str(self.root)}}), \
             mock.patch.object(managed.subprocess, 'call', return_value=130), \
             mock.patch.object(managed, 'open', mock.mock_open(), create=True):
            self.assertEqual(managed.main(['bootstrap']), 0)

    def test_unmanaged_setup_still_records_permissions_for_uninstall(self):
        # cli/uninstall.py reads this same receipt to remove permissions, so an
        # unmanaged install must record them or --remove-permissions silently no-ops.
        from cli import setup
        rules = self.root / 'rules'
        rules.mkdir()
        (rules / '99-uinput.rules').write_text(setup.UINPUT_RULE_CONTENT, encoding='utf-8')
        with mock.patch.object(managed, 'Installation', return_value=self.install):
            os.environ.pop('HYPRWHSPR_GENERATION', None)
            self.assertEqual(setup._select_uinput_rule(rules), rules / '99-uinput.rules')
        permissions = managed.read_json(self.install.receipt)['permissions']
        self.assertEqual(len(permissions), 1)
        self.assertTrue(permissions[0]['adopted_legacy'])
        self.assertFalse(permissions[0]['added'])

    def test_foreign_binding_is_reported_even_beside_an_owned_one(self):
        from cli import install as cli_install
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True, exist_ok=True)
        owned = shlex.quote(str(self.install.data / 'launcher')) + ' --managed-tray'
        # An owned binding and a foreign one side by side: the foreign one still fires.
        path.write_text(f'bindd = SUPER ALT, D, Dictate, exec, {owned} record\n'
                        'bind = SUPER, F, exec, /elsewhere/config/hyprland/hyprwhspr-tray.sh record\n',
                        encoding='utf-8')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.object(managed, 'Installation', return_value=self.install), \
                mock.patch.object(cli_install, 'HYPRWHSPR_ROOT', '/current-release'), \
                mock.patch.object(cli_install, 'log_warning') as warning:
            cli_install._edit_hyprland_bindings()
        self.assertTrue(any('targets another installation' in call.args[0]
                            for call in warning.call_args_list),
                        f'foreign binding was not reported: {warning.call_args_list}')


class TwentyFirstReviewTests(ReviewHelpers, ManagedFixture):
    """Defects found in the twentieth-round fixes."""

    def test_systemd_disable_works_with_a_customized_unit(self):
        from cli import systemd as cli_systemd
        dest = Path(os.environ['XDG_CONFIG_HOME']) / 'systemd/user/hyprwhspr.service'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('[Service]\n# hand edited\n', encoding='utf-8')
        # setup_systemd probes HYPRWHSPR_ROOT for bin/hyprwhspr and the unit
        # template before any mode branching, so it must point at a fake tree or
        # the test silently depends on a real install at /usr/lib/hyprwhspr.
        fake_root = self.root / 'release'
        (fake_root / 'bin').mkdir(parents=True)
        (fake_root / 'bin/hyprwhspr').write_text('#!/bin/sh\n', encoding='utf-8')
        (fake_root / 'bin/hyprwhspr').chmod(0o755)
        (fake_root / 'config/systemd').mkdir(parents=True)
        (fake_root / 'config/systemd/hyprwhspr.service').write_text('[Service]\n', encoding='utf-8')
        calls = []
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
             mock.patch.object(cli_systemd, 'HYPRWHSPR_ROOT', str(fake_root)), \
             mock.patch.object(cli_systemd, 'USER_SYSTEMD_DIR', dest.parent), \
             mock.patch.object(cli_systemd, '_validate_hyprwhspr_root', return_value=True), \
             mock.patch.object(cli_systemd, 'run_command', side_effect=lambda *a, **k: calls.append(a[0]) or mock.Mock(returncode=0)), \
             mock.patch.object(managed, 'Installation', return_value=self.install):
            cli_systemd.setup_systemd('disable')
        # The unmodifiable unit must not stop disable from reaching systemctl.
        self.assertTrue(any('disable' in call for call in calls), calls)
        self.assertIn('hand edited', dest.read_text(encoding='utf-8'))

    def test_reconstruction_does_not_adopt_a_repointed_unit(self):
        root = self.install.data / 'releases/active'
        (root / 'scripts').mkdir(parents=True)
        (root / 'scripts/managed-launcher.sh').write_text('launcher\n', encoding='utf-8')
        managed.atomic_json(self.install.current, {'root': str(root), 'version': 'v1.0.0',
                                                   'python': {'path': '/usr/bin/python3'}})
        launcher = self.install.data / 'launcher'
        launcher.write_text('launcher\n', encoding='utf-8')
        (self.install.data / 'interpreter').write_text('/usr/bin/python3\n', encoding='utf-8')
        unit = Path(os.environ['XDG_CONFIG_HOME']) / 'systemd/user/hyprwhspr.service'
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text('[Service]\nEnvironment="XDG_CONFIG_HOME=/user-repointed"\nExecStart="/l"\n',
                        encoding='utf-8')
        generated = '[Service]\nEnvironment="XDG_CONFIG_HOME=/generated"\nExecStart="/l"\n'
        receipt = {}
        with mock.patch.object(managed, 'verify_payload', return_value={'version': 'v1.0.0'}), \
                mock.patch.object(managed, 'service_content', return_value=generated):
            self.install.recover_core_receipts(receipt)
        # The launcher matches and is adopted; the re-pointed unit must not be.
        self.assertIn(str(launcher), receipt.get('files', {}))
        self.assertNotIn(str(unit), receipt.get('files', {}))

    def test_config_save_survives_a_missing_managed_module(self):
        import types
        import config_manager
        config = Path(os.environ['XDG_CONFIG_HOME']) / 'hyprwhspr/config.json'
        config.parent.mkdir(parents=True, exist_ok=True)
        manager = config_manager.ConfigManager()
        manager.config_dir, manager.config_file = config.parent, config
        stub = types.ModuleType('managed_install')  # no record_file attribute
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), \
                mock.patch.dict(sys.modules, {'managed_install': stub}):
            self.assertTrue(manager.save_config())
        self.assertTrue(config.exists())


class SettledDesignTests(ReviewHelpers, ManagedFixture):
    """The three design questions, pinned so they stop being re-litigated."""

    # --- Q1: the ownership receipt is shared, not managed-only -----------------

    def test_unmanaged_config_save_records_ownership(self):
        import config_manager
        config = Path(os.environ['XDG_CONFIG_HOME']) / 'hyprwhspr/config.json'
        config.parent.mkdir(parents=True, exist_ok=True)
        manager = config_manager.ConfigManager()
        manager.config_dir, manager.config_file = config.parent, config
        os.environ.pop('HYPRWHSPR_GENERATION', None)
        self.assertTrue(manager.save_config())
        files = managed.Installation().read_receipt().get('files', {})
        self.assertEqual(files.get(str(config), {}).get('kind'), 'personal')

    def test_unmanaged_model_download_records_ownership_so_purge_works(self):
        # cli/uninstall.py --purge selects models purely by receipt kind.
        model = self.root / 'ggml-base.bin'
        model.write_bytes(b'\x00model')
        os.environ.pop('HYPRWHSPR_GENERATION', None)
        self.assertTrue(managed.record_ownership('file', model, 'model'))
        files = managed.Installation().read_receipt().get('files', {})
        self.assertEqual(files[str(model)]['kind'], 'model')

    def test_corrupt_receipt_is_recovered_rather_than_failing_the_record(self):
        install = managed.Installation()
        install.receipt.parent.mkdir(parents=True, exist_ok=True)
        install.receipt.write_text('{truncated', encoding='utf-8')
        target = self.root / 'thing'
        target.write_text('x', encoding='utf-8')
        self.assertTrue(managed.record_ownership('file', target, 'personal'))
        self.assertIn(str(target), install.read_receipt()['files'])
        self.assertTrue(list(install.state.glob('ownership.json.unreadable-*')))

    def test_record_ownership_reports_failure_without_raising(self):
        target = self.root / 'thing'
        target.write_text('x', encoding='utf-8')
        with mock.patch.object(managed, 'record_file', side_effect=OSError('state is read-only')):
            self.assertFalse(managed.record_ownership('file', target, 'personal'))
        self.assertIsInstance(managed.record_ownership.last_error, OSError)

    # --- Q2: the generation is the authority for XDG paths --------------------

    def test_xdg_roots_are_pinned_at_first_install(self):
        install = managed.Installation()
        pinned = install.xdg_roots()
        self.assertEqual(pinned['config'], os.environ['XDG_CONFIG_HOME'])
        # A later run from a different session must not repoint anything.
        managed.atomic_json(install.current, {'version': 'v1.0.0', 'xdg': pinned})
        os.environ['XDG_CONFIG_HOME'] = str(self.root / 'a-tty-session')
        self.assertEqual(str(install.config_home()), pinned['config'])
        self.assertEqual(install.xdg_roots({'xdg': pinned})['config'], pinned['config'])

    def test_unit_is_rendered_from_the_recorded_roots_not_the_environment(self):
        install = managed.Installation()
        pinned = install.xdg_roots()
        managed.atomic_json(install.current, {'version': 'v1.0.0', 'xdg': pinned})
        root = self.root / 'release'
        (root / 'config/systemd').mkdir(parents=True)
        (root / 'config/systemd/hyprwhspr.service').write_text(
            '[Unit]\n[Service]\nExecStart=/usr/lib/hyprwhspr/bin/hyprwhspr\n', encoding='utf-8')
        os.environ['XDG_CONFIG_HOME'] = str(self.root / 'a-tty-session')
        unit = managed.service_content(root, install.data / 'launcher')
        self.assertIn(f'Environment="XDG_CONFIG_HOME={pinned["config"]}"', unit)
        self.assertNotIn('a-tty-session', unit)

    def test_config_home_falls_back_to_the_environment_before_first_install(self):
        install = managed.Installation()
        self.assertFalse(install.current.exists())
        self.assertEqual(str(install.config_home()), os.environ['XDG_CONFIG_HOME'])

    def test_every_installer_config_reader_follows_the_recorded_root(self):
        install = managed.Installation()
        pinned = install.xdg_roots()
        managed.atomic_json(install.current, {'version': 'v1.0.0', 'xdg': pinned})
        # Seed a binding and a bar file under the pinned root, then move the
        # ambient root away. Every reader must still find them.
        hypr = Path(pinned['config']) / 'hypr'
        hypr.mkdir(parents=True, exist_ok=True)
        (hypr / 'hyprland.conf').write_text(
            f"bind = SUPER, D, exec, {install.data}/src/config/hyprland/hyprwhspr-tray.sh record\n",
            encoding='utf-8')
        decoy = self.root / 'a-tty-session/hypr'
        decoy.mkdir(parents=True)
        (decoy / 'hyprland.conf').write_text('# never read\n', encoding='utf-8')
        os.environ['XDG_CONFIG_HOME'] = str(self.root / 'a-tty-session')

        self.assertIn(pinned['config'], str(install.legacy_bar_paths()[0]))
        found = {str(path) for path in install.hyprland_files()}
        self.assertIn(str(hypr / 'hyprland.conf'), found)
        self.assertNotIn(str(decoy / 'hyprland.conf'), found)
        self.assertIn(hypr / 'hyprland.conf', install.binding_migrations())

    def test_launch_exports_the_pinned_config_root_to_children(self):
        install = managed.Installation()
        pinned = str(self.root / 'pinned-config')
        root = self.root / 'release'
        (root / 'lib').mkdir(parents=True)
        (root / 'lib/cli.py').write_text('', encoding='utf-8')
        env_path = self.root / 'cli-env'
        (env_path / 'bin').mkdir(parents=True)
        (env_path / 'bin/python').write_text('', encoding='utf-8')
        managed.atomic_json(install.current, {
            'version': 'v1.0.0', 'root': str(root), 'cli': {'path': str(env_path)},
            'backend': {'path': str(env_path)}, 'xdg': {'config': pinned}})
        seen = {}
        with mock.patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root / 'shell-config')}), \
             mock.patch.object(managed.subprocess, 'run',
                               side_effect=lambda cmd, **kw: seen.update(kw.get('env') or {})
                               or mock.Mock(returncode=0)), \
             mock.patch.object(managed.Installation, 'recover', lambda s, **k: None):
            managed.launch(['config'])
        # The child must see the installation's root, not the invoking shell's.
        self.assertEqual(seen.get('XDG_CONFIG_HOME'), pinned)
