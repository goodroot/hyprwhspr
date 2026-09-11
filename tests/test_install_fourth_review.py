"""Fourth review: recovery errors and repeated setup must not destroy user state."""
from contextlib import ExitStack, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

from tests.test_managed_install import ManagedFixture
from tests.test_install_second_review import cli_install, noctalia, ROOT
from tests.test_install_review_regressions import systemd
import managed_install as managed
import managed_integrations as integrations
import backend_installer as backend


class FourthReviewTests(ManagedFixture):
    def test_rollback_command_failures_restore_pointer_and_leave_repair_available(self):
        for failure in ('stop', 'start', 'daemon-reload', 'reload'):
            with self.subTest(failure=failure):
                staged = self.install.data / 'environments/staged'
                staged.mkdir(parents=True, exist_ok=True)
                old = {'root': 'previous'}
                managed.atomic_json(self.install.current, {'root': 'replacement'})
                managed.atomic_json(self.install.journal, {'phase': 'activated', 'old': old,
                    'created': [str(staged)], 'running': True, 'legacy_unit': 'unit', 'reload_hyprland': True})
                def command(args, **kwargs):
                    if failure in args:
                        raise subprocess.CalledProcessError(1, args, stderr='desktop unavailable')
                def service(action):
                    command(['systemctl', action])
                with mock.patch.object(managed, 'run', side_effect=command), mock.patch.object(self.install, 'service', side_effect=service), mock.patch.object(self.install, 'service_state', return_value={'ActiveState': 'active'}), mock.patch.object(self.install, 'daemon_running', return_value=False), redirect_stderr(io.StringIO()) as output:
                    self.install.recover()
                    self.assertFalse(self.install.journal.exists())
                    self.assertEqual(managed.read_json(self.install.current), old)
                    self.assertEqual(staged.exists(), failure == 'stop')
                    self.install.recover()
                self.assertIn('desktop unavailable', output.getvalue())
                self.assertFalse(staged.exists())

    def test_post_commit_cleanup_failure_is_not_retried_or_reported_as_failed_activation(self):
        root = self.install.data / 'releases/current'
        root.mkdir(parents=True)
        old_cli = self.install.data / 'environments/old-cli'
        old_cli.mkdir(parents=True)
        new_cli = self.install.data / 'environments/new-cli'
        new_cli.mkdir()
        old = {'version': 'v1.2.3', 'root': str(root), 'python': {'path': '/python'},
               'cli': {'path': str(old_cli)}, 'backend': None, 'selection': None}
        managed.atomic_json(self.install.current, old)
        actual_recover = self.install.recover
        calls = []
        def recover(**kwargs):
            calls.append(managed.read_json(self.install.journal, {}).get('phase'))
            if calls[-1] == 'committed':
                raise RuntimeError('garbage permission denied')
            return actual_recover(**kwargs)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(managed, 'interpreter', return_value=old['python']))
            stack.enter_context(mock.patch.object(managed, 'verify_payload', return_value={'version': old['version']}))
            stack.enter_context(mock.patch.object(self.install, 'recover', side_effect=recover))
            stack.enter_context(mock.patch.object(self.install, 'build', return_value={'path': str(new_cli)}))
            for name, result in [('legacy_selection', None), ('validate_config', None), ('check_integrations', None), ('service_state', {}), ('daemon_running', False), ('install_launcher', None)]:
                stack.enter_context(mock.patch.object(self.install, name, return_value=result))
            with self.assertRaisesRegex(managed.CommittedCleanupError, 'is active; cleanup incomplete') as caught:
                self.install.update(local_payload=True)
        self.assertEqual(calls, [None, 'committed'])
        self.assertEqual(managed.read_json(self.install.current), caught.exception.generation)
        self.assertIn('garbage permission denied', str(caught.exception.__cause__))
        self.assertTrue(self.install.journal.exists())

    def test_backend_cleanup_failure_reports_live_generation_and_nonzero(self):
        generation = {'version': 'v1.2.3', 'backend': {'path': str(self.root / 'active-backend')}}
        error = managed.CommittedCleanupError(generation, 'permission denied')
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': json.dumps({'version': 'v1.2.3'})}), mock.patch.object(managed, 'Installation', return_value=self.install), mock.patch.object(self.install, 'update', side_effect=error), mock.patch.object(backend, 'VENV_DIR', self.root / 'old'):
            with self.assertRaisesRegex(SystemExit, 'Backend activated successfully'):
                backend._managed_select('cpu', None, False)
            self.assertEqual(backend.VENV_DIR, Path(generation['backend']['path']))
            self.assertEqual(json.loads(os.environ['HYPRWHSPR_GENERATION']), generation)

    def test_repeated_setup_does_not_own_intervening_user_insertions(self):
        path = self.root / 'style.css'
        path.write_text('original\n')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text('app-one\noriginal\n')
            path.write_text(path.read_text() + 'user-added\n')
            with integrations.edit_files([path], [path]):
                path.write_text('app-two\n' + path.read_text())
            with integrations.edit_files([path], [path], 'remove'):
                pass
        self.assertEqual(path.read_text(), 'original\nuser-added\n')

    def test_repeated_json_rewrite_cannot_restore_pre_user_baseline(self):
        path = self.root / 'config.jsonc'
        path.write_text('{"original":true}')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text('{"original":true,"app":1}')
            path.write_text('{"original":true,"app":1,"user":true}')
            with integrations.edit_files([path], [path]):
                path.write_text('{"original":true,"app":2,"user":true}')
            with integrations.edit_files([path], [path], 'remove'):
                pass
        self.assertIn('"user":true', path.read_text())
        self.assertIn(str(path), managed.read_json(self.install.receipt)['files'])

    def test_shared_symlinked_binding_allows_legacy_editor(self):
        path = self.root / 'config/hypr/bindings.conf'
        path.parent.mkdir(parents=True)
        target = self.root / 'dotfiles'
        target.write_text('user config')
        path.symlink_to(target)
        with mock.patch.dict(os.environ, {'HYPRWHSPR_GENERATION': '{}'}), mock.patch.object(integrations, 'Installation', return_value=self.install), mock.patch.object(cli_install, '_edit_hyprland_bindings', return_value=True) as editor:
            self.assertTrue(cli_install._setup_hyprland_bindings())
        editor.assert_called_once()
        self.assertEqual(target.read_text(), 'user config')

    def test_large_import_set_gets_larger_cold_import_budget(self):
        imports = ('sounddevice', 'numpy', 'soxr', 'soundfile', 'pandas', 'scipy', 'numba', 'sklearn', 'transformers', 'torch', 'librosa')
        with mock.patch.object(managed, 'run') as run:
            self.assertTrue(self.install.healthy('/env', imports))
        self.assertGreaterEqual(run.call_args.kwargs['timeout'], 600)

    def test_launcher_prefers_recorded_interpreter_and_validates_it(self):
        base = self.root / 'data/hyprwhspr'
        base.mkdir(parents=True)
        root = self.root / 'release'
        module = root / 'lib/src/managed_install.py'
        module.parent.mkdir(parents=True)
        module.write_text('print("resolved")\n')
        (base / 'current.json').write_text(json.dumps({'root': str(root)}))
        python = self.root / 'recorded-python'
        marker = self.root / 'interpreter-used'
        import shlex
        python.write_text('#!/bin/sh\nprintf used >> ' + shlex.quote(str(marker)) + '\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python.chmod(0o755)
        (base / 'interpreter').write_text(str(python) + '\n')
        result = subprocess.run(['bash', str(ROOT / 'scripts/managed-launcher.sh')], env=dict(os.environ, XDG_DATA_HOME=str(base.parent)), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker.read_text(), 'usedused')
        self.assertIn('resolved', result.stdout)

    def test_luau_widget_handles_unicode_quotes_and_template_drift(self):
        template = (ROOT / 'config/noctalia/plugin/widget.luau').read_text()
        result = integrations.managed_widget_content(template, Path('/home/é/quote"/back\\slash/\x01'))
        self.assertIn('/home/é/', result)
        self.assertNotIn('\\u00', result)
        self.assertIn('quote\\"', result)
        self.assertIn('back\\\\slash', result)
        self.assertIn('\\001', result)
        for marker in ('local candidates = {}', 'local TRAY_REL = "/config/hyprland/hyprwhspr-tray.sh"'):
            with self.assertRaisesRegex(RuntimeError, 'template changed'):
                integrations.managed_widget_content(template.replace(marker, 'changed'), '/tray')

    def test_older_ambiguous_receipt_does_not_authorize_user_block_removal(self):
        path = self.root / 'style.css'
        path.write_text('base\napp\nuser\n')
        entry = {'original': 'base\n', 'sha256': managed.digest(path), 'insertions': ['app\nuser\n']}
        self.assertFalse(integrations.remove_entry(path, entry))
        self.assertIn('user\n', path.read_text())

    def test_first_setup_output_can_still_restore_original_json(self):
        path = self.root / 'config.jsonc'
        path.write_text('{"original":true}')
        with mock.patch.object(integrations, 'Installation', return_value=self.install):
            with integrations.edit_files([path], [path]):
                path.write_text('{"original":true,"app":1}')
            with integrations.edit_files([path], [path], 'remove'):
                pass
        self.assertEqual(path.read_text(), '{"original":true}')
