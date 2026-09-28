"""ONNX selection: one readiness check, one switch rule, read-only status."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import ANY, Mock, patch

from tests.test_setup_command_scope import setup, install
from cli import onnx, models
import onnx_model
import orukeet
from onnx_model import resolve_model, load_model, DEFAULT_MODEL, NEEDS_DOWNLOAD


def result(code, stderr=''):
    return types.SimpleNamespace(returncode=code, stdout='', stderr=stderr)


class SelectionTests(unittest.TestCase):
    def test_resolution_keeps_the_users_quantization_for_every_model(self):
        for model in (DEFAULT_MODEL, 'orukeet', 'custom/repo'):
            settings = {'onnx_asr_model': model, 'onnx_asr_quantization': None}
            with patch.object(setup, 'Prompt', types.SimpleNamespace(ask=lambda *a, **k: '')):
                self.assertEqual(setup._prompt_onnx_model_selection(settings), (model, None, True))
        self.assertEqual(resolve_model({}), (DEFAULT_MODEL, 'int8', True))
        with patch.object(setup, 'Prompt', types.SimpleNamespace(ask=lambda *a, **k: '2')):
            self.assertEqual(setup._prompt_onnx_model_selection({'onnx_asr_quantization': None}),
                             ('orukeet', None, True))

    def test_save_writes_only_backend_and_model(self):
        config = Mock()
        config.save_config.return_value = False
        with self.assertRaisesRegex(OSError, 'retry setup'):
            onnx.save_selection(config, ('orukeet', None, True))
        self.assertEqual([c.args[0] for c in config.set_setting.call_args_list],
                         ['transcription_backend', 'onnx_asr_model'])


class LoaderTests(unittest.TestCase):
    def test_vad_is_only_loaded_when_enabled(self):
        for enabled in (False, True):
            runtime = Mock()
            with patch.dict(sys.modules, {'onnx_asr': runtime}):
                direct, vad = load_model(DEFAULT_MODEL, None, enabled)
            runtime.load_model.assert_called_once_with(DEFAULT_MODEL)
            self.assertEqual(runtime.load_vad.call_count, int(enabled))
            self.assertIs(direct, runtime.load_model.return_value)
            if not enabled:
                self.assertIsNone(vad)

    def test_orukeet_always_loads_int8_whatever_the_setting(self):
        runtime = Mock()
        with patch.dict(sys.modules, {'onnx_asr': runtime}), \
             patch.object(orukeet, 'download_model', return_value=Path('/tmp/o')) as download:
            load_model('orukeet', None, False)
        download.assert_called_once_with(offline=False, repair=False)
        runtime.load_model.assert_called_once_with('nemo-conformer-tdt', path=Path('/tmp/o'), quantization='int8')

    def test_prepare_exit_codes_separate_fixable_cache_problems_from_real_errors(self):
        selection = ('orukeet', 'int8', True)
        for error in (FileNotFoundError('miss'), orukeet.ChecksumError('bad')):
            with patch.object(onnx_model, 'load_model', side_effect=error):
                self.assertEqual(onnx_model.prepare(selection, offline=True), NEEDS_DOWNLOAD)
                # Online there is nothing left to download, so the error is real.
                with self.assertRaises(type(error)):
                    onnx_model.prepare(selection, offline=False)
        with patch.object(onnx_model, 'load_model', side_effect=ImportError('libcudnn')):
            with self.assertRaises(ImportError):
                onnx_model.prepare(selection, offline=True)
        with patch.object(onnx_model, 'load_model') as load:
            self.assertEqual(onnx_model.prepare(selection, offline=True), 0)
            onnx_model.prepare(selection, offline=False)
        self.assertEqual([c.kwargs for c in load.call_args_list],
                         [{'offline': True, 'repair': False}, {'offline': False, 'repair': True}])


class PrepareModelTests(unittest.TestCase):
    def _prepare(self, results, venv=None, mise=False):
        """Run prepare_model with run_command returning (or raising) the given results."""
        venv = venv or onnx.backend_installer.VENV_DIR
        with patch.dict(os.environ, {'HF_HOME': '/tmp/test-hf', 'HYPRWHSPR_BACKEND_ENV': ''}, clear=True), \
             patch.object(onnx.backend_installer, 'VENV_DIR', venv), \
             patch.object(onnx.backend_installer, '_check_mise_active', return_value=mise), \
             patch.object(onnx.backend_installer, '_create_mise_free_environment',
                          return_value={'HF_HOME': '/tmp/test-hf', 'PATH': '/mise-free'}), \
             patch.object(onnx, 'run_command', side_effect=results) as run, \
             patch.object(onnx, 'log_error') as error:
            ready = onnx.prepare_model(('orukeet', 'int8', False))
        return ready, run.call_args_list, error

    def test_warm_cache_loads_once_offline_in_backend_env(self):
        # VENV_DIR already resolves HYPRWHSPR_BACKEND_ENV at import; preparation must reuse it.
        for venv in (onnx.backend_installer.VENV_DIR, Path('/tmp/managed-backend')):
            ready, calls, _ = self._prepare([result(0)], venv)
            self.assertTrue(ready)
            (probe,) = calls
            self.assertEqual(probe.args[0][0], str(venv / 'bin' / 'python'))
            # cwd kept off the path (-P) but the service's environment intact; only lib/src added.
            self.assertEqual(probe.args[0][1:3], ['-P', '-c'])
            self.assertIn(repr(str(Path(onnx.__file__).resolve().parents[1])), probe.args[0][3])
            self.assertEqual(probe.args[0][-1], '1')
            self.assertEqual(probe.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertEqual(probe.kwargs['env']['HF_HOME'], '/tmp/test-hf')

    def test_fixable_miss_downloads_once_with_visible_progress(self):
        ready, (probe, download), _ = self._prepare([result(NEEDS_DOWNLOAD), result(0)])
        self.assertTrue(ready)
        self.assertEqual(download.args[0][-1], '0')
        self.assertNotIn('HF_HUB_OFFLINE', download.kwargs['env'])
        self.assertTrue(download.kwargs['verbose'])
        self.assertTrue(download.kwargs['check'])

    def test_real_error_is_shown_and_never_downloads(self):
        ready, calls, error = self._prepare([result(1, 'Traceback...\nImportError: libcudnn.so.9')])
        self.assertFalse(ready)
        self.assertEqual(len(calls), 1)
        self.assertIn('libcudnn.so.9', error.call_args.args[0])

    def test_mise_free_environment_when_mise_is_active(self):
        ready, calls, _ = self._prepare([result(NEEDS_DOWNLOAD), result(0)], mise=True)
        self.assertTrue(ready)
        for call in calls:
            self.assertEqual(call.kwargs['env']['PATH'], '/mise-free')

    def test_download_failure_fails(self):
        for effect in (subprocess.CalledProcessError(1, 'python'), FileNotFoundError('python')):
            ready, _, _ = self._prepare([result(NEEDS_DOWNLOAD), effect])
            self.assertFalse(ready)


class SwitchRuleTests(unittest.TestCase):
    def test_rule(self):
        new, old = ('orukeet', 'int8', True), (DEFAULT_MODEL, 'int8', True)
        # (prepared, current) -> selection to save; setup never aborts over the model.
        for prepared, current, saved in ((True, None, new), (True, old, new),
                                         (False, None, new),   # new backend: env already replaced
                                         (False, new, new),    # same model, cache evicted
                                         (False, old, old)):   # failed switch keeps the working model
            with patch.object(onnx, 'prepare_model', return_value=prepared), \
                 patch.object(onnx, 'log_warning') as warning:
                self.assertEqual(onnx.prepare_selection(new, current), saved)
            self.assertEqual(warning.called, not prepared)


class InteractiveSetupTests(unittest.TestCase):
    def _run(self, current_backend, prepared=False, decline_install=False, installed=True, choice=''):
        """Run interactive setup; save_selection raises to stop before any host mutation.

        Returns (setup_result, prepare_mock, save_mock); setup_result is 'saved' when
        setup reached save_selection.
        """
        existing = {'transcription_backend': current_backend} if current_backend else {}
        confirm = lambda prompt, *a, **k: not (decline_install and 'backend installation' in prompt)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(setup, '_check_mise_active', return_value=(False, '')))
            stack.enter_context(patch.object(setup, '_setup_command_symlink'))
            stack.enter_context(patch.object(setup, '_run_keyboard_selection', return_value=[]))
            stack.enter_context(patch.object(setup, '_load_existing_setup_config', return_value=existing))
            stack.enter_context(patch.object(setup, '_prompt_backend_selection', return_value=('onnx-asr', False, False)))
            stack.enter_context(patch.object(setup, '_detect_current_backend', return_value=current_backend))
            stack.enter_context(patch.object(setup, 'Prompt', types.SimpleNamespace(ask=lambda *a, **k: choice)))
            stack.enter_context(patch.object(setup, 'Confirm', types.SimpleNamespace(ask=confirm)))
            stack.enter_context(patch.object(setup, '_cleanup_backend', return_value=True))
            stack.enter_context(patch.object(setup, 'install_backend', return_value=installed))
            stack.enter_context(patch.object(onnx, 'log_warning'))
            prepare = stack.enter_context(patch.object(onnx, 'prepare_model', return_value=prepared))
            save = stack.enter_context(patch.object(setup, 'save_selection', side_effect=RuntimeError('stop')))
            service = stack.enter_context(patch.object(setup, 'setup_systemd'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                outcome = setup.setup_command(python_path='/tmp/mock-python')
            except SystemExit:
                outcome = 'saved'
        service.assert_not_called()
        return outcome, prepare, save

    def test_parakeet_engine_switch_keeps_venv(self):
        for current, engine, expected in (('onnx-asr', '2', 'parakeet-cpp'), ('parakeet-cpp', '1', 'onnx-asr')):
            answers = iter(['1', engine])
            confirm = Mock()
            with (patch.object(setup, '_detect_current_backend', return_value=current),
                  patch.object(setup, 'Prompt', types.SimpleNamespace(ask=lambda *a, **k: next(answers))),
                  patch.object(setup, 'Confirm', types.SimpleNamespace(ask=confirm)),
                  contextlib.redirect_stdout(io.StringIO()) as out):
                self.assertEqual(setup._prompt_backend_selection({}), (expected, False, False))
            confirm.assert_not_called()
            self.assertNotIn('recreate the venv', out.getvalue())

    def test_failed_preparation_saves_whenever_the_backend_changed(self):
        # Fresh installs, local switches and cloud switches all replace the environment.
        for current in (None, 'faster-whisper', 'rest-api', 'realtime-ws'):
            outcome, prepare, save = self._run(current)
            self.assertEqual(outcome, 'saved', current)
            prepare.assert_called_once_with((DEFAULT_MODEL, 'int8', True))
            save.assert_called_once_with(ANY, (DEFAULT_MODEL, 'int8', True))

    def test_failed_preparation_on_same_backend_keeps_current_model_and_continues(self):
        # Rerunning setup (e.g. offline, for bar changes) must not abort over the model.
        for choice, attempted in (('', DEFAULT_MODEL), ('2', 'orukeet')):
            outcome, prepare, save = self._run('onnx-asr', choice=choice)
            self.assertEqual(outcome, 'saved', choice)
            prepare.assert_called_once_with((attempted, 'int8', True))
            save.assert_called_once_with(ANY, (DEFAULT_MODEL, 'int8', True))

    def test_declined_install_skips_preparation_and_saves(self):
        outcome, prepare, save = self._run(None, decline_install=True)
        self.assertEqual(outcome, 'saved')
        prepare.assert_not_called()

    def test_failed_backend_install_reports_failure(self):
        # lib/cli.py exits 1 only on False; a bare return used to exit 0.
        outcome, prepare, save = self._run(None, installed=False)
        self.assertIs(outcome, False)
        prepare.assert_not_called()
        save.assert_not_called()


class AutoSetupTests(unittest.TestCase):
    def _run(self, previous, prepared, model=None, settings=None, backend='onnx-asr', saved=True):
        import backend_installer
        import config_manager
        settings = dict(settings or {}, transcription_backend=previous)
        events = []
        config = Mock()
        config.default_config = {'recording_mode': 'toggle'}
        config.get_setting.side_effect = settings.get
        config.get_all_settings.return_value = settings.copy()
        config.set_setting.side_effect = settings.__setitem__
        config.save_config.side_effect = lambda: events.append('save') or saved
        args = types.SimpleNamespace(backend=backend, model=model, no_waybar=True, no_systemd=True)
        with patch.object(install, '_check_mise_active', return_value=(False, '')), \
             patch.object(backend_installer, 'install_backend', return_value=True) as installer, \
             patch.object(config_manager, 'ConfigManager', return_value=config), \
             patch.object(install, '_verify_installation_step', return_value=True), \
             patch.object(onnx, 'prepare_model', side_effect=lambda s: events.append(s) or prepared), \
             patch.object(onnx, 'log_warning'), \
             patch.object(install, 'validate_command'), \
             patch.object(install, 'systemd_command') as service, \
             contextlib.redirect_stdout(io.StringIO()):
            outcome = install.omarchy_command(args)
        service.assert_not_called()
        return outcome, events, settings, installer

    def test_success_prepares_then_saves_and_keeps_user_settings(self):
        for model, recording in ((DEFAULT_MODEL, 'hold'), ('orukeet', 'toggle')):
            outcome, events, settings, _ = self._run(
                'pywhispercpp', True, model,
                {'onnx_asr_vad_min_duration': 77, 'recording_mode': recording, 'onnx_asr_quantization': None})
            self.assertTrue(outcome)
            self.assertEqual(events, [(model, None, True), 'save'])
            self.assertEqual(settings['onnx_asr_model'], model)
            self.assertIsNone(settings['onnx_asr_quantization'])
            self.assertEqual(settings['onnx_asr_vad_min_duration'], 77)
            # A customised mode survives; the untouched default gets setup auto's 'auto'.
            self.assertEqual(settings['recording_mode'], {'hold': 'hold', 'toggle': 'auto'}[recording])
            self.assertNotIn('mic_osd_enabled', settings)

    def test_failed_preparation_follows_the_switch_rule(self):
        # Same backend: keep the working model. New backend: save the new one.
        for previous, kept in (('onnx-asr', DEFAULT_MODEL), ('pywhispercpp', 'orukeet')):
            outcome, events, settings, _ = self._run(previous, False, 'orukeet')
            self.assertTrue(outcome, previous)
            self.assertEqual(events[-1], 'save')
            self.assertEqual(settings['transcription_backend'], 'onnx-asr')
            self.assertEqual(settings['onnx_asr_model'], kept)

    def test_config_save_failure_prevents_service_start(self):
        outcome, _, _, _ = self._run('pywhispercpp', True, 'orukeet', saved=False)
        self.assertFalse(outcome)

    def test_orukeet_on_a_whisper_backend_stops_before_installing(self):
        outcome, events, settings, installer = self._run('pywhispercpp', True, 'orukeet', backend='cpu')
        self.assertFalse(outcome)
        installer.assert_not_called()
        self.assertEqual(events, [])


class ModelCommandTests(unittest.TestCase):
    def test_download_uses_config_or_explicit_model_without_saving(self):
        config = Mock()
        config.get_setting.return_value = 'onnx-asr'
        config.get_all_settings.return_value = {'onnx_asr_model': 'custom', 'onnx_asr_use_vad': False}
        for explicit, expected in ((None, 'custom'), ('orukeet', 'orukeet'), (DEFAULT_MODEL, DEFAULT_MODEL)):
            with patch.object(models, 'ConfigManager', return_value=config), \
                 patch.object(models, 'prepare_model', return_value=True) as prepare:
                self.assertTrue(models.model_command('download', explicit))
            self.assertEqual(prepare.call_args.args[0], (expected, 'int8', False))
            config.save_config.assert_not_called()

    def test_download_of_another_model_says_it_is_not_selected(self):
        config = Mock()
        config.get_setting.return_value = 'onnx-asr'
        config.get_all_settings.return_value = {'onnx_asr_model': DEFAULT_MODEL}
        for explicit, noted in ((None, False), (DEFAULT_MODEL, False), ('orukeet', True)):
            with patch.object(models, 'ConfigManager', return_value=config), \
                 patch.object(models, 'prepare_model', return_value=True), \
                 patch.object(models, 'log_info') as info:
                self.assertTrue(models.model_command('download', explicit))
            self.assertEqual(any('not selected' in c.args[0] for c in info.call_args_list), noted, explicit)

    def test_orukeet_download_requires_onnx_backend(self):
        config = Mock()
        config.get_setting.return_value = 'pywhispercpp'
        with patch.object(models, 'ConfigManager', return_value=config), \
             patch.object(models, 'download_model') as whisper_download, \
             patch.object(models, 'prepare_model') as prepare, \
             patch.object(onnx, 'log_error') as error:
            self.assertFalse(models.model_command('download', 'orukeet'))
        whisper_download.assert_not_called()
        prepare.assert_not_called()
        self.assertIn('onnx-asr', error.call_args.args[0])


class StatusTests(unittest.TestCase):
    def _status(self, selected, hub):
        config = Mock()
        config.get_setting.return_value = selected
        with patch.object(orukeet, 'hub_cache_dir', return_value=Path(hub)), \
             patch.object(models, 'log_info') as info, \
             patch.object(models, 'log_success') as success, \
             patch.object(models, 'log_warning') as warning:
            models.onnx_asr_model_status(config)
        lines = [c.args[0] for c in info.call_args_list + success.call_args_list]
        return lines, warning.called

    def test_aliases_match_exact_repos_only(self):
        with tempfile.TemporaryDirectory() as hub:
            # Look-alike caches from other tools must not count as present.
            for name in ('models--istupakov--parakeet-tdt-0.6b-v2-onnx', 'models--nvidia--parakeet-tdt-0.6b-v3',
                         'models--Systran--faster-whisper-base'):
                (Path(hub) / name / 'snapshots').mkdir(parents=True)
            for selected, present, warns in (('nemo-parakeet-tdt-0.6b-v2', True, False),
                                             (DEFAULT_MODEL, False, True),
                                             ('whisper-base', False, False),
                                             ('gigaam-v2-ctc', False, False),
                                             ('org/missing', False, True)):
                lines, warned = self._status(selected, hub)
                self.assertEqual(any('Cache files present' in line for line in lines), present, selected)
                self.assertEqual(warned, warns, selected)

    def test_orukeet_status_reads_the_stamp_without_hashing(self):
        with tempfile.TemporaryDirectory() as hub:
            with patch.object(orukeet, 'cache_state', side_effect=['verified', 'unverified', 'missing']), \
                 patch.object(orukeet, 'download_model') as download, \
                 patch.object(orukeet, 'sha256') as digest:
                verified, _ = self._status('orukeet', hub)
                unverified, _ = self._status('orukeet', hub)
                _, missing_warns = self._status('orukeet', hub)
            download.assert_not_called()
            digest.assert_not_called()
        self.assertTrue(any('verified' in line for line in verified))
        self.assertTrue(any('model download' in line for line in unverified))
        self.assertTrue(missing_warns)

    def test_status_does_not_import_backends_or_numpy(self):
        blocked = {name: None for name in list(sys.modules) if name == 'backends' or name.startswith('backends.')}
        blocked['numpy'] = None
        for selected in (DEFAULT_MODEL, 'orukeet'):
            with patch.dict(sys.modules, blocked):
                self._status(selected, '/nonexistent-hf-cache')


if __name__ == '__main__':
    unittest.main()
