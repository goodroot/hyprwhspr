import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib' / 'src'))
import diagnostics as diag
from recording_control_server import RecordingControlServer

SECRET = 'SECRET-private-label-token-user-path'


@unittest.skipUnless(importlib.util.find_spec("jsonschema") is not None, "jsonschema is required for schema validation tests")
class ValidationTests(unittest.TestCase):
    def validate(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(value))
            before = path.read_bytes()
            with mock.patch.object(diag, 'CONFIG_FILE', path), mock.patch(
                    'config_manager.ConfigManager', side_effect=AssertionError('must not construct')):
                result = diag.validate_config()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            return result

    def test_missing_uses_fresh_defaults_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, findings, failed = diag.validate_config(Path(tmp) / 'missing' / 'config.json')
            self.assertFalse(failed)
            self.assertEqual(findings[0]['check_id'], 'config.missing')
            self.assertEqual(list(Path(tmp).iterdir()), [])
        first = diag.build_default_config()
        first['word_overrides'].clear()
        self.assertTrue(diag.build_default_config()['word_overrides'])

    def test_malformed_json_and_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{\n"secret":')
            _, findings, unusable = diag.validate_config(path)
            self.assertTrue(unusable)
            with mock.patch.object(diag, 'CONFIG_FILE', path), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.run_diagnostics(), 2)
            self.assertIn('line 2, column 10', findings[0]['explanation'])
        with mock.patch.object(Path, 'read_text', side_effect=PermissionError(SECRET)):
            _, findings, unusable = diag.validate_config()
            self.assertTrue(unusable)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.run_diagnostics(), 2)
            self.assertNotIn(SECRET, json.dumps(findings))
            self.assertEqual(findings[0]['check_id'], 'config.read')

    def test_structure_types_nested_ranges_and_finite(self):
        for value in ([], {'audio_volume': 2}, {'grab_keys': 'false'},
                      {'threads': True}, {'recording_mode': 'invalid'},
                      {'applications': {SECRET: {'auto_paste': 12}}},
                      {'audio_volume': float('nan')}, {'rest_timeout': float('inf')},
                      {'push_to_talk': 'false'}, {'audio_device': []}):
            with self.subTest(value=value):
                _, findings, _ = self.validate(value)
                self.assertTrue(any(f['severity'] == 'error' for f in findings))
                self.assertNotIn(SECRET, json.dumps(findings))

    def test_legacy_aliases_and_prompts(self):
        config, findings, _ = self.validate({'push_to_talk': True, 'audio_device': 2,
            'whisper_prompt_de': SECRET, SECRET: SECRET})
        self.assertEqual(config['recording_mode'], 'push_to_talk')
        self.assertEqual(config['audio_device_id'], 2)
        self.assertEqual(sum(f['check_id'] == 'config.unknown' for f in findings), 1)
        self.assertFalse(any(f['severity'] == 'error' for f in findings))
        self.assertNotIn(SECRET, json.dumps(findings))

    def test_inherited_prompt_finding_survives_other_aliases(self):
        from config_manager import ENGLISH_PROMPT
        for alias, value in (('rest_api_key', SECRET), ('shift_paste', True),
                             ('push_to_talk', True), ('audio_device', 2)):
            _, findings, _ = self.validate({alias: value, 'whisper_prompt': ENGLISH_PROMPT})
            self.assertIn('config.deprecated_prompt', {f['check_id'] for f in findings})
            self.assertNotIn(SECRET, json.dumps(findings))

    def test_missing_validator_is_unusable_but_invalid_setting_is_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{}')
            with mock.patch.object(diag, 'CONFIG_FILE', path), mock.patch.dict(
                    sys.modules, {'jsonschema': None}), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.run_diagnostics(), 2)
            path.write_text('{"threads": false}')
            with mock.patch.object(diag, 'CONFIG_FILE', path), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.run_diagnostics(), 1)

    def test_legacy_backend_names(self):
        for backend in ('local', 'remote', 'amd'):
            _, findings, _ = self.validate({'transcription_backend': backend})
            self.assertFalse(any(f['severity'] == 'error' for f in findings))
            self.assertIn('config.deprecated_backend', {f['check_id'] for f in findings})

    def test_environment_without_coercion_or_disclosure(self):
        with mock.patch.dict(diag.os.environ, {'TOKEN': SECRET}, clear=True):
            config, findings, _ = self.validate({'rest_api_key': '${TOKEN}',
                'threads': '${THREADS}', 'whisper_prompt_fr': '${MISSING}'})
        self.assertEqual(config['rest_api_key'], SECRET)
        self.assertTrue(any(f['check_id'] == 'config.environment' for f in findings))
        self.assertTrue(any(f['severity'] == 'error' for f in findings))
        self.assertNotIn(SECRET, json.dumps(findings))
        self.assertNotIn('THREADS', json.dumps(findings))

    def test_schema_url_is_never_followed(self):
        _, findings, _ = self.validate({'$schema': 'https://' + SECRET})
        self.assertFalse(any(f['severity'] == 'error' for f in findings))

    def test_compatibility(self):
        _, findings, _ = self.validate({'secondary_language': 'fr', 'grab_keys': True,
            'use_hypr_bindings': True, 'transcription_backend': 'realtime-ws',
            'websocket_provider': 'openai', 'websocket_model': 'gpt-transcribe',
            'recording_mode': 'continuous', 'realtime_mode': 'converse'})
        ids = {f['check_id'] for f in findings}
        self.assertTrue({'config.secondary_shortcut', 'config.keyboard',
            'config.realtime_mode', 'config.realtime_continuous'} <= ids)
        _, findings, _ = self.validate({'transcription_backend': 'realtime-ws',
            'websocket_provider': SECRET, 'websocket_model': SECRET})
        self.assertIn('config.realtime_model', {f['check_id'] for f in findings})
        self.assertNotIn(SECRET, json.dumps(findings))


class SnapshotTests(unittest.TestCase):
    def app(self):
        return SimpleNamespace(_recording_lock=threading.Lock(), is_recording=True,
            is_processing=False, whisper_manager=SimpleNamespace(_model_lock=threading.Lock(),
                ready=True, _backend=SimpleNamespace(is_loaded=True)),
            audio_capture=SimpleNamespace(recovery_lock=threading.Lock(), device_id=SECRET,
                sample_rate=48000, recovery_in_progress=False, _input_selection_error=SECRET),
            text_injector=SimpleNamespace(wtype_available=True, ydotool_available=False,
                xdotool_available=False, _last_text=SECRET))

    def test_snapshot_privacy_and_concurrent_recovery(self):
        app = self.app()
        snapshot = diag.daemon_snapshot(app)
        self.assertTrue(snapshot['recording'])
        self.assertTrue(snapshot['selection_failed'])
        self.assertNotIn(SECRET, json.dumps(snapshot))
        self.assertNotIn('microphone_index', snapshot)
        for lock, section in ((app._recording_lock, 'recording_state'),
                              (app.audio_capture.recovery_lock, 'audio_state'),
                              (app.whisper_manager._model_lock, 'backend_state')):
            lock.acquire()
            try:
                self.assertEqual(diag.daemon_snapshot(app)[section], 'busy')
            finally:
                lock.release()

    def test_partial_initialization_keeps_other_sections_and_releases_lock(self):
        for section, owner, attribute in (
                ('audio_state', 'audio_capture', '_input_selection_error'),
                ('backend_state', 'whisper_manager', '_backend'),
                ('recording_state', None, 'is_processing')):
            app = self.app()
            delattr(getattr(app, owner) if owner else app, attribute)
            snapshot = diag.daemon_snapshot(app)
            self.assertEqual(snapshot[section], 'unavailable')
            for other in ('recording_state', 'backend_state', 'audio_state'):
                if other != section:
                    self.assertEqual(snapshot[other], 'available')
            for lock in (app._recording_lock, app.audio_capture.recovery_lock,
                         app.whisper_manager._model_lock):
                self.assertTrue(lock.acquire(blocking=False))
                lock.release()
            self.assertNotIn(SECRET, json.dumps(snapshot))

    def test_request_shutdown_and_private_exceptions(self):
        callback = mock.Mock(return_value=diag.daemon_snapshot(self.app()))
        server = RecordingControlServer('/unused', '/unused', mock.Mock(), mock.Mock(),
                                        on_diagnostics=callback)
        event = threading.Event()
        server._stop_event = event
        for stopped, broken in ((False, False), (True, False), (False, True)):
            event.clear()
            if stopped:
                event.set()
            callback.side_effect = RuntimeError(SECRET) if broken else None
            conn = mock.Mock()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                server._handle_json_request(conn, '{"verb":"diagnostics"}', event)
            conn.settimeout.assert_called_once_with(None)
            response = json.loads(conn.sendall.call_args.args[0])
            self.assertEqual(response['ok'], not stopped and not broken)
            self.assertNotIn(SECRET, json.dumps(response) + output.getvalue())
        server._on_command.assert_not_called()
        server._is_recording.assert_not_called()

    def test_untrusted_snapshot_allowlist(self):
        result = diag.sanitize_snapshot({'format_version': 1, 'recording': SECRET,
            'sample_rate': float('nan'), 'display': SECRET, 'transcript': SECRET,
            'delivery_routes': [SECRET, 'wtype']})
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertEqual(result['delivery_routes'], ['wtype'])
        self.assertIsNone(diag.sanitize_snapshot({'error': SECRET}))


class ReportTests(unittest.TestCase):
    def test_cloud_report_privacy_in_both_formats(self):
        config = diag.build_default_config()
        config.update(transcription_backend='rest-api', rest_api_key=SECRET,
                      rest_endpoint_url=SECRET, applications={SECRET: {}},
                      post_transcription_hook=SECRET, audio_device_name=SECRET)
        calls = []
        def probe(args):
            calls.append(args)
            if '-c' in args:
                modules = args[3:]
                return SimpleNamespace(stdout=json.dumps({'python': [3, 14, 0],
                    'modules': {name: True for name in modules},
                    'metadata': {name: True for name in modules}}), returncode=0)
            return SimpleNamespace(stdout=SECRET, stderr=SECRET, returncode=1)
        with mock.patch.object(diag, 'validate_config', return_value=(config, [], False)), \
                mock.patch.object(diag.Probes, 'run', side_effect=probe), \
                mock.patch.object(diag.Probes, 'live', side_effect=OSError(SECRET)), \
                mock.patch.object(diag.shutil, 'which', return_value=None), \
                mock.patch.object(diag.Path, 'glob', return_value=[]):
            for as_json in (False, True):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(diag.run_diagnostics(True, as_json), 0)
                self.assertNotIn(SECRET, output.getvalue())
                self.assertNotIn('pywhispercpp', output.getvalue())
                if as_json:
                    self.assertEqual(json.loads(output.getvalue())['format_version'], 1)
        self.assertIn('requests', calls[0])
        self.assertNotIn('pywhispercpp', calls[0])

    def test_broken_interpreter_is_error_and_retains_partial_report(self):
        import subprocess
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = Path(tmp) / 'venv' / 'bin' / 'python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/sh\nprintf "%s" "' + SECRET + '" >&2\nexit 7\n')
            interpreter.chmod(0o755)
            def probe(args):
                if '-c' in args:
                    return real_run(args, capture_output=True, text=True, timeout=2)
                return SimpleNamespace(stdout='inactive', returncode=0)
            with mock.patch.object(diag, 'DATA_DIR', Path(tmp)), \
                    mock.patch.object(diag, 'validate_config', return_value=(diag.build_default_config(), [], False)), \
                    mock.patch.object(diag.Probes, 'run', side_effect=probe), \
                    mock.patch.object(diag.Probes, 'live', return_value=None), \
                    mock.patch.object(diag.shutil, 'which', return_value=None), \
                    mock.patch.object(diag.Path, 'glob', return_value=[]):
                for as_json in (False, True):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(diag.run_diagnostics(True, as_json), 1)
                    self.assertNotIn(SECRET, output.getvalue())
                    self.assertIn('dependencies.probe', output.getvalue())
                    self.assertIn('service.state', output.getvalue())
                    self.assertIn('daemon.snapshot', output.getvalue())

    def test_old_python_is_actionable_and_invalid_probe_is_error(self):
        config = diag.build_default_config()
        config['transcription_backend'] = 'rest-api'
        modules = diag.PLAN_SPECS['rest'][1]
        valid = {'python': [3, 9, 9], 'modules': {m: True for m in modules},
                 'metadata': {m: True for m in modules}}
        for payload in (valid, {}, {'python': SECRET}, {'python': [3, 14, 0]}):
            with mock.patch.object(diag, 'validate_config', return_value=(config, [], False)), \
                    mock.patch.object(diag.Probes, 'run', return_value=SimpleNamespace(
                        stdout=json.dumps(payload), returncode=0)), \
                    mock.patch.object(diag.Probes, 'live', return_value=None), \
                    mock.patch.object(diag.shutil, 'which', return_value=None), \
                    mock.patch.object(diag.Path, 'glob', return_value=[]):
                report, _ = diag.build_report()
            errors = [f for f in report['findings'] if f['severity'] == 'error']
            self.assertTrue(errors)
            self.assertNotIn(SECRET, json.dumps(report))
            if payload is valid:
                self.assertEqual(errors[0]['check_id'], 'python.version')
                self.assertIn('Python 3.10', errors[0]['explanation'])
                self.assertIn('Python 3.10', errors[0]['next_step'])

    def test_timeouts_continue_and_total_budget(self):
        with mock.patch.object(diag, 'validate_config', return_value=(diag.build_default_config(), [], False)), \
                mock.patch.object(diag.Probes, 'run', side_effect=TimeoutError(SECRET)), \
                mock.patch.object(diag.Probes, 'live', side_effect=TimeoutError(SECRET)), \
                mock.patch.object(diag.shutil, 'which', return_value=None), \
                mock.patch.object(diag.Path, 'glob', return_value=[]):
            report, _ = diag.build_report()
        ids = {f['check_id'] for f in report['findings']}
        self.assertTrue({'dependencies.probe', 'service.state', 'daemon.snapshot'} <= ids)
        self.assertNotIn(SECRET, json.dumps(report))
        with mock.patch.object(diag.time, 'monotonic', side_effect=[0, 9, 11]):
            probe = diag.Probes()
            self.assertEqual(probe.timeout(), 1)
            with self.assertRaises(TimeoutError):
                probe.timeout()

    def test_exit_codes(self):
        for severity, unusable, expected in [('info', False, 0), ('warning', False, 0),
                                            ('error', False, 1), ('error', True, 2)]:
            with mock.patch.object(diag, 'validate_config', return_value=(
                    {}, [diag.finding('test', severity, 'Safe message.')], unusable)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diag.run_diagnostics(), expected)

class CommandTests(unittest.TestCase):
    def test_cli_routes_without_legacy_handlers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('diagnostic_cli',
            Path(__file__).resolve().parents[1] / 'lib' / 'cli.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for args, code in ((['config', 'validate', '--json'], 0),
                           (['status', '--report', '--json'], 0),
                           (['status', '--json'], 2)):
            with mock.patch.object(sys, 'argv', ['hyprwhspr'] + args), \
                    mock.patch.object(cli, '_get_version', side_effect=AssertionError('extra probe')), \
                    mock.patch.object(diag, 'run_diagnostics', return_value=0) as run, \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                cli.main()
            self.assertEqual(raised.exception.code, code)
            if code == 0:
                self.assertTrue(run.call_args.kwargs['json_output'])
            else:
                run.assert_not_called()

    def test_live_old_daemon_and_safe_new_snapshot(self):
        client = mock.MagicMock()
        client.__enter__.return_value = client
        for response, available in (({'ok': False, 'error': SECRET}, False),
                                    ({'ok': True, 'snapshot': {'format_version': 1,
                                      'recording': True, 'transcript': SECRET}}, True)):
            client.recv.return_value = json.dumps(response).encode() + b'\n'
            with mock.patch.object(diag.socket, 'socket', return_value=client):
                snapshot = diag.Probes().live()
            self.assertEqual(snapshot is not None, available)
            self.assertNotIn(SECRET, json.dumps(snapshot))
            client.sendall.assert_called_with(b'{"verb":"diagnostics"}\n')
