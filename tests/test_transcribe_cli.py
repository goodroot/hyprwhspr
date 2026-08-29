import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

from cli import transcribe  # noqa: E402


class FakeConfig:
    def __init__(self, backend='pywhispercpp'):
        self.backend = backend

    def get_setting(self, name, default=None):
        if name == 'transcription_backend':
            return self.backend
        return default


class TranscribeCliTests(unittest.TestCase):
    def test_daemon_transcript_is_the_only_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        def daemon_response(*_args):
            print('backend progress')
            return 'hello world'

        with mock.patch.object(transcribe, 'ConfigManager', return_value=FakeConfig()), \
                mock.patch.object(transcribe, '_request_daemon', side_effect=daemon_response), \
                mock.patch.object(transcribe, '_transcribe_standalone') as standalone, \
                mock.patch('sys.stdout', stdout), mock.patch('sys.stderr', stderr):
            self.assertTrue(transcribe.transcribe_command('/tmp/input.wav'))
        self.assertEqual(stdout.getvalue(), 'hello world\n')
        self.assertEqual(stderr.getvalue(), 'backend progress\n')
        standalone.assert_not_called()

    def test_standalone_fallback_receives_language_and_clean_flag(self):
        with mock.patch.object(transcribe, 'ConfigManager', return_value=FakeConfig()) as config, \
                mock.patch.object(transcribe, '_request_daemon', return_value=None), \
                mock.patch.object(transcribe, '_transcribe_standalone', return_value='bonjour') as standalone, \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertTrue(transcribe.transcribe_command(
                '/tmp/input.mp3', language='fr', clean=True
            ))
        standalone.assert_called_once_with(
            Path('/tmp/input.mp3'), 'fr', True, config.return_value
        )

    def test_writes_utf8_output_atomically_and_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / 'transcript.txt'
            output.write_text('old', encoding='utf-8')
            with mock.patch.object(transcribe, 'ConfigManager', return_value=FakeConfig()), \
                    mock.patch.object(transcribe, '_request_daemon', return_value='Grüße'):
                self.assertTrue(transcribe.transcribe_command('/tmp/input.wav', output))
            self.assertEqual(output.read_text(encoding='utf-8'), 'Grüße\n')
            self.assertEqual(list(Path(tempdir).glob('*.tmp')), [])

    def test_output_keeps_destination_mode_and_umask_for_new_files(self):
        import os
        with tempfile.TemporaryDirectory() as tempdir:
            existing = Path(tempdir) / 'existing.txt'
            existing.write_text('old', encoding='utf-8')
            os.chmod(existing, 0o644)
            transcribe._write_output('text', existing)
            self.assertEqual(existing.stat().st_mode & 0o777, 0o644)

            created = Path(tempdir) / 'created.txt'
            transcribe._write_output('text', created)
            umask = os.umask(0)
            os.umask(umask)
            self.assertEqual(created.stat().st_mode & 0o777, 0o666 & ~umask)

    def test_rejects_realtime_empty_result_and_input_overwrite(self):
        stderr = io.StringIO()
        with mock.patch.object(transcribe, 'ConfigManager', return_value=FakeConfig('realtime-ws')), \
                mock.patch('sys.stderr', stderr):
            self.assertFalse(transcribe.transcribe_command('/tmp/input.wav'))
        self.assertIn('live capture only', stderr.getvalue())

        stderr = io.StringIO()
        with mock.patch.object(transcribe, 'ConfigManager', return_value=FakeConfig()), \
                mock.patch.object(transcribe, '_request_daemon', return_value=''), \
                mock.patch('sys.stderr', stderr):
            self.assertFalse(transcribe.transcribe_command('/tmp/input.wav'))
        self.assertIn('produced no text', stderr.getvalue())

        stderr = io.StringIO()
        with mock.patch('sys.stderr', stderr):
            self.assertFalse(transcribe.transcribe_command(
                '/tmp/input.wav', '/tmp/../tmp/input.wav'
            ))
        self.assertIn('must differ', stderr.getvalue())

    def test_standalone_transcription_always_cleans_up_manager(self):
        instances = []

        class FakeManager:
            def __init__(self, config_manager):
                self.config = config_manager
                self.cleaned = False
                instances.append(self)

            def initialize(self):
                return True

            def transcribe_audio(self, audio, sample_rate, language_override):
                self.call = (audio, sample_rate, language_override)
                return ' raw '

            def cleanup(self):
                self.cleaned = True

        module = types.SimpleNamespace(WhisperManager=FakeManager)
        with mock.patch.dict(sys.modules, {'whisper_manager': module}), \
                mock.patch.object(transcribe, 'decode_audio_file', return_value=('audio', 22050)), \
                mock.patch.object(transcribe, 'preprocess_text', return_value='clean') as preprocess:
            result = transcribe._transcribe_standalone(
                Path('/tmp/input.wav'), 'de', True, FakeConfig()
            )
        self.assertEqual(result, 'clean')
        self.assertEqual(instances[0].call, ('audio', 22050, 'de'))
        self.assertTrue(instances[0].cleaned)
        preprocess.assert_called_once_with('raw', instances[0].config)

        instances.clear()
        FakeManager.initialize = lambda self: False
        with mock.patch.dict(sys.modules, {'whisper_manager': module}), \
                mock.patch.object(transcribe, 'decode_audio_file', return_value=('audio', 16000)):
            with self.assertRaisesRegex(
                    transcribe.TranscribeCommandError, 'Failed to initialize'):
                transcribe._transcribe_standalone(
                    Path('/tmp/input.wav'), None, False, FakeConfig()
                )
        self.assertTrue(instances[0].cleaned)

    def test_unexpected_failure_is_reported_without_escaping(self):
        stderr = io.StringIO()
        with mock.patch.object(
                transcribe, 'ConfigManager', side_effect=RuntimeError('backend boom')), \
                mock.patch('sys.stderr', stderr):
            self.assertFalse(transcribe.transcribe_command('/tmp/input.wav'))
        self.assertIn('File transcription failed: backend boom', stderr.getvalue())

    def test_daemon_response_has_a_finite_timeout(self):
        class FakeSocket:
            def __init__(self):
                self.timeouts = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def settimeout(self, value):
                self.timeouts.append(value)

            def connect(self, _path):
                return None

            def sendall(self, _payload):
                return None

            def recv(self, _size):
                raise transcribe.socket.timeout()

        client = FakeSocket()
        socket_path = mock.Mock()
        socket_path.exists.return_value = True
        with mock.patch.object(transcribe, 'SOCKET_FILE', socket_path), \
                mock.patch.object(transcribe.socket, 'socket', return_value=client):
            with self.assertRaisesRegex(
                    transcribe.TranscribeCommandError, 'timed out after 30 minutes'):
                transcribe._request_daemon(Path('/tmp/audio.wav'), None, False)
        self.assertEqual(
            client.timeouts,
            [2.0, transcribe.DAEMON_RESPONSE_TIMEOUT_SECONDS],
        )

    def test_native_stdout_is_redirected_away_from_transcript_fd(self):
        script = f'''\
import os
import sys
sys.path.insert(0, {str(ROOT / "lib" / "src")!r})
from cli.transcribe import _redirect_native_stdout_to_stderr
with _redirect_native_stdout_to_stderr():
    os.write(1, b"native log")
os.write(1, b"transcript")
'''
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, check=True
        )
        self.assertEqual(result.stdout, b'transcript')
        self.assertEqual(result.stderr, b'native log')


if __name__ == '__main__':
    unittest.main()
