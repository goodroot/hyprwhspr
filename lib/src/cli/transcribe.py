"""Batch audio-file transcription command."""

import contextlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

try:
    from ..audio_file import AudioFileError, decode_audio_file
    from ..backend_utils import normalize_backend
    from ..config_manager import ConfigManager
    from ..paths import SOCKET_FILE
    from ..text_injector import preprocess_text
except ImportError:
    from audio_file import AudioFileError, decode_audio_file
    from backend_utils import normalize_backend
    from config_manager import ConfigManager
    from paths import SOCKET_FILE
    from text_injector import preprocess_text


class TranscribeCommandError(RuntimeError):
    """An expected user-facing failure from file transcription."""


DAEMON_RESPONSE_TIMEOUT_SECONDS = 30 * 60


@contextlib.contextmanager
def _redirect_native_stdout_to_stderr():
    """Keep native-library writes to fd 1 out of transcript stdout."""
    saved_stdout = None
    try:
        sys.stdout.flush()
        saved_stdout = os.dup(1)
        os.dup2(2, 1)
    except (AttributeError, OSError, ValueError):
        if saved_stdout is not None:
            os.close(saved_stdout)
        yield
        return
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            os.dup2(saved_stdout, 1)
        finally:
            os.close(saved_stdout)


def _configured_backend(config):
    return normalize_backend(config.get_setting('transcription_backend', 'pywhispercpp'))


def _request_daemon(input_path, language, clean):
    """Return a daemon response, or None when no healthy daemon is reachable."""
    if not SOCKET_FILE.exists():
        return None
    request = json.dumps({
        'verb': 'transcribe_file',
        'path': str(input_path),
        'language': language,
        'clean': bool(clean),
    }, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with client:
            client.settimeout(2.0)
            try:
                client.connect(str(SOCKET_FILE))
            except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
                return None
            client.sendall(request)
            client.settimeout(DAEMON_RESPONSE_TIMEOUT_SECONDS)
            chunks = []
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except socket.timeout as exc:
        raise TranscribeCommandError(
            'Daemon transcription timed out after 30 minutes'
        ) from exc
    except OSError as exc:
        raise TranscribeCommandError(f'Daemon request failed: {exc}') from exc
    try:
        response = json.loads(b''.join(chunks).decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TranscribeCommandError('Daemon returned an invalid response') from exc
    if not isinstance(response, dict) or not isinstance(response.get('ok'), bool):
        raise TranscribeCommandError('Daemon returned an invalid response')
    if not response['ok']:
        raise TranscribeCommandError(str(response.get('error') or 'Daemon rejected the request'))
    text = response.get('text')
    if not isinstance(text, str):
        raise TranscribeCommandError('Daemon returned an invalid transcript')
    return text


def _transcribe_standalone(input_path, language, clean, config):
    if _configured_backend(config) == 'realtime-ws':
        raise TranscribeCommandError(
            "The realtime-ws backend supports live capture only; configure a local "
            "or REST backend to transcribe files"
        )
    from whisper_manager import WhisperManager

    with _redirect_native_stdout_to_stderr():
        manager = WhisperManager(config_manager=config)
        try:
            audio_data, sample_rate = decode_audio_file(input_path)
            if not manager.initialize():
                raise TranscribeCommandError('Failed to initialize transcription backend')
            text = manager.transcribe_audio(
                audio_data, sample_rate=sample_rate, language_override=language
            )
            text = text.strip() if text else ''
            if not text:
                raise TranscribeCommandError('Transcription produced no text')
            return preprocess_text(text, config) if clean else text
        finally:
            manager.cleanup()


def _output_mode(destination):
    """Mode for a transcript written via replace: keep the destination's own."""
    try:
        return os.stat(destination).st_mode & 0o7777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _write_output(text, output_path):
    payload = text + '\n'
    if output_path is None:
        sys.stdout.write(payload)
        sys.stdout.flush()
        return

    destination = Path(output_path).expanduser()
    if not destination.parent.exists():
        raise TranscribeCommandError(
            f"Output directory does not exist: {destination.parent}"
        )
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f'.{destination.name}.', suffix='.tmp', dir=str(destination.parent)
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates 0600; replacing an existing file must not silently
        # tighten its permissions, and a new file should follow the umask
        os.chmod(temp_path, _output_mode(destination))
        os.replace(temp_path, destination)
    except OSError as exc:
        raise TranscribeCommandError(f"Could not write output file '{destination}': {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def transcribe_command(input_path, output_path=None, language=None, clean=False):
    """Transcribe one WAV/MP3 file and write its text result."""
    source = Path(input_path).expanduser().resolve()
    try:
        if output_path is not None:
            destination = Path(output_path).expanduser().resolve()
            if destination == source:
                raise TranscribeCommandError('Output path must differ from the input file')
        # Many backends still use print() for operational messages. Keep stdout
        # reserved for the transcript so the command remains pipe-friendly.
        with contextlib.redirect_stdout(sys.stderr):
            config = ConfigManager()
            if _configured_backend(config) == 'realtime-ws':
                raise TranscribeCommandError(
                    "The realtime-ws backend supports live capture only; configure a "
                    "local or REST backend to transcribe files"
                )
            text = _request_daemon(source, language, clean)
            if text is None:
                text = _transcribe_standalone(source, language, clean, config)
        if not text or not text.strip():
            raise TranscribeCommandError('Transcription produced no text')
        _write_output(text, output_path)
        return True
    except (AudioFileError, TranscribeCommandError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"Error: File transcription failed: {exc}", file=sys.stderr)
        return False
