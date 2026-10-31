"""
Deepgram live streaming protocol: raw PCM16 over a WebSocket.

Session options travel in the URL query; there is no config message. Binary
audio frames follow, then {"type":"Finalize"} and {"type":"CloseStream"} back
to back: Finalize flushes the tail, CloseStream then sends Metadata and closes.
The server answers with Results events (interim, then is_final). Idle streams
close after 10 seconds without audio, so KeepAlive is sent while waiting. A
stream can't be cleared or resumed, so each recording gets its own connection,
as with Phonon.
"""

from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from .phonon_realtime_client import PhononRealtimeClient
except ImportError:
    from phonon_realtime_client import PhononRealtimeClient


class DeepgramRealtimeClient(PhononRealtimeClient):
    """Phonon's one-stream-per-recording lifecycle with Deepgram's wire format."""

    LOG_TAG = '[DEEPGRAM]'
    PROVIDER_NAME = 'Deepgram'
    # The server handles these in order, so the tail is flushed without a wait
    END_OF_INPUT = ({'type': 'Finalize'}, {'type': 'CloseStream'})
    KEEPALIVE = {'type': 'KeepAlive'}

    def __init__(self, mode: str = 'transcribe'):
        super().__init__(mode)
        self._url_language = None
        self._server_close = None

    def _prepare_connect(self):
        super()._prepare_connect()
        with self.lock:
            self._server_close = None

    def _ws_connect_params(self):
        self._url_language = self.language
        # No language means auto-detect, which Deepgram streams as multilingual
        language = self.language or 'multi'
        defaults = {
            'model': self.model,
            'language': language,
            'encoding': 'linear16',
            'sample_rate': self.sample_rate,
            'channels': 1,
            'interim_results': 'true',
            'smart_format': 'true',
        }
        if language == 'multi':
            # Deepgram's recommendation for code-switching
            defaults['endpointing'] = 100
        parts = urlsplit(self.url)
        # The URL's own query wins, and may repeat keys (keyterm=a&keyterm=b)
        user = parse_qsl(parts.query, keep_blank_values=True)
        overridden = {key for key, _ in user}
        pairs = [(key, value) for key, value in defaults.items() if key not in overridden] + user
        url = urlunsplit(parts._replace(query=urlencode(pairs)))
        headers = {'Authorization': f'Token {self.api_key}'} if self.api_key else None
        return url, headers

    def update_language(self, language: Optional[str]):
        # Takes effect on the next connection; see needs_fresh_session
        self.language = language

    @property
    def needs_fresh_session(self):
        """The language is part of the URL: a stream opened with another needs replacing."""
        return super().needs_fresh_session or self._url_language != self.language

    def _on_open(self, ws, generation=None):
        with self.lock:
            if self._is_active_connection(ws, generation) and not self._failure:
                self.connected = True

    @staticmethod
    def _server_close_frame(error):
        """(code, reason) if websocket-client reported a server close frame as an error, else None.

        1.9.2+ passes an exception carrying status_code and reason, then repeats
        them to on_close. 1.9.0/1.9.1 pass the raw close frame, and on_close gets
        None, None. A handshake failure has status_code but no reason: not a close.
        """
        if hasattr(error, 'status_code') and hasattr(error, 'reason'):
            return error.status_code, error.reason
        if getattr(error, 'opcode', None) == 0x8:
            data = getattr(error, 'data', b'') or b''
            if len(data) < 2:
                return None, None
            return int.from_bytes(data[:2], 'big'), data[2:].decode('utf-8', 'replace')
        return None

    def _on_error(self, ws, error, generation=None):
        # A server close frame arrives here first; on_close follows and judges it
        close = self._server_close_frame(error)
        if close is not None:
            with self.lock:
                if self._is_active_connection(ws, generation):
                    self._server_close = close
            return
        super()._on_error(ws, error, generation)

    def _on_close(self, ws, close_status_code, close_msg, generation=None):
        with self.lock:
            if close_status_code is None and self._is_active_connection(ws, generation) and self._server_close:
                close_status_code, close_msg = self._server_close
            # After CloseStream a normal close ends the stream, Metadata or not
            if (self._is_active_connection(ws, generation) and self._ending
                    and close_status_code == 1000 and not self._failure):
                self._complete_stream_locked()
        super()._on_close(ws, close_status_code, close_msg, generation)

    def _complete_stream_locked(self):
        if not self.response_complete:
            self._complete_locked(self._committed_text_locked())

    def _handle_event_locked(self, event):
        event_type = event.get('type')
        if event_type == 'Results':
            alternatives = (event.get('channel') or {}).get('alternatives') or [{}]
            text = alternatives[0].get('transcript') or ''
            if not isinstance(text, str):
                raise ValueError('transcript text must be a string')
            text = text.strip()
            if not event.get('is_final'):
                self._partial_transcript = text
                return
            if text:
                self._segments[len(self._segments)] = text
            self._partial_transcript = ''
        elif event_type == 'Metadata':
            # Summary of a stream that CloseStream ended: nothing follows it.
            if self._ending:
                self._complete_stream_locked()
        elif event_type == 'Error':
            self._fail_locked(f'Server rejected stream: {event.get("description", event.get("message"))}')
