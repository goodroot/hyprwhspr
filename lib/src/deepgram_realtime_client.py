"""
Deepgram live streaming protocol: raw PCM16 over a WebSocket.

Session options travel in the URL query; there is no config message. Binary
audio frames follow, then {"type":"CloseStream"}. The server answers with
Results events (interim, then is_final) and, after CloseStream, flushes the
last Results, sends Metadata and closes. A stream can't be cleared or resumed,
so each recording gets its own connection, as with Phonon.
"""

from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from .phonon_realtime_client import PhononRealtimeClient
    from .text_script import join_segments
except ImportError:
    from phonon_realtime_client import PhononRealtimeClient
    from text_script import join_segments


class DeepgramRealtimeClient(PhononRealtimeClient):
    """Phonon's one-stream-per-recording lifecycle with Deepgram's wire format."""

    LOG_TAG = '[DEEPGRAM]'
    PROVIDER_NAME = 'Deepgram'
    END_OF_INPUT = {'type': 'CloseStream'}
    # The language is part of the connection URL
    supports_mid_session_language_update = False

    def _ws_connect_params(self):
        # No language means auto-detect, which Deepgram streams as multilingual
        language = self.language or 'multi'
        params = {
            'model': self.model,
            'language': language,
            'encoding': 'linear16',
            'sample_rate': self.sample_rate,
            'channels': 1,
            'interim_results': 'true',
            'smart_format': 'true',
            'punctuate': 'true',
        }
        if language == 'multi':
            # Deepgram's recommendation for code-switching
            params['endpointing'] = 100
        parts = urlsplit(self.url)
        params.update(parse_qsl(parts.query))
        url = urlunsplit(parts._replace(query=urlencode(params)))
        headers = {'Authorization': f'Token {self.api_key}'} if self.api_key else None
        return url, headers

    def update_language(self, language: Optional[str]):
        self.language = language

    def _on_open(self, ws, generation=None):
        with self.lock:
            if self._is_active_connection(ws, generation) and not self._failure:
                self.connected = True

    def _dropped_idle_locked(self, ws, generation):
        """Deepgram drops streams with no audio for 10 seconds: nothing was
        recorded, so this isn't a failure. The next recording reconnects."""
        if (not self._is_active_connection(ws, generation) or not self.connected
                or self._audio_activity_id or self._ending):
            return False
        self.connected = False
        self._sender_running = False
        self._queue_cond.notify_all()
        # The transport reports the same drop twice, as an error and a close
        self.ws = None
        return True

    def _on_error(self, ws, error, generation=None):
        with self.lock:
            if self._dropped_idle_locked(ws, generation):
                return
        super()._on_error(ws, error, generation)

    def _on_close(self, ws, close_status_code, close_msg, generation=None):
        with self.lock:
            if self._dropped_idle_locked(ws, generation):
                return
            if (self._is_active_connection(ws, generation) and self._ending
                    and close_status_code == 1000 and not self._failure):
                self._complete_locked()
        super()._on_close(ws, close_status_code, close_msg, generation)

    def _complete_locked(self):
        if self.response_complete:
            return
        self.current_response_text = join_segments(
            [self._segments[key] for key in sorted(self._segments)]
        )
        self._partial_transcript = ''
        self.response_complete = True
        self.response_event.set()

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
                self._complete_locked()
        elif event_type == 'Error':
            self._fail_locked(f'Server rejected stream: {event.get("description", event.get("message"))}')
