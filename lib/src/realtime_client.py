"""
Generic WebSocket client.
Provider-agnostic design, use whatever.
"""

import json
from collections import deque
from typing import Optional

try:
    from .realtime_base import WebSocketRealtimeClientBase
except ImportError:
    from realtime_base import WebSocketRealtimeClientBase


class RealtimeClient(WebSocketRealtimeClientBase):
    """WebSocket client for the OpenAI Realtime transcription API"""

    LOG_TAG = '[REALTIME]'
    VALID_TRANSCRIPTION_DELAYS = {'minimal', 'low', 'medium', 'high', 'xhigh'}

    def __init__(self, mode: str = 'transcribe'):
        """
        Realtime client for transcription or conversation.

        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        super().__init__(mode=mode)
        self.transcription_delay = 'low'
        self.partial_transcript_callback = None
        self.sample_rate = 24000  # OpenAI Realtime API requires 24kHz

        # Track if buffer was committed (by VAD or manual)
        # Prevents double-commit error when VAD auto-commits on speech end
        self._buffer_committed = False

        # Items from previous/cancelled takes; late transcripts for them are dropped
        self._session_item_ids = set()
        self._retired_item_ids = deque(maxlen=64)

    # ------------------------------------------------------------------
    # Transport hooks
    # ------------------------------------------------------------------

    def _ws_connect_params(self):
        return self.url, {'Authorization': f'Bearer {self.api_key}'}

    def _on_connect_success(self):
        # Always send session.update with audio format configuration
        self._send_session_update()

    def _after_open(self):
        with self.lock:
            self.connected = True
            self.connecting = False
            # Wake sender thread in case audio was queued just before connect.
            self._queue_cond.notify_all()

        # Start sender thread (drains audio queue)
        self._start_sender_thread()

    def _audio_ws_message(self, base64_audio: str) -> dict:
        return {'type': 'input_audio_buffer.append', 'audio': base64_audio}

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: dict):
        """Handle a single event from the server"""
        event_type = event.get('type', '')

        # Log session events
        if event_type in ('session.created', 'session.updated'):
            self._log(f'Session event: {event_type}')

        # Response events (conversational API)
        elif event_type == 'response.created':
            self._log('Response created')
            with self.lock:
                self.current_response_text = ""
                self.response_complete = False
            self.response_event.clear()

        elif event_type == 'response.output_text.delta':
            # Accumulate text deltas
            delta = event.get('delta', '')
            if delta:
                with self.lock:
                    self.current_response_text += delta

        elif event_type == 'response.output_text.done':
            # Final text available
            text = event.get('text', '')
            with self.lock:
                if text:
                    self.current_response_text = text
                text_len = len(self.current_response_text)
            self._log(f'Response text done ({text_len} chars)')

        elif event_type == 'response.done':
            # Response complete
            with self.lock:
                self.response_complete = True
            self.response_event.set()
            self._log('Response done')

        # Transcription events (fallback/alternative)
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            if self._is_retired_item(event):
                return
            transcript = event.get('transcript', '') or ''
            transcript = transcript.strip()
            with self.lock:
                if not transcript:
                    transcript = self._partial_transcript.strip()
                if transcript:
                    self._committed_segments.append(transcript)
                self._transcript_generation += 1
                self._last_transcript_audio_activity_id = self._audio_activity_id
                self._partial_transcript = ""
                # Keep legacy fields coherent
                self.current_response_text = transcript
                self.response_complete = True
            self._notify_partial_transcript("")
            self.response_event.set()
            self._log(f'Transcription completed ({len(transcript)} chars)')

        elif event_type == 'conversation.item.input_audio_transcription.delta':
            if self._is_retired_item(event):
                return
            delta = event.get('delta', '') or ''
            if delta:
                with self.lock:
                    self._partial_transcript += delta
                    partial = self._partial_transcript
                self._notify_partial_transcript(partial)

        elif event_type == 'input_audio_buffer.committed':
            self._log('Audio buffer committed')
            with self.lock:
                self._buffer_committed = True
                self._track_item_locked(event)

        elif event_type == 'input_audio_buffer.speech_started':
            self._log('Speech detected')
            # Reset commit tracking - new speech means we haven't committed THIS audio yet
            with self.lock:
                self._buffer_committed = False
                self._partial_transcript = ""
                self._track_item_locked(event)
            self._notify_partial_transcript("")

        elif event_type == 'input_audio_buffer.speech_stopped':
            self._log('Speech ended')

        elif event_type == 'error':
            error = event.get('error', {})
            error_message = error.get('message', 'Unknown error')
            self._log(f'Server error: {error_message}')
            with self.lock:
                self._partial_transcript = ""
            self._notify_partial_transcript("")
            self.response_complete = True
            self.response_event.set()  # Unblock waiting thread

    def _track_item_locked(self, event: dict):
        """Record the conversation item id for the current take (call under lock)."""
        item_id = event.get('item_id')
        if item_id:
            self._session_item_ids.add(item_id)

    def _is_retired_item(self, event: dict) -> bool:
        """Whether a transcription event belongs to a previous/cancelled take."""
        item_id = event.get('item_id')
        with self.lock:
            if item_id and item_id in self._retired_item_ids:
                self._log('Dropping stale transcript for retired item')
                return True
        return False

    def _send_session_update(self):
        """Send session.update event based on mode"""
        if not self.connected or not self.ws:
            return

        if self.mode == 'transcribe':
            # Transcription-only session
            # Build transcription config - omit language for auto-detect
            model = self.model or 'gpt-4o-mini-transcribe'
            transcription_config = {'model': model}
            if self.language:
                transcription_config['language'] = self.language

            is_realtime_whisper = model == 'gpt-realtime-whisper'
            if is_realtime_whisper:
                transcription_config['delay'] = self._validated_transcription_delay()

            session_data = {
                'type': 'transcription',
                'audio': {
                    'input': {
                        'format': {
                            'type': 'audio/pcm',
                            'rate': 24000
                        },
                        'transcription': transcription_config,
                        'turn_detection': None if is_realtime_whisper else {
                            'type': 'server_vad',
                            'threshold': 0.5,
                            'prefix_padding_ms': 300,
                            'silence_duration_ms': 500
                        }
                    }
                }
            }
        else:
            # Conversational session (voice-to-AI) - no VAD, manual commit
            session_data = {
                'type': 'realtime',
                'output_modalities': ['text'],  # Text output only (no audio response)
                'audio': {
                    'input': {
                        'format': {
                            'type': 'audio/pcm',
                            'rate': 24000
                        },
                        'turn_detection': None  # Manual commit on stop
                    }
                },
                'instructions': self.instructions or 'You are a helpful assistant. Respond to the user based on what they say.'
            }

        event = {
            'type': 'session.update',
            'session': session_data
        }

        try:
            self.ws.send(json.dumps(event))
            self._log('Sent session.update')
        except Exception as e:
            self._log(f'Failed to send session.update: {e}')

    def update_language(self, language: Optional[str]):
        """Update the language for transcription and resend session.update

        Args:
            language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
        """
        self.language = language
        if self.connected:
            self._send_session_update()

    def set_transcription_delay(self, delay: str):
        """Set gpt-realtime-whisper transcription delay."""
        self.transcription_delay = self._normalize_transcription_delay(delay)
        if self.connected:
            self._send_session_update()

    def set_partial_transcript_callback(self, callback):
        """Register a callback for live transcription deltas."""
        self.partial_transcript_callback = callback

    def _normalize_transcription_delay(self, delay: str) -> str:
        delay = (delay or 'low').strip().lower()
        if delay not in self.VALID_TRANSCRIPTION_DELAYS:
            self._log(f"Invalid realtime_transcription_delay '{delay}', using 'low'")
            return 'low'
        return delay

    def _validated_transcription_delay(self) -> str:
        self.transcription_delay = self._normalize_transcription_delay(self.transcription_delay)
        return self.transcription_delay

    def _notify_partial_transcript(self, text: str):
        callback = self.partial_transcript_callback
        if not callback:
            return
        try:
            callback(text)
        except Exception as e:
            self._log(f'Partial transcript callback failed: {e}')

    # ------------------------------------------------------------------
    # Commit hooks
    # ------------------------------------------------------------------

    def clear_audio_buffer(self):
        """Clear the server-side audio buffer before starting a new recording."""
        with self.lock:
            # Retire the previous take's items so late transcripts are dropped
            self._retired_item_ids.extend(self._session_item_ids)
            self._session_item_ids.clear()
        if not self.connected or not self.ws:
            return
        try:
            event = {'type': 'input_audio_buffer.clear'}
            self.ws.send(json.dumps(event))
            with self.lock:
                self._reset_stream_state_locked()
                self._buffer_committed = False  # Reset commit tracking for new recording
                self._partial_transcript = ""
            self.response_event.clear()
            self._notify_partial_transcript("")
        except Exception as e:
            self._log(f'Failed to clear buffer: {e}')

    def _on_commit_fast_path_locked(self):
        self._buffer_committed = False

    def _capture_commit_context_locked(self) -> dict:
        # Capture buffer committed state and reset it
        # This prevents "buffer too small" error when VAD auto-commits on speech end
        buffer_was_committed = self._buffer_committed
        self._buffer_committed = False
        return {'buffer_was_committed': buffer_was_committed}

    def _request_transcript(self, ctx: dict):
        # Only send commit if buffer hasn't already been committed by VAD
        if not ctx.get('buffer_was_committed'):
            commit_event = {'type': 'input_audio_buffer.commit'}
            self.ws.send(json.dumps(commit_event))
            self._log('Committed audio buffer')
        else:
            self._log('Skipping commit (VAD already committed)')

        # For converse mode, request a response from the model
        # For transcribe mode, transcription happens automatically via VAD
        if self.mode == 'converse':
            response_event = {
                'type': 'response.create',
                'response': {
                    'output_modalities': ['text']  # Text only, no audio response
                }
            }
            self.ws.send(json.dumps(response_event))
            self._log('Requested response, waiting...')
        else:
            self._log('Waiting for transcription...')
