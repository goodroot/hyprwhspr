"""
Generic WebSocket client.
Provider-agnostic design, use whatever.
"""

import json
import uuid
from collections import deque
from typing import Optional

try:
    from .openai_realtime_models import (
        is_continuous,
        uses_language_context,
        uses_manual_commit,
    )
    from .realtime_base import WebSocketRealtimeClientBase
except ImportError:
    from openai_realtime_models import (
        is_continuous,
        uses_language_context,
        uses_manual_commit,
    )
    from realtime_base import WebSocketRealtimeClientBase


class RealtimeClient(WebSocketRealtimeClientBase):
    """WebSocket client for the OpenAI Realtime transcription API"""

    LOG_TAG = '[REALTIME]'
    VALID_TRANSCRIPTION_DELAYS = {'minimal', 'low', 'medium', 'high', 'xhigh'}
    VALID_CONVERSATION_HISTORY = {'session', 'turn'}

    def __init__(self, mode: str = 'transcribe'):
        """
        Realtime client for transcription or conversation.

        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        super().__init__(mode=mode)
        self.transcription_delay = 'low'
        self.transcription_prompt = None
        self.conversation_history = 'turn'
        self.partial_transcript_callback = None
        self.sample_rate = 24000  # OpenAI Realtime API requires 24kHz

        # Track if buffer was committed (by VAD or manual)
        # Prevents double-commit error when VAD auto-commits on speech end
        self._buffer_committed = False

        # Items from previous/cancelled takes; late transcripts for them are dropped
        self._take_item_ids = set()
        self._retired_item_ids = deque(maxlen=64)
        self._active_response_request_id = None
        self._active_response_id = None
        self._response_output_item_ids = set()
        self._retired_response_request_ids = deque(maxlen=64)
        self._retired_response_ids = deque(maxlen=64)
        self._logged_uncorrelated_response_compatibility = False

    # ------------------------------------------------------------------
    # Transport hooks
    # ------------------------------------------------------------------

    def _ws_connect_params(self):
        return self.url, {'Authorization': f'Bearer {self.api_key}'}

    def _on_connect_success(self):
        # Always send session.update with audio format configuration
        self._send_session_update()

    def _after_open(self, ws):
        with self.lock:
            if ws is not self.ws:
                return
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
            if not self._is_active_converse_response(event):
                return
            self._log('Response created')
            with self.lock:
                self._active_response_id = self._event_response_id(event)
                self.current_response_text = ""
                self.response_complete = False
            self.response_event.clear()

        elif event_type == 'response.output_text.delta':
            if not self._is_active_converse_response(event):
                return
            # Accumulate text deltas
            delta = event.get('delta', '')
            if delta:
                with self.lock:
                    self.current_response_text += delta
                    self._track_output_item_locked(event)

        elif event_type == 'response.output_text.done':
            if not self._is_active_converse_response(event):
                return
            # Final text available
            text = event.get('text', '')
            with self.lock:
                self._track_output_item_locked(event)
                if text:
                    self.current_response_text = text
                text_len = len(self.current_response_text)
            self._log(f'Response text done ({text_len} chars)')

        elif event_type in ('response.output_item.added', 'response.output_item.done'):
            if not self._is_active_converse_response(event):
                return
            with self.lock:
                self._track_output_item_locked(event)

        elif event_type == 'response.done':
            if not self._is_active_converse_response(event):
                return
            # Response complete
            with self.lock:
                self.response_complete = True
            self._clear_completed_turn_history(event)
            with self.lock:
                if self._active_response_request_id:
                    self._retired_response_request_ids.append(self._active_response_request_id)
                if self._active_response_id:
                    self._retired_response_ids.append(self._active_response_id)
                self._active_response_request_id = None
                self._active_response_id = None
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
            self._take_item_ids.add(item_id)

    def _track_output_item_locked(self, event: dict):
        """Record an output item seen while handling the active response."""
        item = event.get('item') or {}
        item_id = event.get('item_id') or item.get('id')
        if item_id:
            self._response_output_item_ids.add(item_id)

    @staticmethod
    def _event_response_id(event: dict):
        response = event.get('response') or {}
        return response.get('id') or event.get('response_id')

    @staticmethod
    def _event_response_request_id(event: dict):
        response = event.get('response') or {}
        metadata = response.get('metadata') or event.get('metadata') or {}
        return metadata.get('hyprwhspr_request_id')

    def _is_active_converse_response(self, event: dict) -> bool:
        """Require request metadata or a response ID linked by response.created."""
        if self.mode != 'converse':
            return False

        request_id = self._event_response_request_id(event)
        response_id = self._event_response_id(event)
        with self.lock:
            if request_id and request_id == self._active_response_request_id:
                if self._active_response_id and response_id and response_id != self._active_response_id:
                    self._log('Dropping response event for a different response')
                    return False
                return True
            if response_id and response_id == self._active_response_id:
                return True
            if request_id in self._retired_response_request_ids or response_id in self._retired_response_ids:
                self._log('Dropping stale response event for retired request')
                return False
            if not self._logged_uncorrelated_response_compatibility:
                self._logged_uncorrelated_response_compatibility = True
                self._log('Ignoring uncorrelated response event; endpoint must echo response metadata')
            return False

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

            language_context = uses_language_context(model)
            if self.language:
                if language_context:
                    transcription_config['languages'] = [self.language]
                else:
                    transcription_config['language'] = self.language

            if language_context and self.transcription_prompt:
                transcription_config['prompt'] = self.transcription_prompt

            if is_continuous(model):
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
                        'turn_detection': None if uses_manual_commit(model) else {
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
            with self._ws_send_lock:
                self.ws.send(json.dumps(event))
            self._log('Sent session.update')
        except Exception as e:
            self._log(f'Failed to send session.update: {e}')

    def update_transcription_config(
        self,
        language: Optional[str],
        prompt: Optional[str],
    ):
        """Update new-model transcription language and prompt together."""
        self.language = language
        self.transcription_prompt = prompt
        if self.connected:
            self._send_session_update()

    def update_language(self, language: Optional[str]):
        """Update the language for transcription and resend session.update

        Args:
            language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
        """
        self.update_transcription_config(language, self.transcription_prompt)

    def set_transcription_delay(self, delay: str):
        """Set the OpenAI streaming transcription delay."""
        self.transcription_delay = self._normalize_transcription_delay(delay)
        if self.connected:
            self._send_session_update()

    def set_conversation_history(self, history: str):
        """Set whether conversational turns retain server-side history."""
        history = (history or 'turn').strip().lower()
        if history not in self.VALID_CONVERSATION_HISTORY:
            self._log(f"Invalid realtime_conversation_history '{history}', using 'turn'")
            history = 'turn'
        self.conversation_history = history

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

    def _clear_completed_turn_history(self, event: dict):
        """Delete this completed conversational turn without closing the socket."""
        if self.mode != 'converse' or self.conversation_history != 'turn':
            return

        response = event.get('response') or {}
        output_items = response.get('output') or []
        output_ids = {
            item.get('id')
            for item in output_items
            if isinstance(item, dict) and item.get('id')
        }
        with self.lock:
            item_ids = self._take_item_ids | self._response_output_item_ids | output_ids
            self._take_item_ids.clear()
            self._response_output_item_ids.clear()
            self._retired_item_ids.extend(item_ids)

        if not self.connected or not self.ws:
            return
        for item_id in item_ids:
            try:
                with self._ws_send_lock:
                    self.ws.send(json.dumps({
                        'type': 'conversation.item.delete',
                        'item_id': item_id,
                    }))
            except Exception as e:
                self._log(f'Failed to delete conversation item: {e}')

    # ------------------------------------------------------------------
    # Commit hooks
    # ------------------------------------------------------------------

    def clear_audio_buffer(self):
        """Clear the server-side audio buffer before starting a new recording."""
        with self.lock:
            # Retire the previous take's items so late transcripts are dropped
            self._retired_item_ids.extend(self._take_item_ids)
            self._take_item_ids.clear()
        if not self.connected or not self.ws:
            return
        try:
            event = {'type': 'input_audio_buffer.clear'}
            with self._ws_send_lock:
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

    def _on_response_timeout(self):
        """Cancel and retire a timed-out converse response so it cannot satisfy a later take."""
        if self.mode != 'converse':
            return

        with self.lock:
            request_id = self._active_response_request_id
            response_id = self._active_response_id
            if request_id:
                self._retired_response_request_ids.append(request_id)
            if response_id:
                self._retired_response_ids.append(response_id)
            self._active_response_request_id = None
            self._active_response_id = None
            self.current_response_text = ""
            self.response_complete = False
        self.response_event.clear()

        self._clear_completed_turn_history({})

        if not request_id or not self.connected or not self.ws:
            return
        event = {'type': 'response.cancel'}
        if response_id:
            event['response_id'] = response_id
        try:
            with self._ws_send_lock:
                self.ws.send(json.dumps(event))
            self._log('Cancelled timed-out response')
        except Exception as e:
            self._log(f'Failed to cancel timed-out response: {e}')

    def _request_transcript(self, ctx: dict):
        # Only send commit if buffer hasn't already been committed by VAD
        if not ctx.get('buffer_was_committed'):
            commit_event = {'type': 'input_audio_buffer.commit'}
            with self._ws_send_lock:
                self.ws.send(json.dumps(commit_event))
            self._log('Committed audio buffer')
        else:
            self._log('Skipping commit (VAD already committed)')

        # For converse mode, request a response from the model
        # For transcribe mode, transcription happens automatically via VAD
        if self.mode == 'converse':
            request_id = uuid.uuid4().hex
            with self.lock:
                self._active_response_request_id = request_id
                self._active_response_id = None
                self._response_output_item_ids.clear()
            response_event = {
                'type': 'response.create',
                'response': {
                    'output_modalities': ['text'],  # Text only, no audio response
                    'metadata': {'hyprwhspr_request_id': request_id},
                }
            }
            with self._ws_send_lock:
                self.ws.send(json.dumps(response_event))
            self._log('Requested response, waiting...')
        else:
            self._log('Waiting for transcription...')
