"""
Gemini Live API WebSocket client.
Provides streaming speech-to-text using Google's Gemini Live API.
"""

import json
import base64
import threading
from typing import Optional

try:
    from .realtime_base import WebSocketRealtimeClientBase
except ImportError:
    from realtime_base import WebSocketRealtimeClientBase

import numpy as np


class GeminiRealtimeClient(WebSocketRealtimeClientBase):
    """WebSocket client for Gemini Live API realtime transcription"""

    LOG_TAG = '[GEMINI]'

    # Language is baked into the setup message at connect time.
    # Changing it mid-session requires a full reconnect, which would
    # drop any already-streamed audio. _transcribe_realtime checks
    # this flag and skips update_language() after audio has been sent.
    supports_mid_session_language_update = False

    def __init__(self, mode: str = 'transcribe'):
        """
        Gemini Live API client for transcription or conversation.

        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        super().__init__(mode=mode)
        self.sample_rate = 16000  # Gemini output stream rate
        self._setup_complete = threading.Event()

    # ------------------------------------------------------------------
    # Transport hooks
    # ------------------------------------------------------------------

    def _ws_connect_params(self):
        # Gemini uses API key as query parameter
        separator = '&' if '?' in self.url else '?'
        return f'{self.url}{separator}key={self.api_key}', None

    def _prepare_connect(self):
        self._setup_complete.clear()

    def _after_open(self):
        """Send the setup/config message; 'connected' is set on setupComplete."""
        self._log('WebSocket opened, sending config...')

        # Build setup message per BidiGenerateContentSetup proto
        # Native audio models REQUIRE responseModalities: ["AUDIO"].
        # Text transcription comes via inputAudioTranscription / outputAudioTranscription.
        setup_config = {
            'model': f'models/{self.model}',
            'generationConfig': {
                'responseModalities': ['AUDIO'],
                'speechConfig': {
                    'voiceConfig': {
                        'prebuiltVoiceConfig': {
                            'voiceName': 'Puck'
                        }
                    }
                },
            },
            'inputAudioTranscription': {},
            'outputAudioTranscription': {},
        }

        # Add system instruction
        if self.instructions:
            setup_config['systemInstruction'] = {
                'parts': [{'text': self.instructions}]
            }
        elif self.mode == 'transcribe':
            instruction = 'You are a transcription assistant. Acknowledge audio input briefly.'
            if self.language:
                instruction += f' The user speaks {self.language}.'
            setup_config['systemInstruction'] = {
                'parts': [{'text': instruction}]
            }

        # Send setup message
        setup_message = {'setup': setup_config}
        try:
            self.ws.send(json.dumps(setup_message))
            self._log('Config sent')
        except Exception as e:
            self._log(f'Failed to send config: {e}')

    def _audio_ws_message(self, base64_audio: str) -> dict:
        # Gemini Live API audio format
        return {
            'realtimeInput': {
                'audio': {
                    'data': base64_audio,
                    'mimeType': f'audio/pcm;rate={self.sample_rate}'
                }
            }
        }

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: dict):
        """Handle a single event from the Gemini server"""

        # Setup complete acknowledgment
        if 'setupComplete' in event:
            self._log('Setup complete')
            with self.lock:
                self.connected = True
                self.connecting = False
                self._queue_cond.notify_all()
            self._setup_complete.set()
            self._start_sender_thread()
            return

        # Server content (transcription, model responses)
        if 'serverContent' in event:
            server_content = event['serverContent']
            if server_content:
                self._handle_server_content(server_content)
            # Also check for usageMetadata at the top level alongside serverContent
            if 'usageMetadata' in event:
                pass  # silently consume
            return

        # Session resumption updates (ignore silently)
        if 'sessionResumptionUpdate' in event:
            return

        # Tool calls (not used for transcription, but log for completeness)
        if 'toolCall' in event:
            self._log('Received tool call (not handled)')
            return

        # Usage metadata (ignore silently)
        if 'usageMetadata' in event:
            return

        # Unknown event type - log for debugging
        keys = list(event.keys())
        if keys:
            self._log(f'Unknown event: {keys}')

    def _handle_server_content(self, content: dict):
        """Handle serverContent events"""

        # Input transcription (user's speech-to-text)
        input_transcription = content.get('inputTranscription')
        if input_transcription:
            text = input_transcription.get('text', '')
            if text and text.strip():
                with self.lock:
                    self._committed_segments.append(text.strip())
                    self._transcript_generation += 1
                    self._last_transcript_audio_activity_id = self._audio_activity_id
                    self.current_response_text = text.strip()
                    self.response_complete = True
                self.response_event.set()
                self._log(f'Input transcription ({len(text)} chars)')

        # Output transcription (model's spoken response as text)
        output_transcription = content.get('outputTranscription')
        if output_transcription:
            text = output_transcription.get('text', '')
            if text and self.mode == 'converse':
                with self.lock:
                    self.current_response_text += text
                self._log(f'Output transcription ({len(text)} chars)')

        # Model turn (text parts from model response)
        model_turn = content.get('modelTurn')
        if model_turn:
            parts = model_turn.get('parts', [])
            for part in parts:
                text = part.get('text', '')
                if text and self.mode == 'converse':
                    with self.lock:
                        self.current_response_text += text

        # Turn complete
        turn_complete = content.get('turnComplete', False)
        if turn_complete:
            if self.mode == 'converse':
                with self.lock:
                    self.response_complete = True
                self.response_event.set()
                self._log('Turn complete')

    # ------------------------------------------------------------------
    # Commit hooks
    # ------------------------------------------------------------------

    def _send_turn_complete(self):
        """Signal end of user audio input.

        Gemini's VAD handles turn detection for realtimeInput audio.
        We send a brief silence tail to help the VAD detect speech end,
        then let the server process and return the transcription.
        """
        if not self.connected or not self.ws:
            return

        try:
            # Send ~2s of silence to help VAD detect speech end on short recordings
            silence = np.zeros(int(self.sample_rate * 2.0), dtype=np.float32)
            pcm_bytes = self._float32_to_pcm16(silence)
            base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')

            self.ws.send(json.dumps(self._audio_ws_message(base64_audio)))
            self._log('Sent silence tail for VAD flush')
        except Exception as e:
            self._log(f'Failed to send silence tail: {e}')

    def _request_transcript(self, ctx: dict):
        # Signal end of user turn
        self._send_turn_complete()

    def update_language(self, language: Optional[str]):
        """Update language and reconnect to apply it (Gemini requires re-setup)."""
        if self.language == language:
            return
        self.language = language
        self._log(f'Language set to: {language or "auto-detect"}')

        if not self.connected or not self.ws:
            return

        # Gemini doesn't support mid-session language changes, reconnect
        self._log('Reconnecting to apply language change')
        try:
            self.ws.close()
        except Exception:
            pass
        self._connect_internal()
