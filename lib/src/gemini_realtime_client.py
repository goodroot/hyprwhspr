"""
Gemini Live API WebSocket client.
Provides streaming speech-to-text using Google's Gemini Live API.
"""

import sys
import json
import base64
import threading
import time
from typing import Optional
from queue import Queue, Empty
from collections import deque

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import websocket
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: websocket-client is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pip install websocket-client>=1.6.0", file=sys.stderr)
    sys.exit(1)


class GeminiRealtimeClient:
    """WebSocket client for Gemini Live API realtime transcription"""

    def __init__(self, mode: str = 'transcribe'):
        """
        Gemini Live API client for transcription or conversation.

        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        self.ws = None
        self.url = None
        self.api_key = None
        self.model = None
        self.instructions = None
        self.mode = mode
        self.language = None

        # Threading
        self.lock = threading.Lock()

        # Connection state
        self.connected = False
        self.connecting = False
        self.receiver_thread = None
        self.receiver_running = False
        self._setup_complete = threading.Event()

        # Event handling
        self.event_queue = Queue()
        self.response_event = threading.Event()
        self.current_response_text = ""
        self.response_complete = False

        # Transcription assembly (transcribe mode)
        self._transcript_generation = 0
        self._committed_segments = []

        # Track audio activity
        self._audio_activity_id = 0
        self._last_transcript_audio_activity_id = 0

        # Audio streaming
        self._audio_queue = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self.input_sample_rate = 16000
        self.sample_rate = 16000  # Gemini accepts 16kHz natively

        self._queue_cond = None
        self._sender_thread = None
        self._sender_running = False
        self._dropped_chunks = 0
        self._last_drop_log_time = 0.0

        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]
        self._queue_cond = threading.Condition(self.lock)

    def _start_sender_thread(self):
        """Start background sender thread (once)."""
        thread_to_join = None
        with self.lock:
            if self._sender_thread and self._sender_thread.is_alive():
                if not self._sender_running:
                    thread_to_join = self._sender_thread
                else:
                    return

        if thread_to_join:
            try:
                thread_to_join.join(timeout=1.0)
            except Exception:
                pass

        with self.lock:
            if self._sender_thread and self._sender_thread.is_alive() and self._sender_running:
                return
            self._sender_running = True
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

    def _sender_loop(self):
        """Background thread: drain queued audio and send over WebSocket."""
        while True:
            with self.lock:
                self._queue_cond.wait_for(
                    lambda: (not self._sender_running)
                    or (self.connected and self.ws and len(self._audio_queue) > 0)
                )

                if not self._sender_running:
                    return

                audio_chunk = self._audio_queue.popleft()
                chunk_duration = len(audio_chunk) / float(self.input_sample_rate)
                self.audio_buffer_seconds = max(
                    0.0, self.audio_buffer_seconds - chunk_duration
                )
                ws = self.ws

                if not self._audio_queue:
                    self._queue_cond.notify_all()

            try:
                pcm_bytes = self._float32_to_pcm16(audio_chunk)
                base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')

                # Gemini Live API audio format
                event = {
                    'realtimeInput': {
                        'audio': {
                            'data': base64_audio,
                            'mimeType': f'audio/pcm;rate={self.sample_rate}'
                        }
                    }
                }
                ws.send(json.dumps(event))
            except Exception as e:
                print(f'[GEMINI] Failed to send queued audio: {e}', flush=True)

    def connect(self, url: str, api_key: str, model: str, instructions: Optional[str] = None) -> bool:
        """
        Establish WebSocket connection to Gemini Live API.

        Args:
            url: WebSocket base URL (API key will be appended)
            api_key: Google AI API key
            model: Model identifier (e.g., 'gemini-3.1-flash-live-preview')
            instructions: Optional system instructions

        Returns:
            True if connection successful, False otherwise
        """
        self.url = url
        self.api_key = api_key
        self.model = model
        self.instructions = instructions

        return self._connect_internal()

    def _connect_internal(self) -> bool:
        """Internal connection logic with reconnection support"""
        if self.connecting:
            return False

        self.connecting = True
        self._setup_complete.clear()

        try:
            # Gemini uses API key as query parameter
            separator = '&' if '?' in self.url else '?'
            ws_url = f'{self.url}{separator}key={self.api_key}'

            print(f'[GEMINI] Connecting to {self.url}...', flush=True)

            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for setup complete (config ack)
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if self.connected:
                print('[GEMINI] Connected successfully', flush=True)
                self.reconnect_attempts = 0
                return True
            else:
                print('[GEMINI] Connection timeout', flush=True)
                try:
                    self.ws.close()
                except Exception:
                    pass
                return False

        except Exception as e:
            print(f'[GEMINI] Connection error: {e}', flush=True)
            return False
        finally:
            self.connecting = False

    def _on_open(self, _ws):
        """WebSocket connection opened - send config message and start receiver"""
        print('[GEMINI] WebSocket opened, sending config...', flush=True)

        # Start receiver thread to process events (including setupComplete)
        start_receiver = False
        with self.lock:
            if not self.receiver_running:
                self.receiver_running = True
                start_receiver = True
        if start_receiver:
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

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
            print('[GEMINI] Config sent', flush=True)
        except Exception as e:
            print(f'[GEMINI] Failed to send config: {e}', flush=True)

    def _on_message(self, _ws, message):
        """Handle incoming WebSocket message"""
        try:
            event = json.loads(message)
            self.event_queue.put(event)
        except json.JSONDecodeError as e:
            print(f'[GEMINI] Failed to parse event: {e}', flush=True)

    def _on_error(self, _ws, error):
        """Handle WebSocket error"""
        print(f'[GEMINI] WebSocket error: {error}', flush=True)

    def _on_close(self, _ws, close_status_code, _close_msg):
        """Handle WebSocket close"""
        with self.lock:
            self.connected = False
            self._sender_running = False
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()

        print(f'[GEMINI] WebSocket closed (code: {close_status_code})', flush=True)

        if self.receiver_running and close_status_code != 1000:
            self._attempt_reconnect()

    def _receiver_loop(self):
        """Background thread to process incoming events"""
        while self.receiver_running:
            try:
                event = self.event_queue.get(timeout=0.1)
                self._handle_event(event)
            except Empty:
                continue
            except Exception as e:
                print(f'[GEMINI] Error in receiver loop: {e}', flush=True)

    def _handle_event(self, event: dict):
        """Handle a single event from the Gemini server"""

        # Setup complete acknowledgment
        if 'setupComplete' in event:
            print('[GEMINI] Setup complete', flush=True)
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
            print('[GEMINI] Received tool call (not handled)', flush=True)
            return

        # Usage metadata (ignore silently)
        if 'usageMetadata' in event:
            return

        # Unknown event type - log for debugging
        keys = list(event.keys())
        if keys:
            print(f'[GEMINI] Unknown event: {keys}', flush=True)

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
                print(
                    f'[GEMINI] Input transcription ({len(text)} chars)',
                    flush=True,
                )

        # Output transcription (model's spoken response as text)
        output_transcription = content.get('outputTranscription')
        if output_transcription:
            text = output_transcription.get('text', '')
            if text and self.mode == 'converse':
                with self.lock:
                    self.current_response_text += text
                print(
                    f'[GEMINI] Output transcription ({len(text)} chars)',
                    flush=True,
                )

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
                print('[GEMINI] Turn complete', flush=True)

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

            event = {
                'realtimeInput': {
                    'audio': {
                        'data': base64_audio,
                        'mimeType': f'audio/pcm;rate={self.sample_rate}'
                    }
                }
            }
            self.ws.send(json.dumps(event))
            print('[GEMINI] Sent silence tail for VAD flush', flush=True)
        except Exception as e:
            print(f'[GEMINI] Failed to send silence tail: {e}', flush=True)

    def update_language(self, language: Optional[str]):
        """Update language and reconnect to apply it (Gemini requires re-setup)."""
        if self.language == language:
            return
        self.language = language
        print(f'[GEMINI] Language set to: {language or "auto-detect"}', flush=True)

        if not self.connected or not self.ws:
            return

        # Gemini doesn't support mid-session language changes, reconnect
        print('[GEMINI] Reconnecting to apply language change', flush=True)
        try:
            self.ws.close()
        except Exception:
            pass
        self._connect_internal()

    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print('[GEMINI] Max reconnection attempts reached', flush=True)
            return False

        delay = self.reconnect_delays[min(self.reconnect_attempts, len(self.reconnect_delays) - 1)]
        self.reconnect_attempts += 1

        print(f'[GEMINI] Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s...', flush=True)
        time.sleep(delay)

        if self._connect_internal():
            return True

        return False

    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes"""
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def clear_audio_buffer(self):
        """Clear state before starting a new recording"""
        with self.lock:
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self.current_response_text = ""
            self.response_complete = False
            self._transcript_generation = 0
            self._committed_segments = []
            self._audio_activity_id = 0
            self._last_transcript_audio_activity_id = 0
            self._dropped_chunks = 0
            self._last_drop_log_time = 0.0
            self._queue_cond.notify_all()
        self.response_event.clear()

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to stream.

        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        if not self.connected or not self.ws:
            return

        drop_msg = None
        with self.lock:
            chunk_duration = len(audio_chunk) / float(self.input_sample_rate)

            while (
                (self.audio_buffer_seconds + chunk_duration) > self.max_buffer_seconds
                and self._audio_queue
            ):
                dropped = self._audio_queue.popleft()
                dropped_duration = len(dropped) / float(self.input_sample_rate)
                self.audio_buffer_seconds = max(
                    0.0, self.audio_buffer_seconds - dropped_duration
                )
                self._dropped_chunks += 1

            if (self.audio_buffer_seconds + chunk_duration) > self.max_buffer_seconds:
                self._dropped_chunks += 1
            else:
                self._audio_queue.append(audio_chunk)
                self.audio_buffer_seconds += chunk_duration
                self._audio_activity_id += 1
                self._queue_cond.notify_all()

            now = time.time()
            if self._dropped_chunks and (now - self._last_drop_log_time) > 2.0:
                drop_msg = (
                    f'[GEMINI] Dropping audio chunk(s) (queued>{self.max_buffer_seconds:.1f}s). '
                    f'dropped_chunks={self._dropped_chunks}'
                )
                self._last_drop_log_time = now

        if drop_msg:
            print(drop_msg, flush=True)

    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Signal end of turn and wait for transcription/response.

        Args:
            timeout: Maximum time to wait (seconds)

        Returns:
            Transcribed text or model response, or empty string on timeout
        """
        if not self.connected or not self.ws:
            print('[GEMINI] Not connected, cannot commit', flush=True)
            return ""

        try:
            with self.lock:
                def _full_committed_text_locked() -> str:
                    parts = [p for p in self._committed_segments if p]
                    return ' '.join(parts).strip()

                existing_generation = self._transcript_generation
                existing_transcript = (
                    _full_committed_text_locked() if self.mode == 'transcribe' else ""
                )
                has_new_audio_since_transcript = (
                    self._audio_activity_id != self._last_transcript_audio_activity_id
                )
                has_queued_audio = len(self._audio_queue) > 0

                # Fast path: VAD already produced transcript before user stops
                if (
                    self.mode == 'transcribe'
                    and existing_transcript
                    and (not has_new_audio_since_transcript)
                    and (not has_queued_audio)
                ):
                    result = existing_transcript
                    self._committed_segments = []
                    self._transcript_generation = 0
                    self.current_response_text = ""
                    self.response_complete = False
                    self.response_event.clear()
                    self.audio_buffer_seconds = 0.0
                    print(
                        f'[GEMINI] Using existing transcript ({len(result)} chars)',
                        flush=True,
                    )
                    return result

                self.current_response_text = ""
                self.response_complete = False
                self.response_event.clear()

                queued_seconds = float(self.audio_buffer_seconds)
                max_backlog = float(self.max_buffer_seconds)

            # Wait for queued audio to drain before signaling turn complete
            drain_timeout = min(
                max_backlog + 1.0,
                max(0.5, timeout * 0.5, queued_seconds + 0.25),
            )
            with self.lock:
                self._queue_cond.wait_for(
                    lambda: len(self._audio_queue) == 0, timeout=drain_timeout
                )

            time.sleep(0.05)

            # Signal end of user turn
            self._send_turn_complete()

            if self.mode == 'transcribe':
                deadline = time.time() + max(0.0, timeout)
                best_generation = existing_generation
                best_text = ""

                while time.time() < deadline:
                    remaining = max(0.0, deadline - time.time())
                    if not self.response_event.wait(timeout=remaining):
                        break

                    with self.lock:
                        if self._transcript_generation > best_generation:
                            best_generation = self._transcript_generation
                            best_text = _full_committed_text_locked()

                    # Settle briefly to catch final updates
                    settle_deadline = min(deadline, time.time() + 0.6)
                    self.response_event.clear()
                    while time.time() < settle_deadline:
                        settle_remaining = max(0.0, settle_deadline - time.time())
                        if not self.response_event.wait(timeout=settle_remaining):
                            break
                        with self.lock:
                            if self._transcript_generation > best_generation:
                                best_generation = self._transcript_generation
                                best_text = _full_committed_text_locked()
                        self.response_event.clear()

                    result = (best_text or "").strip()
                    with self.lock:
                        self._committed_segments = []
                        self._transcript_generation = 0
                        self.audio_buffer_seconds = 0.0
                    print(
                        f'[GEMINI] Transcript received ({len(result)} chars)',
                        flush=True,
                    )
                    return result

                print(f'[GEMINI] Timeout waiting for transcript ({timeout}s)', flush=True)
                with self.lock:
                    fallback = _full_committed_text_locked()
                    self._committed_segments = []
                    self._transcript_generation = 0
                return (fallback or "").strip()

            # Converse mode: wait for model response
            if self.response_event.wait(timeout=timeout):
                if self.response_complete:
                    result = self.current_response_text.strip()
                    print(
                        f'[GEMINI] Response received ({len(result)} chars)',
                        flush=True,
                    )
                    self.audio_buffer_seconds = 0.0
                    return result

                print('[GEMINI] Event set but response not complete', flush=True)
                return ""

            print(f'[GEMINI] Timeout waiting for response ({timeout}s)', flush=True)
            return ""

        except Exception as e:
            print(f'[GEMINI] Error in commit_and_get_text: {e}', flush=True)
            return ""

    def close(self):
        """Close WebSocket connection and cleanup"""
        with self.lock:
            self._sender_running = False
            self.receiver_running = False
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()

        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)

        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)

        with self.lock:
            self.connected = False

        print('[GEMINI] Connection closed', flush=True)

    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds"""
        self.max_buffer_seconds = max(1.0, seconds)
