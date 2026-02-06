"""
Generic WebSocket client.
Provider-agnostic design, use whatever.
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


class RealtimeClient:
    """Generic WebSocket client for realtime transcription APIs"""
    
    def __init__(self, mode: str = 'transcribe'):
        """
        Realtime client for transcription or conversation.
        
        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        self.ws = None
        self.url = None
        self.api_key = None
        self.model = None
        self.instructions = None
        self.mode = mode
        self.language = None  # Language code for transcription (None = auto-detect)

        # Threading
        self.lock = threading.Lock()
        
        # Connection state
        self.connected = False
        self.connecting = False
        self.receiver_thread = None
        self.receiver_running = False
        
        # Event handling
        self.event_queue = Queue()
        self.response_event = threading.Event()
        self.current_response_text = ""
        self.response_complete = False
        
        # Transcription assembly (transcribe mode)
        self._transcript_generation = 0
        self._committed_segments = []

        # Track whether new audio has been queued since the last received transcript.
        # This helps avoid returning stale mid-stream text on stop.
        self._audio_activity_id = 0
        self._last_transcript_audio_activity_id = 0

        # Audio streaming
        # IMPORTANT: append_audio() is called from the sounddevice callback thread.
        # It must be fast and non-blocking: no websocket I/O or heavy resampling work here.
        self._audio_queue = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self.input_sample_rate = 16000  # AudioCapture provides 16kHz
        self.sample_rate = 24000  # OpenAI Realtime API requires 24kHz

        self._queue_cond = None
        self._sender_thread = None
        self._sender_running = False
        self._dropped_chunks = 0
        self._last_drop_log_time = 0.0
        
        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff
        self._queue_cond = threading.Condition(self.lock)

        # Track if buffer was committed (by VAD or manual)
        # Prevents double-commit error when VAD auto-commits on speech end
        self._buffer_committed = False

    def _start_sender_thread(self):
        """Start background sender thread (once)."""
        thread_to_join = None
        with self.lock:
            if self._sender_thread and self._sender_thread.is_alive():
                # If a previous sender is still alive but we marked it stopped,
                # wait briefly for it to exit so we don't end up with no active sender.
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
            # Re-check after join attempt
            if self._sender_thread and self._sender_thread.is_alive() and self._sender_running:
                return
            self._sender_running = True
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

    def _sender_loop(self):
        """Background thread: drain queued audio and send over the WebSocket."""
        try:
            from scipy import signal as _signal
        except Exception:
            _signal = None

        while True:
            with self.lock:
                self._queue_cond.wait_for(
                    lambda: (not self._sender_running)
                    or (self.connected and self.ws and len(self._audio_queue) > 0)
                )

                if not self._sender_running:
                    return

                # At this point, the wait predicate guarantees we're connected and have queued audio.
                audio_chunk = self._audio_queue.popleft()
                chunk_duration = len(audio_chunk) / float(self.input_sample_rate)
                self.audio_buffer_seconds = max(
                    0.0, self.audio_buffer_seconds - chunk_duration
                )
                ws = self.ws

                if not self._audio_queue:
                    self._queue_cond.notify_all()

            try:
                # Resample 16kHz -> 24kHz (ratio 3/2) outside the audio callback thread.
                if self.input_sample_rate == 16000 and self.sample_rate == 24000:
                    if _signal is None:
                        # scipy should be present on Arch install; if not, fall back to best-effort
                        resampled = audio_chunk
                    else:
                        resampled = _signal.resample_poly(audio_chunk, up=3, down=2)
                        resampled = resampled.astype(np.float32, copy=False)
                else:
                    resampled = audio_chunk

                pcm_bytes = self._float32_to_pcm16(resampled)
                base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')

                event = {'type': 'input_audio_buffer.append', 'audio': base64_audio}
                ws.send(json.dumps(event))

            except Exception as e:
                print(f'[REALTIME] Failed to send queued audio: {e}', flush=True)
        
    def connect(self, url: str, api_key: str, model: str, instructions: Optional[str] = None) -> bool:
        """
        Establish WebSocket connection with authentication.
        
        Args:
            url: WebSocket URL (e.g., 'wss://api.openai.com/v1/realtime?model=gpt-realtime-mini-2025-12-15')
            api_key: API key for authentication
            model: Model identifier
            instructions: Optional session instructions/prompt
        
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
        
        try:
            # Prepare headers with authentication
            headers = {
                'Authorization': f'Bearer {self.api_key}'
            }
            
            print(f'[REALTIME] Connecting to {self.url}...', flush=True)
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                header=[f'{k}: {v}' for k, v in headers.items()],
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            
            # Wait for connection (with timeout)
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f'[REALTIME] Connected successfully', flush=True)
                self.reconnect_attempts = 0
                
                # Always send session.update with audio format configuration
                self._send_session_update()
                
                return True
            else:
                print(f'[REALTIME] Connection timeout', flush=True)
                return False
                
        except Exception as e:
            print(f'[REALTIME] Connection error: {e}', flush=True)
            return False
        finally:
            self.connecting = False
    
    def _on_open(self, _ws):
        """WebSocket connection opened"""
        start_receiver = False
        with self.lock:
            self.connected = True
            self.connecting = False
            if not self.receiver_running:
                self.receiver_running = True
                start_receiver = True
            # Wake sender thread in case audio was queued just before connect.
            self._queue_cond.notify_all()

        # Start receiver thread
        if start_receiver:
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

        # Start sender thread (drains audio queue)
        self._start_sender_thread()
    
    def _on_message(self, _ws, message):
        """Handle incoming WebSocket message"""
        try:
            event = json.loads(message)
            self.event_queue.put(event)
        except json.JSONDecodeError as e:
            print(f'[REALTIME] Failed to parse event: {e}', flush=True)
    
    def _on_error(self, _ws, error):
        """Handle WebSocket error"""
        print(f'[REALTIME] WebSocket error: {error}', flush=True)
    
    def _on_close(self, _ws, close_status_code, _close_msg):
        """Handle WebSocket close"""
        with self.lock:
            self.connected = False
            # Stop sender thread on disconnect; it will be restarted on next _on_open().
            # This prevents it from waiting indefinitely after an unexpected close.
            self._sender_running = False
            # Drop queued audio on disconnect to avoid sending stale audio after reconnect
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()
        
        print(f'[REALTIME] WebSocket closed (code: {close_status_code})', flush=True)
        
        # Attempt reconnection if not intentionally closed
        if self.receiver_running and close_status_code != 1000:  # 1000 = normal closure
            self._attempt_reconnect()
    
    def _receiver_loop(self):
        """Background thread to process incoming events"""
        while self.receiver_running:
            try:
                # Get event with timeout
                event = self.event_queue.get(timeout=0.1)
                self._handle_event(event)
            except Empty:
                continue
            except Exception as e:
                print(f'[REALTIME] Error in receiver loop: {e}', flush=True)
    
    def _handle_event(self, event: dict):
        """Handle a single event from the server"""
        event_type = event.get('type', '')
        
        # Log session events
        if event_type in ('session.created', 'session.updated'):
            print(f'[REALTIME] Session event: {event_type}', flush=True)
        
        # Response events (conversational API)
        elif event_type == 'response.created':
            print(f'[REALTIME] Response created', flush=True)
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
            print(f'[REALTIME] Response text done ({text_len} chars)', flush=True)

        elif event_type == 'response.done':
            # Response complete
            with self.lock:
                self.response_complete = True
            self.response_event.set()
            print(f'[REALTIME] Response done', flush=True)
        
        # Transcription events (fallback/alternative)
        elif event_type == 'conversation.item.input_audio_transcription.completed':
            transcript = event.get('transcript', '') or ''
            transcript = transcript.strip()
            with self.lock:
                if transcript:
                    self._committed_segments.append(transcript)
                self._transcript_generation += 1
                self._last_transcript_audio_activity_id = self._audio_activity_id
                # Keep legacy fields coherent
                self.current_response_text = transcript
                self.response_complete = True
            self.response_event.set()
            print(
                f'[REALTIME] Transcription completed ({len(transcript)} chars)',
                flush=True,
            )
        
        elif event_type == 'input_audio_buffer.committed':
            print(f'[REALTIME] Audio buffer committed', flush=True)
            with self.lock:
                self._buffer_committed = True
        
        elif event_type == 'input_audio_buffer.speech_started':
            print(f'[REALTIME] Speech detected', flush=True)
            # Reset commit tracking - new speech means we haven't committed THIS audio yet
            with self.lock:
                self._buffer_committed = False
        
        elif event_type == 'input_audio_buffer.speech_stopped':
            print(f'[REALTIME] Speech ended', flush=True)
        
        elif event_type == 'error':
            error = event.get('error', {})
            error_message = error.get('message', 'Unknown error')
            print(f'[REALTIME] Server error: {error_message}', flush=True)
            self.response_complete = True
            self.response_event.set()  # Unblock waiting thread
    
    def _send_session_update(self):
        """Send session.update event based on mode"""
        if not self.connected or not self.ws:
            return
        
        if self.mode == 'transcribe':
            # Transcription-only session
            # Build transcription config - omit language for auto-detect
            transcription_config = {'model': 'gpt-4o-mini-transcribe'}
            if self.language:
                transcription_config['language'] = self.language

            session_data = {
                'type': 'transcription',
                'audio': {
                    'input': {
                        'format': {
                            'type': 'audio/pcm',
                            'rate': 24000
                        },
                        'transcription': transcription_config,
                        'turn_detection': {
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
            print(f'[REALTIME] Sent session.update', flush=True)
        except Exception as e:
            print(f'[REALTIME] Failed to send session.update: {e}', flush=True)
    
    def update_language(self, language: Optional[str]):
        """Update the language for transcription and resend session.update
        
        Args:
            language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
        """
        self.language = language
        if self.connected:
            self._send_session_update()
    
    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f'[REALTIME] Max reconnection attempts reached', flush=True)
            return False
        
        delay = self.reconnect_delays[min(self.reconnect_attempts, len(self.reconnect_delays) - 1)]
        self.reconnect_attempts += 1
        
        print(f'[REALTIME] Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s...', flush=True)
        time.sleep(delay)
        
        if self._connect_internal():
            # Re-send session.update after reconnect (always needed for audio format)
            self._send_session_update()
            return True
        
        return False
    
    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes"""
        # Clip to [-1, 1] range
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        
        # Convert to bytes (little-endian)
        return audio_int16.tobytes()
    
    def clear_audio_buffer(self):
        """Clear the server-side audio buffer before starting a new recording."""
        if not self.connected or not self.ws:
            return
        try:
            event = {'type': 'input_audio_buffer.clear'}
            self.ws.send(json.dumps(event))
            with self.lock:
                self._audio_queue.clear()
                self.audio_buffer_seconds = 0.0
                self._buffer_committed = False  # Reset commit tracking for new recording
                # Clear old transcription state to prevent returning stale results
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
        except Exception as e:
            print(f'[REALTIME] Failed to clear buffer: {e}', flush=True)
    
    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to WebSocket stream.
        
        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        if not self.connected or not self.ws:
            return

        drop_msg = None
        with self.lock:
            chunk_duration = len(audio_chunk) / float(self.input_sample_rate)

            # Drop OLDEST queued chunks until the new chunk fits.
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

            # If we still can't fit (e.g., max_buffer_seconds < chunk duration), drop this chunk.
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
                    f'[REALTIME] Dropping audio chunk(s) (queued>{self.max_buffer_seconds:.1f}s). '
                    f'dropped_chunks={self._dropped_chunks}'
                )
                self._last_drop_log_time = now

        if drop_msg:
            print(drop_msg, flush=True)
    
    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Commit audio buffer and wait for transcription result.

        With server VAD enabled, transcription happens automatically when speech ends.
        This method commits any remaining audio and waits for the transcript.

        Args:
            timeout: Maximum time to wait for transcription (seconds)

        Returns:
            Final transcript text, or empty string on timeout/error
        """
        if not self.connected or not self.ws:
            print('[REALTIME] Not connected, cannot commit', flush=True)
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

                # Common case: server VAD already produced the final transcript before user stops.
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
                    self._buffer_committed = False
                    print(
                        f'[REALTIME] Using existing transcript ({len(result)} chars)',
                        flush=True,
                    )
                    return result

                # Reset response state for manual commit flow
                self.current_response_text = ""
                self.response_complete = False
                self.response_event.clear()

                # Capture buffer committed state and reset it
                # This prevents "buffer too small" error when VAD auto-commits on speech end
                buffer_was_committed = self._buffer_committed
                self._buffer_committed = False

                queued_seconds = float(self.audio_buffer_seconds)
                max_backlog = float(self.max_buffer_seconds)

            # Best-effort: wait for queued audio to drain before committing.
            drain_timeout = min(
                max_backlog + 1.0,
                max(0.5, timeout * 0.5, queued_seconds + 0.25),
            )
            with self.lock:
                self._queue_cond.wait_for(
                    lambda: len(self._audio_queue) == 0, timeout=drain_timeout
                )

            # Small grace to allow any in-flight send to reach the server before commit.
            time.sleep(0.05)

            # Only send commit if buffer hasn't already been committed by VAD
            # (do this outside lock to avoid holding lock during I/O)
            if not buffer_was_committed:
                commit_event = {'type': 'input_audio_buffer.commit'}
                self.ws.send(json.dumps(commit_event))
                print('[REALTIME] Committed audio buffer', flush=True)
            else:
                print('[REALTIME] Skipping commit (VAD already committed)', flush=True)
            
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
                print('[REALTIME] Requested response, waiting...', flush=True)
            else:
                print('[REALTIME] Waiting for transcription...', flush=True)

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

                    # Settle briefly to catch final punctuation updates
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
                        f'[REALTIME] Transcript received ({len(result)} chars)',
                        flush=True,
                    )
                    return result

                print(f'[REALTIME] Timeout waiting for transcript ({timeout}s)', flush=True)
                with self.lock:
                    fallback = _full_committed_text_locked()
                    self._committed_segments = []
                    self._transcript_generation = 0
                return (fallback or "").strip()

            # converse mode: legacy response_event semantics
            if self.response_event.wait(timeout=timeout):
                if self.response_complete:
                    result = self.current_response_text.strip()
                    print(
                        f'[REALTIME] Response received ({len(result)} chars)',
                        flush=True,
                    )
                    self.audio_buffer_seconds = 0.0
                    return result

                print('[REALTIME] Event set but response not complete', flush=True)
                return ""

            print(f'[REALTIME] Timeout waiting for response ({timeout}s)', flush=True)
            return ""
                
        except Exception as e:
            print(f'[REALTIME] Error in commit_and_get_text: {e}', flush=True)
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
        
        print('[REALTIME] Connection closed', flush=True)
    
    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds for backpressure handling"""
        self.max_buffer_seconds = max(1.0, seconds)  # Minimum 1 second

