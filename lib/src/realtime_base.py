"""
Shared base classes for realtime transcription clients.

RealtimeAudioClientBase: audio queueing/backpressure, resampling and PCM16
conversion shared by every realtime provider.

WebSocketRealtimeClientBase: full websocket-client transport skeleton
(connect/reconnect, sender/receiver threads, commit-and-wait machinery)
shared by providers that speak raw WebSocket JSON (OpenAI, Gemini).
Providers built on their own SDK transport (ElevenLabs) subclass only
RealtimeAudioClientBase.
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
    from .audio_resampler import resample_audio
except ImportError:
    from audio_resampler import resample_audio

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

class RealtimeAudioClientBase:
    """Audio queueing, backpressure and format conversion shared by all realtime clients."""

    LOG_TAG = '[REALTIME]'
    IDLE_CLOSE_SECS = 10.0

    def __init__(self):
        self.api_key = None
        self.model = None
        self.language = None

        # Threading
        self.lock = threading.Lock()
        self._queue_cond = threading.Condition(self.lock)

        # Connection state (transport managed by subclass)
        self.connected = False

        # Audio streaming
        # IMPORTANT: append_audio() is called from the sounddevice callback thread.
        # It must be fast and non-blocking: no socket I/O or heavy resampling work here.
        self._audio_queue = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self.input_sample_rate = 16000
        self.sample_rate = 16000

        self._sender_thread = None
        self._sender_running = False
        self._dropped_chunks = 0
        self._last_drop_log_time = 0.0

        # Track whether new audio has been queued since the last received transcript.
        # This helps avoid returning stale mid-stream text on stop.
        self._audio_activity_id = 0
        self._last_transcript_audio_activity_id = 0

        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()
        self._last_audio_chunk_time = 0.0

    def _log(self, msg: str):
        print(f'{self.LOG_TAG} {msg}', flush=True)

    def _transport_ready(self) -> bool:
        """Whether audio can currently be queued/sent. Subclasses override."""
        return self.connected

    def _on_audio_chunk_locked(self):
        """Hook called (under lock) for each chunk entering append_audio."""
        self._last_audio_chunk_time = time.time()

    def _is_idle_close(self) -> bool:
        """Whether a close arrived with no recent audio (idle session)."""
        last = float(self._last_audio_chunk_time or 0.0)
        return (not last) or ((time.time() - last) > self.IDLE_CLOSE_SECS)

    def set_input_sample_rate(self, sample_rate: int):
        """Set the capture rate for incoming AudioCapture chunks."""
        try:
            sample_rate = int(sample_rate)
        except (TypeError, ValueError):
            return
        if sample_rate > 0:
            self.input_sample_rate = sample_rate

    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds for backpressure handling"""
        self.max_buffer_seconds = max(1.0, seconds)  # Minimum 1 second

    def _resample_for_output(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Resample queued capture audio to the provider's configured output rate."""
        return resample_audio(audio_chunk, self.input_sample_rate, self.sample_rate)

    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes (little-endian)"""
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to the outgoing stream.

        Drops OLDEST queued chunks when the backlog exceeds max_buffer_seconds,
        capping worst-case latency while allowing arbitrarily long recordings.

        Args:
            audio_chunk: NumPy array of native-rate audio samples (float32, mono)
        """
        if not self._transport_ready():
            return

        drop_msg = None
        with self.lock:
            self._on_audio_chunk_locked()
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
                    f'{self.LOG_TAG} Dropping audio chunk(s) (queued>{self.max_buffer_seconds:.1f}s). '
                    f'dropped_chunks={self._dropped_chunks}'
                )
                self._last_drop_log_time = now

        if drop_msg:
            print(drop_msg, flush=True)


class WebSocketRealtimeClientBase(RealtimeAudioClientBase):
    """Transport + commit machinery for websocket-client based realtime providers."""

    def __init__(self, mode: str = 'transcribe'):
        """
        Args:
            mode: 'transcribe' for speech-to-text, 'converse' for voice-to-AI
        """
        super().__init__()
        try:
            import websocket as websocket_transport
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                'websocket-client is required for this realtime provider; '
                'install requirements-realtime.txt'
            ) from exc
        self._websocket_transport = websocket_transport
        self.ws = None
        self.url = None
        self.instructions = None
        self.mode = mode

        # Connection state
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
        self._partial_transcript = ""

    # ------------------------------------------------------------------
    # Provider hooks
    # ------------------------------------------------------------------

    def _ws_connect_params(self):
        """Return (url, headers-dict-or-None) for the WebSocket connection."""
        return self.url, None

    def _prepare_connect(self):
        """Hook called at the start of each connection attempt."""

    def _on_connect_success(self):
        """Hook called after the connection is established."""

    def _after_open(self):
        """Hook called from _on_open after the receiver thread is started."""

    def _audio_ws_message(self, base64_audio: str) -> dict:
        """Build the provider-specific JSON message for one audio chunk."""
        raise NotImplementedError

    def _handle_event(self, event: dict):
        """Handle a single event from the server."""
        raise NotImplementedError

    def _on_commit_fast_path_locked(self):
        """Hook (under lock) when commit returns an already-final transcript."""

    def _capture_commit_context_locked(self) -> dict:
        """Hook (under lock) to capture provider state before committing."""
        return {}

    def _request_transcript(self, ctx: dict):
        """Signal end of input to the server (commit / turn complete)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _transport_ready(self) -> bool:
        return bool(self.connected and self.ws)

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
                resampled = self._resample_for_output(audio_chunk)
                pcm_bytes = self._float32_to_pcm16(resampled)
                base64_audio = base64.b64encode(pcm_bytes).decode('utf-8')
                ws.send(json.dumps(self._audio_ws_message(base64_audio)))
            except Exception as e:
                self._log(f'Failed to send queued audio: {e}')

    def connect(self, url: str, api_key: str, model: str, instructions: Optional[str] = None) -> bool:
        """
        Establish WebSocket connection with authentication.

        Args:
            url: WebSocket URL
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
        with self.lock:
            if self.connecting:
                return False
            self.connecting = True

        self._prepare_connect()
        # This attempt's socket. On failure, close only this — self.ws may
        # already belong to a newer attempt started by another thread.
        attempt_ws = None

        try:
            target_url, headers = self._ws_connect_params()
            self._log(f'Connecting to {self.url}...')

            kwargs = {}
            if headers:
                kwargs['header'] = [f'{k}: {v}' for k, v in headers.items()]
            attempt_ws = self._websocket_transport.WebSocketApp(
                target_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                **kwargs,
            )
            with self.lock:
                self.ws = attempt_ws

            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=attempt_ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for connection (with timeout)
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if self.connected:
                self._log('Connected successfully')
                self.reconnect_attempts = 0
                self._on_connect_success()
                return True
            else:
                self._log('Connection timeout')
                self._abandon_attempt(attempt_ws)
                return False

        except Exception as e:
            self._log(f'Connection error: {e}')
            self._abandon_attempt(attempt_ws)
            return False
        finally:
            with self.lock:
                self.connecting = False

    def _abandon_attempt(self, attempt_ws):
        """Close a failed connection attempt's socket without touching a newer one."""
        if attempt_ws is None:
            return
        try:
            attempt_ws.close()
        except Exception:
            pass
        with self.lock:
            if self.ws is attempt_ws:
                self.ws = None

    def _on_open(self, _ws):
        """WebSocket connection opened"""
        start_receiver = False
        with self.lock:
            if not self.receiver_running:
                self.receiver_running = True
                start_receiver = True

        if start_receiver:
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

        self._after_open()

    def _on_message(self, _ws, message):
        """Handle incoming WebSocket message"""
        try:
            event = json.loads(message)
            self.event_queue.put(event)
        except json.JSONDecodeError as e:
            self._log(f'Failed to parse event: {e}')

    def _on_error(self, _ws, error):
        """Handle WebSocket error"""
        self._log(f'WebSocket error: {error}')

    def _on_close(self, _ws, close_status_code, _close_msg):
        """Handle WebSocket close"""
        with self.lock:
            was_connected = self.connected
            self.connected = False
            # Stop sender thread on disconnect; it will be restarted on next connect.
            # This prevents it from waiting indefinitely after an unexpected close.
            self._sender_running = False
            # Drop queued audio on disconnect to avoid sending stale audio after reconnect
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()

        self._log(f'WebSocket closed (code: {close_status_code})')

        # Reconnect only on an unexpected mid-recording close; idle closes are
        # recovered on-demand at the next recording start
        if self.receiver_running and was_connected and close_status_code != 1000:
            if self._is_idle_close():
                self._log('Connection closed (idle) - will reconnect on next recording')
            else:
                self._log('Connection lost unexpectedly')
                threading.Thread(target=self._attempt_reconnect, daemon=True).start()

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
                self._log(f'Error in receiver loop: {e}')

    def _attempt_reconnect(self):
        """Attempt reconnection with exponential backoff (one loop at a time)."""
        if not self._reconnect_lock.acquire(blocking=False):
            return False
        try:
            self._reconnecting = True
            while self.reconnect_attempts < self.max_reconnect_attempts:
                delay = self.reconnect_delays[min(self.reconnect_attempts, len(self.reconnect_delays) - 1)]
                self.reconnect_attempts += 1
                self._log(f'Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s...')
                time.sleep(delay)

                # Abort if the client was closed while we were waiting
                if not self.receiver_running:
                    self._log('Reconnection cancelled (closing)')
                    return False

                if self._connect_internal():
                    return True

            self._log('Max reconnection attempts reached')
            return False
        finally:
            self._reconnecting = False
            self._reconnect_lock.release()

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

        self._log('Connection closed')

    # ------------------------------------------------------------------
    # Transcript assembly / commit
    # ------------------------------------------------------------------

    def _full_committed_text_locked(self) -> str:
        parts = [p for p in self._committed_segments if p]
        return ' '.join(parts).strip()

    def _reset_stream_state_locked(self):
        """Reset per-recording state (call under lock)."""
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
        self._last_audio_chunk_time = 0.0
        self._queue_cond.notify_all()

    def clear_audio_buffer(self):
        """Clear state before starting a new recording"""
        with self.lock:
            self._reset_stream_state_locked()
        self.response_event.clear()

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
            self._log('Not connected, cannot commit')
            return ""

        try:
            with self.lock:
                existing_generation = self._transcript_generation
                existing_transcript = (
                    self._full_committed_text_locked() if self.mode == 'transcribe' else ""
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
                    self._on_commit_fast_path_locked()
                    self._log(f'Using existing transcript ({len(result)} chars)')
                    return result

                # Reset response state for manual commit flow
                self.current_response_text = ""
                self.response_complete = False
                self.response_event.clear()

                ctx = self._capture_commit_context_locked()

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

            self._request_transcript(ctx)

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
                            best_text = self._full_committed_text_locked()

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
                                best_text = self._full_committed_text_locked()
                        self.response_event.clear()

                    result = (best_text or "").strip()
                    with self.lock:
                        self._committed_segments = []
                        self._transcript_generation = 0
                        self.audio_buffer_seconds = 0.0
                    self._log(f'Transcript received ({len(result)} chars)')
                    return result

                self._log(f'Timeout waiting for transcript ({timeout}s)')
                with self.lock:
                    fallback = self._full_committed_text_locked()
                    self._committed_segments = []
                    self._transcript_generation = 0
                return (fallback or "").strip()

            # converse mode: legacy response_event semantics
            if self.response_event.wait(timeout=timeout):
                if self.response_complete:
                    result = self.current_response_text.strip()
                    self._log(f'Response received ({len(result)} chars)')
                    self.audio_buffer_seconds = 0.0
                    return result

                self._log('Event set but response not complete')
                return ""

            self._log(f'Timeout waiting for response ({timeout}s)')
            return ""

        except Exception as e:
            self._log(f'Error in commit_and_get_text: {e}')
            return ""
