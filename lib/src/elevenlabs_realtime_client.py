"""
ElevenLabs Scribe v2 Realtime client using official SDK.
Provides streaming speech-to-text using ElevenLabs' realtime API.
"""

import sys
import base64
import asyncio
import threading
import time
from typing import Optional
from collections import deque

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print(
        'ERROR: python-numpy is not available in this Python environment.',
        file=sys.stderr,
    )
    print(f'ImportError: {e}', file=sys.stderr)
    sys.exit(1)


class ElevenLabsRealtimeClient:
    """Client for ElevenLabs Scribe v2 Realtime transcription using official SDK"""

    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self):
        """Initialize ElevenLabs realtime client"""
        self.api_key = None
        self.model = 'scribe_v2_realtime'
        self.language = None
        self.sample_rate = self.DEFAULT_SAMPLE_RATE

        # Connection state
        self.connected = False
        self._connection = None
        self._elevenlabs = None
        self._connecting = False

        # Async event loop (runs in background thread)
        self._loop = None
        self._loop_thread = None

        # Transcript handling
        self._transcript_event = threading.Event()
        self._current_transcript = ''
        self._partial_transcript = ''
        self._transcript_generation = 0
        self._committed_segments = []

        # Track whether new audio has been queued since last committed transcript.
        # This helps avoid returning a stale mid-stream transcript on stop.
        self._audio_activity_id = 0
        self._last_transcript_audio_activity_id = 0

        # Threading
        self.lock = threading.Lock()
        self._queue_cond = threading.Condition(self.lock)
        self._sender_thread = None
        self._sender_running = False

        # Auto-commit helper:
        # ElevenLabs punctuation often improves on committed transcripts.
        # VAD commit can be too "slow" for continuous dictation, so we also
        # commit after a short detected pause (based on chunk timing) once the
        # send queue is drained.
        self._auto_commit_thread = None
        self._auto_commit_running = False
        self._auto_commit_last_time = 0.0
        self._auto_commit_silence_secs = 0.45
        self._auto_commit_min_interval_secs = 2.0
        self._last_audio_chunk_time = 0.0

        # Reconnection
        self._should_stay_alive = False  # Set True after connect, False on close()
        self._reconnecting = False
        self._reconnect_lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff

        # Buffer tracking
        # NOTE: For ElevenLabs, this tracks UNSENT backlog in seconds (queue depth),
        # not "total recorded since last commit". This allows arbitrarily long recordings
        # while still capping worst-case latency/backpressure.
        self._audio_queue = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self._last_drop_log_time = 0.0
        self._dropped_chunks = 0

    def _start_sender_thread(self):
        """Start background sender thread (once)"""
        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._sender_running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

    def _start_auto_commit_thread(self):
        """Start background auto-commit thread (once)."""
        if self._auto_commit_thread and self._auto_commit_thread.is_alive():
            return
        self._auto_commit_running = True
        self._auto_commit_thread = threading.Thread(
            target=self._auto_commit_loop, daemon=True
        )
        self._auto_commit_thread.start()

    def _auto_commit_loop(self):
        """Background thread: commit after a short pause to improve punctuation/latency."""
        while True:
            time.sleep(0.15)

            with self.lock:
                if not self._auto_commit_running:
                    return

                if not (self.connected and self._connection):
                    continue

                # Don't commit while audio is still queued (we want commit boundaries
                # to reflect what we've already sent).
                if len(self._audio_queue) > 0:
                    continue

                now = time.time()

                # Only commit if there has been new audio since the last committed transcript.
                if self._audio_activity_id == self._last_transcript_audio_activity_id:
                    continue

                # Require a short "quiet" window since last audio chunk was queued.
                if self._last_audio_chunk_time and (now - self._last_audio_chunk_time) < self._auto_commit_silence_secs:
                    continue

                # Avoid rapid-fire commits (docs warn this can degrade performance).
                if self._auto_commit_last_time and (now - self._auto_commit_last_time) < self._auto_commit_min_interval_secs:
                    continue

                connection = self._connection
                self._auto_commit_last_time = now

            try:
                async def _commit():
                    await connection.commit()

                self._run_async(_commit())
                # Intentionally quiet: commit can happen frequently during natural pauses.
            except Exception:
                pass

    def _sender_loop(self):
        """Background thread: drain queued audio and send over the connection."""
        while True:
            with self.lock:
                # Wait until we have work, are connected, or are shutting down
                self._queue_cond.wait_for(
                    lambda: (not self._sender_running)
                    or (self.connected and self._connection and len(self._audio_queue) > 0)
                )

                if not self._sender_running:
                    return

                # If we're not connected, just keep waiting
                if not (self.connected and self._connection):
                    continue

                # Pop one chunk to send (oldest first)
                audio_chunk = self._audio_queue.popleft()
                chunk_duration = len(audio_chunk) / self.sample_rate
                self.audio_buffer_seconds = max(0.0, self.audio_buffer_seconds - chunk_duration)
                connection = self._connection

                # Notify anyone waiting for drain
                if not self._audio_queue:
                    self._queue_cond.notify_all()

            try:
                base64_audio = self._float32_to_pcm16_base64(audio_chunk)

                async def _send():
                    await connection.send(
                        {'audio_base_64': base64_audio, 'sample_rate': self.sample_rate}
                    )

                self._run_async(_send())

            except Exception as e:
                # Drop chunk on send failure (backpressure is best-effort)
                print(f'[ELEVENLABS] Failed to send queued audio: {e}', flush=True)

    def _start_event_loop(self):
        """Start the asyncio event loop in a background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_event_loop_running(self):
        """Ensure the background asyncio loop is running (restart if needed)."""
        try:
            if self._loop is not None and self._loop.is_running():
                return
        except Exception:
            # If loop is in a weird state, just recreate it.
            pass

        # If we previously stopped the loop, discard and recreate.
        self._loop = None
        self._loop_thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._loop_thread.start()
        # Give the loop thread a moment to start.
        time.sleep(0.1)

    def _run_async(self, coro):
        """Run an async coroutine from sync code"""
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError('Event loop not running')
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def connect(
        self, url: str, api_key: str, model: str, instructions: Optional[str] = None
    ) -> bool:
        """
        Establish connection using ElevenLabs SDK.

        Args:
            url: WebSocket URL (ignored - SDK handles this)
            api_key: ElevenLabs API key
            model: Model identifier (e.g., 'scribe_v2_realtime')
            instructions: Ignored for ElevenLabs

        Returns:
            True if connection successful, False otherwise
        """
        self.api_key = api_key
        self.model = model

        try:
            # Import SDK
            try:
                from elevenlabs import ElevenLabs as _ElevenLabs
            except ImportError:
                print(
                    '[ELEVENLABS] SDK not installed. Install with: pip install elevenlabs',
                    flush=True,
                )
                return False

            # Start event loop in background thread (once)
            self._ensure_event_loop_running()

            # Initialize SDK client
            self._elevenlabs = _ElevenLabs(api_key=api_key)

            result = self._connect_internal()
            if result:
                self._should_stay_alive = True
                self.reconnect_attempts = 0
                self._start_sender_thread()
                self._start_auto_commit_thread()
            return result

        except Exception as e:
            print(f'[ELEVENLABS] Connection error: {e}', flush=True)
            import traceback

            traceback.print_exc()
            return False

    def _connect_internal(self) -> bool:
        """Internal connection logic, reusable for reconnection."""
        with self.lock:
            if self._connecting:
                return False
            self._connecting = True

        try:
            from elevenlabs import (
                RealtimeEvents,
                RealtimeAudioOptions,
                AudioFormat,
                CommitStrategy,
            )

            # Connect asynchronously
            async def _do_connect():
                # Build options
                options = RealtimeAudioOptions(
                    model_id=self.model,
                    audio_format=AudioFormat.PCM_16000,
                    sample_rate=16000,
                    commit_strategy=CommitStrategy.VAD,
                )

                if self.language:
                    options['language_code'] = self.language

                connection = await self._elevenlabs.speech_to_text.realtime.connect(
                    options
                )

                # Set up event handlers
                def on_session_started(data):
                    print(
                        f'[ELEVENLABS] Session started: {data.get("session_id", "unknown")}',
                        flush=True,
                    )
                    with self.lock:
                        self.connected = True

                def on_partial_transcript(data):
                    text = data.get('text', '')
                    with self.lock:
                        self._partial_transcript = text

                def on_committed_transcript(data):
                    text = data.get('text', '')
                    print(
                        f'[ELEVENLABS] Committed transcript ({len(text)} chars)',
                        flush=True,
                    )
                    with self.lock:
                        self._current_transcript = text
                        self._partial_transcript = ''
                        if text and text.strip():
                            self._committed_segments.append(text.strip())
                        self._transcript_generation += 1
                        self._last_transcript_audio_activity_id = self._audio_activity_id
                    self._transcript_event.set()

                def on_committed_transcript_with_timestamps(data):
                    text = data.get('text', '')
                    print(
                        f'[ELEVENLABS] Committed transcript with timestamps ({len(text)} chars)',
                        flush=True,
                    )
                    with self.lock:
                        self._current_transcript = text
                        self._partial_transcript = ''
                        if text and text.strip():
                            self._committed_segments.append(text.strip())
                        self._transcript_generation += 1
                        self._last_transcript_audio_activity_id = self._audio_activity_id
                    self._transcript_event.set()

                def on_error(error):
                    print(f'[ELEVENLABS] Error: {error}', flush=True)
                    self._transcript_event.set()

                def on_close():
                    with self.lock:
                        was_connected = self.connected
                        self.connected = False
                        self._connection = None
                        # Clear any queued audio on disconnect to avoid sending stale audio after reconnect
                        self._audio_queue.clear()
                        self.audio_buffer_seconds = 0.0
                        self._queue_cond.notify_all()

                    # If the server closes the socket while we're idle (not recording),
                    # don't flap-reconnect in a tight loop. We'll reconnect on-demand
                    # when the next recording starts.
                    now = time.time()
                    last_audio = float(self._last_audio_chunk_time or 0.0)
                    idle_for = (now - last_audio) if last_audio else None
                    idle_close = (idle_for is None) or (idle_for > 10.0)

                    if self._should_stay_alive and was_connected and (not idle_close):
                        print(
                            '[ELEVENLABS] Connection lost unexpectedly',
                            flush=True,
                        )
                        # Avoid blocking the SDK callback thread (and potentially the asyncio loop).
                        threading.Thread(
                            target=self._attempt_reconnect,
                            daemon=True,
                        ).start()
                    elif self._should_stay_alive and was_connected and idle_close:
                        print('[ELEVENLABS] Connection closed (idle)', flush=True)
                    else:
                        print('[ELEVENLABS] Connection closed', flush=True)

                connection.on(RealtimeEvents.SESSION_STARTED, on_session_started)
                connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial_transcript)
                connection.on(
                    RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed_transcript
                )
                connection.on(
                    RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
                    on_committed_transcript_with_timestamps,
                )
                connection.on(RealtimeEvents.ERROR, on_error)
                connection.on(RealtimeEvents.CLOSE, on_close)

                return connection

            self._connection = self._run_async(_do_connect())

            # Wait for session_started
            timeout = 10.0
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            if self.connected:
                print('[ELEVENLABS] Connected successfully', flush=True)
                return True
            else:
                print('[ELEVENLABS] Connection timeout', flush=True)
                return False

        except Exception as e:
            print(f'[ELEVENLABS] Connection error: {e}', flush=True)
            import traceback

            traceback.print_exc()
            return False
        finally:
            with self.lock:
                self._connecting = False

    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        # Ensure only one reconnect attempt runs at a time
        if not self._reconnect_lock.acquire(blocking=False):
            return
        try:
            if self._reconnecting:
                return
            self._reconnecting = True

            while self.reconnect_attempts < self.max_reconnect_attempts:
                delay = self.reconnect_delays[
                    min(self.reconnect_attempts, len(self.reconnect_delays) - 1)
                ]
                self.reconnect_attempts += 1

                print(
                    f'[ELEVENLABS] Reconnecting (attempt {self.reconnect_attempts}/'
                    f'{self.max_reconnect_attempts}) in {delay}s...',
                    flush=True,
                )
                time.sleep(delay)

                if not self._should_stay_alive:
                    print('[ELEVENLABS] Reconnection cancelled (closing)', flush=True)
                    return

                if self._connect_internal():
                    self.reconnect_attempts = 0
                    print('[ELEVENLABS] Reconnected successfully', flush=True)
                    return

            print('[ELEVENLABS] Max reconnection attempts reached', flush=True)
        finally:
            self._reconnecting = False
            try:
                self._reconnect_lock.release()
            except Exception:
                pass

    def _float32_to_pcm16_base64(self, audio_data: np.ndarray) -> str:
        """Convert float32 numpy array to base64-encoded PCM16"""
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

    def update_language(self, language: Optional[str]):
        """Update the language for transcription (applies on next connection)"""
        self.language = language
        print(f'[ELEVENLABS] Language set to: {language or "auto-detect"}', flush=True)

    def clear_audio_buffer(self):
        """Clear state before starting a new recording"""
        with self.lock:
            self._current_transcript = ''
            self._partial_transcript = ''
            self._transcript_generation = 0
            self._committed_segments = []
            self._audio_activity_id = 0
            self._last_transcript_audio_activity_id = 0
            self._last_audio_chunk_time = 0.0
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._dropped_chunks = 0
            self._last_drop_log_time = 0.0
            self._queue_cond.notify_all()
        self._transcript_event.clear()

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to stream.

        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        # IMPORTANT: this is called from the sounddevice callback thread.
        # It must be fast and non-blocking.
        drop_msg = None
        with self.lock:
            if not self.connected or not self._connection:
                return

            chunk_duration = len(audio_chunk) / self.sample_rate
            self._last_audio_chunk_time = time.time()

            # Drop OLDEST queued chunks until the new chunk fits.
            # This caps worst-case latency while still allowing arbitrarily long recordings.
            while (self.audio_buffer_seconds + chunk_duration) > self.max_buffer_seconds and self._audio_queue:
                dropped = self._audio_queue.popleft()
                dropped_duration = len(dropped) / self.sample_rate
                self.audio_buffer_seconds = max(0.0, self.audio_buffer_seconds - dropped_duration)
                self._dropped_chunks += 1

            # If we still can't fit (e.g., max_buffer_seconds too small for one chunk), drop this chunk.
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
                    f'[ELEVENLABS] Dropping audio chunk(s) (queued>{self.max_buffer_seconds:.1f}s). '
                    f'dropped_chunks={self._dropped_chunks}'
                )
                self._last_drop_log_time = now

        if drop_msg:
            print(drop_msg, flush=True)

    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Commit and wait for transcription result.

        Args:
            timeout: Maximum time to wait for transcription (seconds)

        Returns:
            Final transcript text, or empty string on timeout/error
        """
        if not self.connected or not self._connection:
            print('[ELEVENLABS] Not connected', flush=True)
            return ''

        try:
            def _full_committed_text_locked() -> str:
                """Return concatenated committed transcript for current recording."""
                parts = [p for p in self._committed_segments if p]
                # Use single spaces to stitch segments; preserve order.
                return ' '.join(parts).strip()

            # If we already have a committed transcript AND there is no new queued audio since then,
            # we can return immediately (common case: server VAD committed before user stops).
            with self.lock:
                existing_transcript = _full_committed_text_locked()
                existing_generation = self._transcript_generation
                has_new_audio_since_transcript = (
                    self._audio_activity_id != self._last_transcript_audio_activity_id
                )
                has_queued_audio = len(self._audio_queue) > 0

                if existing_transcript and (not has_new_audio_since_transcript) and (not has_queued_audio):
                    result = existing_transcript
                    self._current_transcript = ''
                    self._committed_segments = []
                    self._transcript_event.clear()
                    self.audio_buffer_seconds = 0.0
                    print(
                        f'[ELEVENLABS] Using existing transcript ({len(result)} chars)',
                        flush=True,
                    )
                    return result

            self._transcript_event.clear()

            # Best-effort: wait for queued audio to drain before committing.
            # This reduces "missing last words" by ensuring the tail audio was sent.
            with self.lock:
                queued_seconds = float(self.audio_buffer_seconds)
                max_backlog = float(self.max_buffer_seconds)

            # Queue is capped at max_buffer_seconds, so waiting up to ~max_backlog+1s is safe.
            drain_timeout = min(max_backlog + 1.0, max(0.5, timeout * 0.5, queued_seconds + 0.25))
            with self.lock:
                self._queue_cond.wait_for(lambda: len(self._audio_queue) == 0, timeout=drain_timeout)

            # Small grace to allow any in-flight send to reach the server before commit.
            time.sleep(0.05)

            # Send commit
            async def _commit():
                await self._connection.commit()

            try:
                self._run_async(_commit())
                print('[ELEVENLABS] Sent commit, waiting for transcript...', flush=True)
            except Exception as e:
                print(f'[ELEVENLABS] Failed to send commit: {e}', flush=True)

            # Wait for transcript
            deadline = time.time() + max(0.0, timeout)
            while time.time() < deadline:
                remaining = max(0.0, deadline - time.time())
                if not self._transcript_event.wait(timeout=remaining):
                    break

                with self.lock:
                    # Prefer a transcript that is newer than what we had when we stopped.
                    if self._transcript_generation > existing_generation:
                        # "Settle" briefly: ElevenLabs may emit one more committed update shortly after stop
                        # (often where punctuation gets finalized). We wait for a short quiet window and
                        # return the latest committed text we see.
                        best_generation = self._transcript_generation
                        best_text = _full_committed_text_locked()

                    # If we didn't have an existing transcript, accept the first one we get.
                    if existing_generation == 0 and self._transcript_generation > 0:
                        best_generation = self._transcript_generation
                        best_text = _full_committed_text_locked()

                    if 'best_generation' in locals():
                        settle_deadline = min(deadline, time.time() + 0.6)
                        while time.time() < settle_deadline:
                            settle_remaining = max(0.0, settle_deadline - time.time())
                            # Wait for another committed transcript (or time out and return the best we have)
                            self._transcript_event.clear()
                            if not self._transcript_event.wait(timeout=settle_remaining):
                                break
                            # Incorporate any newer commit
                            if self._transcript_generation > best_generation:
                                best_generation = self._transcript_generation
                                best_text = _full_committed_text_locked()

                        result = best_text
                        self._current_transcript = ''
                        self._committed_segments = []
                        self.audio_buffer_seconds = 0.0
                        print(
                            f'[ELEVENLABS] Transcript received ({len(result)} chars)',
                            flush=True,
                        )
                        return result

                # Event may have been set by an error or a stale callback; keep waiting.
                self._transcript_event.clear()

            print(
                f'[ELEVENLABS] Timeout waiting for transcript ({timeout}s)',
                flush=True,
            )

            # Fallback: return latest committed transcript if present, else partial, else empty.
            with self.lock:
                full_text = _full_committed_text_locked()
                if full_text:
                    result = full_text
                    self._current_transcript = ''
                    self._committed_segments = []
                    return result
                if self._partial_transcript:
                    result = self._partial_transcript.strip()
                    self._partial_transcript = ''
                    print(
                        f'[ELEVENLABS] Returning partial ({len(result)} chars)',
                        flush=True,
                    )
                    return result
            return ''

        except Exception as e:
            print(f'[ELEVENLABS] Error in commit_and_get_text: {e}', flush=True)
            return ''

    def close(self):
        """Close connection and cleanup"""
        # Signal that this is an intentional shutdown — prevents on_close from reconnecting
        self._should_stay_alive = False
        with self.lock:
            self._sender_running = False
            self._auto_commit_running = False
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()

        if self._connection:
            try:
                # Only attempt async close if the loop is actually running;
                # otherwise we can end up with "coroutine was never awaited" warnings.
                if self._loop is not None and self._loop.is_running():
                    async def _close():
                        await self._connection.close()

                    self._run_async(_close())
            except Exception:
                pass

        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)

        if self._auto_commit_thread and self._auto_commit_thread.is_alive():
            self._auto_commit_thread.join(timeout=1.0)

        with self.lock:
            self.connected = False
            self._connection = None
            # Drop loop references so a later connect() can restart cleanly.
            self._loop = None
            self._loop_thread = None

        print('[ELEVENLABS] Connection closed', flush=True)

    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds"""
        self.max_buffer_seconds = max(1.0, seconds)
