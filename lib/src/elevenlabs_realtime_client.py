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

        # Async event loop (runs in background thread)
        self._loop = None
        self._loop_thread = None

        # Transcript handling
        self._transcript_event = threading.Event()
        self._current_transcript = ''
        self._partial_transcript = ''

        # Threading
        self.lock = threading.Lock()

        # Reconnection
        self._should_stay_alive = False  # Set True after connect, False on close()
        self._reconnecting = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff

        # Buffer tracking
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0

    def _start_event_loop(self):
        """Start the asyncio event loop in a background thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

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
            if self._loop is None:
                self._loop_thread = threading.Thread(
                    target=self._start_event_loop, daemon=True
                )
                self._loop_thread.start()

                # Wait for loop to start
                time.sleep(0.1)

            # Initialize SDK client
            self._elevenlabs = _ElevenLabs(api_key=api_key)

            result = self._connect_internal()
            if result:
                self._should_stay_alive = True
                self.reconnect_attempts = 0
            return result

        except Exception as e:
            print(f'[ELEVENLABS] Connection error: {e}', flush=True)
            import traceback

            traceback.print_exc()
            return False

    def _connect_internal(self) -> bool:
        """Internal connection logic, reusable for reconnection."""
        if self._reconnecting:
            return False

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
                    options.language_code = self.language

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
                    self._transcript_event.set()

                def on_error(error):
                    print(f'[ELEVENLABS] Error: {error}', flush=True)
                    self._transcript_event.set()

                def on_close():
                    with self.lock:
                        was_connected = self.connected
                        self.connected = False

                    if self._should_stay_alive and was_connected:
                        print(
                            '[ELEVENLABS] Connection lost unexpectedly',
                            flush=True,
                        )
                        self._attempt_reconnect()
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

    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self._reconnecting:
            return
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(
                '[ELEVENLABS] Max reconnection attempts reached',
                flush=True,
            )
            return

        self._reconnecting = True
        try:
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
        finally:
            self._reconnecting = False

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
        self._transcript_event.clear()
        self.audio_buffer_seconds = 0.0

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to stream.

        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        if not self.connected or not self._connection:
            return

        try:
            base64_audio = self._float32_to_pcm16_base64(audio_chunk)

            async def _send():
                await self._connection.send(
                    {'audio_base_64': base64_audio, 'sample_rate': self.sample_rate}
                )

            self._run_async(_send())

            # Track buffer size
            chunk_duration = len(audio_chunk) / self.sample_rate
            self.audio_buffer_seconds += chunk_duration
            if self.audio_buffer_seconds > self.max_buffer_seconds:
                self.audio_buffer_seconds = 0.0

        except Exception as e:
            print(f'[ELEVENLABS] Failed to append audio: {e}', flush=True)

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
            # Check if we already have a transcript
            with self.lock:
                if self._current_transcript:
                    result = self._current_transcript.strip()
                    self._current_transcript = ''
                    self._transcript_event.clear()
                    self.audio_buffer_seconds = 0.0
                    print(
                        f'[ELEVENLABS] Using existing transcript ({len(result)} chars)',
                        flush=True,
                    )
                    return result

            self._transcript_event.clear()

            # Send commit
            async def _commit():
                await self._connection.commit()

            try:
                self._run_async(_commit())
                print('[ELEVENLABS] Sent commit, waiting for transcript...', flush=True)
            except Exception as e:
                print(f'[ELEVENLABS] Failed to send commit: {e}', flush=True)

            # Wait for transcript
            if self._transcript_event.wait(timeout=timeout):
                with self.lock:
                    result = self._current_transcript.strip()
                    self._current_transcript = ''
                    self.audio_buffer_seconds = 0.0
                    print(
                        f'[ELEVENLABS] Transcript received ({len(result)} chars)',
                        flush=True,
                    )
                    return result
            else:
                print(
                    f'[ELEVENLABS] Timeout waiting for transcript ({timeout}s)',
                    flush=True,
                )
                # Return partial if available
                with self.lock:
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

        if self._connection:
            try:

                async def _close():
                    await self._connection.close()

                self._run_async(_close())
            except Exception:
                pass

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)

        with self.lock:
            self.connected = False

        print('[ELEVENLABS] Connection closed', flush=True)

    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds"""
        self.max_buffer_seconds = max(1.0, seconds)
