"""
ElevenLabs Scribe v2 Realtime WebSocket client.
Provides streaming speech-to-text using ElevenLabs' realtime API.
"""

import sys
import json
import base64
import threading
import time
from typing import Optional
from queue import Queue, Empty

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print(
        "ERROR: python-numpy is not available in this Python environment.",
        file=sys.stderr,
    )
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import websocket
except (ImportError, ModuleNotFoundError) as e:
    print(
        "ERROR: websocket-client is not available in this Python environment.",
        file=sys.stderr,
    )
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pip install websocket-client>=1.6.0", file=sys.stderr)
    sys.exit(1)


class ElevenLabsRealtimeClient:
    """WebSocket client for ElevenLabs Scribe v2 Realtime transcription API"""

    # ElevenLabs supports: pcm_8000, pcm_16000, pcm_22050, pcm_24000, pcm_44100, pcm_48000
    # Default to 16kHz which matches hyprwhspr's audio capture rate (no resampling needed)
    DEFAULT_SAMPLE_RATE = 16000

    def __init__(self):
        """Initialize ElevenLabs realtime client"""
        self.ws = None
        self.api_key = None
        self.model = "scribe_v2_realtime"
        self.language = None  # Language code for transcription (None = auto-detect)

        # Audio settings - ElevenLabs uses 16kHz by default
        self.sample_rate = self.DEFAULT_SAMPLE_RATE

        # Connection state
        self.connected = False
        self.connecting = False
        self.session_id = None
        self.receiver_thread = None
        self.receiver_running = False

        # Event handling
        self.event_queue = Queue()
        self.response_event = threading.Event()
        self.current_response_text = ""
        self.partial_text = ""  # ElevenLabs provides partial transcripts
        self.response_complete = False

        # Audio streaming
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0

        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff

        # Threading
        self.lock = threading.Lock()

        # Track if we've received a committed transcript
        self._has_committed_transcript = False

    def connect(
        self, url: str, api_key: str, model: str, instructions: Optional[str] = None
    ) -> bool:
        """
        Establish WebSocket connection with authentication.

        Args:
            url: WebSocket URL (will be modified to add query params)
            api_key: ElevenLabs API key
            model: Model identifier (e.g., 'scribe_v2_realtime')
            instructions: Ignored for ElevenLabs (no instruction support)

        Returns:
            True if connection successful, False otherwise
        """
        self.api_key = api_key
        self.model = model

        # Build URL with query parameters (ElevenLabs uses query params, not session.update)
        # Base URL: wss://api.elevenlabs.io/v1/speech-to-text/realtime
        base_url = url.split("?")[0]  # Remove any existing query params

        params = [
            f"model_id={model}",
            f"audio_format=pcm_{self.sample_rate}",
            "commit_strategy=vad",  # Use VAD for automatic commit
            "vad_silence_threshold_secs=0.5",  # Commit after 0.5s of silence
            "vad_threshold=0.4",  # Voice activity detection sensitivity
        ]

        if self.language:
            params.append(f"language_code={self.language}")

        self.url = f"{base_url}?{'&'.join(params)}"

        return self._connect_internal()

    def _connect_internal(self) -> bool:
        """Internal connection logic with reconnection support"""
        if self.connecting:
            return False

        self.connecting = True

        try:
            # Prepare headers with authentication
            # ElevenLabs uses xi-api-key header instead of Authorization: Bearer
            headers = {"xi-api-key": self.api_key}

            print(f"[ELEVENLABS] Connecting to {self.url}...", flush=True)

            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.url,
                header=[f"{k}: {v}" for k, v in headers.items()],
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
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
                print(
                    f"[ELEVENLABS] Connected successfully (session: {self.session_id})",
                    flush=True,
                )
                self.reconnect_attempts = 0
                return True
            else:
                print(f"[ELEVENLABS] Connection timeout", flush=True)
                return False

        except Exception as e:
            print(f"[ELEVENLABS] Connection error: {e}", flush=True)
            return False
        finally:
            self.connecting = False

    def _on_open(self, _ws):
        """WebSocket connection opened"""
        # Note: ElevenLabs sends session_started event after connection
        # We'll set connected=True when we receive that event

        # Start receiver thread
        if not self.receiver_running:
            self.receiver_running = True
            self.receiver_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self.receiver_thread.start()

    def _on_message(self, _ws, message):
        """Handle incoming WebSocket message"""
        try:
            event = json.loads(message)
            self.event_queue.put(event)
        except json.JSONDecodeError as e:
            print(f"[ELEVENLABS] Failed to parse event: {e}", flush=True)

    def _on_error(self, _ws, error):
        """Handle WebSocket error"""
        print(f"[ELEVENLABS] WebSocket error: {error}", flush=True)

    def _on_close(self, _ws, close_status_code, _close_msg):
        """Handle WebSocket close"""
        with self.lock:
            self.connected = False

        print(f"[ELEVENLABS] WebSocket closed (code: {close_status_code})", flush=True)

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
                print(f"[ELEVENLABS] Error in receiver loop: {e}", flush=True)

    def _handle_event(self, event: dict):
        """Handle a single event from the server"""
        message_type = event.get("message_type", "")

        if message_type == "session_started":
            # Session established - now we're truly connected
            self.session_id = event.get("session_id")
            config = event.get("config", {})
            print(f"[ELEVENLABS] Session started: {self.session_id}", flush=True)
            print(
                f"[ELEVENLABS] Config: sample_rate={config.get('sample_rate')}, "
                f"commit_strategy={config.get('commit_strategy')}",
                flush=True,
            )
            with self.lock:
                self.connected = True
                self.connecting = False

        elif message_type == "partial_transcript":
            # Live/interim transcript update
            text = event.get("text", "")
            with self.lock:
                self.partial_text = text
            # Don't log every partial to avoid spam

        elif message_type == "committed_transcript":
            # Final transcript for this speech segment
            text = event.get("text", "")
            print(f"[ELEVENLABS] Committed transcript ({len(text)} chars)", flush=True)
            with self.lock:
                self.current_response_text = text
                self.partial_text = ""
                self.response_complete = True
                self._has_committed_transcript = True
            self.response_event.set()

        elif message_type == "committed_transcript_with_timestamps":
            # Final transcript with word-level timestamps
            text = event.get("text", "")
            words = event.get("words", [])
            language = event.get("language_code")
            print(
                f"[ELEVENLABS] Committed transcript with timestamps ({len(text)} chars, "
                f"{len(words)} words, lang={language})",
                flush=True,
            )
            with self.lock:
                self.current_response_text = text
                self.partial_text = ""
                self.response_complete = True
                self._has_committed_transcript = True
            self.response_event.set()

        # Error handling
        elif message_type in (
            "error",
            "auth_error",
            "quota_exceeded",
            "commit_throttled",
            "unaccepted_terms",
            "rate_limited",
            "queue_overflow",
            "resource_exhausted",
            "session_time_limit_exceeded",
            "input_error",
            "chunk_size_exceeded",
            "insufficient_audio_activity",
            "transcriber_error",
        ):
            error_msg = event.get("error", "Unknown error")
            print(f"[ELEVENLABS] Error ({message_type}): {error_msg}", flush=True)
            with self.lock:
                self.response_complete = True
            self.response_event.set()  # Unblock waiting thread

        else:
            # Unknown event type
            print(f"[ELEVENLABS] Unknown event type: {message_type}", flush=True)

    def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[ELEVENLABS] Max reconnection attempts reached", flush=True)
            return False

        delay = self.reconnect_delays[
            min(self.reconnect_attempts, len(self.reconnect_delays) - 1)
        ]
        self.reconnect_attempts += 1

        print(
            f"[ELEVENLABS] Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) in {delay}s...",
            flush=True,
        )
        time.sleep(delay)

        return self._connect_internal()

    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        """Convert float32 numpy array to PCM16 bytes"""
        # Clip to [-1, 1] range
        audio_clipped = np.clip(audio_data, -1.0, 1.0)

        # Convert to int16
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # Convert to bytes (little-endian)
        return audio_int16.tobytes()

    def update_language(self, language: Optional[str]):
        """Update the language for transcription.

        Note: For ElevenLabs, language is set via query params at connection time.
        This method stores the language for the next connection.

        Args:
            language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
        """
        self.language = language
        # ElevenLabs doesn't support changing language mid-session
        # Would need to reconnect to change language
        print(
            f"[ELEVENLABS] Language set to: {language or 'auto-detect'} (applies on next connection)",
            flush=True,
        )

    def clear_audio_buffer(self):
        """Clear state before starting a new recording.

        Note: ElevenLabs doesn't have an explicit buffer clear command.
        We just reset our local state.
        """
        with self.lock:
            self._has_committed_transcript = False
            self.current_response_text = ""
            self.partial_text = ""
            self.response_complete = False
        self.response_event.clear()
        self.audio_buffer_seconds = 0.0

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Append audio chunk to WebSocket stream.

        Args:
            audio_chunk: NumPy array of audio samples (float32, mono, 16kHz)
        """
        if not self.connected or not self.ws:
            return

        try:
            # Convert to PCM16
            pcm_bytes = self._float32_to_pcm16(audio_chunk)

            # Encode to base64
            base64_audio = base64.b64encode(pcm_bytes).decode("utf-8")

            # Send input_audio_chunk event (ElevenLabs format)
            event = {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64_audio,
                "commit": False,  # Let VAD handle commits
                "sample_rate": self.sample_rate,
            }

            self.ws.send(json.dumps(event))

            # Track buffer size
            chunk_duration = len(audio_chunk) / self.sample_rate
            self.audio_buffer_seconds += chunk_duration

            # Reset counter periodically
            if self.audio_buffer_seconds > self.max_buffer_seconds:
                self.audio_buffer_seconds = 0.0

        except Exception as e:
            print(f"[ELEVENLABS] Failed to append audio: {e}", flush=True)

    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Wait for transcription result.

        With VAD enabled, transcription happens automatically when speech ends.
        This method waits for the committed transcript.

        Args:
            timeout: Maximum time to wait for transcription (seconds)

        Returns:
            Final transcript text, or empty string on timeout/error
        """
        if not self.connected or not self.ws:
            print("[ELEVENLABS] Not connected, cannot get transcript", flush=True)
            return ""

        try:
            # Check if we already have a committed transcript
            with self.lock:
                if self._has_committed_transcript and self.current_response_text:
                    result = self.current_response_text.strip()
                    # Reset state for next recording
                    self.current_response_text = ""
                    self.response_complete = False
                    self._has_committed_transcript = False
                    self.response_event.clear()
                    self.audio_buffer_seconds = 0.0
                    print(
                        f"[ELEVENLABS] Using existing transcript ({len(result)} chars)",
                        flush=True,
                    )
                    return result

                # Reset response state
                self.response_complete = False
            self.response_event.clear()

            # Send a final commit to flush any remaining audio
            # This handles the case where VAD hasn't triggered yet
            try:
                commit_event = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",  # Empty audio
                    "commit": True,  # Force commit
                    "sample_rate": self.sample_rate,
                }
                self.ws.send(json.dumps(commit_event))
                print(
                    "[ELEVENLABS] Sent manual commit, waiting for transcript...",
                    flush=True,
                )
            except Exception as e:
                print(f"[ELEVENLABS] Failed to send commit: {e}", flush=True)

            # Wait for committed_transcript event
            if self.response_event.wait(timeout=timeout):
                with self.lock:
                    if self.response_complete:
                        result = self.current_response_text.strip()
                        print(
                            f"[ELEVENLABS] Transcript received ({len(result)} chars)",
                            flush=True,
                        )
                        # Reset for next recording
                        self.current_response_text = ""
                        self._has_committed_transcript = False
                        self.audio_buffer_seconds = 0.0
                        return result
                    else:
                        print(
                            "[ELEVENLABS] Event set but response not complete",
                            flush=True,
                        )
                        return ""
            else:
                print(
                    f"[ELEVENLABS] Timeout waiting for transcript ({timeout}s)",
                    flush=True,
                )
                # Return partial transcript if available
                with self.lock:
                    if self.partial_text:
                        result = self.partial_text.strip()
                        print(
                            f"[ELEVENLABS] Returning partial transcript ({len(result)} chars)",
                            flush=True,
                        )
                        self.partial_text = ""
                        return result
                return ""

        except Exception as e:
            print(f"[ELEVENLABS] Error in commit_and_get_text: {e}", flush=True)
            return ""

    def close(self):
        """Close WebSocket connection and cleanup"""
        self.receiver_running = False

        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=1.0)

        with self.lock:
            self.connected = False

        print("[ELEVENLABS] Connection closed", flush=True)

    def set_max_buffer_seconds(self, seconds: float):
        """Set maximum buffer size in seconds for backpressure handling"""
        self.max_buffer_seconds = max(1.0, seconds)  # Minimum 1 second
