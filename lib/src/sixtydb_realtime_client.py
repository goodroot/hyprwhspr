"""
60db Realtime STT client.

Streaming speech-to-text over 60db's WebSocket API
(wss://api.60db.ai/ws/stt). Provider-specific protocol, but exposes the same
interface as RealtimeClient / ElevenLabsRealtimeClient so whisper_manager can
drive it interchangeably:

    connect / append_audio / clear_audio_buffer / commit_and_get_text /
    update_language / set_max_buffer_seconds / close   (+ `connected`, `language`)

60db accepts 16kHz linear PCM natively, which is exactly what AudioCapture
produces, so no resampling is needed (same as ElevenLabs Scribe v2).

Protocol summary (see https://docs.60db.ai/websocket-api/stt):
  - Auth via ?apiKey=... query param on the socket URL.
  - Server emits {"connection_established": {...}} after auth.
  - Client sends {"type": "start", languages, config} to open a session.
  - Server replies {"type": "connected", ...} when ready.
  - Client streams {"type": "audio", "audio": <base64 int16 PCM>, ...}.
  - Server emits {"type": "transcription", text, is_final, speech_final, ...}.
    Interim results have is_final=False; finalized results have
    is_final=True AND speech_final=True.
  - Client sends {"type": "stop"} to flush and end the session.
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


class SixtyDbRealtimeClient:
    """WebSocket client for 60db realtime speech-to-text."""

    SAMPLE_RATE = 16000  # 60db supports 16kHz linear PCM natively (no resampling)
    VALID_AUDIO_ENHANCEMENTS = {'off', 'light', 'adaptive'}

    def __init__(self):
        self.ws = None
        self.url = None
        self.api_key = None
        self.model = None  # Unused by 60db STT WS (kept for interface parity)
        self.language = None  # ISO 639-1 code, or None for auto-detect
        self.partial_transcript_callback = None

        # 60db-specific tuning (set by whisper_manager from config before connect)
        self.diarize = False
        self.utterance_end_ms = 500
        self.audio_enhancement = 'adaptive'
        self.continuous_mode = True

        # 60db updates language by resending `config` mid-session, no reconnect needed.
        self.supports_mid_session_language_update = True

        # Threading
        self.lock = threading.Lock()

        # Connection state
        self.connected = False        # True once 60db sends {"type":"connected"}
        self._socket_open = False     # True once the WS transport is open
        self.connecting = False
        self.receiver_thread = None
        self.receiver_running = False

        # Event handling
        self.event_queue = Queue()
        self.response_event = threading.Event()

        # Transcription assembly
        self._transcript_generation = 0
        self._committed_segments = []
        self._partial_transcript = ""

        # Track whether new audio has been queued since the last received transcript,
        # so we don't return stale mid-stream text on stop.
        self._audio_activity_id = 0
        self._last_transcript_audio_activity_id = 0

        # Audio streaming. append_audio() runs on the sounddevice callback thread,
        # so it must be fast and non-blocking: just enqueue here, send elsewhere.
        self._audio_queue = deque()
        self.audio_buffer_seconds = 0.0
        self.max_buffer_seconds = 5.0
        self._dropped_chunks = 0
        self._last_drop_log_time = 0.0

        self._queue_cond = threading.Condition(self.lock)
        self._sender_thread = None
        self._sender_running = False

        # Reconnection
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delays = [1, 2, 4, 8, 16]  # Exponential backoff

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def connect(self, url: str, api_key: str, model: str, instructions: Optional[str] = None) -> bool:
        """
        Establish the WebSocket connection and open a 60db STT session.

        Args:
            url: Base WebSocket URL (e.g. 'wss://api.60db.ai/ws/stt').
            api_key: 60db API key (sent as ?apiKey= query param).
            model: Ignored by 60db STT (kept for interface parity).
            instructions: Ignored.
        """
        self.url = url
        self.api_key = api_key
        self.model = model
        return self._connect_internal()

    def _auth_url(self) -> str:
        """Append the apiKey query param expected by 60db."""
        sep = '&' if ('?' in (self.url or '')) else '?'
        return f'{self.url}{sep}apiKey={self.api_key}'

    def _connect_internal(self) -> bool:
        if self.connecting:
            return False
        self.connecting = True

        try:
            print(f'[60DB] Connecting to {self.url}...', flush=True)

            self.ws = websocket.WebSocketApp(
                self._auth_url(),
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for the session to become ready ({"type":"connected"}).
            timeout = 10.0
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)

            if self.connected:
                print('[60DB] Connected successfully', flush=True)
                self.reconnect_attempts = 0
                return True

            print('[60DB] Connection timeout', flush=True)
            try:
                self.ws.close()
            except Exception:
                pass
            return False

        except Exception as e:
            print(f'[60DB] Connection error: {e}', flush=True)
            return False
        finally:
            self.connecting = False

    def _on_open(self, _ws):
        """Transport opened: start the receiver/sender threads.

        The `start` message is sent only after the server confirms auth with
        `connection_established` (see _handle_event), per the 60db handshake.
        """
        start_receiver = False
        with self.lock:
            self._socket_open = True
            self.connecting = False
            if not self.receiver_running:
                self.receiver_running = True
                start_receiver = True
            self._queue_cond.notify_all()

        if start_receiver:
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()

        self._start_sender_thread()

    def _send_start(self):
        """Send the session-initialization message that 60db requires before audio."""
        if not self._socket_open or not self.ws:
            return

        start_msg = {
            'type': 'start',
            'config': self._build_config(),
        }
        languages = self._languages_list()
        if languages:
            start_msg['languages'] = languages

        try:
            self.ws.send(json.dumps(start_msg))
            print('[60DB] Sent start', flush=True)
        except Exception as e:
            print(f'[60DB] Failed to send start: {e}', flush=True)

    def _build_config(self) -> dict:
        enhancement = self.audio_enhancement
        if enhancement not in self.VALID_AUDIO_ENHANCEMENTS:
            enhancement = 'adaptive'
        return {
            'encoding': 'linear',  # 16-bit linear PCM
            'sample_rate': self.SAMPLE_RATE,
            'utterance_end_ms': max(300, int(self.utterance_end_ms or 500)),
            'continuous_mode': bool(self.continuous_mode),
            'audio_enhancement': enhancement,
            'diarize': bool(self.diarize),
        }

    def _languages_list(self):
        """60db wants an array of ISO codes (max 5); None/empty means auto-detect."""
        if not self.language:
            return None
        return [self.language]

    # ------------------------------------------------------------------ #
    # Sender / receiver threads
    # ------------------------------------------------------------------ #
    def _start_sender_thread(self):
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
        """Drain queued audio and stream it as base64 linear-PCM `audio` messages."""
        while True:
            with self.lock:
                self._queue_cond.wait_for(
                    lambda: (not self._sender_running)
                    or (self.connected and self.ws and len(self._audio_queue) > 0)
                )

                if not self._sender_running:
                    return

                if not (self.connected and self.ws):
                    continue

                audio_chunk = self._audio_queue.popleft()
                chunk_duration = len(audio_chunk) / float(self.SAMPLE_RATE)
                self.audio_buffer_seconds = max(0.0, self.audio_buffer_seconds - chunk_duration)
                ws = self.ws

                if not self._audio_queue:
                    self._queue_cond.notify_all()

            try:
                base64_audio = base64.b64encode(self._float32_to_pcm16(audio_chunk)).decode('utf-8')
                event = {
                    'type': 'audio',
                    'audio': base64_audio,
                    'encoding': 'linear',
                    'sample_rate': self.SAMPLE_RATE,
                }
                ws.send(json.dumps(event))
            except Exception as e:
                print(f'[60DB] Failed to send queued audio: {e}', flush=True)

    def _receiver_loop(self):
        while self.receiver_running:
            try:
                event = self.event_queue.get(timeout=0.1)
                self._handle_event(event)
            except Empty:
                continue
            except Exception as e:
                print(f'[60DB] Error in receiver loop: {e}', flush=True)

    def _on_message(self, _ws, message):
        try:
            event = json.loads(message)
        except json.JSONDecodeError as e:
            print(f'[60DB] Failed to parse event: {e}', flush=True)
            return
        self.event_queue.put(event)

    def _on_error(self, _ws, error):
        print(f'[60DB] WebSocket error: {error}', flush=True)

    def _on_close(self, _ws, close_status_code, _close_msg):
        with self.lock:
            self.connected = False
            self._socket_open = False
            self._sender_running = False
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._queue_cond.notify_all()

        print(f'[60DB] WebSocket closed (code: {close_status_code})', flush=True)

        if self.receiver_running and close_status_code != 1000:  # 1000 = normal
            self._attempt_reconnect()

    # ------------------------------------------------------------------ #
    # Event handling
    # ------------------------------------------------------------------ #
    def _handle_event(self, event: dict):
        event_type = event.get('type', '')

        # Some server frames are unkeyed objects (e.g. {"connection_established": {...}}).
        if not event_type:
            if 'connection_established' in event:
                # Auth confirmed — only now open the transcription session.
                print('[60DB] Authenticated', flush=True)
                self._send_start()
            return

        if event_type == 'connected':
            # Session ready — only now can we safely stream audio.
            with self.lock:
                self.connected = True
                self._queue_cond.notify_all()
            print('[60DB] Session ready', flush=True)

        elif event_type == 'speech_started':
            with self.lock:
                self._partial_transcript = ""
            self._notify_partial_transcript("")

        elif event_type == 'transcription':
            self._handle_transcription(event)

        elif event_type in ('language_changed', 'mode_changed'):
            print(f'[60DB] {event_type}', flush=True)

        elif event_type == 'session_stopped':
            print('[60DB] Session stopped', flush=True)

        elif event_type == 'error':
            message = event.get('message', 'Unknown error')
            print(f'[60DB] Server error: {message}', flush=True)
            with self.lock:
                self._partial_transcript = ""
            self._notify_partial_transcript("")
            self.response_event.set()  # Unblock any waiter

    def _handle_transcription(self, event: dict):
        text = (event.get('text', '') or '').strip()
        is_final = bool(event.get('is_final', False))
        speech_final = bool(event.get('speech_final', False))

        # A finalized utterance: is_final AND speech_final (the canonical result).
        if is_final and speech_final:
            with self.lock:
                if text:  # Empty finals are silence/hallucination rejections — skip.
                    self._committed_segments.append(text)
                self._transcript_generation += 1
                self._last_transcript_audio_activity_id = self._audio_activity_id
                self._partial_transcript = ""
            self._notify_partial_transcript("")
            self.response_event.set()
            print(f'[60DB] Final transcript ({len(text)} chars)', flush=True)
        else:
            # Interim (is_final False) or fast pre-refinement final (speech_final False):
            # treat as a live preview without committing.
            with self.lock:
                self._partial_transcript = text
            self._notify_partial_transcript(text)

    # ------------------------------------------------------------------ #
    # Language / callbacks
    # ------------------------------------------------------------------ #
    def update_language(self, language: Optional[str]):
        """Update transcription language (resends `config` mid-session, no reconnect)."""
        self.language = language
        if not self.connected or not self.ws:
            return
        msg = {'type': 'config', 'languages': self._languages_list() or []}
        try:
            self.ws.send(json.dumps(msg))
            print(f'[60DB] Language set to: {language or "auto-detect"}', flush=True)
        except Exception as e:
            print(f'[60DB] Failed to update language: {e}', flush=True)

    def set_partial_transcript_callback(self, callback):
        """Register a callback for live transcription previews."""
        self.partial_transcript_callback = callback

    def _notify_partial_transcript(self, text: str):
        callback = self.partial_transcript_callback
        if not callback:
            return
        try:
            callback(text)
        except Exception as e:
            print(f'[60DB] Partial transcript callback failed: {e}', flush=True)

    # ------------------------------------------------------------------ #
    # Reconnection
    # ------------------------------------------------------------------ #
    def _attempt_reconnect(self):
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print('[60DB] Max reconnection attempts reached', flush=True)
            return False

        delay = self.reconnect_delays[min(self.reconnect_attempts, len(self.reconnect_delays) - 1)]
        self.reconnect_attempts += 1

        print(
            f'[60DB] Reconnecting (attempt {self.reconnect_attempts}/'
            f'{self.max_reconnect_attempts}) in {delay}s...',
            flush=True,
        )
        time.sleep(delay)
        return self._connect_internal()

    # ------------------------------------------------------------------ #
    # Audio in
    # ------------------------------------------------------------------ #
    def _float32_to_pcm16(self, audio_data: np.ndarray) -> bytes:
        audio_clipped = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)
        return audio_int16.tobytes()  # little-endian int16

    def clear_audio_buffer(self):
        """Reset transcription/audio state before a new recording."""
        with self.lock:
            self._audio_queue.clear()
            self.audio_buffer_seconds = 0.0
            self._transcript_generation = 0
            self._committed_segments = []
            self._partial_transcript = ""
            self._audio_activity_id = 0
            self._last_transcript_audio_activity_id = 0
            self._dropped_chunks = 0
            self._last_drop_log_time = 0.0
            self._queue_cond.notify_all()
        self.response_event.clear()
        self._notify_partial_transcript("")

    def append_audio(self, audio_chunk: np.ndarray):
        """
        Queue an audio chunk for streaming (float32, mono, 16kHz).

        Called from the sounddevice callback thread — fast/non-blocking only.
        Caps the unsent backlog at max_buffer_seconds by dropping the oldest chunks.
        """
        if not self.connected or not self.ws:
            return

        drop_msg = None
        with self.lock:
            chunk_duration = len(audio_chunk) / float(self.SAMPLE_RATE)

            while (
                (self.audio_buffer_seconds + chunk_duration) > self.max_buffer_seconds
                and self._audio_queue
            ):
                dropped = self._audio_queue.popleft()
                dropped_duration = len(dropped) / float(self.SAMPLE_RATE)
                self.audio_buffer_seconds = max(0.0, self.audio_buffer_seconds - dropped_duration)
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
                    f'[60DB] Dropping audio chunk(s) (queued>{self.max_buffer_seconds:.1f}s). '
                    f'dropped_chunks={self._dropped_chunks}'
                )
                self._last_drop_log_time = now

        if drop_msg:
            print(drop_msg, flush=True)

    # ------------------------------------------------------------------ #
    # Commit / fetch transcript
    # ------------------------------------------------------------------ #
    def _full_committed_text_locked(self) -> str:
        parts = [p for p in self._committed_segments if p]
        return ' '.join(parts).strip()

    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        """
        Flush the session and return the final transcript.

        60db finalizes utterances itself (VAD + utterance_end_ms), so a transcript
        is often already available before the user stops. We drain queued audio,
        send {"type":"stop"} to flush the tail, then wait for the newest final.
        """
        if not self.connected or not self.ws:
            print('[60DB] Not connected, cannot commit', flush=True)
            return ""

        try:
            with self.lock:
                existing_generation = self._transcript_generation
                existing_transcript = self._full_committed_text_locked()
                has_new_audio_since_transcript = (
                    self._audio_activity_id != self._last_transcript_audio_activity_id
                )
                has_queued_audio = len(self._audio_queue) > 0

                # Common case: 60db already finalized before the user released.
                if existing_transcript and (not has_new_audio_since_transcript) and (not has_queued_audio):
                    result = existing_transcript
                    self._committed_segments = []
                    self._transcript_generation = 0
                    self.audio_buffer_seconds = 0.0
                    self.response_event.clear()
                    print(f'[60DB] Using existing transcript ({len(result)} chars)', flush=True)
                    return result

                self.response_event.clear()
                queued_seconds = float(self.audio_buffer_seconds)
                max_backlog = float(self.max_buffer_seconds)

            # Best-effort: let queued audio drain before flushing the session.
            drain_timeout = min(max_backlog + 1.0, max(0.5, timeout * 0.5, queued_seconds + 0.25))
            with self.lock:
                self._queue_cond.wait_for(lambda: len(self._audio_queue) == 0, timeout=drain_timeout)

            # Small grace so any in-flight send reaches the server before we flush.
            time.sleep(0.05)

            try:
                self.ws.send(json.dumps({'type': 'stop'}))
                print('[60DB] Sent stop, waiting for transcript...', flush=True)
            except Exception as e:
                print(f'[60DB] Failed to send stop: {e}', flush=True)

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

                # Settle briefly to catch a late LLM-refined / punctuation final.
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
                print(f'[60DB] Transcript received ({len(result)} chars)', flush=True)
                return result

            print(f'[60DB] Timeout waiting for transcript ({timeout}s)', flush=True)

            # Fallback: latest committed text, else last partial, else empty.
            with self.lock:
                fallback = self._full_committed_text_locked()
                if fallback:
                    self._committed_segments = []
                    self._transcript_generation = 0
                    return fallback
                if self._partial_transcript:
                    result = self._partial_transcript.strip()
                    self._partial_transcript = ""
                    return result
            return ""

        except Exception as e:
            print(f'[60DB] Error in commit_and_get_text: {e}', flush=True)
            return ""

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self):
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
            self._socket_open = False

        print('[60DB] Connection closed', flush=True)

    def set_max_buffer_seconds(self, seconds: float):
        self.max_buffer_seconds = max(1.0, seconds)
