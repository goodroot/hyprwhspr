"""
Phonon realtime protocol: raw PCM16 over a self-hosted WebSocket.

On open: {"type":"config","sample_rate":N,"format":"pcm_s16le"}; then binary
audio frames; {"type":"end"} to finish. The server answers with
partial{text}, final{segment,text}, done{text} or error. A stream cannot be
cleared or resumed, so each recording gets its own connection.

Adapted from James Wolfley's Phonon-2 provider (#270).
"""

import json
import time
import threading
import socket
from typing import Optional

import numpy as np

try:
    from .realtime_base import WebSocketRealtimeClientBase
    from .text_script import join_segments
except ImportError:
    from realtime_base import WebSocketRealtimeClientBase
    from text_script import join_segments


class PhononRealtimeClient(WebSocketRealtimeClientBase):
    """One connection per recording, with no unsafe mid-stream replay/reconnect."""

    LOG_TAG = '[PHONON]'
    commit_after_disconnect = True

    def __init__(self, mode: str = 'transcribe'):
        super().__init__(mode='transcribe')
        self.sample_rate = 16000
        self._segments = {}
        self._failure = None
        self._sending_audio = False
        self._ending = False
        self._consumed = False
        self._preview_lock = threading.RLock()

    def _ws_connect_params(self):
        headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else None
        return self.url, headers

    def _prepare_connect(self):
        with self.lock:
            self._reset_stream_state_locked()
            self._segments = {}
            self._failure = None
            self._sending_audio = False
            self._ending = False
            self._consumed = False
            self._partial_transcript = ''
            self.response_event = threading.Event()
            self._ws_threads = {thread for thread in self._ws_threads if thread.is_alive()}

    @property
    def needs_fresh_session(self):
        """A recording skipped by the app still used the server-side stream."""
        with self.lock:
            return bool(self._audio_activity_id or self._ending or self._failure)

    @property
    def busy(self):
        with self.lock:
            return self._ending and not self._consumed and not self._failure

    def update_language(self, language: Optional[str]):
        if language and language.lower() not in ('en', 'english'):
            self.close()
            raise ValueError('Phonon supports English only')
        self.language = language

    def _on_open(self, ws, generation=None):
        # Phonon has no config acknowledgement. Mark ready only after sending it.
        with self.lock:
            if not self._is_active_connection(ws, generation):
                return
        try:
            with self._ws_send_lock:
                ws.send(json.dumps({'type': 'config', 'sample_rate': self.sample_rate,
                                    'format': 'pcm_s16le'}))
            with self.lock:
                if self._is_active_connection(ws, generation) and not self._failure:
                    self.connected = True
        except Exception as exc:
            self._on_error(ws, exc, generation)

    def _on_connect_success(self):
        with self.lock:
            if not self.connected or self._closed:
                raise RuntimeError(self._failure or 'Connection closed during initialization')
            self._sender_running = True
            generation = self._active_generation
            self._sender_thread = threading.Thread(
                target=self._sender_loop, args=(generation,), daemon=True,
            )
            self._sender_thread.start()

    def _abandon_attempt(self, attempt_ws):
        with self.lock:
            generation = self._active_generation if attempt_ws is self.ws else None
        if generation is not None:
            self._shutdown(latch=False, generation=generation)
        else:
            self._close_transport(attempt_ws)

    def _fail_locked(self, message):
        """Wake every waiter, retaining the first failure. Caller owns lock."""
        if not self._failure:
            self._failure = str(message)
            self._log(self._failure)
        self.connected = False
        self._fail_pending_attempt_locked()
        self._sender_running = False
        self._audio_queue.clear()
        self.audio_buffer_seconds = 0.0
        self._queue_cond.notify_all()
        self.response_event.set()

    def _on_error(self, ws, error, generation=None):
        with self.lock:
            if self._is_active_connection(ws, generation):
                self._fail_locked(f'WebSocket failure: {error}')

    def _on_close(self, ws, close_status_code, close_msg, generation=None):
        with self.lock:
            if not self._is_active_connection(ws, generation):
                return
            if not self.response_complete:
                self._fail_locked(f'Stream closed before completion ({close_status_code}): {close_msg}')
            self.connected = False
            self._sender_running = False
            self._queue_cond.notify_all()
        # Reconnection is only safe before the next recording, never during one.

    def _on_message(self, ws, message, generation=None):
        # Process inline so a normal close cannot overtake the queued done event.
        try:
            event = json.loads(message)
            with self.lock:
                if not self._is_active_connection(ws, generation) or self._failure or self.response_complete:
                    return
                self._handle_event_locked(event)
            self._emit_partial_transcript(ws, generation)
        except (ValueError, TypeError, AttributeError) as exc:
            self._on_error(ws, f'Invalid server event: {exc}', generation)

    def _handle_event_locked(self, event):
        event_type = event.get('type')
        text = event.get('text', '')
        if not isinstance(text, str):
            raise ValueError('transcript text must be a string')
        if event_type == 'partial':
            self._partial_transcript = text.strip()
        elif event_type == 'final':
            segment = event.get('segment')
            if not isinstance(segment, int) or isinstance(segment, bool) or segment < 0:
                raise ValueError('final requires a non-negative segment number')
            if segment not in self._segments:
                self._segments[segment] = text.strip()
                self._partial_transcript = ''
        elif event_type == 'done':
            if not self._ending:
                raise ValueError('done received before end of input')
            # The server's full transcript is authoritative, not another segment.
            self.current_response_text = text.strip()
            self._partial_transcript = ''
            self.response_complete = True
            self.response_event.set()
        elif event_type == 'error':
            self._fail_locked(f'Server rejected stream: {event.get("error", event.get("message", text))}')

    def _emit_partial_transcript(self, ws=None, generation=None):
        # Serialize preview delivery with teardown, without holding the state lock
        # while calling application code. Cancellation's clear must arrive last.
        with self._preview_lock:
            with self.lock:
                if ws is not None and not self._is_active_connection(ws, generation):
                    return
                if self._failure or self._closed:
                    committed, tail = '', ''
                else:
                    committed = join_segments([self._segments[key] for key in sorted(self._segments)])
                    tail = self._partial_transcript
            self._publish_live_text(committed, tail)

    def append_audio(self, audio_chunk: np.ndarray):
        # Keep capture non-blocking. Unlike the shared lossy queue policy, fail
        # the recording on overflow rather than returning an incomplete transcript.
        with self.lock:
            if self._failure or self._ending:
                return
            if not self.connected:
                self._fail_locked('Connection lost during recording; audio was not replayed')
                return
            duration = len(audio_chunk) / float(self.input_sample_rate)
            if self.audio_buffer_seconds + duration > self.max_buffer_seconds:
                self._fail_locked('Audio queue overflow; recording discarded')
                return
            self._audio_queue.append(audio_chunk.copy())
            self.audio_buffer_seconds += duration
            self._audio_activity_id += 1
            self._on_audio_chunk_locked()
            self._queue_cond.notify_all()

    def _sender_loop(self, sender_generation):
        while True:
            with self.lock:
                self._queue_cond.wait_for(lambda: not self._sender_running or self._audio_queue or self._ending
                                         or sender_generation != self._active_generation)
                if not self._sender_running or sender_generation != self._active_generation:
                    return
                audio = self._audio_queue.popleft() if self._audio_queue else None
                if audio is not None:
                    self.audio_buffer_seconds = max(
                        0.0, self.audio_buffer_seconds - len(audio) / float(self.input_sample_rate)
                    )
                self._sending_audio = True
                ws = self.ws
                generation = self._active_generation
            try:
                if audio is not None:
                    audio = self._resample_for_output(audio)
                    payload = (np.clip(audio, -1.0, 1.0) * 32767).astype('<i2').tobytes()
                else:
                    payload = json.dumps({'type': 'end'})
                with self._ws_send_lock:
                    with self.lock:
                        active = self._is_active_connection(ws, generation) and not self._failure
                    if active:
                        # websocket-client sends one FIN binary frame, not fragments.
                        ws.send(payload, opcode=2 if audio is not None else 1)
            except Exception as exc:
                self._on_error(ws, f'Audio send failed: {exc}', generation)
            finally:
                with self.lock:
                    if generation == self._active_generation:
                        self._sending_audio = False
                    self._queue_cond.notify_all()
            if audio is None:
                return

    def clear_audio_buffer(self):
        # The backend calls this just before capture. A used connection cannot
        # be cleared server-side; cancellation must close it instead.
        with self.lock:
            if not self.connected or self._audio_activity_id or self._ending or self._failure:
                raise RuntimeError('Phonon requires a fresh connection for each recording')
            self._partial_transcript = ''
        self._emit_partial_transcript()

    def discard_audio(self):
        self.close()

    def close(self):
        self._shutdown(latch=True)

    def _shutdown(self, latch, generation=None):
        # Detach under the lock. A cancelled commit may finish after reconnect;
        # it must never mutate or close that newer recording.
        with self._preview_lock:
            with self.lock:
                if generation is not None and generation != self._active_generation:
                    return
                if self._closed and latch:
                    return
                if not self._consumed:
                    self._fail_locked('Recording cancelled or connection closed')
                self._closed = latch
                if latch:
                    self._stop_event.set()
                self._connection_generation = max(self._connection_generation, self._active_generation) + 1
                self._active_generation = self._connection_generation
                self.connecting = False
                self.connected = False
                self._sender_running = False
                self._queue_cond.notify_all()
                self._partial_transcript = ''
                self._segments = {}
                ws, self.ws = self.ws, None
                sender = self._sender_thread
            self._emit_partial_transcript()
        self._close_transport(ws)
        if sender and sender is not threading.current_thread() and sender.is_alive():
            sender.join(timeout=0.2)

    @staticmethod
    def _close_transport(ws):
        if ws:
            # websocket-client close sends a close frame under its send lock.
            # Interrupt the underlying socket first so a blocked write cannot
            # hold cancellation or the completion timeout hostage.
            sock = getattr(ws, 'sock', None)
            if sock is not None:
                raw_socket = getattr(sock, 'sock', None)
                if raw_socket is not None:
                    try:
                        raw_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                try:
                    sock.shutdown()
                except Exception:
                    pass
            try:
                ws.close()
            except Exception:
                pass

    def commit_and_get_text(self, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.lock:
            generation = self._active_generation
            response_event = self.response_event
        try:
            with self.lock:
                if generation != self._active_generation:
                    raise RuntimeError('Recording cancelled or replaced')
                if self._consumed:
                    return ''
                self._ending = True
                self._queue_cond.notify_all()
                if self._failure:
                    raise RuntimeError(self._failure)
            if not response_event.wait(max(0.0, deadline - time.monotonic())):
                raise RuntimeError('Timeout waiting for Phonon completion')
            with self.lock:
                if generation != self._active_generation:
                    raise RuntimeError('Recording cancelled or replaced')
                if self._failure:
                    raise RuntimeError(self._failure)
                if not self.response_complete:
                    raise RuntimeError('Stream ended without a complete transcript')
                if self._consumed:
                    return ''
                self._consumed = True
                result = self.current_response_text
            return result
        finally:
            # The next recording must reconnect even if the server close has
            # not arrived yet. Invalidates late events from this recording.
            self._shutdown(latch=True, generation=generation)
