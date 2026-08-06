"""RealtimeSTT FastAPI WebSocket protocol adapter.

This client lets hyprwhspr own the microphone connection to RealtimeSTT's
``/ws/transcribe`` endpoint.  The same socket carries binary PCM audio out and
session-scoped interim/final transcript events back.
"""

from __future__ import annotations

import json
import struct
import threading
from collections import OrderedDict
from typing import Optional

try:
    from .realtime_base import WebSocketRealtimeClientBase
except ImportError:
    from realtime_base import WebSocketRealtimeClientBase


class RealtimeSTTRealtimeClient(WebSocketRealtimeClientBase):
    """Client for ``example_fastapi_server`` in the RealtimeSTT repository."""

    LOG_TAG = "[REALTIMESTT]"
    supports_mid_session_language_update = False
    STOP_IDLE_GRACE_SECONDS = 2.0

    def __init__(self):
        super().__init__(mode="transcribe")
        self.sample_rate = 16000
        self.session_id = None
        self.partial_transcript_callback = None
        self._ready_event = threading.Event()
        self._streaming = False
        self._final_by_segment = OrderedDict()
        self._partial_by_segment = {}
        self._latest_sequences = {}
        self._awaiting_clear = False
        self._minimum_segment_id = None
        self._stop_completion_pending = False
        self._stop_idle_timer = None

    # ------------------------------------------------------------------
    # Transport hooks
    # ------------------------------------------------------------------

    def _prepare_connect(self):
        self._ready_event.clear()
        self.session_id = None

    def _after_open(self, _ws):
        # The receiver is already running.  Audio starts only after the server's
        # ready event, not merely after the TCP/WebSocket handshake.
        self._log("WebSocket opened; waiting for server readiness")

    def _audio_ws_message(self, _base64_audio: str) -> dict:
        raise NotImplementedError("RealtimeSTT audio uses binary packets")

    def _send_audio_payload(self, ws, pcm_bytes: bytes):
        metadata = {
            "sampleRate": self.sample_rate,
            "channels": 1,
            "format": "pcm_s16le",
            "frames": len(pcm_bytes) // 2,
        }
        metadata_bytes = json.dumps(
            metadata,
            separators=(",", ":"),
        ).encode("utf-8")
        packet = struct.pack("<I", len(metadata_bytes)) + metadata_bytes + pcm_bytes
        opcode = self._websocket_transport.ABNF.OPCODE_BINARY
        ws.send(packet, opcode=opcode)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_event(self, event: dict):
        event_type = event.get("type", "")

        if event_type == "hello":
            self.session_id = event.get("sessionId") or event.get("clientId")
            self._log(f"Session assigned: {self.session_id or 'unknown'}")
            return

        if event_type == "ready":
            if event.get("ok", True) is False:
                self._log("Server reported unhealthy inference workers")
                return
            with self.lock:
                self.connected = True
                self.connecting = False
                self._queue_cond.notify_all()
            self._ready_event.set()
            self._start_sender_thread()
            self._log("Server ready")
            return

        if event_type == "realtime":
            self._handle_realtime(event)
            return

        if event_type == "final":
            self._handle_final(event)
            return

        if event_type == "clear":
            # Local state was reset synchronously before sending clear.  The
            # acknowledgement establishes a monotonic segment floor so a final
            # already in flight from the previous take cannot leak into this one.
            with self.lock:
                next_segment = event.get("nextSegmentId")
                self._minimum_segment_id = (
                    next_segment if isinstance(next_segment, int) else None
                )
                self._awaiting_clear = False
            return

        if event_type == "status":
            state = event.get("state")
            with self.lock:
                self._streaming = state != "idle"
                if (
                    state == "idle"
                    and self._stop_completion_pending
                    and self._full_committed_text_locked()
                    and self._stop_idle_timer is None
                ):
                    # stop_streaming publishes idle before its optional final
                    # inference job completes. Give that job time to arrive;
                    # if there is no job (only silence followed an earlier VAD
                    # final), release commit with the transcript already held.
                    timer = threading.Timer(
                        self.STOP_IDLE_GRACE_SECONDS,
                        self._complete_idle_stop,
                    )
                    timer.daemon = True
                    self._stop_idle_timer = timer
                    timer.start()
            return

        if event_type == "warning":
            self._log(f"Server warning: {event.get('message', 'Unknown warning')}")
            return

        if event_type == "error":
            message = event.get("message", "Unknown error")
            self._log(f"Server error: {message}")
            if event.get("where") in ("admission", "final", "recorder"):
                self.response_event.set()
            return

    @staticmethod
    def _segment_key(event: dict):
        segment_id = event.get("segmentId")
        return segment_id if segment_id is not None else "current"

    def _handle_realtime(self, event: dict):
        segment = self._segment_key(event)
        sequence = event.get("sequence")
        text = (
            event.get("displayText")
            or event.get("consensusDisplayText")
            or event.get("text")
            or ""
        ).strip()
        if not text:
            return

        with self.lock:
            if not self._accept_segment_locked(segment):
                return
            if sequence is not None:
                previous_sequence = self._latest_sequences.get(segment)
                if previous_sequence is not None and sequence < previous_sequence:
                    return
                self._latest_sequences[segment] = sequence
            self._partial_by_segment[segment] = text
            preview = self._preview_text_locked(segment)
        self._notify_partial_transcript(preview)

    def _handle_final(self, event: dict):
        segment = self._segment_key(event)
        text = (event.get("text") or "").strip()
        if not text:
            return

        with self.lock:
            if not self._accept_segment_locked(segment):
                return
            self._cancel_stop_idle_timer_locked()
            self._stop_completion_pending = False
            self._final_by_segment[segment] = text
            self._partial_by_segment.pop(segment, None)
            self._latest_sequences.pop(segment, None)
            self._committed_segments = list(self._final_by_segment.values())
            self._transcript_generation += 1
            self._last_transcript_audio_activity_id = self._audio_activity_id
            self.current_response_text = text
            self.response_complete = True
            preview = self._preview_text_locked()
        self._notify_partial_transcript(preview)
        self.response_event.set()
        self._log(f"Final segment received ({len(text)} chars)")

    def _cancel_stop_idle_timer_locked(self):
        timer = self._stop_idle_timer
        self._stop_idle_timer = None
        if timer is not None:
            timer.cancel()

    def _complete_idle_stop(self):
        with self.lock:
            self._stop_idle_timer = None
            if not self._stop_completion_pending:
                return
            self._stop_completion_pending = False
        self.response_event.set()

    def _accept_segment_locked(self, segment) -> bool:
        if self._awaiting_clear:
            return False
        if (
            isinstance(segment, int)
            and isinstance(self._minimum_segment_id, int)
            and segment < self._minimum_segment_id
        ):
            return False
        return True

    def _preview_text_locked(self, active_segment=None) -> str:
        parts = list(self._final_by_segment.values())
        if active_segment is not None:
            partial = self._partial_by_segment.get(active_segment, "")
            if partial:
                parts.append(partial)
        elif self._partial_by_segment:
            parts.extend(self._partial_by_segment.values())
        return " ".join(part for part in parts if part).strip()

    def set_partial_transcript_callback(self, callback):
        self.partial_transcript_callback = callback

    def _notify_partial_transcript(self, text: str):
        callback = self.partial_transcript_callback
        if callback is None:
            return
        try:
            callback(text)
        except Exception as exc:
            self._log(f"Partial transcript callback failed: {exc}")

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def _reset_take_locked(self):
        self._cancel_stop_idle_timer_locked()
        self._stop_completion_pending = False
        self._reset_stream_state_locked()
        self._final_by_segment.clear()
        self._partial_by_segment.clear()
        self._latest_sequences.clear()

    def start_audio_stream(self):
        """Reset the server session and begin accepting audio for a new take."""
        with self.lock:
            self._reset_take_locked()
            self._awaiting_clear = True
            self._minimum_segment_id = None
        self.response_event.clear()
        self._notify_partial_transcript("")
        if not self.connected or not self.ws:
            return
        try:
            self.ws.send(json.dumps({"type": "clear"}))
            self.ws.send(json.dumps({"type": "start"}))
            self._streaming = True
        except Exception as exc:
            self._log(f"Failed to start audio stream: {exc}")

    def clear_audio_buffer(self):
        """Compatibility alias used by generic realtime callers."""
        self.start_audio_stream()

    def discard_audio_buffer(self):
        """Cancel a take without leaving RealtimeSTT in streaming mode."""
        with self.lock:
            self._reset_take_locked()
            self._awaiting_clear = True
            self._minimum_segment_id = None
        self.response_event.clear()
        self._notify_partial_transcript("")
        if not self.connected or not self.ws:
            return
        try:
            self.ws.send(json.dumps({"type": "stop"}))
            self.ws.send(json.dumps({"type": "clear"}))
            self._streaming = False
        except Exception as exc:
            self._log(f"Failed to discard audio stream: {exc}")

    def _on_commit_fast_path_locked(self):
        # RealtimeSTT still needs the explicit stop even when VAD finalized the
        # last segment before the hotkey was released.
        self._cancel_stop_idle_timer_locked()
        self._stop_completion_pending = False
        if self.ws:
            self.ws.send(json.dumps({"type": "stop"}))
        self._streaming = False

    def _request_transcript(self, _ctx: dict):
        with self.lock:
            self._cancel_stop_idle_timer_locked()
            self._stop_completion_pending = True
        self.ws.send(json.dumps({"type": "stop"}))
        self._streaming = False
        self._log("Stopped stream; waiting for final segment")

    def close(self):
        with self.lock:
            self._cancel_stop_idle_timer_locked()
            self._stop_completion_pending = False
        super().close()

    def update_language(self, language: Optional[str]):
        # The server chooses language/model at startup.  Keep the value for
        # diagnostics, but do not pretend a per-session update was applied.
        self.language = language
