"""
Realtime WebSocket client for a self-hosted NeMo-Speech.cpp server
(https://github.com/NVIDIA/NeMo-Speech.cpp), e.g. serving
nvidia/nemotron-speech-streaming-en-0.6b.

Event names on the wire (session.created/updated, conversation.item.
input_audio_transcription.delta/.completed, input_audio_buffer.committed,
error) match the OpenAI Realtime transcription events byte-for-byte, so this
subclasses RealtimeClient and reuses its event handling untouched. The two
things that differ are protocol-shape, not event semantics:

1. Audio format: OpenAI's Realtime API is fixed at 24kHz PCM16. NeMo-Speech.cpp
   accepts 8000-96000 Hz, but the shipped streaming models (e.g. the Nemotron
   0.6B streaming checkpoint) are trained at 16kHz, so that is what actually
   gets sent.
2. session.update shape: OpenAI's GA schema nests audio format under
   session.audio.input.format.{type,rate}. NeMo-Speech.cpp's server predates
   that migration and reads flat keys directly under session (sample_rate,
   language, automatic_punctuation, ...) - see docs/api.md in the
   NeMo-Speech.cpp release archive. Sending the nested shape leaves the
   server's flat-key parser with no sample_rate, so it assumes its default
   (also 16000 for the shipped models) while actually receiving whatever rate
   RealtimeClient hardcoded (24000) - a 1.5x time-stretch that decodes to
   nothing. This class fixes both by keeping the base class's 16kHz default
   and sending the flat shape the server actually parses.
"""

import json

try:
    from .realtime_client import RealtimeClient
except ImportError:
    from realtime_client import RealtimeClient


class NemoRealtimeClient(RealtimeClient):
    """RealtimeClient variant for a local/self-hosted NeMo-Speech.cpp server."""

    def __init__(self, mode: str = 'transcribe'):
        super().__init__(mode=mode)
        # Base class defaults to 16000; RealtimeClient.__init__ (OpenAI) bumps
        # it to 24000. Put it back for this provider's actual model rate.
        self.sample_rate = 16000

        # Incremental (append-only) injection: fired once per finalized
        # segment as it lands, instead of waiting for the whole recording to
        # stop. Requires the server started with --endpointing (see
        # docs/asr/configuration.md), which makes it emit multiple
        # conversation.item.input_audio_transcription.completed events per
        # stream - one per detected pause - rather than only when the client
        # sends input_audio_buffer.commit. Without --endpointing this callback
        # simply fires once, same as every other provider.
        self._committed_segment_callback = None
        self._incremental_injected_any = False

    def _ws_connect_params(self):
        # The server only checks auth when started with --api-key; without a
        # stored key, send no header rather than a literal "Bearer None".
        if self.api_key:
            return self.url, {'Authorization': f'Bearer {self.api_key}'}
        return self.url, None

    def set_committed_segment_callback(self, callback):
        """Register a callback invoked with each finalized segment's text as
        it completes mid-stream, for incremental append-only injection."""
        self._committed_segment_callback = callback

    def clear_audio_buffer(self):
        super().clear_audio_buffer()
        self._incremental_injected_any = False

    def _handle_event(self, event: dict):
        # Deliver the segment BEFORE the shared handling runs: that handling
        # sets response_event, which wakes commit_and_get_text() on the main
        # thread. If the callback ran after it, a recording whose only segment
        # is the final one could be pasted twice - once here and once from the
        # joined transcript, before _incremental_injected_any was set.
        if (
            self._committed_segment_callback
            and event.get('type') == 'conversation.item.input_audio_transcription.completed'
            and not self._is_retired_item(event)
        ):
            # NeMo-Speech.cpp always populates `transcript` on this event
            # (unlike providers the base class falls back to the partial
            # buffer for), so reading it directly here is safe.
            transcript = (event.get('transcript') or '').strip()
            if transcript:
                # Only a segment the callback actually took counts as
                # delivered; one it deferred (returned falsy) stays in the
                # joined transcript for the normal end-of-recording path.
                try:
                    if self._committed_segment_callback(transcript):
                        self._incremental_injected_any = True
                except Exception as e:
                    self._log(f'Committed-segment callback failed: {e}')

        # Shared OpenAI-shaped handling: buffers the segment into
        # _committed_segments, so the joined transcript still exists as a
        # fallback for deferred segments and callers without a callback.
        super()._handle_event(event)

    def _send_session_update(self):
        """Send the flat session.update shape NeMo-Speech.cpp's /v1/realtime expects."""
        if not self.connected or not self.ws:
            return

        session_data = {'sample_rate': self.sample_rate}
        if self.language:
            session_data['language'] = self.language

        event = {'type': 'session.update', 'session': session_data}

        try:
            with self._ws_send_lock:
                self.ws.send(json.dumps(event))
            self._log('Sent session.update')
        except Exception as e:
            self._log(f'Failed to send session.update: {e}')
