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

        # Incremental (append-only) injection. This server's cache-aware RNNT
        # never revises a word once it has streamed it as a delta - a
        # finalized segment is the joined deltas plus finalization-time
        # punctuation - so completed words are safe to paste as they arrive,
        # instead of waiting for a pause to finalize the segment. The segment's
        # remaining words (with that punctuation) go out when it finalizes.
        self._stream_text_callback = None
        self._stream_text = ""      # raw deltas of the current segment
        self._stream_delivered = 0  # prefix of _stream_text already delivered
        self._incremental_injected_any = False

    def _ws_connect_params(self):
        # The server only checks auth when started with --api-key; without a
        # stored key, send no header rather than a literal "Bearer None".
        if self.api_key:
            return self.url, {'Authorization': f'Bearer {self.api_key}'}
        return self.url, None

    def set_stream_text_callback(self, callback):
        """Register callback(text, final, whole) -> bool for append-only
        delivery. `text` is whole words only, stripped. `final` marks the tail
        of a finalized segment (the only chunk carrying finalization
        punctuation); `whole` means that tail is the entire segment, nothing
        of it delivered earlier. Return truthy if the text was delivered;
        falsy leaves it for the joined transcript at recording end."""
        self._stream_text_callback = callback

    def clear_audio_buffer(self):
        super().clear_audio_buffer()
        self._reset_stream()
        self._incremental_injected_any = False

    def _reset_stream(self):
        self._stream_text = ""
        self._stream_delivered = 0

    def _deliver(self, text: str, final: bool, whole: bool) -> bool:
        try:
            delivered = bool(self._stream_text_callback(text, final, whole))
        except Exception as e:
            self._log(f'Stream text callback failed: {e}')
            return False
        if delivered:
            self._incremental_injected_any = True
        return delivered

    def _deliver_completed_words(self):
        """Deliver the words finished since the last delivery. A word counts
        as finished once whitespace follows it; the trailing partial word
        waits for the next delta or the segment's finalization."""
        pending = self._stream_text[self._stream_delivered:]
        cut = max(pending.rfind(' '), pending.rfind('\t'), pending.rfind('\n'))
        words = pending[:cut].strip() if cut > 0 else ''
        if words and self._deliver(words, final=False, whole=False):
            self._stream_delivered += cut

    def _deliver_segment_tail(self, transcript: str):
        """On finalization, deliver the words not yet delivered, taken from the
        final transcript so its punctuation lands. Aligned by word count, not
        characters, so a casing/punctuation change on an already-delivered
        word can't cause a duplicate."""
        already = len(self._stream_text[:self._stream_delivered].split())
        final_words = transcript.split()
        if already > len(final_words):
            self._log('Final transcript shorter than delivered text; skipping tail')
            return
        tail = ' '.join(final_words[already:])
        if tail:
            self._deliver(tail, final=True, whole=(already == 0))

    def flush_stream(self):
        """Deliver whatever the current segment has streamed but not yet
        delivered, including a trailing partial word, for when it will never
        finalize."""
        if not self._stream_text_callback:
            return
        pending = self._stream_text[self._stream_delivered:].strip()
        if pending:
            self._deliver(pending, final=True, whole=self._stream_delivered == 0)
        self._reset_stream()

    def _handle_event(self, event: dict):
        event_type = event.get('type')
        if self._stream_text_callback and not self._is_retired_item(event):
            if event_type == 'conversation.item.input_audio_transcription.delta':
                self._stream_text += event.get('delta') or ''
                self._deliver_completed_words()
            elif event_type == 'conversation.item.input_audio_transcription.completed':
                # Deliver BEFORE the shared handling: it sets response_event,
                # which wakes commit_and_get_text() on the main thread. Running
                # after it, a take whose only text is this final tail could be
                # pasted twice - here, and from the joined transcript before
                # _incremental_injected_any was set.
                transcript = (event.get('transcript') or '').strip()
                if transcript:
                    self._deliver_segment_tail(transcript)
                self._reset_stream()

        # Shared OpenAI-shaped handling: buffers finals into
        # _committed_segments, so the joined transcript still exists as a
        # fallback for deferred text and callers without a callback.
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
