"""Live transcript preview for local batch backends.

A local model transcribes the complete recording only after the hotkey stops,
so the recording-status bubble shows nothing until the paste lands. LivePreview
decodes *closed* speech segments while the microphone is still open and hands
the growing text to the status presenter.

The preview is presentation only. The pasted text still comes from one batch
decode of the whole recording, so dictation accuracy is exactly the same as a
plain recording; a preview that disagrees with the final decode is replaced by
it, never merged into it.
"""

import threading
import time

import numpy as np

try:
    from .service_log import log
except ImportError:
    from service_log import log

try:
    from .audio_resampler import resample_audio
except ImportError:
    from audio_resampler import resample_audio


PREVIEW_RATE = 16000
# A segment may still grow while the speaker is inside it: only decode audio a
# following pause has already closed.
_CLOSED_GUARD_SAMPLES = int(0.25 * PREVIEW_RATE)
# Decoded audio is dropped from the buffer, keeping it seconds long instead of
# growing with the dictation. A little pre-roll stays for the next onset.
_DROP_PREROLL_SAMPLES = int(0.20 * PREVIEW_RATE)
_MIN_SEGMENT_SAMPLES = int(0.30 * PREVIEW_RATE)
_MAX_PREVIEW_CHARS = 240
_MIN_PUBLISH_INTERVAL_S = 0.25


class LivePreview:
    """Decode closed speech segments in the background and publish the text.

    `segment(audio16k)` returns `[(start, end), ...]` sample ranges for one
    buffer, `decode(audio16k)` returns the text of one segment, and
    `on_text(text)` receives the accumulated preview. Both callables run on
    this object's worker thread, never on the capture thread.
    """

    def __init__(self, *, segment, decode, on_text=None, interval=0.5):
        self._segment = segment
        self._decode = decode
        self._on_text = on_text
        self._interval = max(0.1, float(interval))

        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)
        # Buffer-relative sample offset past the last decoded segment.
        self._decoded_end = 0
        self._text_parts = []
        self._published = ""
        self._published_at = 0.0
        self._disabled = False

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='live-preview', daemon=True)
        self._thread.start()

    # ---------------------------- capture side ----------------------------

    def feed(self, chunk, sample_rate=PREVIEW_RATE):
        """Accept one capture chunk. Called on the audio callback thread."""
        if chunk is None or self._stop.is_set() or self._disabled:
            return
        try:
            samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return
        if samples.size == 0:
            return
        try:
            rate = int(sample_rate or PREVIEW_RATE)
        except (TypeError, ValueError):
            rate = PREVIEW_RATE
        if rate != PREVIEW_RATE:
            try:
                samples = resample_audio(samples, rate, PREVIEW_RATE)
            except Exception as exc:  # never raise into the capture callback
                self._fail(f'sample-rate conversion failed: {exc}')
                return
        with self._lock:
            self._buffer = samples.copy() if self._buffer.size == 0 else np.concatenate(
                (self._buffer, samples))

    # ---------------------------- worker side -----------------------------

    def _run(self):
        while not self._stop.wait(self._interval):
            self.pump()

    def pump(self):
        """Decode every segment the buffer has closed since the last pass."""
        if self._disabled or self._stop.is_set():
            return
        with self._lock:
            audio = self._buffer
        if audio.size < _MIN_SEGMENT_SAMPLES:
            return
        try:
            segments = list(self._segment(audio) or [])
        except Exception as exc:
            self._fail(f'segmentation failed: {exc}')
            return
        closed_before = audio.size - _CLOSED_GUARD_SAMPLES
        for start, end in segments:
            if self._disabled or self._stop.is_set():
                return
            start, end = int(start), int(end)
            if end > closed_before or start < self._decoded_end:
                continue
            if end - start < _MIN_SEGMENT_SAMPLES:
                self._decoded_end = max(self._decoded_end, end)
                continue
            try:
                text = self._decode(audio[start:end])
            except Exception as exc:
                self._fail(f'segment decode failed: {exc}')
                return
            self._decoded_end = max(self._decoded_end, end)
            text = (text or '').strip()
            if text:
                with self._lock:
                    self._text_parts.append(text)
                self._publish()
        self._drop_decoded()
        # A publish skipped by the interval still has to reach the presenter.
        self._publish()

    def _publish(self):
        if self._on_text is None:
            return
        with self._lock:
            text = ' '.join(part for part in self._text_parts if part)
        if not text or text == self._published:
            return
        now = time.monotonic()
        if now - self._published_at < _MIN_PUBLISH_INTERVAL_S:
            return
        self._published = text
        self._published_at = now
        try:
            self._on_text(text[-_MAX_PREVIEW_CHARS:])
        except Exception as exc:
            log(f'[PREVIEW] Status update failed: {exc}')

    def _drop_decoded(self):
        with self._lock:
            cut = self._decoded_end - _DROP_PREROLL_SAMPLES
            if cut > 0 and cut < self._buffer.size:
                self._buffer = self._buffer[cut:].copy()
                self._decoded_end -= cut

    def _fail(self, reason):
        if self._disabled:
            return
        self._disabled = True
        self._stop.set()
        log(f'[PREVIEW] Live preview disabled: {reason}')

    # ------------------------------ teardown ------------------------------

    def stop(self, timeout=3.0):
        """Stop the worker; safe to call from any thread and more than once."""
        self._stop.set()
        thread = self._thread
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    @property
    def disabled(self):
        return self._disabled

    @property
    def stopped(self):
        return self._stop.is_set()

    @property
    def text(self):
        with self._lock:
            return ' '.join(part for part in self._text_parts if part)
