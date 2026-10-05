"""
Live typing: type append-only realtime text while the user is still speaking.

Only for models whose emitted words never change (live_text 'append_only').
Nothing typed is ever revised (#219); at stop, the final transcript's remaining
words are typed, aligned by word count, so finalization punctuation lands
without duplicating what's already there.
"""

import queue
import threading
from typing import List, Optional, Tuple

try:
    from .hallucination import could_be_hallucination
    from .service_log import log
    from .text_injector import InjectionOutcome
except ImportError:
    from hallucination import could_be_hallucination
    from service_log import log
    from text_injector import InjectionOutcome


class LiveTyper:
    """Word accounting: which stable words are new since the last update.

    Opening words are held while they could still grow into a hallucination
    marker ("Thank" of "thank you"); once they can't, everything goes out.
    """

    def __init__(self, markers=None):
        self.markers = markers
        self.typed = 0
        self.held_phantom = True

    @staticmethod
    def stable_words(committed: str, tail: str) -> List[str]:
        """Committed words plus finished tail words; the last tail word may still grow."""
        tail = tail or ''
        tail_words = tail.split()
        if tail_words and not tail[-1].isspace():
            tail_words = tail_words[:-1]
        return (committed or '').split() + tail_words

    def update(self, committed: str, tail: str) -> Tuple[List[str], bool]:
        """New words to type, and whether the segment is final (no live tail)."""
        final = not (tail or '').strip()
        words = self.stable_words(committed, tail)
        if self.held_phantom:
            if not words or could_be_hallucination(' '.join(words), self.markers):
                return [], final
            self.held_phantom = False
        new = words[self.typed:]
        self.typed = max(self.typed, len(words))
        return new, final

    def finish(self, final_text: str) -> List[str]:
        """Words of the final transcript not typed yet."""
        words = (final_text or '').split()
        if len(words) < self.typed:
            log(f'[LIVE] Final transcript shorter than typed text ({len(words)} < {self.typed}); nothing added')
            return []
        new = words[self.typed:]
        self.typed = len(words)
        return new


class LiveTypingSession:
    """One recording's live typing: intake from the receiver thread, paste on a worker.

    Pasting takes a clipboard round-trip per chunk, so it never runs on the
    websocket receiver thread. The worker keeps chunks in order.
    """

    FINISH_TIMEOUT_SECS = 10.0

    def __init__(self, injector, markers=None):
        self._injector = injector
        self._typer = LiveTyper(markers)
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._open = True
        self._failed = False
        self._done = threading.Event()
        self._handle = injector.begin_stream()
        self._outcome = None
        self._started = False
        self._worker = threading.Thread(target=self._run, name='live-typing', daemon=True)
        try:
            self._worker.start()
        except Exception:
            injector.release_stream(self._handle)
            raise

    def on_live_text(self, committed: str, tail: str) -> None:
        """Live-text listener; called on the realtime receiver thread."""
        with self._lock:
            if not self._open:
                return
            words, final = self._typer.update(committed, tail)
            if words or (final and self._typer.typed):
                self._queue.put((words, final))

    def _run(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if self._handle.cancelled.is_set():
                    continue
                words, final = item
                try:
                    outcome = self._injector.inject_stream_chunk(
                        self._handle, ' '.join(words), final=final)
                except Exception as e:
                    log(f'[LIVE] Typing failed: {e}')
                    outcome = InjectionOutcome.FAILED
                if outcome == InjectionOutcome.FAILED:
                    self._failed = True
                    self._handle.failed = True
            self._outcome = self._injector.end_stream(
                self._handle, submit=self._started and not self._failed
                and not self._handle.cancelled.is_set())
        except Exception as e:
            log(f'[LIVE] Finishing typed text failed: {e}')
            self._failed = True
        finally:
            try:
                self._injector.release_stream(self._handle)
            finally:
                self._done.set()

    def _wait(self, recovery_text=None):
        if not self._done.wait(self.FINISH_TIMEOUT_SECS):
            if recovery_text is not None:
                self._handle.recovery_text = recovery_text
            self._handle.cancelled.set()
            self._failed = True
            log('[LIVE] Text delivery timed out; stopping pending typing')
            return False
        return True

    def finish(self, final_text: str) -> Optional[InjectionOutcome]:
        """Wait for delivery and cleanup; timeout cancels all pending typing."""
        with self._lock:
            if self._open:
                self._open = False
                self._started = self._typer.typed > 0
                # Retain the complete transcript if the bounded wait times out,
                # even when queued words never reached the injector.
                self._final_text = final_text
                if self._started:
                    remainder = self._typer.finish(final_text)
                    if remainder:
                        self._queue.put((remainder, True))
                self._queue.put(None)
        if not self._wait(getattr(self, '_final_text', None)):
            return InjectionOutcome.FAILED
        if self._failed or self._outcome == InjectionOutcome.FAILED:
            return InjectionOutcome.FAILED
        if not self._started:
            return None
        return self._outcome or InjectionOutcome.INJECTED

    def cancel(self) -> None:
        """Stop pending typing and wait a bounded time for worker cleanup."""
        with self._lock:
            if self._done.is_set():
                return
            self._failed = True
            self._handle.cancelled.set()
            if self._open:
                self._open = False
                self._queue.put(None)
        self._wait()
