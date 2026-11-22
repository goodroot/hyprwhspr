"""
Desktop-notification status presenter for hyprwhspr.

Drop-in alternative to MicOSDRunner for compositors that do NOT support the
layer-shell protocol (notably GNOME/Mutter). The layer-shell overlay degrades
to a focus-stealing toplevel window there, which swallows the post-dictation
paste keystroke. Notifications never take keyboard focus, so they show
recording status without interfering with injection.

Exposes the same method surface main.py calls on the OSD runner, so it can be
assigned to self._mic_osd_runner unchanged.
"""

import shutil
import sys
import threading
from pathlib import Path

try:
    from desktop_notify import STATUS_DETAIL_TEXT, close_notification, send_notification_with_id
except ImportError:
    src_dir = Path(__file__).resolve().parents[1] / 'src'
    sys.path.insert(0, str(src_dir))
    from desktop_notify import STATUS_DETAIL_TEXT, close_notification, send_notification_with_id


class NotificationPresenter:
    """Show recording status as a desktop notification (no focus stealing)."""

    _STATE_TEXT = {
        'recording': '🎤 Recording…',
        'paused': '⏸ Paused',
        'processing': 'Transcribing…',
        'success': '✓ Inserted',
        'error': '✗ Error',
    }
    _DETAIL_TEXT = STATUS_DETAIL_TEXT
    # Details that describe a wait; the rest are results and auto-dismiss.
    _WAIT_DETAILS = {'loading_recording', 'loading_captured'}
    # Results the user has to act on stay up as long as an alert would.
    _ACTION_DETAILS = {'clipboard', 'empty'}
    # States that represent a finished action and should auto-dismiss.
    _TRANSIENT_STATES = {'success', 'error'}
    # States that last until the next state replaces or closes them (timeout 0).
    _PERSISTENT_STATES = {'processing'}
    _APP_NAME = 'hyprwhspr'
    # All status bubbles carry the transient hint (see _send), so they never
    # land in the notification center. These timeouts only govern how long the
    # banner lingers on screen (GNOME ignores -t and uses its own duration).
    _ACTIVE_TIMEOUT_MS = 5000
    _TRANSIENT_TIMEOUT_MS = 2000
    # Live preview rides in the same bubble; keep it to the newest words.
    _PREVIEW_MAX_CHARS = 240

    def __init__(self, active_timeout_ms: int = None):
        self._nid = None  # active notification id (for in-place replace)
        self._state = 'recording'
        self._detail = None
        self._preview = ''
        # The wake thread and the recording thread both touch the bubble
        self._lock = threading.RLock()
        # Banner duration for active states; falls back to the class default.
        self._active_timeout_ms = self._ACTIVE_TIMEOUT_MS if active_timeout_ms is None else active_timeout_ms

    @staticmethod
    def is_available() -> bool:
        return shutil.which('notify-send') is not None or shutil.which('gdbus') is not None

    def _send(self, body: str, timeout_ms: int):
        """Show or replace the status notification. Captures/keeps the id so
        subsequent calls replace the same bubble instead of stacking."""
        # transient: bypass the server's persistence so status bubbles never
        # accumulate in the notification center (mirrors the ephemeral overlay).
        nid = send_notification_with_id(
            self._APP_NAME, body, urgency='normal', timeout_ms=timeout_ms,
            app_name=self._APP_NAME, replaces_id=self._nid, transient=True)
        if nid is not None:
            self._nid = nid

    def _close(self, nid: int):
        """Actively dismiss the notification so it doesn't linger in the
        notification center. GNOME keeps even transient notifications in its
        list, so letting it expire is not enough — we must close it by id."""
        close_notification(nid)

    # ---- interface mirrored from MicOSDRunner ----

    def show(self) -> bool:
        if not self.is_available():
            return False
        with self._lock:
            self._state = 'recording'
            self._detail = None
            self._send(self._STATE_TEXT['recording'], self._active_timeout_ms)
        return True

    def _state_timeout(self, state: str) -> int:
        if state in self._TRANSIENT_STATES:
            return self._TRANSIENT_TIMEOUT_MS
        if state in self._PERSISTENT_STATES:
            return 0
        return self._active_timeout_ms

    def set_state(self, state: str):
        if not self.is_available():
            return
        with self._lock:
            # Don't spawn a bubble from a state change before show() created one,
            # except for terminal states (success/error) which may stand alone.
            if self._nid is None and state not in self._TRANSIENT_STATES:
                return
            self._state = state
            self._detail = None
            self._send(self._body(state), self._state_timeout(state))

    def set_detail(self, detail: str) -> bool:
        """Reword the bubble for a wait or a result. False if it cannot show it.

        A wait keeps the current state's lifetime, so a model load that holds
        up "Transcribing…" stays until the result replaces it.
        """
        body = self._DETAIL_TEXT.get(detail)
        if body is None or not self.is_available():
            return False
        with self._lock:
            if detail in self._WAIT_DETAILS:
                if self._nid is None:
                    return False  # no bubble to update; the caller alerts instead
                timeout = self._state_timeout(self._state)
            elif detail in self._ACTION_DETAILS:
                timeout = max(self._TRANSIENT_TIMEOUT_MS, self._active_timeout_ms)
            else:
                timeout = self._TRANSIENT_TIMEOUT_MS
            self._detail = detail
            self._send(body, timeout)
        return True

    def clear_detail(self):
        """Drop a wait's wording and show the current state again."""
        with self._lock:
            if self._detail not in self._WAIT_DETAILS:
                return
            self._detail = None
            if self._nid is None:
                return
            self._send(self._body(self._state), self._state_timeout(self._state))

    def hide(self):
        # Actively close the bubble so it does not persist in the notification
        # center, then forget the id so the next recording starts fresh.
        with self._lock:
            nid, self._nid = self._nid, None
            self._detail = None
            self._preview = ''
        if nid is not None:
            self._close(nid)

    def clear_state(self):
        pass

    def _body(self, state: str) -> str:
        """State line, plus live preview text when there is any."""
        body = self._STATE_TEXT.get(state, self._STATE_TEXT['recording'])
        return f'{body}\n{self._preview}' if self._preview else body

    def set_preview_text(self, text: str):
        """Show live preview text under the status line (local backends).

        Preview never reaches the paste; it only fills the bubble while the
        user is still speaking.
        """
        text = (text or '').strip()
        if not text or not self.is_available():
            return
        with self._lock:
            if self._nid is None:
                return  # no bubble to update
            text = text[-self._PREVIEW_MAX_CHARS:]
            if text == self._preview:
                return
            self._preview = text
            self._send(self._body(self._state), self._state_timeout(self._state))

    def clear_preview_text(self):
        with self._lock:
            if not self._preview:
                return
            self._preview = ''
            if self._nid is not None:
                self._send(self._body(self._state), self._state_timeout(self._state))

    def stop(self):
        self.hide()
