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

from desktop_notify import close_notification, send_notification_with_id


class NotificationPresenter:
    """Show recording status as a desktop notification (no focus stealing)."""

    _STATE_TEXT = {
        'recording': '🎤 Recording…',
        'paused': '⏸ Paused',
        'processing': 'Transcribing…',
        'success': '✓ Inserted',
        'error': '✗ Error',
    }
    # States that represent a finished action and should auto-dismiss.
    _TRANSIENT_STATES = {'success', 'error'}
    _APP_NAME = 'hyprwhspr'
    # All status bubbles carry the transient hint (see _send), so they never
    # land in the notification center. These timeouts only govern how long the
    # banner lingers on screen (GNOME ignores -t and uses its own duration).
    _ACTIVE_TIMEOUT_MS = 5000
    _TRANSIENT_TIMEOUT_MS = 2000

    def __init__(self, active_timeout_ms: int = None):
        self._nid = None  # active notification id (for in-place replace)
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
        self._send(self._STATE_TEXT['recording'], self._active_timeout_ms)
        return True

    def set_state(self, state: str):
        if not self.is_available():
            return
        # Don't spawn a bubble from a state change before show() created one,
        # except for terminal states (success/error) which may stand alone.
        if self._nid is None and state not in self._TRANSIENT_STATES:
            return
        body = self._STATE_TEXT.get(state, self._STATE_TEXT['recording'])
        timeout = (self._TRANSIENT_TIMEOUT_MS if state in self._TRANSIENT_STATES
                   else self._active_timeout_ms)
        self._send(body, timeout)

    def hide(self):
        # Actively close the bubble so it does not persist in the notification
        # center, then forget the id so the next recording starts fresh.
        if self._nid is not None:
            self._close(self._nid)
        self._nid = None

    def clear_state(self):
        pass

    def set_preview_text(self, text: str):
        # Local backends produce no live partials; notification shows status only.
        pass

    def clear_preview_text(self):
        pass

    def stop(self):
        if self._nid is not None:
            self._close(self._nid)
        self._nid = None
