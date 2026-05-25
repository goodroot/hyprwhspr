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
import subprocess


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

    def __init__(self):
        self._nid = None  # active notification id (for in-place replace)

    @staticmethod
    def is_available() -> bool:
        return shutil.which('notify-send') is not None

    def _send(self, body: str, timeout_ms: int):
        """Show or replace the status notification. Captures/keeps the id so
        subsequent calls replace the same bubble instead of stacking."""
        cmd = ['notify-send', '-a', self._APP_NAME, '-u', 'normal',
               '-t', str(timeout_ms), '-p']
        if self._nid is not None:
            cmd += ['-r', str(self._nid)]
        cmd += [self._APP_NAME, body]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            out = (result.stdout or b'').decode('utf-8', 'ignore').strip()
            if out:
                try:
                    self._nid = int(out)
                except ValueError:
                    pass
        except Exception:
            pass

    # ---- interface mirrored from MicOSDRunner ----

    # Finite timeouts so notifications auto-expire and never accumulate.
    # (-t 0 / "never expire" notifications could pile up and wedge GNOME's
    # banner display.) GNOME hides the banner after a few seconds regardless;
    # the timeout governs how long it lingers in the list.
    _ACTIVE_TIMEOUT_MS = 5000
    _TRANSIENT_TIMEOUT_MS = 2000

    def show(self) -> bool:
        if not self.is_available():
            return False
        self._send(self._STATE_TEXT['recording'], self._ACTIVE_TIMEOUT_MS)
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
                   else self._ACTIVE_TIMEOUT_MS)
        self._send(body, timeout)

    def hide(self):
        # Let the last notification expire on its own timeout; just forget the
        # id so the next recording starts a fresh (banner-popping) notification.
        self._nid = None

    def clear_state(self):
        pass

    def set_preview_text(self, text: str):
        # Local backends produce no live partials; notification shows status only.
        pass

    def clear_preview_text(self):
        pass

    def stop(self):
        self._nid = None
