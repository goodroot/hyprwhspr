"""Desktop notifications that don't pile up in the notification center.

GNOME Shell keeps notifications in its persistent list even when the freedesktop
`transient` hint is set or the expire-timeout elapses (verified on GNOME 49 —
the hint arrives over D-Bus as `boolean true` but is ignored for the list). So
non-critical notifications are actively dismissed via CloseNotification a short
time after they are shown. Critical notifications are left to persist until the
user dismisses them.

mako/dunst/swaync honor the transient hint and drop the notification on their
own; the explicit close is then a harmless no-op.
"""

import shutil
import subprocess
import threading


def close_notification(nid: int):
    """Remove a notification (banner + notification-center entry) by its id."""
    try:
        subprocess.run(
            ['gdbus', 'call', '--session',
             '--dest', 'org.freedesktop.Notifications',
             '--object-path', '/org/freedesktop/Notifications',
             '--method', 'org.freedesktop.Notifications.CloseNotification',
             str(nid)],
            capture_output=True, timeout=2, check=False)
    except Exception:
        pass


def _close_later(nid: int, delay_s: float):
    timer = threading.Timer(delay_s, close_notification, args=(nid,))
    timer.daemon = True
    timer.start()


def notify(title: str, message: str, urgency: str = "normal",
           timeout_ms: int = 5000):
    """Show a desktop notification.

    Non-critical notifications auto-dismiss after `timeout_ms`; critical ones
    persist in the notification center until the user dismisses them.
    """
    if shutil.which('notify-send') is None:
        return
    try:
        if urgency == "critical":
            # Real error: persist in the center, never expire.
            subprocess.run(
                ['notify-send', '-u', 'critical', '-t', '0', title, message],
                timeout=2, check=False, capture_output=True)
            return
        # Informational: transient hint (honored by mako/dunst/swaync) plus the
        # printed id so we can actively close it later (GNOME ignores the hint).
        result = subprocess.run(
            ['notify-send', '-u', urgency, '-t', str(timeout_ms),
             '-h', 'boolean:transient:true', '-p', title, message],
            timeout=2, check=False, capture_output=True)
        out = (result.stdout or b'').decode('utf-8', 'ignore').strip()
        if out:
            try:
                # Close no sooner than GNOME's own banner duration (~4s) so the
                # banner is actually seen before it leaves the list.
                _close_later(int(out), max(timeout_ms, 4000) / 1000.0)
            except ValueError:
                pass
    except Exception:
        pass
