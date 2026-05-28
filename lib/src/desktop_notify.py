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
import re
from typing import Optional


_URGENCY_BYTE = {
    'low': 0,
    'normal': 1,
    'critical': 2,
}


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


def _parse_notification_id(output) -> Optional[int]:
    text = (output or b'').decode('utf-8', 'ignore').strip()
    if not text:
        return None
    match = re.search(r'uint32\s+(\d+)', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    match = re.search(r'^\d+$', text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def send_notification_with_id(
    title: str,
    message: str,
    urgency: str = "normal",
    timeout_ms: int = 5000,
    app_name: str = "hyprwhspr",
    replaces_id: Optional[int] = None,
    icon: str = "",
    transient: bool = True,
) -> Optional[int]:
    """Show a notification and return its server id when available.

    Modern notify-send supports `-p`; older libnotify 0.7 does not, so fall back
    to the freedesktop Notifications D-Bus method directly.
    """
    if shutil.which('notify-send') is not None:
        cmd = ['notify-send', '-a', app_name, '-u', urgency, '-t', str(timeout_ms)]
        if transient:
            cmd += ['-h', 'boolean:transient:true']
        if icon:
            cmd += ['-i', icon]
        if replaces_id is not None:
            cmd += ['-r', str(replaces_id)]
        cmd += ['-p', title, message]
        try:
            result = subprocess.run(cmd, timeout=2, check=False, capture_output=True)
            nid = _parse_notification_id(result.stdout)
            if nid is not None:
                return nid
        except Exception:
            pass

    if shutil.which('gdbus') is None:
        return None

    hints = [f"'urgency': <byte {_URGENCY_BYTE.get(urgency, 1)}>"]
    if transient:
        hints.append("'transient': <true>")
    replace = int(replaces_id or 0)
    try:
        result = subprocess.run(
            ['gdbus', 'call', '--session',
             '--dest', 'org.freedesktop.Notifications',
             '--object-path', '/org/freedesktop/Notifications',
             '--method', 'org.freedesktop.Notifications.Notify',
             app_name, str(replace), icon, title, message, '[]',
             '{' + ', '.join(hints) + '}', str(timeout_ms)],
            timeout=2, check=False, capture_output=True)
        if result.returncode == 0:
            return _parse_notification_id(result.stdout)
    except Exception:
        pass
    return None


def notify(title: str, message: str, urgency: str = "normal",
           timeout_ms: int = 5000):
    """Show a desktop notification.

    Non-critical notifications auto-dismiss after `timeout_ms`; critical ones
    persist in the notification center until the user dismisses them.
    """
    if shutil.which('notify-send') is None and shutil.which('gdbus') is None:
        return
    try:
        if urgency == "critical":
            # Real error: persist in the center, never expire.
            send_notification_with_id(
                title, message, urgency='critical', timeout_ms=0,
                transient=False)
            return
        # Informational: transient hint (honored by mako/dunst/swaync) plus the
        # printed id so we can actively close it later (GNOME ignores the hint).
        nid = send_notification_with_id(
            title, message, urgency=urgency, timeout_ms=timeout_ms)
        if nid is not None:
            # Close no sooner than GNOME's own banner duration (~4s) so the
            # banner is actually seen before it leaves the list.
            _close_later(nid, max(timeout_ms, 4000) / 1000.0)
    except Exception:
        pass
