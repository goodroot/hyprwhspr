"""Desktop notifications that don't pile up in the notification center.

GNOME Shell keeps notifications in its persistent list even when the freedesktop
`transient` hint is set or the expire-timeout elapses (verified on GNOME 49 —
the hint arrives over D-Bus as `boolean true` but is ignored for the list). So
non-critical notifications are actively dismissed via CloseNotification a short
time after they are shown. Critical notifications are left to persist until the
user dismisses them.

mako/dunst/swaync honor the transient hint and drop the notification on their
own; the explicit close is then a harmless no-op.

notification_level gates everything sent through notify() and
NotificationSlots. The recording-status bubble (mic_osd_enabled) calls
send_notification_with_id directly and is not gated.
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

LEVELS = ('all', 'action_required', 'off')

# What a notification is, for notification_level:
#   lifecycle  an automatic announcement (model load/unload, recovery progress)
#   result     the outcome of something the user just asked for
#   action     a problem the user has to act on
KINDS = ('lifecycle', 'result', 'action')

_level_provider = None


def set_level_provider(provider) -> None:
    """Install the callable that returns the configured notification_level."""
    global _level_provider
    _level_provider = provider


def notification_level() -> str:
    try:
        level = _level_provider() if _level_provider is not None else 'all'
    except Exception:
        level = 'all'
    return level if level in LEVELS else 'all'


def allowed(kind: str = 'action') -> bool:
    """Whether notification_level lets a notification of this kind through."""
    level = notification_level()
    if level == 'off':
        return False
    if level == 'action_required':
        return kind != 'lifecycle'
    return True


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
           timeout_ms: int = 5000, kind: str = "action"):
    """Show a desktop notification.

    Non-critical notifications auto-dismiss after `timeout_ms`; critical ones
    persist in the notification center until the user dismisses them.
    """
    if not allowed(kind):
        return
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


# Wording that refines the recording status without changing its state, shared
# by the status bubble and the alerts that stand in for it. "Inserted" is only
# shown for a confirmed paste; every other outcome names what happened.
STATUS_DETAIL_TEXT = {
    'loading_recording': 'Model loading — keep speaking',
    'loading_captured': 'Model loading — audio captured',
    'clipboard': 'Text copied — paste to insert',
    'consumed': 'Handled by hook',
    'empty': 'No speech recognized — try recording again',
    'done': '✓ Done',
}


class NotificationSlots:
    """One replaceable notification per named slot.

    A slot stands for one subject (the model, a recovery episode, text
    delivery): each message replaces the last instead of stacking. A temporary
    message closes itself after its timeout; a retained one stays until it is
    dismissed, replaced or cleared.

    Every show and clear starts a new generation, so the close scheduled for
    an earlier message can never close the one that replaced it.
    """

    def __init__(self, timer_factory=threading.Timer):
        self._lock = threading.Lock()
        self._timer_factory = timer_factory
        self._slots = {}  # name -> {'nid', 'generation', 'timer'}

    def _entry(self, slot):
        return self._slots.setdefault(slot, {'nid': None, 'generation': 0, 'timer': None})

    def _supersede(self, entry):
        entry['generation'] += 1
        timer, entry['timer'] = entry['timer'], None
        if timer is not None:
            timer.cancel()

    def show(self, slot: str, message: str, kind: str = 'action', retained: bool = False,
             urgency: str = 'normal', timeout_ms: int = 5000, title: str = 'hyprwhspr') -> bool:
        """Replace the slot's notification. False when notification_level hid it.

        A hidden message still retires what it would have replaced: that
        notification describes a state that is over.
        """
        if not allowed(kind):
            self.clear(slot)
            return False
        with self._lock:
            entry = self._entry(slot)
            self._supersede(entry)
            generation = entry['generation']
            if retained:
                nid = send_notification_with_id(
                    title, message, urgency='critical', timeout_ms=0,
                    replaces_id=entry['nid'], transient=False)
            else:
                nid = send_notification_with_id(
                    title, message, urgency=urgency, timeout_ms=timeout_ms,
                    replaces_id=entry['nid'])
            if nid is not None:
                entry['nid'] = nid
            if not retained and entry['nid'] is not None:
                # No sooner than GNOME's own banner duration; see notify()
                timer = self._timer_factory(
                    max(timeout_ms, 4000) / 1000.0, self._expire, args=(slot, generation))
                timer.daemon = True
                entry['timer'] = timer
                timer.start()
        return True

    def _expire(self, slot, generation):
        with self._lock:
            entry = self._slots.get(slot)
            if entry is None or entry['generation'] != generation:
                return  # replaced or cleared since; not ours to close
            nid, entry['nid'] = entry['nid'], None
            entry['timer'] = None
        if nid is not None:
            close_notification(nid)

    def clear(self, slot: str) -> None:
        """Close the slot's notification, if one may still be showing."""
        with self._lock:
            entry = self._slots.get(slot)
            if entry is None:
                return
            self._supersede(entry)
            nid, entry['nid'] = entry['nid'], None
        if nid is not None:
            close_notification(nid)

    def active(self, slot: str) -> bool:
        with self._lock:
            entry = self._slots.get(slot)
            return entry is not None and entry['nid'] is not None
