"""
MPRIS media pausing for hyprwhspr
Pauses players that are actively playing during recording, and resumes them after.

Ducking a podcast to 50% doesn't help: you still miss what was said. Players that
speak MPRIS (Firefox, Chromium, Spotify, mpv, VLC, ...) can simply be paused and
resumed at the same position, leaving every volume untouched.

Only players hyprwhspr actually paused are resumed, and only if the same bus
connection still owns the name - a player that quit and restarted mid-recording
must not be handed a Play() it never asked for.
"""

import threading
from contextlib import contextmanager

try:
    import dbus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False


MPRIS_NAME_PREFIX = 'org.mpris.MediaPlayer2.'
MPRIS_PATH = '/org/mpris/MediaPlayer2'
MPRIS_PLAYER_IFACE = 'org.mpris.MediaPlayer2.Player'
PROPERTIES_IFACE = 'org.freedesktop.DBus.Properties'

# Every D-Bus call is made on the recording-start path, so a wedged player must
# never stall a recording. Keep these short.
CALL_TIMEOUT = 1.0


@contextmanager
def _session_bus():
    """A short-lived private session-bus connection.

    Private, because these calls are made from the recording thread while
    suspend_monitor.py runs a GLib main loop on the shared connections; a
    connection of our own keeps blocking calls out of that machinery.
    """
    bus = dbus.SessionBus(private=True)
    try:
        yield bus
    finally:
        try:
            bus.close()
        except Exception:
            pass


class MediaPauser:
    """Pauses/resumes MPRIS media players around a recording"""

    def __init__(self):
        # (well-known name, unique owner) pairs we paused, in pause order
        self._paused_players = []
        self._lock = threading.Lock()
        self._is_paused = False

        if not DBUS_AVAILABLE:
            print("[MEDIA_PAUSER] dbus-python not available, media pausing disabled")

    @staticmethod
    def _player_proxies(bus, name):
        """(properties interface, player interface) for one MPRIS bus name"""
        proxy = bus.get_object(name, MPRIS_PATH)
        return (dbus.Interface(proxy, PROPERTIES_IFACE),
                dbus.Interface(proxy, MPRIS_PLAYER_IFACE))

    @staticmethod
    def _connection_pid(bus, name):
        """PID behind a bus name, or None if the daemon won't say."""
        try:
            daemon = dbus.Interface(
                bus.get_object('org.freedesktop.DBus', '/org/freedesktop/DBus'),
                'org.freedesktop.DBus')
            return int(daemon.GetConnectionUnixProcessID(name, timeout=CALL_TIMEOUT))
        except Exception:
            return None

    def pause_all(self) -> list:
        """
        Pause every MPRIS player that is currently playing.

        Returns:
            PIDs of the players that were paused (empty list if none/unavailable).
            Callers use these to avoid also ducking those players' streams.
        """
        if not DBUS_AVAILABLE:
            return []

        with self._lock:
            if self._is_paused:
                return []  # Already paused; nothing new to report

            paused_pids = []
            try:
                with _session_bus() as bus:
                    names = [n for n in bus.list_names() if str(n).startswith(MPRIS_NAME_PREFIX)]

                    for name in names:
                        name = str(name)
                        try:
                            props, player = self._player_proxies(bus, name)
                            status = props.Get(MPRIS_PLAYER_IFACE, 'PlaybackStatus',
                                               timeout=CALL_TIMEOUT)
                            if str(status) != 'Playing':
                                continue
                            if not bool(props.Get(MPRIS_PLAYER_IFACE, 'CanPause',
                                                  timeout=CALL_TIMEOUT)):
                                continue

                            owner = str(bus.get_name_owner(name))
                            player.Pause(timeout=CALL_TIMEOUT)
                            self._paused_players.append((name, owner))

                            pid = self._connection_pid(bus, name)
                            if pid is not None:
                                paused_pids.append(pid)
                        except Exception as e:
                            # One uncooperative player must not abort the sweep
                            print(f"[MEDIA_PAUSER] Could not pause {name}: {e}", flush=True)

                self._is_paused = bool(self._paused_players)
                if self._paused_players:
                    print(f"[MEDIA_PAUSER] Paused {len(self._paused_players)} player(s)",
                          flush=True)
                return paused_pids

            except Exception as e:
                print(f"[MEDIA_PAUSER] Failed to pause media: {e}", flush=True)
                # Anything already paused stays recorded so restore() can resume it
                self._is_paused = bool(self._paused_players)
                return paused_pids

    def resume_all(self) -> bool:
        """
        Resume only the players hyprwhspr paused, and only if the same connection
        still owns the name and the player is still paused.

        Returns:
            True if resumption completed without error, False otherwise
        """
        if not DBUS_AVAILABLE:
            return False

        with self._lock:
            if not self._is_paused:
                return True  # Nothing paused, nothing to do

            ok = True
            try:
                with _session_bus() as bus:
                    resumed = 0
                    for name, owner in self._paused_players:
                        try:
                            if not bus.name_has_owner(name):
                                continue  # Player quit while we recorded
                            if str(bus.get_name_owner(name)) != owner:
                                continue  # Player restarted; not ours to resume
                            props, player = self._player_proxies(bus, name)
                            status = props.Get(MPRIS_PLAYER_IFACE, 'PlaybackStatus',
                                               timeout=CALL_TIMEOUT)
                            if str(status) != 'Paused':
                                continue  # User already moved on
                            player.Play(timeout=CALL_TIMEOUT)
                            resumed += 1
                        except Exception as e:
                            print(f"[MEDIA_PAUSER] Could not resume {name}: {e}", flush=True)
                            ok = False

                    if resumed > 0:
                        print(f"[MEDIA_PAUSER] Resumed {resumed} player(s)", flush=True)

            except Exception as e:
                print(f"[MEDIA_PAUSER] Failed to resume media: {e}", flush=True)
                ok = False

            # Clear state either way to avoid stuck pausing
            self._paused_players = []
            self._is_paused = False
            return ok

    @property
    def is_paused(self) -> bool:
        """Check if players are currently paused by hyprwhspr"""
        with self._lock:
            return self._is_paused
