import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import media_pauser


class FakePlayer:
    """One MPRIS player on the fake bus."""

    def __init__(self, status="Playing", can_pause=True, pid=1000,
                 pause_error=None, play_error=None):
        self.status = status
        self.can_pause = can_pause
        self.pid = pid
        self.pause_error = pause_error
        self.play_error = play_error
        self.paused = 0
        self.played = 0

    def Get(self, iface, prop, timeout=None):
        if prop == "PlaybackStatus":
            return self.status
        if prop == "CanPause":
            return self.can_pause
        raise KeyError(prop)

    def Pause(self, timeout=None):
        if self.pause_error:
            raise self.pause_error
        self.paused += 1
        self.status = "Paused"

    def Play(self, timeout=None):
        if self.play_error:
            raise self.play_error
        self.played += 1
        self.status = "Playing"


class FakeBus:
    def __init__(self, players, owners=None):
        self.players = players  # {bus name: FakePlayer}
        self.owners = owners or {name: f":1.{i}" for i, name in enumerate(players)}

    def list_names(self):
        return list(self.players) + ["org.freedesktop.DBus", "org.gnome.Shell"]

    def get_object(self, name, path):
        if name == "org.freedesktop.DBus":
            return "daemon"
        return self.players[name]

    def get_name_owner(self, name):
        return self.owners[name]

    def name_has_owner(self, name):
        return name in self.owners

    def close(self):
        self.closed = True


class FakeDaemon:
    def __init__(self, bus):
        self._bus = bus

    def GetConnectionUnixProcessID(self, name, timeout=None):
        return self._bus.players[name].pid


def fake_dbus(bus):
    """A media_pauser.dbus stand-in whose Interface() hands back the target itself."""
    def interface(target, iface_name):
        if target == "daemon":
            return FakeDaemon(bus)
        return target

    return types.SimpleNamespace(
        SessionBus=mock.Mock(return_value=bus),  # ignores private=True
        Interface=interface,
    )


class MediaPauserTests(unittest.TestCase):
    def _pauser(self, bus):
        pauser = media_pauser.MediaPauser()
        return pauser, mock.patch.multiple(
            media_pauser,
            DBUS_AVAILABLE=True,
            dbus=fake_dbus(bus),
            create=True,  # dbus-python is optional; the name may not exist
        )

    def test_pauses_only_playing_pausable_players(self):
        playing = FakePlayer(status="Playing", pid=111)
        stopped = FakePlayer(status="Stopped", pid=222)
        unpausable = FakePlayer(status="Playing", can_pause=False, pid=333)
        bus = FakeBus({
            "org.mpris.MediaPlayer2.firefox": playing,
            "org.mpris.MediaPlayer2.vlc": stopped,
            "org.mpris.MediaPlayer2.weird": unpausable,
        })
        pauser, patch = self._pauser(bus)

        with patch:
            pids = pauser.pause_all()

        self.assertEqual(pids, [111])
        self.assertEqual(playing.paused, 1)
        self.assertEqual(stopped.paused, 0)
        self.assertEqual(unpausable.paused, 0)
        self.assertTrue(pauser.is_paused)

    def test_one_broken_player_does_not_abort_the_sweep(self):
        broken = FakePlayer(pid=111, pause_error=RuntimeError("no reply"))
        healthy = FakePlayer(pid=222)
        bus = FakeBus({
            "org.mpris.MediaPlayer2.broken": broken,
            "org.mpris.MediaPlayer2.spotify": healthy,
        })
        pauser, patch = self._pauser(bus)

        with patch:
            pids = pauser.pause_all()

        self.assertEqual(pids, [222])
        self.assertEqual(healthy.paused, 1)

    def test_resume_skips_player_whose_owner_changed(self):
        restarted = FakePlayer(pid=111)
        kept = FakePlayer(pid=222)
        bus = FakeBus({
            "org.mpris.MediaPlayer2.firefox": restarted,
            "org.mpris.MediaPlayer2.mpv": kept,
        }, owners={
            "org.mpris.MediaPlayer2.firefox": ":1.5",
            "org.mpris.MediaPlayer2.mpv": ":1.6",
        })
        pauser, patch = self._pauser(bus)

        with patch:
            pauser.pause_all()
            # firefox quit and came back with a fresh connection
            bus.owners["org.mpris.MediaPlayer2.firefox"] = ":1.9"
            self.assertTrue(pauser.resume_all())

        self.assertEqual(restarted.played, 0)
        self.assertEqual(kept.played, 1)
        self.assertFalse(pauser.is_paused)

    def test_resume_is_quiet_about_a_player_that_quit(self):
        player = FakePlayer(pid=111)
        bus = FakeBus({"org.mpris.MediaPlayer2.mpv": player})
        pauser, patch = self._pauser(bus)

        with patch:
            pauser.pause_all()
            bus.owners.pop("org.mpris.MediaPlayer2.mpv")
            self.assertTrue(pauser.resume_all())  # a quit player is not a failure

        self.assertEqual(player.played, 0)

    def test_resume_skips_player_no_longer_paused(self):
        player = FakePlayer(pid=111)
        bus = FakeBus({"org.mpris.MediaPlayer2.mpv": player})
        pauser, patch = self._pauser(bus)

        with patch:
            pauser.pause_all()
            player.status = "Stopped"  # user moved on during the recording
            pauser.resume_all()

        self.assertEqual(player.played, 0)

    def test_resume_clears_state_even_when_a_call_raises(self):
        player = FakePlayer(pid=111, play_error=RuntimeError("gone"))
        bus = FakeBus({"org.mpris.MediaPlayer2.mpv": player})
        pauser, patch = self._pauser(bus)

        with patch:
            pauser.pause_all()
            self.assertFalse(pauser.resume_all())

        self.assertFalse(pauser.is_paused)
        self.assertEqual(pauser._paused_players, [])

    def test_sweep_that_paused_nothing_leaves_no_state_to_restore(self):
        bus = FakeBus({"org.mpris.MediaPlayer2.vlc": FakePlayer(status="Paused")})
        pauser, patch = self._pauser(bus)

        with patch:
            self.assertEqual(pauser.pause_all(), [])

        self.assertFalse(pauser.is_paused)

    def test_pause_is_reentrant(self):
        player = FakePlayer(pid=111)
        bus = FakeBus({"org.mpris.MediaPlayer2.mpv": player})
        pauser, patch = self._pauser(bus)

        with patch:
            pauser.pause_all()
            self.assertEqual(pauser.pause_all(), [])

        self.assertEqual(player.paused, 1)

    def test_bus_is_opened_once_and_never_closed(self):
        # A connection closed (or GC'd) under DBusGMainLoop leaves the GLib loop
        # dispatching on freed memory and segfaults the service mid-recording.
        player = FakePlayer(pid=111)
        bus = FakeBus({"org.mpris.MediaPlayer2.mpv": player})
        pauser = media_pauser.MediaPauser()
        fake = fake_dbus(bus)

        with mock.patch.multiple(media_pauser, DBUS_AVAILABLE=True, dbus=fake, create=True):
            pauser.pause_all()
            pauser.resume_all()
            pauser.pause_all()
            pauser.resume_all()

        self.assertEqual(fake.SessionBus.call_count, 1)
        self.assertFalse(getattr(bus, "closed", False))

    def test_no_dbus_is_a_clean_noop(self):
        with mock.patch.object(media_pauser, "DBUS_AVAILABLE", False):
            pauser = media_pauser.MediaPauser()
            self.assertEqual(pauser.pause_all(), [])
            self.assertFalse(pauser.resume_all())
            self.assertFalse(pauser.is_paused)


if __name__ == "__main__":
    unittest.main()
