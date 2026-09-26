import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import audio_ducker


class FakeStream:
    def __init__(self, index, pid=None, name="player", binary="player",
                 volume=1.0, corked=False):
        self.index = index
        self.proplist = {
            'application.process.id': str(pid) if pid is not None else None,
            'application.name': name,
            'application.process.binary': binary,
        }
        self.volume = types.SimpleNamespace(values=[volume, volume])
        self.corked = corked


class FakePulse:
    def __init__(self, streams):
        self.streams = streams
        self.set_volumes = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sink_input_list(self):
        return self.streams

    def volume_set_all_chans(self, stream, volume):
        self.set_volumes[stream.index] = volume
        stream.volume = types.SimpleNamespace(volume=volume, values=[volume, volume])


def patched(pulse):
    return mock.patch.multiple(
        audio_ducker,
        PULSECTL_AVAILABLE=True,
        pulsectl=types.SimpleNamespace(Pulse=mock.Mock(return_value=pulse)),
        create=True,
    )


class AudioDuckerTests(unittest.TestCase):
    def test_ducks_other_streams_but_not_our_own_feedback_sounds(self):
        music = FakeStream(1, pid=100, name="Firefox", binary="firefox")
        ping = FakeStream(2, pid=101, name="paplay", binary="paplay")
        pulse = FakePulse([music, ping])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            self.assertTrue(ducker.duck())

        self.assertEqual(pulse.set_volumes, {1: 0.5})
        self.assertTrue(ducker.is_ducked)

    def test_skip_pids_leaves_paused_players_alone(self):
        paused = FakeStream(1, pid=100, name="Firefox", binary="firefox")
        game = FakeStream(2, pid=200, name="Game", binary="game")
        pulse = FakePulse([paused, game])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse), mock.patch.object(
                audio_ducker, "_pid_ancestry", side_effect=lambda pid, **kw: [pid]):
            ducker.duck(skip_pids=[100])

        self.assertEqual(pulse.set_volumes, {2: 0.5})

    def test_skip_pids_matches_a_child_audio_process(self):
        # Firefox owns the MPRIS name in the parent; audio comes from a child
        child = FakeStream(1, pid=555, name="Firefox", binary="firefox")
        pulse = FakePulse([child])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse), mock.patch.object(
                audio_ducker, "_pid_ancestry", return_value=[555, 100]):
            ducker.duck(skip_pids=[100])

        self.assertEqual(pulse.set_volumes, {})

    def test_corked_streams_are_never_ducked(self):
        # Nothing to gain (they're silent) and something to lose: if the stream
        # goes away, stream-restore hands the ducked volume to its replacement.
        corked = FakeStream(1, pid=100, corked=True)
        playing = FakeStream(2, pid=200)
        pulse = FakePulse([corked, playing])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()

        self.assertEqual(pulse.set_volumes, {2: 0.5})

    def test_partial_duck_failure_stays_restorable(self):
        first = FakeStream(1, pid=100)
        pulse = FakePulse([first, FakeStream(2, pid=200)])
        original_set = pulse.volume_set_all_chans

        def explode_after_first(stream, volume):
            if stream.index == 2:
                raise RuntimeError("pulse died")
            original_set(stream, volume)

        pulse.volume_set_all_chans = explode_after_first
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            self.assertFalse(ducker.duck())
            # The stream we already lowered must still be restorable
            self.assertTrue(ducker.is_ducked)
            pulse.volume_set_all_chans = original_set
            pulse.streams = [first]
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[1], 1.0)

    def test_restore_skips_a_reused_stream_index(self):
        stream = FakeStream(1, pid=100, name="Firefox", binary="firefox")
        pulse = FakePulse([stream])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            imposter = FakeStream(1, pid=999, name="Other", binary="other", volume=0.2)
            pulse.streams = [imposter]
            pulse.set_volumes.clear()
            self.assertTrue(ducker.restore())

        self.assertEqual(pulse.set_volumes, {})
        self.assertFalse(ducker.is_ducked)

    def test_restore_returns_volume_and_clears_state(self):
        stream = FakeStream(1, pid=100, volume=0.8)
        pulse = FakePulse([stream])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            pulse.set_volumes.clear()
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[1], 0.8)
        self.assertFalse(ducker.is_ducked)


class DuckerSelfHealTests(unittest.TestCase):
    """A ducked stream that ends before restore() leaves PulseAudio's
    stream-restore database holding the ducked volume for its app, so the
    app's next stream starts ducked. The ducker must heal that, not adopt it
    as the app's real volume."""

    def test_replacement_stream_at_ducked_level_is_healed_on_restore(self):
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            # Chrome ended the stream mid-recording and started a new one,
            # which stream-restore opened at the ducked 0.5.
            pulse.streams = [FakeStream(2, pid=101, name="Chrome", binary="chrome", volume=0.5)]
            pulse.set_volumes.clear()
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[2], 1.0)

    def test_app_stuck_at_ducked_level_heals_on_the_next_cycle(self):
        # The reported failure: the ducked stream is gone at restore with no
        # replacement yet, so the next duck used to record 0.5 as "original".
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            pulse.streams = []
            ducker.restore()

            pulse.streams = [FakeStream(3, pid=102, name="Chrome", binary="chrome", volume=0.5)]
            ducker.duck()
            self.assertAlmostEqual(pulse.set_volumes[3], 0.5)  # 50% of the real 1.0
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[3], 1.0)

    def test_volume_changed_by_the_user_is_not_overridden(self):
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            pulse.streams = []
            ducker.restore()

            # Not at the ducked level: the user picked 0.7 since.
            pulse.streams = [FakeStream(4, pid=103, name="Chrome", binary="chrome", volume=0.7)]
            ducker.duck()
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[4], 0.7)

    def test_other_app_at_the_ducked_level_is_left_alone(self):
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            pulse.streams = [FakeStream(5, pid=200, name="Firefox", binary="firefox", volume=0.5)]
            pulse.set_volumes.clear()
            ducker.restore()

        self.assertNotIn(5, pulse.set_volumes)


class PidAncestryTests(unittest.TestCase):
    def test_walks_parents_from_proc(self):
        chain = {5: 4, 4: 3, 3: 1}

        def fake_open(path, *args, **kwargs):
            pid = int(str(path).split('/')[2])
            return mock.mock_open(read_data=f"Name:\tx\nPPid:\t{chain[pid]}\n")()

        with mock.patch("builtins.open", fake_open):
            self.assertEqual(audio_ducker._pid_ancestry(5), [5, 4, 3])

    def test_unreadable_proc_entry_ends_the_walk(self):
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(audio_ducker._pid_ancestry(42), [42])


if __name__ == "__main__":
    unittest.main()
