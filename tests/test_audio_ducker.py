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
        # Corked streams need no ducking.
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
    """Recover replacements that inherit ducked volumes."""

    def test_replacement_stream_at_ducked_level_is_healed_on_restore(self):
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)

        with patched(pulse):
            ducker.duck()
            # The replacement inherits the ducked level.
            pulse.streams = [FakeStream(2, pid=101, name="Chrome", binary="chrome", volume=0.5)]
            pulse.set_volumes.clear()
            ducker.restore()

        self.assertAlmostEqual(pulse.set_volumes[2], 1.0)

    def test_app_stuck_at_ducked_level_heals_on_the_next_cycle(self):
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


class DuckerSelfHealEdgeTests(unittest.TestCase):
    def _leave_chrome_pending(self, ducker, pulse):
        pulse.streams = [FakeStream(1, pid=100, name="Chrome", binary="chrome")]
        ducker.duck()
        pulse.streams = []  # ducked stream ended before restore
        ducker.restore()

    def test_pending_survives_a_fresh_stream_of_the_same_app_listed_first(self):
        # A fresh stream must not consume another stream's recovery.
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            self._leave_chrome_pending(ducker, pulse)
            pulse.streams = [
                FakeStream(10, pid=101, name="Chrome", binary="chrome", volume=1.0),
                FakeStream(11, pid=102, name="Chrome", binary="chrome", volume=0.5),
            ]
            ducker.duck()
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[10], 1.0)
        self.assertAlmostEqual(pulse.set_volumes[11], 1.0)

    def test_every_stuck_stream_of_the_app_is_healed(self):
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            self._leave_chrome_pending(ducker, pulse)
            pulse.streams = [
                FakeStream(12, pid=101, name="Chrome", binary="chrome", volume=0.5),
                FakeStream(13, pid=102, name="Chrome", binary="chrome", volume=0.5),
            ]
            ducker.duck()
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[12], 1.0)
        self.assertAlmostEqual(pulse.set_volumes[13], 1.0)

    def test_reduction_change_between_duck_and_restore_still_heals(self):
        # Match the applied level, even after a configuration change.
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            ducker.duck()
            pulse.streams = []
            ducker.set_reduction_percent(20)
            ducker.restore()
            pulse.streams = [FakeStream(2, pid=101, name="Chrome", binary="chrome", volume=0.5)]
            ducker.duck()
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[2], 1.0)

    def test_failed_restore_keeps_unrestored_streams_pending(self):
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome")])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            ducker.duck()
            original_set = pulse.volume_set_all_chans

            def fail(stream, volume):
                raise RuntimeError("pulse died")

            pulse.volume_set_all_chans = fail
            self.assertFalse(ducker.restore())
            pulse.volume_set_all_chans = original_set
            pulse.streams = [FakeStream(2, pid=101, name="Chrome", binary="chrome", volume=0.5)]
            ducker.duck()
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[2], 1.0)

    def test_streams_naming_no_application_are_never_tracked(self):
        pulse = FakePulse([FakeStream(1, pid=100, name=None, binary=None)])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            ducker.duck()
            # An unrelated anonymous stream happens to sit at 0.5.
            pulse.streams = [FakeStream(2, pid=200, name=None, binary=None, volume=0.5)]
            pulse.set_volumes.clear()
            ducker.restore()
        self.assertNotIn(2, pulse.set_volumes)

    def test_pending_entry_lives_through_the_very_next_cycle(self):
        # Unrelated streams in the same cycle must not age it out early.
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            self._leave_chrome_pending(ducker, pulse)
            pulse.streams = [
                FakeStream(20, pid=200, name="Game", binary="game"),
                FakeStream(21, pid=101, name="Chrome", binary="chrome", volume=0.5),
            ]
            ducker.duck()
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[21], 1.0)

    def test_pending_entry_expires_after_one_silent_cycle(self):
        # Expired levels are taken at face value.
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            self._leave_chrome_pending(ducker, pulse)
            pulse.streams = [FakeStream(20, pid=200, name="Game", binary="game")]
            ducker.duck()
            ducker.restore()

            pulse.streams = [FakeStream(22, pid=101, name="Chrome", binary="chrome", volume=0.5)]
            ducker.duck()
            self.assertAlmostEqual(pulse.set_volumes[22], 0.25)
            ducker.restore()
        self.assertAlmostEqual(pulse.set_volumes[22], 0.5)

    def test_quiet_neighbour_is_not_mistaken_for_a_ducked_stream(self):
        # Nearby quiet levels must remain distinct.
        pulse = FakePulse([FakeStream(1, pid=100, name="Chrome", binary="chrome", volume=0.02)])
        ducker = audio_ducker.AudioDucker(reduction_percent=50)
        with patched(pulse):
            ducker.duck()
            pulse.streams = [FakeStream(3, pid=101, name="Chrome", binary="chrome", volume=0.018)]
            pulse.set_volumes.clear()
            ducker.restore()
        self.assertNotIn(3, pulse.set_volumes)


class DuckerPendingPairTests(unittest.TestCase):
    def _leave_pending(self, ducker, pulse, volumes):
        pulse.streams = [FakeStream(i, pid=100 + i, volume=volume)
                         for i, volume in enumerate(volumes)]
        self.assertTrue(ducker.duck())
        pulse.streams = []
        self.assertTrue(ducker.restore())

    def test_distinct_levels_heal_regardless_of_order_and_arrival(self):
        for reverse in (False, True):
            for next_cycle in (False, True):
                with self.subTest(reverse=reverse, next_cycle=next_cycle):
                    streams = [FakeStream(1, volume=1.0), FakeStream(2, volume=0.6)]
                    pulse = FakePulse(streams[::-1] if reverse else streams)
                    ducker = audio_ducker.AudioDucker(50)
                    with patched(pulse):
                        ducker.duck()
                        if next_cycle:
                            pulse.streams = []
                            ducker.restore()
                        replacements = [FakeStream(3, volume=0.5), FakeStream(4, volume=0.3)]
                        pulse.streams = replacements[::-1] if reverse else replacements
                        if next_cycle:
                            ducker.duck()
                        self.assertTrue(ducker.restore())
                    self.assertEqual([s.volume.values[0] for s in replacements], [1.0, 0.6])
                    self.assertFalse(ducker._pending_restores)

    def test_pending_match_survives_for_late_or_corked_sibling(self):
        for corked in (False, True):
            with self.subTest(corked=corked):
                pulse = FakePulse([])
                ducker = audio_ducker.AudioDucker(50)
                with patched(pulse):
                    self._leave_pending(ducker, pulse, [1.0])
                    first = FakeStream(10, volume=0.5)
                    sibling = FakeStream(11, volume=0.5, corked=corked)
                    pulse.streams = [first, sibling] if corked else [first]
                    ducker.duck()
                    if not corked:
                        pulse.streams.append(sibling)
                    self.assertTrue(ducker.restore())
                self.assertEqual(first.volume.values, [1.0, 1.0])
                self.assertEqual(sibling.volume.values, [1.0, 1.0])
                self.assertFalse(ducker._pending_restores)

    def test_consuming_one_pair_preserves_another(self):
        pulse = FakePulse([FakeStream(1, volume=1.0), FakeStream(2, volume=0.6)])
        ducker = audio_ducker.AudioDucker(50)
        with patched(pulse):
            ducker.duck()
            pulse.streams = [FakeStream(10, volume=0.5)]
            ducker.restore()
            self.assertEqual(ducker._pending_restores, {('player', 'player'): {(0.6, 0.3): 1}})
            pulse.streams = [FakeStream(11, volume=0.3)]
            ducker.duck()
            ducker.restore()
        self.assertEqual(pulse.streams[0].volume.values, [0.6, 0.6])

    def test_pairs_expire_independently(self):
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(50)
        with patched(pulse):
            self._leave_pending(ducker, pulse, [1.0])
            self._leave_pending(ducker, pulse, [0.6])
            pulse.streams = [FakeStream(10, volume=0.5), FakeStream(11, volume=0.3)]
            ducker.duck()
            ducker.restore()
        self.assertEqual([s.volume.values[0] for s in pulse.streams], [0.5, 0.6])
        self.assertFalse(ducker._pending_restores)

    def test_duplicate_pairs_heal_all_replacements(self):
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(50)
        with patched(pulse):
            self._leave_pending(ducker, pulse, [1.0, 1.0])
            self.assertEqual(ducker._pending_restores, {('player', 'player'): {(1.0, 0.5): 1}})
            pulse.streams = [FakeStream(10, volume=0.5), FakeStream(11, volume=0.5)]
            ducker.duck()
            ducker.restore()
        self.assertEqual([s.volume.values[0] for s in pulse.streams], [1.0, 1.0])

    def test_ambiguous_matches_are_not_healed(self):
        for reduction, originals, replacement in ((100, [1.0, 0.6], 0.0),
                                                  (50, [1.0, 0.99], 0.497)):
            for next_cycle in (False, True):
                with self.subTest(reduction=reduction, next_cycle=next_cycle):
                    pulse = FakePulse([FakeStream(i, volume=v) for i, v in enumerate(originals)])
                    ducker = audio_ducker.AudioDucker(reduction)
                    with patched(pulse):
                        ducker.duck()
                        if next_cycle:
                            pulse.streams = []
                            ducker.restore()
                        pulse.streams = [FakeStream(10, volume=replacement)]
                        if next_cycle:
                            ducker.duck()
                        ducker.restore()
                    self.assertEqual(pulse.streams[0].volume.values, [replacement, replacement])

    def test_disappearing_matched_stream_renews_recovery(self):
        pulse = FakePulse([])
        ducker = audio_ducker.AudioDucker(50)
        with patched(pulse):
            self._leave_pending(ducker, pulse, [1.0])
            self._leave_pending(ducker, pulse, [0.5])
            pulse.streams = [FakeStream(10, volume=0.5)]
            ducker.duck()
            ducker.restore()
        self.assertEqual(pulse.streams[0].volume.values, [1.0, 1.0])

    def test_partial_healing_failure_retains_both_pairs(self):
        pulse = FakePulse([FakeStream(1, volume=1.0), FakeStream(2, volume=0.6)])
        ducker = audio_ducker.AudioDucker(50)
        with patched(pulse):
            ducker.duck()
            pulse.streams = [FakeStream(10, volume=0.5), FakeStream(11, volume=0.3)]
            original_set = pulse.volume_set_all_chans

            def fail_second(stream, volume):
                if stream.index == 11:
                    raise RuntimeError("pulse died")
                original_set(stream, volume)

            with mock.patch.object(pulse, 'volume_set_all_chans', side_effect=fail_second):
                self.assertFalse(ducker.restore())
            self.assertFalse(ducker.is_ducked)
            self.assertEqual(len(ducker._pending_restores[('player', 'player')]), 2)
            ducker.duck()
            ducker.restore()
        self.assertEqual([s.volume.values[0] for s in pulse.streams], [1.0, 0.6])


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
