import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import playback_suppressor


class PlaybackSuppressorTests(unittest.TestCase):
    def setUp(self):
        self.suppressor = playback_suppressor.PlaybackSuppressor(reduction_percent=40)
        self.ducker = mock.Mock()
        self.ducker.duck.return_value = True
        self.ducker.restore.return_value = True
        self.ducker.is_ducked = True
        self.pauser = mock.Mock()
        self.pauser.pause_all.return_value = [111]
        self.pauser.resume_all.return_value = True
        self.pauser.is_paused = True
        self.suppressor._ducker = self.ducker
        self.suppressor._pauser = self.pauser
        # Don't pay the cork settle delay in tests
        patcher = mock.patch.object(playback_suppressor.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_duck_mode_never_touches_media_players(self):
        self.assertTrue(self.suppressor.suppress(mode="duck"))

        self.ducker.duck.assert_called_once_with()
        self.pauser.pause_all.assert_not_called()

    def test_pause_mode_pauses_then_ducks_the_remainder(self):
        self.assertTrue(self.suppressor.suppress(mode="pause"))

        self.pauser.pause_all.assert_called_once_with()
        self.ducker.duck.assert_called_once_with(skip_pids=[111])
        self.sleep.assert_called_once_with(playback_suppressor.PAUSE_SETTLE_SECONDS)

    def test_pause_mode_skips_settle_when_nothing_was_paused(self):
        self.pauser.pause_all.return_value = []
        self.pauser.is_paused = False

        self.suppressor.suppress(mode="pause")

        self.sleep.assert_not_called()
        self.ducker.duck.assert_called_once_with(skip_pids=[])

    def test_pause_mode_settles_even_when_no_pid_was_resolved(self):
        # Bus wouldn't name the PID; corking is the only guard left, so wait for it
        self.pauser.pause_all.return_value = []
        self.pauser.is_paused = True

        self.suppressor.suppress(mode="pause")

        self.sleep.assert_called_once_with(playback_suppressor.PAUSE_SETTLE_SECONDS)

    def test_unknown_mode_falls_back_to_ducking(self):
        self.suppressor.suppress(mode="banana")

        self.ducker.duck.assert_called_once_with()
        self.pauser.pause_all.assert_not_called()

    def test_reduction_percent_is_applied_live(self):
        self.suppressor.suppress(mode="duck", reduction_percent=80)

        self.ducker.set_reduction_percent.assert_called_once_with(80)

    def test_restore_resumes_media_even_when_unducking_raises(self):
        self.ducker.restore.side_effect = RuntimeError("pulse gone")

        self.assertFalse(self.suppressor.restore())
        self.pauser.resume_all.assert_called_once_with()

    def test_restore_skips_halves_that_are_not_active(self):
        self.ducker.is_ducked = False
        self.pauser.is_paused = False

        self.assertTrue(self.suppressor.restore())
        self.ducker.restore.assert_not_called()
        self.pauser.resume_all.assert_not_called()

    def test_is_active_covers_either_mechanism(self):
        self.ducker.is_ducked = False
        self.pauser.is_paused = True
        self.assertTrue(self.suppressor.is_active)

        self.pauser.is_paused = False
        self.assertFalse(self.suppressor.is_active)


if __name__ == "__main__":
    unittest.main()
