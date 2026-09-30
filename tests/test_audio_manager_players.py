"""Beep players: looked up once, fast player first, configured volume honored."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import audio_manager  # noqa: E402
from audio_manager import AudioManager  # noqa: E402
from tests.text_injector_helpers import ConfigStub  # noqa: E402


class BeepPlayerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for name in ('beep.ogg', 'beep.mp3'):
            (self.dir / name).write_bytes(b'x')
        self.commands = []
        self.lookups = []
        patcher = mock.patch.object(AudioManager, '_run_audio_command',
                                    lambda _self, cmd, tool: self.commands.append(cmd) or True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _manager(self, tools, sound='beep.ogg', volume=0.4):
        def which(tool):
            self.lookups.append(tool)
            return f'/usr/bin/{tool}' if tool in tools else None
        patcher = mock.patch.object(audio_manager.shutil, 'which', side_effect=which)
        patcher.start()
        self.addCleanup(patcher.stop)
        with mock.patch('builtins.print'):
            return AudioManager(ConfigStub({'audio_feedback': True, 'start_sound_volume': volume,
                                            'start_sound_path': str(self.dir / sound)}))

    def test_ogg_prefers_pw_play_with_volume(self):
        manager = self._manager({'pw-play', 'paplay', 'ffplay'})
        self.assertTrue(manager.play_start_sound())
        self.assertEqual(self.commands, [['pw-play', '--volume=0.40', str(self.dir / 'beep.ogg')]])

    def test_paplay_gets_its_linear_scale(self):
        manager = self._manager({'paplay', 'ffplay'})
        manager.play_start_sound()
        self.assertEqual(self.commands[0][:2], ['paplay', f'--volume={int(0.4 * 65536)}'])

    def test_other_formats_still_go_to_ffplay_first(self):
        manager = self._manager({'pw-play', 'ffplay'}, sound='beep.mp3')
        manager.play_start_sound()
        self.assertEqual(self.commands[0][0], 'ffplay')

    def test_without_ffplay_other_formats_fall_back_to_pw_play(self):
        manager = self._manager({'pw-play'}, sound='beep.mp3')
        manager.play_start_sound()
        self.assertEqual(self.commands[0][0], 'pw-play')

    def test_players_are_looked_up_once_without_a_subprocess(self):
        # `which` ran as a blocking subprocess right before the start cue.
        manager = self._manager({'pw-play'})
        with mock.patch.object(audio_manager.subprocess, 'run') as run:
            for _ in range(3):
                manager.play_start_sound()
        run.assert_not_called()
        self.assertEqual(self.lookups, ['pw-play'])
        self.assertEqual(len(self.commands), 3)


if __name__ == '__main__':
    unittest.main()
