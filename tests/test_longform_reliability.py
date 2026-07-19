import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from longform_controller import LongFormController


class ImmediateTimer:
    def __init__(self, interval, callback):
        self.callback = callback
        self.daemon = False

    def start(self):
        self.callback()

    def cancel(self):
        pass


class LongformReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _controller(self, timer_factory=ImmediateTimer):
        persisted = np.array([0.1, 0.2], dtype=np.float32)
        segment_manager = SimpleNamespace(
            save_segment=mock.Mock(return_value=None),
            concatenate_all=mock.Mock(return_value=persisted),
            concatenate_readable=mock.Mock(
                side_effect=lambda extra=None: np.concatenate(
                    [persisted] + ([extra] if extra is not None and len(extra) else [])
                )
            ),
            has_segments=mock.Mock(return_value=True),
            clear_session=mock.Mock(),
            start_session=mock.Mock(),
        )
        audio_capture = SimpleNamespace(
            sample_rate=16000,
            start_recording=mock.Mock(return_value=True),
            resume_recording=mock.Mock(return_value=True),
            pause_recording=mock.Mock(),
            stop_recording=mock.Mock(),
            get_current_audio_copy=mock.Mock(),
            clear_buffer=mock.Mock(),
        )
        audio_manager = SimpleNamespace(
            play_start_sound=mock.Mock(),
            play_stop_sound=mock.Mock(),
            play_error_sound=mock.Mock(),
        )
        controller = LongFormController(
            config=SimpleNamespace(get_setting=lambda key, default=None: 1),
            audio_capture=audio_capture,
            audio_manager=audio_manager,
            whisper_manager=SimpleNamespace(
                transcribe_audio=mock.Mock(return_value="hello")
            ),
            inject_text=mock.Mock(return_value=True),
            notify_capture=mock.Mock(),
            set_visualizer_state=mock.Mock(),
            show_mic_osd=mock.Mock(),
            hide_mic_osd=mock.Mock(),
            show_result_and_hide=mock.Mock(),
            write_recording_status=mock.Mock(),
            set_processing=mock.Mock(),
            hallucination_markers={"silence"},
            timer_factory=timer_factory,
            state_file=Path(self.temp_dir.name) / "longform_state",
        )
        controller.segment_manager = segment_manager
        controller.state = 'RECORDING'
        return controller, persisted

    def test_start_pause_resume_transitions_are_owned_by_controller(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'IDLE'

        controller.start_recording(language_override="en")
        self.assertEqual(controller.state, 'RECORDING')
        self.assertEqual(controller.language_override, "en")
        controller.segment_manager.start_session.assert_called_once_with()

        controller.segment_manager.save_segment.return_value = Path("segment.wav")
        controller.audio_capture.pause_recording.return_value = np.array([0.3], dtype=np.float32)
        controller.pause_recording()
        self.assertEqual(controller.state, 'PAUSED')

        controller.resume_recording()
        self.assertEqual(controller.state, 'RECORDING')

    def test_pause_persistence_failure_preserves_complete_audio_and_errors(self):
        controller, persisted = self._controller()
        unsaved = np.array([0.3, 0.4], dtype=np.float32)
        controller.audio_capture.pause_recording.return_value = unsaved

        controller.pause_recording()

        np.testing.assert_array_equal(controller.error_audio, np.concatenate([persisted, unsaved]))
        self.assertEqual(controller.state, 'ERROR')
        controller.segment_manager.clear_session.assert_not_called()

    def test_autosave_failure_freezes_without_clearing_buffer(self):
        controller, persisted = self._controller()
        snapshot = np.array([0.3], dtype=np.float32)
        frozen = np.array([0.3, 0.4], dtype=np.float32)
        controller.audio_capture.get_current_audio_copy.return_value = snapshot
        controller.audio_capture.pause_recording.return_value = frozen

        controller.start_auto_save_timer()

        np.testing.assert_array_equal(controller.error_audio, np.concatenate([persisted, frozen]))
        self.assertEqual(controller.state, 'ERROR')
        controller.audio_capture.clear_buffer.assert_not_called()

    def test_failed_final_write_can_submit_combined_audio(self):
        controller, persisted = self._controller()
        final = np.array([0.3], dtype=np.float32)
        controller.audio_capture.pause_recording.return_value = final

        controller.submit_shortcut()

        submitted = controller.whisper_manager.transcribe_audio.call_args.args[0]
        np.testing.assert_array_equal(submitted, np.concatenate([persisted, final]))
        controller.segment_manager.clear_session.assert_called_once_with()

    def test_injection_failure_retains_audio_until_successful_retry(self):
        controller, persisted = self._controller()
        controller.inject_text.side_effect = [False, True]

        controller.submit()

        self.assertEqual(controller.state, 'ERROR')
        np.testing.assert_array_equal(controller.error_audio, persisted)
        controller.segment_manager.clear_session.assert_not_called()

        controller.submit(retry=True)

        self.assertEqual(controller.state, 'IDLE')
        self.assertIsNone(controller.error_audio)
        controller.segment_manager.clear_session.assert_called_once_with()

    def test_cancel_discards_unrecoverable_error_session(self):
        controller, _ = self._controller()
        controller.state = 'ERROR'
        controller.error_audio = None

        controller.request_cancel()

        controller.audio_capture.stop_recording.assert_called_once_with()
        controller.segment_manager.clear_session.assert_called_once_with()
        self.assertEqual(controller.state, 'IDLE')
        self.assertIsNone(controller.error_audio)
        self.assertEqual(controller.state_file.read_text(), 'IDLE')


if __name__ == "__main__":
    unittest.main()
