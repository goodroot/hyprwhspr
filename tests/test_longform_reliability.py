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
from text_injector import InjectionOutcome


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
            inject_text=mock.Mock(return_value=InjectionOutcome.INJECTED),
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

    def test_start_is_refused_while_the_backend_is_claimed_elsewhere(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'IDLE'
        controller.claim_recording = mock.Mock(return_value=False)

        controller.start_recording()

        self.assertEqual(controller.state, 'IDLE')
        controller.audio_capture.start_recording.assert_not_called()
        controller.notify_capture.assert_called_once_with("", final=True)

    def test_backend_claim_is_held_only_while_capturing_or_processing(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'IDLE'
        controller.claim_recording = mock.Mock(return_value=True)
        controller.release_recording = mock.Mock()

        controller.start_recording()
        controller.release_recording.assert_not_called()

        # A paused session can persist indefinitely, so it must not keep the
        # backend claimed away from file transcription or a model unload
        controller.audio_capture.pause_recording.return_value = np.array([0.3], dtype=np.float32)
        controller.segment_manager.save_segment.return_value = Path("segment.wav")
        controller.pause_recording()
        controller.release_recording.assert_called_once_with()

        controller.resume_recording()
        self.assertEqual(controller.claim_recording.call_count, 2)
        self.assertEqual(controller.state, 'RECORDING')

    def test_resume_and_submit_are_refused_while_the_backend_is_claimed(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'PAUSED'
        controller.claim_recording = mock.Mock(return_value=False)

        controller.resume_recording()
        self.assertEqual(controller.state, 'PAUSED')
        controller.audio_capture.resume_recording.assert_not_called()
        controller.notify_capture.assert_called_with("", final=True)

        controller.submit()
        self.assertEqual(controller.state, 'PAUSED')
        controller.whisper_manager.transcribe_audio.assert_not_called()

    def test_resume_exception_releases_the_claim_and_capture_client(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'PAUSED'
        controller.release_recording = mock.Mock()
        controller.audio_capture.resume_recording.side_effect = RuntimeError('mic gone')

        with self.assertRaises(RuntimeError):
            controller.resume_recording()

        self.assertEqual(controller.state, 'PAUSED')
        controller.release_recording.assert_called_once_with()
        controller.notify_capture.assert_called_once_with("", final=True)

    def test_submit_exception_before_processing_releases_the_claim(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'PAUSED'
        controller.release_recording = mock.Mock()
        controller.segment_manager.concatenate_all.side_effect = RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            controller.submit()

        controller.release_recording.assert_called_once_with()
        controller.notify_capture.assert_called_once_with("", final=True)
        controller.whisper_manager.transcribe_audio.assert_not_called()

    def test_refused_submit_from_recording_pauses_and_disarms_auto_save(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'RECORDING'
        controller.claim_recording = mock.Mock(return_value=False)
        timer = mock.Mock()
        controller.auto_save_timer = timer

        controller.submit()

        self.assertEqual(controller.state, 'PAUSED')
        controller.whisper_manager.transcribe_audio.assert_not_called()
        # A timer left armed here dies silently on its next fire and would run
        # alongside the chain a later resume starts
        timer.cancel.assert_called_once_with()
        self.assertIsNone(controller.auto_save_timer)

    def test_failed_capture_start_releases_the_claim_and_the_capture_client(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'IDLE'
        controller.release_recording = mock.Mock()
        controller.audio_capture.start_recording.return_value = False

        controller.start_recording()

        self.assertEqual(controller.state, 'IDLE')
        controller.release_recording.assert_called_once_with()
        controller.notify_capture.assert_called_once_with("", final=True)

    def test_startup_exception_releases_the_claim_and_stops_the_stream(self):
        controller, _ = self._controller(timer_factory=mock.Mock())
        controller.state = 'IDLE'
        controller.release_recording = mock.Mock()
        controller.segment_manager.start_session.side_effect = RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            controller.start_recording()

        self.assertEqual(controller.state, 'IDLE')
        controller.release_recording.assert_called_once_with()
        controller.audio_capture.stop_recording.assert_called_once_with()
        controller.notify_capture.assert_called_once_with("", final=True)

        controller.release_recording.reset_mock()
        controller.audio_capture.start_recording.side_effect = RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            controller.start_recording()
        controller.release_recording.assert_called_once_with()

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
        controller.inject_text.side_effect = [InjectionOutcome.FAILED, InjectionOutcome.INJECTED]

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
