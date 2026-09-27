"""chunked_transcription: continuous-mode flushing that holds text until stop."""

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import whisper_manager  # noqa: E402
from tests.test_suspend_resume_recovery import _import_main_isolated  # noqa: E402


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def get_hallucination_markers(self):
        return ['thank you']


class PromptContextRoutingTests(unittest.TestCase):
    """WhisperManager forwards prompt_context only to backends that take it."""

    def _manager(self, supports):
        mgr = whisper_manager.WhisperManager.__new__(whisper_manager.WhisperManager)
        mgr.config = FakeConfig({'transcription_backend': 'pywhispercpp'})
        mgr.ready = True
        mgr._model_lock = threading.Lock()
        mgr._last_use_time = 0
        mgr._backend = mock.Mock(supports_prompt_context=supports, return_value='text')
        mgr._backend.transcribe.return_value = 'text'
        return mgr

    def test_supporting_backend_gets_prompt_context(self):
        mgr = self._manager(True)
        mgr.transcribe_audio(np.full(1600, 0.1, dtype=np.float32), 16000, prompt_context='before')
        self.assertEqual(mgr._backend.transcribe.call_args.kwargs['prompt_context'], 'before')

    def test_other_backends_never_see_it(self):
        mgr = self._manager(False)
        mgr.transcribe_audio(np.full(1600, 0.1, dtype=np.float32), 16000, prompt_context='before')
        self.assertNotIn('prompt_context', mgr._backend.transcribe.call_args.kwargs)


class ChunkedRecordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, transcripts=()):
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app.config = FakeConfig({'continuous_silence_threshold': 0.01})
        app._recording_lock = threading.Lock()
        app._recording_finalizing = threading.Event()
        app._current_language_override = None
        app.is_recording = True
        app.is_processing = False
        app._chunk_texts = None
        app._chunk_tail_has_sound = True
        app._continuous_silence_thread = None
        app._continuous_silence_stop = threading.Event()
        app._continuous_flush_lock = threading.Lock()
        app._continuous_transcription_done = threading.Event()
        app._continuous_transcription_done.set()
        app._continuous_cancelled = False
        app.audio_capture = types.SimpleNamespace(
            sample_rate=16000,
            rolling_avg_level=0.0,
            buffered_seconds=lambda: 0.0,
            flush_buffer=lambda: np.full(16000 * 20, 0.1, dtype=np.float32),
            stop_recording=mock.Mock(return_value=None),
        )
        app.whisper_manager = types.SimpleNamespace(
            transcribe_audio=mock.Mock(side_effect=list(transcripts)))
        app.audio_manager = mock.Mock()
        app.playback_suppressor = types.SimpleNamespace(is_active=False)
        app._inject_text = mock.Mock(return_value=self.main.InjectionOutcome.INJECTED)
        app._notify_capture = mock.Mock()
        for name in ('_autostop_stop_silence_monitor', '_clear_mic_osd_preview_text',
                     '_set_visualizer_state', '_stop_audio_level_monitoring',
                     '_write_recording_status', '_show_result_and_hide', '_notify_zero_volume'):
            setattr(app, name, mock.Mock())
        return app

    def _flush_and_wait(self, app):
        app._continuous_flush_audio()
        self.assertTrue(app._continuous_transcription_done.wait(2))

    def test_pieces_are_held_and_prompted_with_earlier_text(self):
        app = self._app(['First part.', 'Second part.'])
        app._chunk_texts = []
        self._flush_and_wait(app)
        self._flush_and_wait(app)

        self.assertEqual(app._chunk_texts, ['First part.', 'Second part.'])
        calls = app.whisper_manager.transcribe_audio.call_args_list
        self.assertIsNone(calls[0].kwargs['prompt_context'])
        self.assertEqual(calls[1].kwargs['prompt_context'], 'First part.')
        app._inject_text.assert_not_called()
        # A capture client must stay attached until the whole recording is pasted
        app._notify_capture.assert_not_called()

    def test_continuous_mode_still_pastes_each_piece(self):
        app = self._app(['Hello.'])
        self._flush_and_wait(app)
        app._inject_text.assert_called_once_with('Hello.')
        self.assertIsNone(app.whisper_manager.transcribe_audio.call_args.kwargs['prompt_context'])

    def test_stop_transcribes_tail_after_pieces_and_pastes_once(self):
        app = self._app(['tail words'])
        app._chunk_texts = ['First part.']
        app.audio_capture.stop_recording.return_value = np.full(16000, 0.1, dtype=np.float32)

        app._stop_recording()

        self.assertEqual(
            app.whisper_manager.transcribe_audio.call_args.kwargs['prompt_context'], 'First part.')
        app._inject_text.assert_called_once_with('First part. tail words')
        self.assertIsNone(app._chunk_texts)

    def test_stop_with_nothing_after_last_piece_still_pastes(self):
        app = self._app()
        app._chunk_texts = ['First part.', 'Second part.']
        app.audio_capture.stop_recording.return_value = None  # buffer was just flushed

        app._stop_recording()

        app.whisper_manager.transcribe_audio.assert_not_called()
        app._inject_text.assert_called_once_with('First part. Second part.')
        app._notify_zero_volume.assert_not_called()

    def test_silent_tail_is_not_transcribed(self):
        app = self._app()
        app._chunk_texts = ['Real words.']
        app._chunk_tail_has_sound = False  # the monitor heard nothing since the last cut
        app.audio_capture.stop_recording.return_value = np.full(16000, 0.001, dtype=np.float32)

        app._stop_recording()

        app.whisper_manager.transcribe_audio.assert_not_called()
        app._inject_text.assert_called_once_with('Real words.')

    def test_phantom_tail_is_dropped(self):
        app = self._app(['Thank you'])
        app._chunk_texts = ['Real words.']
        app.audio_capture.stop_recording.return_value = np.full(16000, 0.1, dtype=np.float32)

        app._stop_recording()

        app._inject_text.assert_called_once_with('Real words.')

    def test_stop_waits_for_the_piece_in_flight(self):
        app = self._app(['tail'])
        app._chunk_texts = ['one']
        app.audio_capture.stop_recording.return_value = np.full(16000, 0.1, dtype=np.float32)
        app._continuous_transcription_done.clear()

        def finish_piece():
            time.sleep(0.1)
            app._chunk_texts.append('two')
            app._continuous_transcription_done.set()

        threading.Thread(target=finish_piece).start()
        app._stop_recording()

        app._inject_text.assert_called_once_with('one two tail')

    def test_monitor_waits_for_minimum_piece_length(self):
        app = self._app()
        app.config.values['chunked_min_seconds'] = 10.0
        buffered = [5.0]
        app.audio_capture.buffered_seconds = lambda: buffered[0]
        app._continuous_flush_audio = mock.Mock()
        app._POLL_INTERVAL = 0.01

        app._continuous_start_silence_monitor(chunked=True)
        try:
            time.sleep(0.15)
            app._continuous_flush_audio.assert_not_called()  # silent, but piece too short
            buffered[0] = 10.0
            deadline = time.monotonic() + 2
            while not app._continuous_flush_audio.called and time.monotonic() < deadline:
                time.sleep(0.01)
            app._continuous_flush_audio.assert_called()
        finally:
            app.is_recording = False
            app._continuous_stop_silence_monitor()

    def test_monitor_tracks_sound_after_each_cut(self):
        app = self._app()
        app.audio_capture.buffered_seconds = lambda: 20.0
        app._continuous_flush_audio = mock.Mock()
        app._POLL_INTERVAL = 0.01

        def wait_for(condition):
            deadline = time.monotonic() + 2
            while not condition() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(condition())

        app._continuous_start_silence_monitor(chunked=True)
        try:
            wait_for(lambda: app._continuous_flush_audio.called)
            self.assertFalse(app._chunk_tail_has_sound)  # still silent after the cut
            app.audio_capture.rolling_avg_level = 0.5
            wait_for(lambda: app._chunk_tail_has_sound)
        finally:
            app.is_recording = False
            app._continuous_stop_silence_monitor()


if __name__ == "__main__":
    unittest.main()
