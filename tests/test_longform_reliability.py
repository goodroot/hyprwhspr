import ast
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lib"), str(ROOT / "lib" / "src")]

SOURCE = ROOT / "lib" / "main.py"


class LongformMethods:
    pass


tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
method_names = {
    "_ensure_longform_initialized",
    "_on_longform_submit_triggered",
    "_longform_pause_recording",
    "_longform_persistence_failed",
    "_longform_submit",
    "_start_longform_auto_save_timer",
    "_stop_longform_auto_save_timer",
}
class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "hyprwhsprApp")
namespace = {
    "np": np,
    "threading": threading,
    "_HALLUCINATION_MARKERS": {
        "blank audio", "blank", "silence", "no speech", "you", "thank you",
        "thanks for watching", "thank you for watching", "video playback",
        "music", "music playing", "keyboard clicking",
    },
}
for node in class_node.body:
    if isinstance(node, ast.FunctionDef) and node.name in method_names:
        function_node = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=[],
            returns=node.returns,
            type_comment=node.type_comment,
        )
        ast.fix_missing_locations(function_node)
        exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        setattr(LongformMethods, node.name, namespace[node.name])


class ImmediateTimer:
    def __init__(self, interval, callback):
        self.callback = callback
        self.daemon = False

    def start(self):
        self.callback()

    def cancel(self):
        pass


class LongformReliabilityTests(unittest.TestCase):
    def _app(self):
        app = LongformMethods()
        persisted = np.array([0.1, 0.2], dtype=np.float32)
        app._longform_segment_manager = SimpleNamespace(
            save_segment=mock.Mock(return_value=None),
            concatenate_all=mock.Mock(return_value=persisted),
            concatenate_readable=mock.Mock(
                side_effect=lambda extra=None: np.concatenate(
                    [persisted] + ([extra] if extra is not None and len(extra) else [])
                )
            ),
            has_segments=mock.Mock(return_value=True),
            clear_session=mock.Mock(),
        )
        app.audio_capture = SimpleNamespace(
            sample_rate=16000,
            pause_recording=mock.Mock(),
            get_current_audio_copy=mock.Mock(),
            clear_buffer=mock.Mock(),
        )
        app.audio_manager = SimpleNamespace(
            play_error_sound=mock.Mock(), play_stop_sound=mock.Mock()
        )
        app.whisper_manager = SimpleNamespace(transcribe_audio=mock.Mock(return_value="hello"))
        app.config = SimpleNamespace(get_setting=lambda key, default=None: 1)
        app._longform_state = 'RECORDING'
        app._longform_error_audio = None
        app._longform_language_override = None
        app._longform_auto_save_timer = None
        app._longform_lock = threading.Lock()
        app._write_longform_state = mock.Mock()
        app._set_visualizer_state = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._hide_mic_osd = mock.Mock()
        app._notify_capture_subscriber = mock.Mock()
        app.is_processing = False
        return app, persisted

    def test_pause_persistence_failure_preserves_complete_audio_and_errors(self):
        app, persisted = self._app()
        unsaved = np.array([0.3, 0.4], dtype=np.float32)
        app.audio_capture.pause_recording.return_value = unsaved

        app._longform_pause_recording()

        np.testing.assert_array_equal(app._longform_error_audio, np.concatenate([persisted, unsaved]))
        self.assertEqual(app._longform_state, 'ERROR')
        app._longform_segment_manager.clear_session.assert_not_called()

    def test_autosave_failure_freezes_without_clearing_buffer(self):
        app, persisted = self._app()
        snapshot = np.array([0.3], dtype=np.float32)
        frozen = np.array([0.3, 0.4], dtype=np.float32)
        app.audio_capture.get_current_audio_copy.return_value = snapshot
        app.audio_capture.pause_recording.return_value = frozen

        with mock.patch.object(threading, "Timer", ImmediateTimer):
            app._start_longform_auto_save_timer()

        np.testing.assert_array_equal(app._longform_error_audio, np.concatenate([persisted, frozen]))
        self.assertEqual(app._longform_state, 'ERROR')
        app.audio_capture.clear_buffer.assert_not_called()

    def test_failed_final_write_can_submit_combined_audio(self):
        app, persisted = self._app()
        final = np.array([0.3], dtype=np.float32)
        combined = np.concatenate([persisted, final])
        app.audio_capture.pause_recording.return_value = final
        app._inject_text = mock.Mock(return_value=True)

        app._on_longform_submit_triggered()

        submitted = app.whisper_manager.transcribe_audio.call_args.args[0]
        np.testing.assert_array_equal(submitted, combined)
        app._longform_segment_manager.clear_session.assert_called_once_with()

    def test_injection_failure_retains_audio_until_successful_retry(self):
        app, persisted = self._app()
        app._inject_text = mock.Mock(side_effect=[False, True])

        app._longform_submit()

        self.assertEqual(app._longform_state, 'ERROR')
        np.testing.assert_array_equal(app._longform_error_audio, persisted)
        app._longform_segment_manager.clear_session.assert_not_called()

        app._longform_submit(retry=True)

        self.assertEqual(app._longform_state, 'IDLE')
        self.assertIsNone(app._longform_error_audio)
        app._longform_segment_manager.clear_session.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
