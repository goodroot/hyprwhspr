import sys
import types
import unittest
from pathlib import Path
from unittest import mock


LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

from cli import test_cmd


class FakeConfig:
    def get_setting(self, key, default=None):
        return default


class MonitorAudioCapture:
    @staticmethod
    def get_available_input_devices():
        return [{"id": 7, "name": "pulse"}]

    def __init__(self, device_id=None, config_manager=None):
        pass

    def get_input_selection_error(self):
        return (
            "System default input is an output monitor, not a microphone. "
            "Select a microphone in sound settings or set audio_device_name, "
            "then run: hyprwhspr test --live"
        )


class WorkingAudioCapture(MonitorAudioCapture):
    def get_input_selection_error(self):
        return None

    def is_available(self):
        return True

    def get_current_device_info(self):
        return {"name": "mic"}


class TestCommandAudioDiagnosticsTests(unittest.TestCase):
    def test_mic_only_reports_monitor_selection_error(self):
        audio_module = types.SimpleNamespace(AudioCapture=MonitorAudioCapture)

        with (
            mock.patch.object(test_cmd, "ConfigManager", return_value=FakeConfig()),
            mock.patch.dict(sys.modules, {"audio_capture": audio_module}),
            mock.patch.object(test_cmd, "log_error") as log_error,
        ):
            result = test_cmd.test_command(mic_only=True)

        self.assertFalse(result)
        message = " ".join(str(call.args[0]) for call in log_error.call_args_list)
        self.assertIn("output monitor", message)
        self.assertIn("hyprwhspr test --live", message)

    def test_missing_parakeet_cpp_fails_and_skips_transcription(self):
        # A working mic, so the backend result alone decides the outcome.
        audio_module = types.SimpleNamespace(AudioCapture=WorkingAudioCapture)
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: (
            "parakeet-cpp" if key == "transcription_backend" else default)
        with (
            mock.patch.object(test_cmd, "ConfigManager", return_value=config),
            mock.patch.dict(sys.modules, {"audio_capture": audio_module}),
            mock.patch("parakeet_cpp_runtime.is_installed", return_value=False),
            mock.patch.object(test_cmd, "log_error") as log_error,
            mock.patch.object(test_cmd, "log_warning") as log_warning,
        ):
            self.assertFalse(test_cmd.test_command())
        self.assertIn("Parakeet.cpp unavailable",
                      " ".join(str(c.args[0]) for c in log_error.call_args_list))
        self.assertIn("Skipping transcription test",
                      " ".join(str(c.args[0]) for c in log_warning.call_args_list))


if __name__ == "__main__":
    unittest.main()
