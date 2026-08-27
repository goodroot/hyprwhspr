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


if __name__ == "__main__":
    unittest.main()
