import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeCompleted:
    def __init__(self, source):
        self.returncode = 0
        self.stdout = f"{source}\n"


class FakeStream:
    def __init__(self, owner, device, callback=None):
        self.owner = owner
        self.device = device
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.owner.started_devices.append(self.device)
        self.started = True
        if self.owner.fail_once_for_device == self.device:
            self.owner.fail_once_for_device = None
            raise RuntimeError("Unanticipated host error")
        if self.owner.on_stream_start:
            self.owner.on_stream_start()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSoundDevice(types.ModuleType):
    NO_FAIL_DEVICE = object()

    def __init__(self):
        super().__init__("sounddevice")
        self.default = types.SimpleNamespace(
            samplerate=None,
            channels=None,
            dtype=None,
            device=[None, None],
        )
        self.devices = [
            {
                "name": "alsa_input.old mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "alsa_input.new mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Preferred USB Mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Output Only",
                "max_input_channels": 0,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ]
        self.streams = []
        self.started_devices = []
        self.fail_once_for_device = self.NO_FAIL_DEVICE
        self.on_stream_start = None

    def query_devices(self, device=None, kind=None):
        if device is None:
            return list(self.devices)
        return self.devices[device]

    def query_hostapis(self, hostapi):
        return {"name": "PulseAudio"}

    def InputStream(self, device=None, samplerate=None, channels=None,
                    dtype=None, blocksize=None, callback=None):
        stream = FakeStream(self, device, callback=callback)
        self.streams.append(stream)
        return stream


class FakeNumpy(types.ModuleType):
    def __init__(self):
        super().__init__("numpy")
        self.float32 = "float32"
        self.ndarray = object

    def concatenate(self, arrays, axis=0):
        result = []
        for array in arrays:
            result.extend(array)
        return result

    def sqrt(self, value):
        return value ** 0.5

    def mean(self, values):
        return sum(values) / len(values)

    def any(self, values):
        return any(values)

    def isnan(self, values):
        return [False for _ in values]

    def isinf(self, values):
        return [False for _ in values]

    def ascontiguousarray(self, values, dtype=None):
        return values


class AudioDeviceRefreshTests(unittest.TestCase):
    def _load_audio_capture(self, fake_sd):
        self._saved_audio_capture = sys.modules.get("audio_capture")
        self._saved_sounddevice = sys.modules.get("sounddevice")
        self._saved_numpy = sys.modules.get("numpy")
        sys.modules["sounddevice"] = fake_sd
        sys.modules["numpy"] = FakeNumpy()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))
        import audio_capture
        return importlib.reload(audio_capture)

    def tearDown(self):
        if hasattr(self, "_saved_sounddevice"):
            if self._saved_sounddevice is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = self._saved_sounddevice
        if hasattr(self, "_saved_numpy"):
            if self._saved_numpy is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = self._saved_numpy
        if hasattr(self, "_saved_audio_capture"):
            if self._saved_audio_capture is None:
                sys.modules.pop("audio_capture", None)
            else:
                sys.modules["audio_capture"] = self._saved_audio_capture

    def _run_sources(self, sources):
        source_iter = iter(sources)

        def run(*args, **kwargs):
            return FakeCompleted(next(source_iter))

        return run

    def test_system_default_init_does_not_make_default_preferred(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        self.assertIsNone(capture.preferred_device_id)
        self.assertEqual(capture.device_id, 0)

    def test_record_start_reresolves_default_even_when_old_device_exists(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.device_id, 1)
        self.assertIn(1, fake_sd.started_devices)
        self.assertIsNone(capture.preferred_device_id)

    def test_explicit_device_id_does_not_follow_default_change(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.new mic"])):
            capture = module.AudioCapture(device_id=0, config_manager=FakeConfig({"audio_device_id": 0, "stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.device_id, 0)
        self.assertIn(0, fake_sd.started_devices)

    def test_configured_name_fallback_does_not_become_preferred(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"audio_device_name": "Missing Mic"}))

        self.assertEqual(capture.device_id, 0)
        self.assertIsNone(capture.preferred_device_id)
        with mock.patch.object(capture, "_get_pulse_default_source_device_id") as get_default:
            self.assertFalse(capture.refresh_default_input("test"))
        get_default.assert_not_called()

    def test_keepalive_moves_to_refreshed_default_device(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        config = FakeConfig({"keepalive_stream": True})

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=config)
            old_stream = capture._keepalive_stream
            self.assertEqual(old_stream.device, 0)
            self.assertTrue(capture.refresh_default_input("test"))

        self.assertTrue(old_stream.stopped)
        self.assertTrue(old_stream.closed)
        self.assertEqual(capture._keepalive_stream.device, 1)

    def test_keepalive_cycles_when_pulse_source_changes_on_same_portaudio_device(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "pulse",
            "max_input_channels": 32,
            "default_samplerate": 44100,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)
        config = FakeConfig({"keepalive_stream": True})

        with mock.patch("subprocess.run", side_effect=self._run_sources([
            "alsa_input.pci-0000_00_1f.3.analog-stereo",
            "bluez_input.68:F2:1F:03:3F:C9",
        ])):
            capture = module.AudioCapture(config_manager=config)
            old_stream = capture._keepalive_stream
            self.assertEqual(capture.device_id, 4)
            self.assertTrue(capture.refresh_default_input("test"))

        self.assertTrue(old_stream.stopped)
        self.assertTrue(old_stream.closed)
        self.assertEqual(capture.device_id, 4)
        self.assertEqual(capture._keepalive_stream.device, 4)

    def test_retriable_open_failure_reresolves_default_before_retry(self):
        fake_sd = FakeSoundDevice()
        fake_sd.fail_once_for_device = 0
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(fake_sd.started_devices[:2], [0, 1])
        self.assertEqual(capture.device_id, 1)

    def test_unmatched_pulse_default_clears_stale_concrete_device(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "bluez_input.68:F2:1F:03:3F:C9"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertEqual(capture.device_id, 0)
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)
        self.assertEqual(fake_sd.default.device[0], None)
        self.assertIn(None, fake_sd.started_devices)

    def test_refresh_default_input_skips_during_recovery(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        capture.recovery_in_progress = True
        with mock.patch.object(capture, "_get_pulse_default_source_device_id") as get_default:
            self.assertFalse(capture.refresh_default_input("test"))

        get_default.assert_not_called()

    def test_sounddevice_fallback_rejects_output_only_default(self):
        fake_sd = FakeSoundDevice()
        fake_sd.default.device = [3, None]
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=1, stdout="")):
            capture = module.AudioCapture(config_manager=FakeConfig())

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)

    def test_pulse_default_rejects_output_only_match(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        capture.device_id = None
        capture.device_info = None
        fake_sd.default.device = [3, None]
        with (
            mock.patch.object(capture, "_get_pulse_default_source_device_id", return_value=3),
            mock.patch.object(capture, "_set_system_default_device"),
        ):
            self.assertFalse(capture.refresh_default_input("test"))

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)


if __name__ == "__main__":
    unittest.main()
