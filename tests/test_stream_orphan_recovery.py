"""Regression tests for orphaned PortAudio streams during recovery (#209).

A PulseAudio/PipeWire restart mid-recording used to drop the stream
reference without stop/close, leaking a live C-level stream whose callback
kept appending to the unbounded audio_data buffer.
"""

import importlib
import sys
import time
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


class FakeChunk(list):
    def copy(self):
        return FakeChunk(self)

    def __pow__(self, exp):
        return [value ** exp for value in self]


class FakeIndata:
    """Stands in for the numpy indata block: indata[:, 0] yields the chunk."""

    def __init__(self, samples):
        self._samples = FakeChunk(samples)

    def __getitem__(self, key):
        return self._samples


class FakeStream:
    def __init__(self, owner, device, callback=None):
        self.owner = owner
        self.device = device
        self.callback = callback
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.raise_on_abort = False
        self.wedge_seconds = 0  # abort()/close() block this long (0 = healthy)

    def start(self):
        self.started = True
        self.owner.started_devices.append(self.device)

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True
        if self.wedge_seconds:
            time.sleep(self.wedge_seconds)
        if self.raise_on_abort:
            raise RuntimeError("PortAudio error: stream is stopped")

    def close(self):
        if self.wedge_seconds:
            time.sleep(self.wedge_seconds)
        self.closed = True


class FakeSoundDevice(types.ModuleType):
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
                "name": "alsa_input.test mic",
                "max_input_channels": 1,
                "default_samplerate": 16000,
                "hostapi": 0,
            },
        ]
        self.streams = []
        self.started_devices = []
        self.terminated = False
        self.initialized = False

    def _terminate(self):
        self.terminated = True

    def _initialize(self):
        self.initialized = True

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


class StreamOrphanRecoveryTests(unittest.TestCase):
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
                # _load_audio_capture reloaded this module in place against the
                # fake sounddevice/numpy; reload once more against the restored
                # real modules to avoid leaking the fakes into other tests.
                if self._saved_sounddevice is not None and self._saved_numpy is not None:
                    importlib.reload(self._saved_audio_capture)

    @staticmethod
    def _wait_for(predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def _start_recording(self, module, fake_sd):
        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        with mock.patch("subprocess.run", side_effect=run):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertTrue(capture.start_recording())
        self.assertTrue(self._wait_for(lambda: capture.stream is not None and capture.stream.started))
        return capture

    def _stop_and_join(self, capture):
        with capture.lock:
            capture.is_recording = False
        if capture.record_thread is not None:
            capture.record_thread.join(timeout=3.0)

    def test_recovery_closes_stream_even_when_abort_raises(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream
        stream.raise_on_abort = True

        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        with mock.patch("subprocess.run", side_effect=run):
            self.assertTrue(capture.recover_audio_capture("pulse_server_restart"))

        self.assertTrue(stream.aborted)
        self.assertTrue(stream.closed)
        self.assertIsNone(capture.stream)
        self._stop_and_join(capture)

    def test_thread_abort_leaves_stream_for_recovery(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream

        capture._abort_cleanup = True
        self._stop_and_join(capture)

        # Not dropped (that orphaned the C stream); recovery pops it later
        self.assertIs(capture.stream, stream)
        capture._cleanup_stream()
        self.assertTrue(stream.closed)

    def test_recovery_resets_portaudio_when_stream_is_wedged(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        capture.stream.wedge_seconds = 3  # close() returns after the reset
        unrecoverable = []
        capture.on_unrecoverable_stream = lambda: unrecoverable.append(True)

        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        with mock.patch("subprocess.run", side_effect=run):
            self.assertTrue(capture.recover_audio_capture("pulse_server_restart"))

        self.assertTrue(fake_sd.terminated)
        self.assertTrue(fake_sd.initialized)
        self.assertEqual(unrecoverable, [])
        self.assertIsNone(capture.stream)
        self._stop_and_join(capture)

    def test_recovery_reports_stream_that_survives_portaudio_reset(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        capture.stream.wedge_seconds = 10  # close() outlives reset + recheck
        unrecoverable = []
        capture.on_unrecoverable_stream = lambda: unrecoverable.append(True)

        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        with mock.patch("subprocess.run", side_effect=run):
            capture.recover_audio_capture("pulse_server_restart")

        self.assertTrue(fake_sd.terminated)
        self.assertEqual(unrecoverable, [True])
        self._stop_and_join(capture)

    def test_stale_stream_callback_stops_writing_shared_state(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream

        stream.callback(FakeIndata([0.1, 0.2]), 2, None, None)
        self.assertEqual(len(capture.audio_data), 1)
        frames_before = capture.frames_since_start

        capture._cleanup_stream()
        self.assertIsNone(capture.stream)
        self.assertTrue(stream.closed)

        # Orphaned stream keeps firing; its chunks must be dropped
        stream.callback(FakeIndata([0.3, 0.4]), 2, None, None)
        self.assertEqual(len(capture.audio_data), 1)
        self.assertEqual(capture.frames_since_start, frames_before)
        self._stop_and_join(capture)

    def test_buffer_cap_bounds_audio_data_until_flush(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream

        # Cap the buffer at 3 samples
        module._MAX_BUFFER_SECONDS = 3.0 / capture.sample_rate
        for _ in range(4):
            stream.callback(FakeIndata([0.1, 0.2]), 2, None, None)

        self.assertEqual(len(capture.audio_data), 2)
        self.assertTrue(capture._buffer_capped)

        # Draining the buffer (continuous mode) resets the cap
        capture.clear_buffer()
        stream.callback(FakeIndata([0.5, 0.6]), 2, None, None)
        self.assertEqual(len(capture.audio_data), 1)
        self._stop_and_join(capture)


if __name__ == "__main__":
    unittest.main()
