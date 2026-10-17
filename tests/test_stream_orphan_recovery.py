"""Regression tests for orphaned PortAudio streams during recovery (#209).

A PulseAudio/PipeWire restart mid-recording used to drop the stream
reference without stop/close, leaking a live C-level stream whose callback
kept appending to the unbounded audio_data buffer.
"""

import importlib
import sys
import threading
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
    ndim = 1
    dtype = "float32"
    flags = {'C_CONTIGUOUS': True}

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
        result = FakeChunk()
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
            capture._record_stop_event.set()
        if capture.record_thread is not None:
            capture.record_thread.join(timeout=3.0)

    def _fast_teardowns(self, capture):
        """Use real workers/events but skip bounded joins in lifecycle tests."""
        started = threading.Event()
        teardown = capture._teardown_stream_with_timeout

        def begin(stream, context, abort=False, wait=True, **kwargs):
            worker = teardown(stream, context, abort=abort, wait=False, **kwargs)
            join = worker.join
            worker.join = lambda timeout=None: join(None if timeout is None else 0)
            started.set()
            return worker

        patcher = mock.patch.object(capture, "_teardown_stream_with_timeout", side_effect=begin)
        patcher.start()
        self.addCleanup(patcher.stop)
        return started

    def _stop_while_teardown_pending(self, capture, started):
        capture._record_stop_event.set()
        self.assertTrue(started.wait(2.0))
        with mock.patch.object(capture.record_thread, "join"):
            return capture.stop_recording()

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

    def test_recovery_waits_out_a_slow_teardown(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream
        release = threading.Event()
        stream.abort = release.wait
        teardown = capture._teardown_stream_with_timeout

        def begin(*args, **kwargs):
            worker = teardown(*args, **kwargs, wait=False)
            join = worker.join

            def finish(timeout=None):
                release.set()
                join(timeout=2.0)

            worker.join = finish
            return worker

        capture.on_unrecoverable_stream = mock.Mock()
        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        try:
            with mock.patch("subprocess.run", side_effect=run), \
                    mock.patch.object(capture, "_teardown_stream_with_timeout", side_effect=begin):
                self.assertTrue(capture.recover_audio_capture("pulse_server_restart"))
            self.assertTrue(stream.closed)
            self.assertFalse(fake_sd.terminated)
            capture.on_unrecoverable_stream.assert_not_called()
        finally:
            release.set()
            self._stop_and_join(capture)

    def test_recovery_wedge_reports_at_once_and_allows_recording_after_late_close(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream
        release = threading.Event()
        stream.abort = lambda: release.wait()
        reported = mock.Mock()
        capture.on_unrecoverable_stream = reported
        self._fast_teardowns(capture)

        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        try:
            with mock.patch("subprocess.run", side_effect=run):
                self.assertFalse(capture.recover_audio_capture("pulse_server_restart"))
            self.assertFalse(fake_sd.terminated)
            # Server already gone: no 30s wait before escalating (#209)
            reported.assert_called_once_with()
            capture.check_pending_teardowns()
            reported.assert_called_once_with()
            self.assertFalse(capture.start_recording())
            self.assertFalse(capture.recover_audio_capture("pulse_server_restart"))
            reported.assert_called_once_with()

            release.set()
            self.assertTrue(self._wait_for(lambda: not capture._has_pending_teardowns()))
            self.assertTrue(stream.closed)
            with mock.patch("subprocess.run", side_effect=run):
                self.assertTrue(capture.start_recording())
            self.assertTrue(self._wait_for(lambda: capture.stream is not None and capture.stream.started))
        finally:
            release.set()
            self._stop_and_join(capture)

    def test_normal_stop_preserves_audio_and_resumes_after_late_close(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream
        stream.callback(FakeIndata([0.1, 0.2]), 2, None, None)
        release = threading.Event()
        stream.stop = lambda: release.wait()
        reported = threading.Event()
        capture.on_unrecoverable_stream = reported.set
        started = self._fast_teardowns(capture)

        try:
            with mock.patch.object(module, "log") as log:
                self.assertEqual(self._stop_while_teardown_pending(capture, started), [0.1, 0.2])
                with capture.lock:
                    before_deadline = max(capture._pending_teardowns.values()) - 1.0
                with mock.patch.object(module.time, "monotonic", return_value=before_deadline):
                    capture.check_pending_teardowns()
                self.assertFalse(reported.is_set())
            self.assertNotIn(
                mock.call("[WARN] Recording thread did not exit cleanly after 3 seconds"),
                log.call_args_list)
            self.assertFalse(stream.closed)
            self.assertFalse(capture._cleanup_complete.is_set())
            self.assertFalse(capture.start_recording())
            self.assertFalse(capture.recover_audio_capture("pulse_server_restart"))
            self.assertFalse(fake_sd.terminated)
            self.assertEqual(len(fake_sd.streams), 1)

            release.set()
            capture.record_thread.join(timeout=2.0)
            self.assertFalse(capture.record_thread.is_alive())
            self.assertTrue(stream.closed)
            self.assertTrue(capture._cleanup_complete.is_set())
            self.assertFalse(capture._has_pending_teardowns())
            self.assertFalse(reported.is_set())
            run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
            with mock.patch("subprocess.run", side_effect=run):
                self.assertTrue(capture.start_recording())
            self.assertTrue(self._wait_for(lambda: capture.stream is not None and capture.stream.started))
        finally:
            release.set()
            self._stop_and_join(capture)

    def test_late_cleanup_does_not_complete_a_newer_recording(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = self._start_recording(module, fake_sd)
        stream = capture.stream
        release = threading.Event()
        stream.stop = lambda: release.wait()
        old_worker = capture.record_thread
        started = self._fast_teardowns(capture)
        try:
            self._stop_while_teardown_pending(capture, started)
            with capture.lock:
                capture._record_stop_event = threading.Event()
                capture._cleanup_complete.clear()
            with mock.patch.object(capture, "_start_keepalive") as keepalive:
                release.set()
                old_worker.join(timeout=2.0)
                self.assertFalse(old_worker.is_alive())
                self.assertTrue(stream.closed)
                self.assertFalse(capture._cleanup_complete.is_set())
                keepalive.assert_not_called()
        finally:
            release.set()
            old_worker.join(timeout=2.0)

    def test_async_cleanup_blocks_until_all_streams_close_without_escalating(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = module.AudioCapture(config_manager=FakeConfig())
        releases = [threading.Event(), threading.Event()]
        streams = [FakeStream(fake_sd, 0), FakeStream(fake_sd, 0)]
        reported = mock.Mock()
        capture.on_unrecoverable_stream = reported
        try:
            for stream, release in zip(streams, releases):
                stream.abort = release.wait
                capture._teardown_stream_async(stream, "post-thread cleanup")
            self.assertTrue(capture._has_pending_teardowns())
            self.assertFalse(capture.start_recording())

            releases[0].set()
            self.assertTrue(self._wait_for(lambda: streams[0].closed))
            self.assertTrue(capture._has_pending_teardowns())
            self.assertFalse(capture.recover_audio_capture("pulse_server_restart"))
            self.assertFalse(capture.start_recording())
            reported.assert_not_called()
        finally:
            for release in releases:
                release.set()
            self.assertTrue(self._wait_for(lambda: not capture._has_pending_teardowns()))
        self.assertTrue(all(stream.closed for stream in streams))

    def test_start_retry_does_not_open_another_stream_during_teardown(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
        stream = fake_sd.InputStream(device=0)
        release = threading.Event()
        stopping = threading.Event()
        stream.start = mock.Mock(side_effect=RuntimeError("Unanticipated host error"))

        def stop():
            stopping.set()
            release.wait()

        stream.stop = stop
        self._fast_teardowns(capture)
        run = lambda *args, **kwargs: FakeCompleted("alsa_input.test mic")
        try:
            with mock.patch("subprocess.run", side_effect=run), \
                    mock.patch.object(fake_sd, "InputStream", return_value=stream) as open_stream:
                self.assertTrue(capture.start_recording())
                self.assertTrue(stopping.wait(2.0))
                with mock.patch.object(capture.record_thread, "join"):
                    capture.stop_recording()
                open_stream.assert_called_once()
                self.assertFalse(capture._cleanup_complete.is_set())
                self.assertFalse(capture.start_recording())
                release.set()
                capture.record_thread.join(timeout=2.0)
                self.assertFalse(capture.record_thread.is_alive())
                self.assertTrue(stream.closed)
                self.assertTrue(capture._cleanup_complete.is_set())
        finally:
            release.set()
            self._stop_and_join(capture)

    def test_keepalive_teardown_is_serialized_and_timeout_protected(self):
        # Inline stop()/close() could hang the record thread or recovery (#282)
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = module.AudioCapture(config_manager=FakeConfig())
        keepalive = FakeStream(fake_sd, 0)
        capture._keepalive_stream = keepalive

        with mock.patch.object(capture, "_teardown_stream_with_timeout") as teardown:
            capture._stop_keepalive()

        teardown.assert_called_once_with(keepalive, "keepalive")
        self.assertIsNone(capture._keepalive_stream)
        self.assertFalse(keepalive.stopped)

    def test_losing_keepalive_race_discards_through_protected_teardown(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = module.AudioCapture(config_manager=FakeConfig({"keepalive_stream": True}))
        winner = FakeStream(fake_sd, 0)
        capture.device_id = 0
        capture._keepalive_stream = None  # init may have opened one

        def open_stream(**kwargs):
            # Another thread installs its keepalive while ours is starting
            capture._keepalive_stream = winner
            return FakeStream(fake_sd, 0)

        with mock.patch.object(capture, "_is_multiplexed_audio_server", return_value=True), \
                mock.patch.object(fake_sd, "InputStream", side_effect=open_stream), \
                mock.patch.object(capture, "_teardown_stream_with_timeout") as teardown:
            capture._start_keepalive()

        self.assertIs(capture._keepalive_stream, winner)
        teardown.assert_called_once()
        discarded, context = teardown.call_args.args
        self.assertIsNot(discarded, winner)
        self.assertEqual(context, "keepalive")

    def test_close_waits_for_a_stalled_stop(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        capture = module.AudioCapture(config_manager=FakeConfig())
        entered = threading.Event()
        release = threading.Event()
        stream = mock.Mock()

        def stop():
            entered.set()
            release.wait()

        stream.stop.side_effect = stop
        worker = capture._teardown_stream_with_timeout(stream, "thread cleanup", wait=False)
        try:
            self.assertTrue(entered.wait(2.0))
            stream.close.assert_not_called()
        finally:
            release.set()
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        stream.close.assert_called_once_with()

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
