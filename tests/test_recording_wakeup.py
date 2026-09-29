"""Per-recording wake-up and stream ownership without audio hardware."""
import threading
import unittest
from unittest import mock

from tests.test_stream_orphan_recovery import (
    StreamOrphanRecoveryTests, FakeSoundDevice, FakeConfig, FakeCompleted,
)


class RecordingWakeupTests(unittest.TestCase):
    _load_audio_capture = StreamOrphanRecoveryTests._load_audio_capture
    tearDown = StreamOrphanRecoveryTests.tearDown

    def capture(self):
        self.sd = FakeSoundDevice()
        module = self._load_audio_capture(self.sd)
        patcher = mock.patch('subprocess.run', return_value=FakeCompleted('alsa_input.test mic'))
        patcher.start()
        self.addCleanup(patcher.stop)
        capture = module.AudioCapture(config_manager=FakeConfig())
        capture._start_keepalive = mock.Mock()
        capture._stop_keepalive = mock.Mock()
        return capture

    def start(self, capture):
        # Capture the newly allocated event before the worker starts, allowing a
        # deterministic assertion that the worker blocks on that exact event.
        original = threading.Thread
        entered = threading.Event()
        def thread(*args, **kwargs):
            event = kwargs['args'][0]
            wait = event.wait
            def waiting(timeout=None):
                entered.set()
                return wait(timeout)
            event.wait = waiting
            return original(*args, **kwargs)
        with mock.patch('threading.Thread', side_effect=thread):
            self.assertTrue(capture.start_recording())
        self.assertTrue(entered.wait(2))
        return capture._record_stop_event

    def test_stop_and_pause_wake_worker_and_restart_uses_fresh_event(self):
        capture = self.capture()
        for operation in ('stop_recording', 'pause_recording'):
            event = self.start(capture)
            worker = capture.record_thread
            stream = capture.stream
            getattr(capture, operation)()
            self.assertTrue(event.is_set())
            self.assertFalse(worker.is_alive())
            self.assertTrue(stream.closed)
            self.assertTrue(capture._cleanup_complete.is_set())
            new_event = self.start(capture)
            self.assertIsNot(new_event, event)
            event.set()
            self.assertFalse(new_event.is_set())
            self.assertTrue(capture.record_thread.is_alive())
            capture.stop_recording()

    def test_start_failure_releases_cleanup_and_signals_event(self):
        capture = self.capture()
        with mock.patch('threading.Thread.start', side_effect=RuntimeError('failed')):
            self.assertFalse(capture.start_recording())
        self.assertTrue(capture._record_stop_event.is_set())
        self.assertTrue(capture._cleanup_complete.is_set())
        self.assertFalse(capture.is_recording)

    def test_abandoned_start_never_publishes_or_closes_new_stream(self):
        capture = self.capture()
        entered, release = threading.Event(), threading.Event()
        create = self.sd.InputStream
        old_streams = []
        def blocked(**kwargs):
            stream = create(**kwargs)
            old_streams.append(stream)
            entered.set()
            release.wait(2)
            return stream
        with mock.patch.object(self.sd, 'InputStream', side_effect=blocked):
            self.assertTrue(capture.start_recording())
            worker = capture.record_thread
            self.assertTrue(entered.wait(2))
            capture._record_stop_event.set()
            # Recovery has abandoned the old worker and a new session owns state.
            new_event = threading.Event()
            newer = mock.Mock()
            with capture.lock:
                capture._record_stop_event = new_event
                capture.stream = newer
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(old_streams[0].closed)
        self.assertIs(capture.stream, newer)
        newer.close.assert_not_called()
        self.assertFalse(new_event.is_set())
        self.assertFalse(capture._cleanup_complete.is_set())
