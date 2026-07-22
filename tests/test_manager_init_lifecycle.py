import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

import whisper_manager as wm  # noqa: E402
from whisper_manager import WhisperManager  # noqa: E402


class FakeConfig:
    def get_setting(self, key, default=None):
        return default

    def get_temp_directory(self):
        return "/tmp"


def make_fake_backend_cls(events, name='pywhispercpp', init_result=True,
                          init_gate=None, cleanup_error=None):
    class FakeBackend:
        def __init__(self, manager):
            self._manager = manager
            self.name = name
            self.is_loaded = False
            events.append(('create', id(self)))

        def initialize(self):
            if init_gate is not None:
                init_gate.wait(timeout=5)
            events.append(('initialize', id(self)))
            if init_result:
                self.is_loaded = True
                self._manager.ready = True
            return init_result

        def cleanup(self):
            events.append(('cleanup', id(self)))
            if cleanup_error is not None:
                raise cleanup_error

        def apply_partial_callback(self, callback):
            events.append(('partial_callback', callback))

    return FakeBackend


class InitLifecycleTests(unittest.TestCase):
    def _manager(self, backend_cls):
        manager = WhisperManager(config_manager=FakeConfig())
        patcher = mock.patch.dict(wm.BACKENDS, {'pywhispercpp': backend_cls})
        patcher.start()
        self.addCleanup(patcher.stop)
        return manager

    def test_initialize_closes_previous_backend_before_creating_new(self):
        events = []
        manager = self._manager(make_fake_backend_cls(events))

        self.assertTrue(manager.initialize())
        first_id = events[0][1]
        self.assertTrue(manager.initialize())

        kinds = [(kind, ident == first_id) for kind, ident in events]
        self.assertEqual(kinds[:4], [
            ('create', True), ('initialize', True),
            ('cleanup', True), ('create', False),
        ])

    def test_concurrent_initialize_is_single_flight(self):
        events = []
        gate = threading.Event()
        manager = self._manager(make_fake_backend_cls(events, init_gate=gate))

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(manager.initialize()))
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        gate.set()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(results, [True, True])
        self.assertEqual([kind for kind, _ in events].count('create'), 1)

    def test_failed_initialize_keeps_backend_for_missing_client_gate(self):
        events = []
        manager = self._manager(
            make_fake_backend_cls(events, name='realtime-ws', init_result=False))

        self.assertFalse(manager.initialize())
        self.assertIsNotNone(manager._backend)
        self.assertTrue(manager.realtime_client_missing())

    def test_realtime_callback_is_reapplied_after_client_initializes(self):
        events = []
        manager = self._manager(make_fake_backend_cls(
            events,
            name='realtime-ws',
        ))
        callback = lambda text: None
        manager.set_realtime_partial_callback(callback)

        self.assertTrue(manager.initialize())

        self.assertIn(('partial_callback', callback), events)

    def test_cleanup_exception_does_not_abort_initialize(self):
        events = []
        manager = self._manager(
            make_fake_backend_cls(events, cleanup_error=RuntimeError("boom")))

        self.assertTrue(manager.initialize())
        self.assertTrue(manager.initialize())
        self.assertEqual([kind for kind, _ in events].count('create'), 2)

    def test_ready_cleared_during_swap(self):
        events = []
        gate = threading.Event()
        manager = self._manager(make_fake_backend_cls(events, init_gate=gate))

        gate.set()
        self.assertTrue(manager.initialize())
        self.assertTrue(manager.ready)

        gate.clear()
        seen_ready = []
        thread = threading.Thread(
            target=lambda: manager.initialize())
        thread.start()
        # Wait until the re-init is past the swap (old cleaned up, new created)
        for _ in range(100):
            if [kind for kind, _ in events].count('create') == 2:
                break
            threading.Event().wait(0.01)
        seen_ready.append(manager.ready)
        gate.set()
        thread.join(timeout=5)

        self.assertEqual(seen_ready, [False])
        self.assertTrue(manager.ready)

    def test_initialize_waits_for_active_model_operation(self):
        events = []
        manager = self._manager(make_fake_backend_cls(events))
        self.assertTrue(manager.initialize())

        with manager._model_lock:
            thread = threading.Thread(target=manager.initialize)
            thread.start()
            threading.Event().wait(0.05)
            self.assertTrue(thread.is_alive())
            self.assertEqual([kind for kind, _ in events].count('cleanup'), 0)

        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual([kind for kind, _ in events].count('cleanup'), 1)


if __name__ == '__main__':
    unittest.main()
