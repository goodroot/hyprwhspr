"""Regression tests for the suspend/resume recovery fallback path (#244).

When backend reinitialization fails after resume, _on_system_resume used to
reference undefined names (`backend`, `pywhispercpp_variants`), raising a
NameError that the outer handler swallowed - so the background retry thread
was never started and a realtime-ws backend stayed disconnected.
"""

import importlib
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import whisper_manager  # noqa: E402


class _AutoCodes:
    """evdev.ecodes stand-in: any KEY_* lookup yields a distinct int."""

    def __init__(self):
        self._codes = {}
        self.ecodes = {}
        self.KEY = {}

    def __getattr__(self, name):
        return self._codes.setdefault(name, len(self._codes) + 1)


def _hardware_stubs():
    """Stubs for the modules main pulls in that touch real hardware.

    audio_capture binds sounddevice through require_package(), which calls
    sys.exit(1) when the package is missing, and importing it for real starts
    PortAudio; global_shortcuts imports evdev unguarded and reads ecodes in a
    class body. Both are stubbed so this test needs neither package nor
    hardware. pyudev/dbus are already guarded by try/except at their import
    sites, so they need no stub.
    """
    return {
        "sounddevice": {
            "query_devices": lambda *a, **k: [],
            "InputStream": object,
            "PortAudioError": Exception,
            "default": types.SimpleNamespace(device=(None, None)),
        },
        "evdev": {
            "InputDevice": object,
            "UInput": object,
            "categorize": lambda *a, **k: None,
            "list_devices": lambda: [],
            "ecodes": _AutoCodes(),
        },
    }


def _import_main_isolated():
    """Import lib/main.py against stub hardware modules, then restore sys.modules.

    main and everything it drags in are dropped from sys.modules afterwards, so
    the stubs can't leak into tests that import those modules for real; the
    returned module object keeps its own references alive.
    """
    saved = dict(sys.modules)
    for name, attrs in _hardware_stubs().items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
    try:
        return importlib.import_module("main")
    finally:
        for name in set(sys.modules) - set(saved):
            del sys.modules[name]
        sys.modules.update(saved)


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class ActiveBackendIsLocalTests(unittest.TestCase):
    """active_backend_is_local() classifies live and not-yet-built backends."""

    def _make(self, backend_obj, configured):
        mgr = whisper_manager.WhisperManager.__new__(
            whisper_manager.WhisperManager)
        mgr._backend = backend_obj
        mgr.config = FakeConfig({'transcription_backend': configured})
        return mgr

    def test_live_backend_object_wins(self):
        backend = types.SimpleNamespace(is_local=False, name='realtime-ws')
        self.assertFalse(self._make(backend, 'realtime-ws').active_backend_is_local())

        backend = types.SimpleNamespace(is_local=True, name='pywhispercpp')
        self.assertTrue(self._make(backend, 'cpu').active_backend_is_local())

    def test_falls_back_to_configured_backend(self):
        self.assertFalse(self._make(None, 'realtime-ws').active_backend_is_local())
        self.assertFalse(self._make(None, 'rest-api').active_backend_is_local())
        self.assertTrue(self._make(None, 'faster-whisper').active_backend_is_local())
        # Legacy alias and unknown names both resolve to the local default.
        self.assertTrue(self._make(None, 'local').active_backend_is_local())
        self.assertTrue(self._make(None, 'not-a-backend').active_backend_is_local())


class ResumeBackendFailureTests(unittest.TestCase):
    """A failed backend reinit must still arm the background retry."""

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _make_app(self, is_local):
        main = self.main

        app = main.hyprwhsprApp.__new__(main.hyprwhsprApp)
        app.audio_capture = types.SimpleNamespace(
            recover_audio_capture=lambda reason: True)
        app.whisper_manager = types.SimpleNamespace(
            reinitialize_after_resume=lambda: False,
            active_backend_is_local=lambda: is_local,
        )
        app._mic_state_lock = threading.Lock()
        app._mic_disconnected = True
        app._background_recovery_needed = threading.Event()
        app._background_recovery_thread = None
        app._resync_shortcut_keyboards = lambda reason: None
        app._background_recovery_retry = lambda: None
        app.results = []
        app._write_recovery_result = lambda ok, reason: app.results.append((ok, reason))
        return app

    def _resume(self, app):
        with mock.patch.object(self.main.time, 'sleep'):
            app._on_system_resume()
        if app._background_recovery_thread is not None:
            app._background_recovery_thread.join(timeout=5)

    def test_realtime_backend_failure_starts_background_retry(self):
        app = self._make_app(is_local=False)
        self._resume(app)

        self.assertEqual(app.results, [(False, 'suspend_resume_websocket')])
        self.assertTrue(app._background_recovery_needed.is_set())
        self.assertIsNotNone(app._background_recovery_thread)

    def test_local_backend_failure_reports_model_reason(self):
        app = self._make_app(is_local=True)
        self._resume(app)

        self.assertEqual(app.results, [(False, 'suspend_resume_model')])
        self.assertTrue(app._background_recovery_needed.is_set())
        self.assertIsNotNone(app._background_recovery_thread)


if __name__ == "__main__":
    unittest.main()
