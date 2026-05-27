import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import ydotoold_session  # noqa: E402
from ydotoold_session import YdotooldSession  # noqa: E402


class FakeProc:
    """Minimal stand-in for subprocess.Popen."""

    def __init__(self, alive=True):
        self._alive = alive
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True
        self._alive = False

    def die(self):
        self._alive = False


def _spawn_factory(socket_path, *, create_socket=True, alive=True, record=None):
    """Build a spawn callable that simulates ydotoold registering its socket."""
    def _spawn():
        proc = FakeProc(alive=alive)
        if record is not None:
            record.append(proc)
        if create_socket:
            open(socket_path, "w").close()  # os.path.exists() → socket ready
        return proc
    return _spawn


class YdotooldSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sock = os.path.join(self.tmp.name, "hyprwhspr-ydotool.sock")

    def tearDown(self):
        self.tmp.cleanup()

    def test_lazy_no_spawn_until_ensure(self):
        record = []
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, record=record))
        self.assertEqual(record, [])          # constructing must not spawn
        self.assertFalse(s.is_running())      # is_running() must not spawn either
        self.assertEqual(record, [])

    def test_ensure_running_spawns_and_waits_for_socket(self):
        record = []
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, record=record))
        self.assertTrue(s.ensure_running())
        self.assertEqual(len(record), 1)
        self.assertTrue(s.is_running())
        self.assertTrue(os.path.exists(self.sock))

    def test_ensure_running_idempotent(self):
        record = []
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, record=record))
        self.assertTrue(s.ensure_running())
        self.assertTrue(s.ensure_running())   # alive + socket present → no respawn
        self.assertEqual(len(record), 1)

    def test_respawn_after_death(self):
        record = []
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, record=record))
        self.assertTrue(s.ensure_running())
        record[0].die()                        # daemon crashed between injections
        self.assertFalse(s.is_running())
        self.assertTrue(s.ensure_running())    # next use restarts it
        self.assertEqual(len(record), 2)

    def test_restart_once_then_give_up_when_socket_never_appears(self):
        # Spawn succeeds (proc alive) but the socket never shows up → timeout → False.
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, create_socket=False),
                            socket_timeout=0.15, poll_interval=0.02)
        self.assertFalse(s.ensure_running())

    def test_spawn_failure_returns_false(self):
        def _boom():
            raise OSError("ydotoold not found")
        s = YdotooldSession(socket_path=self.sock, spawn=_boom)
        self.assertFalse(s.ensure_running())

    def test_default_spawn_builds_correct_argv(self):
        captured = {}
        sock = self.sock

        class _Rec:
            def __init__(self_inner, argv, **kw):
                captured['argv'] = argv
                open(sock, "w").close()

            def poll(self_inner):
                return None

        orig = ydotoold_session.subprocess.Popen
        ydotoold_session.subprocess.Popen = _Rec
        try:
            s = YdotooldSession(socket_path=sock)  # uses _default_spawn
            self.assertTrue(s.ensure_running())
        finally:
            ydotoold_session.subprocess.Popen = orig
        self.assertEqual(captured['argv'],
                         ['ydotoold', '-p', sock, '-P', '0600'])

    def test_socket_env_points_at_private_socket(self):
        s = YdotooldSession(socket_path=self.sock, spawn=_spawn_factory(self.sock))
        env = s.socket_env()
        self.assertEqual(env['YDOTOOL_SOCKET'], self.sock)

    def test_close_terminates_and_unlinks_and_is_idempotent(self):
        record = []
        s = YdotooldSession(socket_path=self.sock,
                            spawn=_spawn_factory(self.sock, record=record))
        s.ensure_running()
        self.assertTrue(os.path.exists(self.sock))
        s.close()
        self.assertTrue(record[0].terminated)
        self.assertFalse(os.path.exists(self.sock))
        self.assertFalse(s.is_running())
        s.close()  # second close must be a harmless no-op


if __name__ == "__main__":
    unittest.main()
