"""Service-unit detection and the single-instance guards."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import instance_detection  # noqa: E402
from tests.test_suspend_resume_recovery import _import_main_isolated  # noqa: E402

OWN_UNIT = '0::/user.slice/user-1000.slice/user@1000.service/app.slice/hyprwhspr.service\n'
COMPOSITOR_UNIT = ('0::/user.slice/user-1000.slice/user@1000.service/session.slice/'
                   'wayland-wm@hyprland.desktop.service\n')
INHERITED_ENV = {'INVOCATION_ID': '1ec56bf2997940a69abc236d8a53fb27',
                 'JOURNAL_STREAM': '12:40978'}


def fake_cgroup(content):
    real_path = Path
    cgroup_file = mock.Mock()
    if content is None:
        cgroup_file.read_text.side_effect = OSError('unreadable')
    else:
        cgroup_file.read_text.return_value = content

    def path(p):
        return cgroup_file if p == '/proc/self/cgroup' else real_path(p)
    return mock.patch.object(instance_detection, 'Path', path)


class ServiceUnitDetectionTests(unittest.TestCase):
    def _detect(self, cgroup):
        with fake_cgroup(cgroup), mock.patch.dict(os.environ, INHERITED_ENV):
            return instance_detection.running_as_service_unit()

    def test_own_unit(self):
        self.assertTrue(self._detect(OWN_UNIT))

    def test_own_unit_on_hybrid_cgroup_v1(self):
        self.assertTrue(self._detect(
            '12:pids:/user.slice\n'
            '1:name=systemd:/user.slice/user-1000.slice/user@1000.service/'
            'app.slice/hyprwhspr.service\n'))

    def test_inherited_invocation_id_from_compositor_unit(self):
        self.assertFalse(self._detect(COMPOSITOR_UNIT))

    def test_unreadable_cgroup(self):
        self.assertFalse(self._detect(None))


class InheritedSystemdEnvGuardTests(unittest.TestCase):
    def test_process_detection_still_runs(self):
        with fake_cgroup(COMPOSITOR_UNIT), mock.patch.dict(os.environ, INHERITED_ENV), \
                mock.patch.object(instance_detection, 'is_service_active_via_systemd',
                                  return_value=True), \
                mock.patch.object(instance_detection, 'find_hyprwhspr_processes',
                                  return_value=[4242]), \
                mock.patch.object(instance_detection, 'verify_process_is_valid',
                                  return_value=True):
            running, how = instance_detection.is_hyprwhspr_running()
        self.assertTrue(running)
        self.assertIn('4242', how)

    def test_service_unit_skips_process_detection(self):
        with fake_cgroup(OWN_UNIT), \
                mock.patch.object(instance_detection, 'find_hyprwhspr_processes') as find:
            self.assertEqual(instance_detection.is_hyprwhspr_running(), (False, None))
        find.assert_not_called()


class LockFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.runtime = Path(temp.name)
        self.lock = self.runtime / 'hyprwhspr' / 'hyprwhspr.lock'
        # Module function, not an app method: patch_app_global doesn't apply.
        for patcher in (mock.patch.dict(vars(self.main), {'LOCK_FILE': self.lock}),
                        mock.patch.object(self.main.atexit, 'register')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.main._release_lock_file)

    def test_takes_the_lock_whatever_the_env_or_unit(self):
        for cgroup in (COMPOSITOR_UNIT, OWN_UNIT):
            with self.subTest(cgroup=cgroup), fake_cgroup(cgroup), \
                    mock.patch.dict(os.environ, INHERITED_ENV):
                self.assertEqual(self.main._acquire_lock_file(), (True, None))
                self.assertEqual(self.lock.read_text(), str(os.getpid()))
                self.main._release_lock_file()

    def test_duplicate_is_refused_and_keeps_the_holder_pid(self):
        self.assertEqual(self.main._acquire_lock_file(), (True, None))
        holder = self.main._lock_file
        self.main._lock_file = None
        self.addCleanup(holder.close)
        self.assertEqual(self.main._acquire_lock_file(), (False, f'lock file (PID: {os.getpid()})'))
        self.assertEqual(self.lock.read_text(), str(os.getpid()))

    def test_release_keeps_the_file_and_frees_the_lock(self):
        self.main._acquire_lock_file()
        self.main._release_lock_file()
        self.assertTrue(self.lock.exists())
        self.assertEqual(self.main._acquire_lock_file(), (True, None))

    def test_updater_sees_the_running_daemon(self):
        import managed_install
        installation = managed_install.Installation(self.runtime / 'data', self.runtime / 'state')
        with mock.patch.dict(os.environ, {'XDG_RUNTIME_DIR': str(self.runtime)}):
            self.main._acquire_lock_file()
            self.assertTrue(installation.daemon_running())
            self.main._release_lock_file()
            self.assertFalse(installation.daemon_running())

    def test_allow_second_instance_skips_process_detection(self):
        with mock.patch.dict(os.environ, {'HYPRWHSPR_ALLOW_SECOND_INSTANCE': '1'}), \
                mock.patch.dict(sys.modules, {'instance_detection': instance_detection}), \
                mock.patch.object(instance_detection, 'is_hyprwhspr_running') as check:
            self.assertEqual(self.main._is_hyprwhspr_running(), (False, None))
        check.assert_not_called()


if __name__ == '__main__':
    unittest.main()
