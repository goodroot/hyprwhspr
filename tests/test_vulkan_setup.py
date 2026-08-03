import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402


class FakeResult:
    def __init__(self, returncode=0, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


GPU_SUMMARY = b"GPU0:\n\tapiVersion = 1.4.354\n\tdeviceName = Radeon RX 9070 XT\n\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"


class SetupVulkanSupportTests(unittest.TestCase):
    """The package install is a convenience; `vulkaninfo` is the real check."""

    def _run(self, pacman_result=None, pacman_exc=None, has_gpu=True):
        def fake_which(name):
            return f"/usr/bin/{name}" if name in ("pacman", "vulkaninfo") else None

        sudo = mock.Mock(side_effect=pacman_exc) if pacman_exc else mock.Mock(return_value=pacman_result)

        with mock.patch.object(backend_installer.shutil, "which", side_effect=fake_which), \
             mock.patch.object(backend_installer, "run_sudo_command", sudo), \
             mock.patch.object(backend_installer, "run_command", return_value=FakeResult(0, GPU_SUMMARY)), \
             mock.patch.object(backend_installer, "vulkaninfo_has_hardware_gpu", return_value=has_gpu):
            return backend_installer.setup_vulkan_support()

    def test_failed_package_install_still_detects_working_vulkan(self):
        self.assertTrue(self._run(pacman_result=FakeResult(returncode=1)))

    def test_sudo_exception_still_detects_working_vulkan(self):
        self.assertTrue(self._run(pacman_exc=PermissionError("a password is required")))

    def test_no_result_from_package_install_still_detects_working_vulkan(self):
        self.assertTrue(self._run(pacman_result=None))

    def test_successful_package_install_detects_working_vulkan(self):
        self.assertTrue(self._run(pacman_result=FakeResult(returncode=0)))

    def test_still_returns_false_when_no_hardware_gpu(self):
        """The capability check must remain authoritative in both directions."""
        self.assertFalse(self._run(pacman_result=FakeResult(returncode=1), has_gpu=False))


if __name__ == "__main__":
    unittest.main()
