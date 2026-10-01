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

    def test_pacman_installs_spirv_headers_for_source_builds(self):
        sudo = mock.Mock(return_value=FakeResult(0))
        with mock.patch.object(backend_installer.shutil, "which", side_effect=lambda n: f"/usr/bin/{n}"), \
             mock.patch.object(backend_installer, "run_sudo_command", sudo), \
             mock.patch.object(backend_installer, "run_command", return_value=FakeResult(0, GPU_SUMMARY)), \
             mock.patch.object(backend_installer, "vulkaninfo_has_hardware_gpu", return_value=True):
            backend_installer.setup_vulkan_support()
        self.assertIn("spirv-headers", sudo.call_args.args[0])

    def test_missing_build_headers_do_not_block_detection(self):
        """Without pacman, a wheel install needs no headers; only vulkaninfo decides."""
        with mock.patch.object(backend_installer.shutil, "which",
                               side_effect=lambda n: "/usr/bin/vulkaninfo" if n == "vulkaninfo" else None), \
             mock.patch.object(backend_installer.Path, "exists", return_value=False), \
             mock.patch.object(backend_installer, "run_command", return_value=FakeResult(0, GPU_SUMMARY)), \
             mock.patch.object(backend_installer, "vulkaninfo_has_hardware_gpu", return_value=True):
            self.assertTrue(backend_installer.setup_vulkan_support())


class VulkanBuildToolsTests(unittest.TestCase):
    """ggml-vulkan's CMake needs the headers, glslc and the SPIRV-Headers config."""

    def _missing(self, header=True, glslc=True, spirv=True):
        with mock.patch.object(backend_installer.Path, "exists", return_value=header), \
             mock.patch.object(backend_installer.shutil, "which",
                               side_effect=lambda n: "/usr/bin/glslc" if glslc and n == "glslc" else None), \
             mock.patch.object(backend_installer.glob, "glob",
                               side_effect=lambda p: [p] if spirv and p.startswith("/usr/share/") else []):
            return backend_installer._missing_vulkan_build_tools()

    def test_complete_toolchain(self):
        self.assertEqual(self._missing(), [])

    def test_each_missing_piece_is_named(self):
        self.assertEqual(self._missing(header=False), ["Vulkan headers"])
        self.assertEqual(self._missing(glslc=False), ["glslc"])
        self.assertEqual(self._missing(spirv=False), ["SPIRV-Headers"])


class InstallVulkanTests(unittest.TestCase):

    def _install(self, missing, wheel=None, glibc=(2, 41)):
        prepare = mock.Mock(return_value=False)
        with mock.patch.object(backend_installer, "_glibc_version", return_value=glibc), \
             mock.patch.object(backend_installer, "download_pywhispercpp_wheel", return_value=wheel), \
             mock.patch.object(backend_installer, "install_pywhispercpp_from_wheel", return_value=True), \
             mock.patch.object(backend_installer, "_missing_vulkan_build_tools", return_value=missing), \
             mock.patch.object(backend_installer, "install_system_dependencies") as sysdeps, \
             mock.patch.object(backend_installer, "_prepare_pywhispercpp_sources", prepare):
            result = backend_installer.install_pywhispercpp_vulkan(Path("/venv/bin/pip"))
        return result, sysdeps, prepare

    def test_missing_tools_fail_before_the_source_build(self):
        result, sysdeps, prepare = self._install(["SPIRV-Headers"])
        self.assertFalse(result)
        sysdeps.assert_not_called()
        prepare.assert_not_called()

    def test_wheel_needs_no_build_tools(self):
        result, _, prepare = self._install(["SPIRV-Headers"], wheel=Path("/cache/x.whl"))
        self.assertTrue(result)
        prepare.assert_not_called()

    def test_complete_toolchain_proceeds_to_source_build(self):
        _, sysdeps, prepare = self._install([], glibc=(2, 39))
        sysdeps.assert_called_once()
        prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
