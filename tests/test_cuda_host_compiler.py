import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402


class CudaHostCompilerTests(unittest.TestCase):
    def _toolkit(self, tmp, header):
        root = Path(tmp) / "cuda"
        nvcc = root / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()
        config = root / "include" / "crt" / "host_config.h"
        config.parent.mkdir(parents=True)
        config.write_text(header)
        return nvcc

    def _detect(self, nvcc, executables, versions):
        def which(name):
            return executables.get(name)

        def run(command, **kwargs):
            compiler = command[0]
            version = versions.get(compiler)
            if version is None:
                return subprocess.CompletedProcess(command, 1, stdout=b"")
            return subprocess.CompletedProcess(command, 0, stdout=f"{version}\n".encode())

        with mock.patch.dict(os.environ, {"CUDACXX": str(nvcc)}, clear=True), \
             mock.patch.object(backend_installer.shutil, "which", side_effect=which), \
             mock.patch.object(backend_installer, "run_command", side_effect=run):
            return backend_installer.detect_cuda_host_compiler()

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "custom-g++"
            override.touch()
            override.chmod(0o755)
            with mock.patch.dict(
                os.environ, {"HYPRWHSPR_CUDA_HOST": str(override)}, clear=True
            ):
                self.assertEqual(backend_installer.detect_cuda_host_compiler(), str(override))

    def test_gcc16_uses_gcc15_when_toolkit_allows_gcc15(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++", "g++-15": "/custom/g++-15"},
                    {"/custom/g++": 16, "/custom/g++-15": 15},
                ),
                "/custom/g++-15",
            )

    def test_cuda12_ceiling_skips_newer_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 14\n#error -- unsupported GNU version!\n#endif\n")
            self.assertEqual(
                self._detect(
                    nvcc,
                    {
                        "g++": "/custom/g++",
                        "g++-14": "/custom/g++-14",
                    },
                    {"/custom/g++": 16, "/custom/g++-14": 14},
                ),
                "/custom/g++-14",
            )

    def test_target_layout_header_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cuda"
            nvcc = root / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.touch()
            config = root / "targets" / "x86_64-linux" / "include" / "crt" / "host_config.h"
            config.parent.mkdir(parents=True)
            config.write_text("#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++", "g++-15": "/custom/g++-15"},
                    {"/custom/g++": 16, "/custom/g++-15": 15},
                ),
                "/custom/g++-15",
            )

    def test_path_discovered_nvcc_is_used_when_cudacxx_is_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            executables = {
                "nvcc": str(nvcc),
                "g++": "/custom/g++",
                "g++-15": "/custom/g++-15",
            }
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(backend_installer.shutil, "which", side_effect=executables.get), \
                 mock.patch.object(
                     backend_installer,
                     "run_command",
                     side_effect=lambda command, **kwargs: subprocess.CompletedProcess(
                         command,
                         0,
                         stdout={
                             "/custom/g++": b"16.1.0\n",
                             "/custom/g++-15": b"15.2.0\n",
                         }.get(command[0], b""),
                     ),
                 ):
                self.assertEqual(backend_installer.detect_cuda_host_compiler(), "/custom/g++-15")

    def test_candidate_version_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            self.assertEqual(
                self._detect(
                    nvcc,
                    {
                        "g++": "/custom/g++",
                        "g++-15": "/custom/g++-15",
                        "g++-14": "/custom/g++-14",
                    },
                    {
                        "/custom/g++": 16,
                        "/custom/g++-15": 16,
                        "/custom/g++-14": 14,
                    },
                ),
                "/custom/g++-14",
            )

    def test_minor_version_ceiling_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(
                tmp,
                "#if __GNUC__ > 13 || (__GNUC__ == 13 && __GNUC_MINOR__ > 2)\n"
                "#error -- unsupported GNU version!\n#endif\n",
            )
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++", "g++-13": "/custom/g++-13"},
                    {"/custom/g++": "13.3.0", "/custom/g++-13": "13.2.1"},
                ),
                "/custom/g++-13",
            )

    def test_unrelated_gcc_guard_does_not_change_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(
                tmp,
                "#if __GNUC__ > 7\n#endif\n"
                "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n",
            )
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++", "g++-15": "/custom/g++-15"},
                    {"/custom/g++": 16, "/custom/g++-15": 15},
                ),
                "/custom/g++-15",
            )

    def test_nested_gcc_guard_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(
                tmp,
                "#if defined(__GNUC__)\n"
                "#if __GNUC__ > 14\n"
                "#error -- unsupported GNU version! later than 14\n"
                "#endif\n#endif\n",
            )
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++", "g++-14": "/custom/g++-14"},
                    {"/custom/g++": 15, "/custom/g++-14": 14},
                ),
                "/custom/g++-14",
            )

    def test_compatible_default_compiler_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            self.assertEqual(
                self._detect(
                    nvcc,
                    {"g++": "/custom/g++"},
                    {"/custom/g++": 15},
                ),
                "/custom/g++",
            )

    def test_unparseable_header_preserves_default_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "/* compiler limit unavailable */\n")
            with mock.patch.object(backend_installer, "log_warning") as warning:
                self.assertEqual(
                    self._detect(
                        nvcc,
                        {"g++": "/custom/g++"},
                        {"/custom/g++": 16},
                    ),
                    "/custom/g++",
                )
            warning.assert_called_once_with(
                "Could not determine CUDA's supported GCC version; using system g++ "
                "(set HYPRWHSPR_CUDA_HOST to override)"
            )

    def test_unusable_override_warns_before_auto_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 15\n#error -- unsupported GNU version!\n#endif\n")
            bad_override = Path(tmp) / "not-executable-g++"
            bad_override.touch()
            with mock.patch.dict(
                os.environ,
                {"CUDACXX": str(nvcc), "HYPRWHSPR_CUDA_HOST": str(bad_override)},
                clear=True,
            ), mock.patch.object(
                backend_installer.shutil,
                "which",
                side_effect={"g++": "/custom/g++"}.get,
            ), mock.patch.object(
                backend_installer,
                "run_command",
                return_value=subprocess.CompletedProcess([], 0, stdout=b"15.1.0\n"),
            ), mock.patch.object(backend_installer, "log_warning") as warning:
                self.assertEqual(backend_installer.detect_cuda_host_compiler(), "/custom/g++")
            warning.assert_any_call(
                f"HYPRWHSPR_CUDA_HOST is not an executable file: {bad_override}; auto-detecting"
            )

    def test_no_compatible_compiler_falls_back_to_cpu_with_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = self._toolkit(tmp, "#if __GNUC__ > 14\n#error -- unsupported GNU version!\n#endif\n")
            with mock.patch.object(backend_installer, "log_warning") as warning:
                self.assertEqual(
                    self._detect(
                        nvcc,
                        {"g++": "/custom/g++"},
                        {"/custom/g++": 16},
                    ),
                    None,
                )
            warning.assert_any_call(
                "No installed g++ is compatible with CUDA's GCC <= 14 requirement"
            )
            warning.assert_any_call("Install a GCC 14 toolchain (on Arch: yay -S gcc14 gcc14-libs)")
            warning.assert_any_call("Or set HYPRWHSPR_CUDA_HOST to a compatible g++ executable")


if __name__ == "__main__":
    unittest.main()
