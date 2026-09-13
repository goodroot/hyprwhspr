"""Safety and path contracts for the pinned llama.cpp sidecar runtime."""

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402
import qwen3_asr_runtime  # noqa: E402


PREFIX = qwen3_asr_runtime.LLAMA_CPP_ARCHIVE_ROOT


def build_archive(path, members):
    """members: (name, kind, payload_or_linkname, mode) tuples."""
    with tarfile.open(path, "w:gz") as bundle:
        for name, kind, payload, mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "file":
                data = payload if isinstance(payload, bytes) else payload.encode()
                info.size = len(data)
                bundle.addfile(info, io.BytesIO(data))
            elif kind == "sym":
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                bundle.addfile(info)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                bundle.addfile(info)


class ExtractorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.archive = self.base / "runtime.tar.gz"
        self.dest = self.base / "out"
        self.dest.mkdir()

    def extract(self, members):
        build_archive(self.archive, members)
        backend_installer._extract_llama_runtime(self.archive, self.dest)

    def test_happy_path_preserves_exec_bit_and_symlinks(self):
        self.extract([
            (f"{PREFIX}/", "dir", None, 0o755),
            (f"{PREFIX}/llama-server", "file", b"ELF", 0o755),
            (f"{PREFIX}/libggml.so.0", "file", b"so", 0o755),
            (f"{PREFIX}/libggml.so", "sym", "libggml.so.0", 0o777),
        ])
        server = self.dest / "llama-server"
        self.assertTrue(os.access(server, os.X_OK))
        self.assertEqual(server.stat().st_mode & 0o777, 0o755)
        link = self.dest / "libggml.so"
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), "libggml.so.0")
        self.assertTrue(link.resolve().is_file())

    def test_member_outside_the_archive_root_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.extract([("elsewhere/llama-server", "file", b"ELF", 0o755)])

    def test_parent_traversal_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.extract([(f"{PREFIX}/../escape", "file", b"x", 0o644)])

    def test_absolute_member_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.extract([("/etc/passwd", "file", b"x", 0o644)])

    def test_symlink_escaping_the_runtime_is_rejected(self):
        for target in ("../outside", "/etc/passwd", ".."):
            with self.subTest(target=target):
                self.setUp()
                with self.assertRaises(RuntimeError):
                    self.extract([(f"{PREFIX}/evil", "sym", target, 0o777)])

    def test_oversized_member_is_rejected(self):
        with mock.patch.object(backend_installer, "_LLAMA_MEMBER_LIMIT", 8):
            with self.assertRaises(RuntimeError):
                self.extract([(f"{PREFIX}/big", "file", b"x" * 64, 0o644)])

    def test_total_size_limit_is_enforced(self):
        with mock.patch.object(backend_installer, "_LLAMA_TOTAL_LIMIT", 16):
            with self.assertRaises(RuntimeError):
                self.extract([
                    (f"{PREFIX}/a", "file", b"x" * 12, 0o644),
                    (f"{PREFIX}/b", "file", b"x" * 12, 0o644),
                ])

    def test_unsupported_member_type_is_rejected(self):
        build_archive(self.archive, [(f"{PREFIX}/keep", "file", b"x", 0o644)])
        with tarfile.open(self.archive, "w:gz") as bundle:
            info = tarfile.TarInfo(f"{PREFIX}/fifo")
            info.type = tarfile.FIFOTYPE
            bundle.addfile(info)
        with self.assertRaises(RuntimeError):
            backend_installer._extract_llama_runtime(self.archive, self.dest)


class RuntimePathTests(unittest.TestCase):
    def test_runtime_lives_under_the_uninstallable_runtime_tree(self):
        # cli/uninstall.py clears DATA_DIR/runtime wholesale; keeping the
        # sidecar there is what makes uninstall cover it.
        from paths import DATA_DIR
        self.assertEqual(
            qwen3_asr_runtime.QWEN3_ASR_RUNTIMES_DIR.parent, DATA_DIR / "runtime")
        self.assertTrue(
            qwen3_asr_runtime.QWEN3_ASR_MODELS_DIR.is_relative_to(DATA_DIR / "qwen3-asr"))

    def test_layout_is_flat_with_no_bin_or_lib_split(self):
        device = "vulkan"
        root = qwen3_asr_runtime.runtime_dir(device)
        self.assertEqual(qwen3_asr_runtime.server_path(device), root / "llama-server")
        self.assertEqual(qwen3_asr_runtime.library_dir(device), root)

    def test_every_device_has_a_pinned_asset(self):
        self.assertEqual(set(qwen3_asr_runtime.QWEN3_ASR_DEVICES),
                         set(qwen3_asr_runtime.LLAMA_CPP_ASSETS))
        for device, (name, size, digest) in qwen3_asr_runtime.LLAMA_CPP_ASSETS.items():
            with self.subTest(device=device):
                self.assertIn(qwen3_asr_runtime.LLAMA_CPP_RELEASE, name)
                self.assertGreater(size, 0)
                self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_cuda_is_not_offered(self):
        # Upstream ships no Linux CUDA asset; offering it would select a runtime
        # that can never be installed.
        self.assertNotIn("cuda", qwen3_asr_runtime.QWEN3_ASR_DEVICES)

    def test_runtime_must_be_an_executable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = Path(tmp) / "llama-server"
            server.write_bytes(b"ELF")
            with mock.patch.object(qwen3_asr_runtime, "server_path", return_value=server):
                server.chmod(0o644)
                self.assertFalse(qwen3_asr_runtime.runtime_installed("cpu"))
                server.chmod(0o755)
                self.assertTrue(qwen3_asr_runtime.runtime_installed("cpu"))


class DeviceResolutionTests(unittest.TestCase):
    class Config:
        def __init__(self, device):
            self.device = device

        def get_setting(self, key, default=None):
            return self.device if key == "qwen3_asr_device" else default

    def test_explicit_device_wins(self):
        self.assertEqual(
            qwen3_asr_runtime.resolve_device(self.Config("cpu")), "cpu")
        self.assertEqual(
            qwen3_asr_runtime.resolve_device(self.Config("vulkan")), "vulkan")

    def test_auto_prefers_an_installed_runtime_over_a_probe(self):
        with mock.patch.object(qwen3_asr_runtime, "runtime_installed",
                               side_effect=lambda d: d == "cpu"), \
                mock.patch.object(qwen3_asr_runtime, "detect_device",
                                  return_value="vulkan") as probe:
            self.assertEqual(qwen3_asr_runtime.resolve_device(self.Config("auto")), "cpu")
        probe.assert_not_called()

    def test_auto_prefers_vulkan_when_both_runtimes_are_installed(self):
        # The regression: QWEN3_ASR_DEVICES was serving as both the membership
        # set and the preference order, so a machine with both runtimes on disk
        # silently ran on CPU.
        with mock.patch.object(qwen3_asr_runtime, "runtime_installed", return_value=True):
            self.assertEqual(
                qwen3_asr_runtime.resolve_device(self.Config("auto")), "vulkan")

    def test_preference_order_is_accelerated_first_and_covers_every_device(self):
        self.assertEqual(qwen3_asr_runtime.DEVICE_PREFERENCE[0], "vulkan")
        self.assertEqual(set(qwen3_asr_runtime.DEVICE_PREFERENCE),
                         set(qwen3_asr_runtime.QWEN3_ASR_DEVICES))

    def test_force_probe_ignores_what_is_already_installed(self):
        # An explicit reinstall must be able to move a CPU-only box to Vulkan.
        with mock.patch.object(qwen3_asr_runtime, "runtime_installed",
                               side_effect=lambda d: d == "cpu"), \
                mock.patch.object(qwen3_asr_runtime, "detect_device", return_value="vulkan"):
            self.assertEqual(
                qwen3_asr_runtime.resolve_device(self.Config("auto"), force_probe=True),
                "vulkan")

    def test_force_probe_still_respects_an_explicit_device(self):
        with mock.patch.object(qwen3_asr_runtime, "detect_device", return_value="vulkan"):
            self.assertEqual(
                qwen3_asr_runtime.resolve_device(self.Config("cpu"), force_probe=True), "cpu")

    def test_auto_probes_when_nothing_is_installed(self):
        with mock.patch.object(qwen3_asr_runtime, "runtime_installed", return_value=False), \
                mock.patch.object(qwen3_asr_runtime, "detect_device", return_value="vulkan"):
            self.assertEqual(qwen3_asr_runtime.resolve_device(self.Config("auto")), "vulkan")

    def test_detect_falls_back_to_cpu_without_vulkaninfo(self):
        with mock.patch.object(qwen3_asr_runtime.shutil, "which", return_value=None):
            self.assertEqual(qwen3_asr_runtime.detect_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
