import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402
import config_manager  # noqa: E402
import whisper_manager  # noqa: E402
from backends import pywhispercpp_backend  # noqa: E402
from backends import PywhispercppBackend  # noqa: E402


class WheelVariantTests(unittest.TestCase):
    def test_no_cuda_returns_none(self):
        self.assertIsNone(backend_installer._get_wheel_variant(None))
        self.assertIsNone(backend_installer._get_wheel_variant(""))

    def test_cuda12_minors_all_map_to_one_wheel(self):
        for v in ("12.0", "12.4", "12.6", "12.9"):
            self.assertEqual(backend_installer._get_wheel_variant(v), "cuda12")

    def test_cuda11_and_13_fall_back_to_source(self):
        with mock.patch.object(backend_installer, "log_info"):
            for v in ("10.2", "11.8", "13.0"):
                self.assertIsNone(backend_installer._get_wheel_variant(v))


class WheelFilenameTests(unittest.TestCase):
    def test_download_and_pip_names_per_python(self):
        ver = backend_installer.PYWHISPERCPP_VERSION
        build = backend_installer.PYWHISPERCPP_WHEEL_BUILD
        for py in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            tag = "cp" + py.replace(".", "")
            base = f"pywhispercpp-{ver}-{build}-{tag}-{tag}-linux_x86_64"
            self.assertEqual(
                backend_installer._get_wheel_filename(py, "cuda12", True),
                f"{base}+cuda12.whl",
            )
            self.assertEqual(
                backend_installer._get_wheel_filename(py, "cuda12", False),
                f"{base}.whl",
            )


    def test_vulkan_download_name_carries_the_variant_suffix(self):
        ver = backend_installer.PYWHISPERCPP_VERSION
        build = backend_installer.PYWHISPERCPP_WHEEL_BUILD
        variant = backend_installer.VULKAN_WHEEL_VARIANT
        self.assertEqual(variant, "vulkan")
        self.assertEqual(
            backend_installer._get_wheel_filename("3.14", variant, True),
            f"pywhispercpp-{ver}-{build}-cp314-cp314-linux_x86_64+vulkan.whl",
        )

    def test_build_number_is_a_wheel_build_tag(self):
        build = backend_installer.PYWHISPERCPP_WHEEL_BUILD
        self.assertIsInstance(build, int)
        self.assertGreaterEqual(build, 1)


class WheelChecksumTests(unittest.TestCase):
    WHEEL = b"wheel bytes"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name)
        self.name = backend_installer._get_wheel_filename("3.14", "vulkan", True)
        self.pip_name = backend_installer._get_wheel_filename("3.14", "vulkan", False)

    def _download(self, published, served=WHEEL):
        """Run download_pywhispercpp_wheel('vulkan'); returns (path, urlretrieve mock)."""
        def retrieve(url, path, reporthook=None):
            Path(path).write_bytes(served)

        with mock.patch.object(backend_installer, "WHEEL_CACHE_DIR", self.cache), \
                mock.patch.object(backend_installer, "_detect_venv_python_version", return_value="3.14"), \
                mock.patch.object(backend_installer, "_published_wheel_sha256", return_value=published), \
                mock.patch.object(backend_installer.urllib.request, "urlretrieve",
                                  side_effect=retrieve) as urlretrieve, \
                mock.patch.object(backend_installer, "log_info"), \
                mock.patch.object(backend_installer, "log_success"), \
                mock.patch.object(backend_installer, "log_warning"):
            return backend_installer.download_pywhispercpp_wheel("vulkan"), urlretrieve

    def _sha(self, data):
        import hashlib
        return hashlib.sha256(data).hexdigest()

    def test_verified_download_is_returned_under_its_pip_name(self):
        path, _ = self._download(self._sha(self.WHEEL))
        self.assertEqual(path, self.cache / "vulkan" / self.pip_name)
        self.assertEqual(path.read_bytes(), self.WHEEL)

    def test_checksum_mismatch_rejects_and_removes_the_download(self):
        path, _ = self._download(self._sha(b"other bytes"))
        self.assertIsNone(path)
        self.assertEqual(list((self.cache / "vulkan").iterdir()), [])

    def test_unpublished_wheel_is_not_downloaded(self):
        path, urlretrieve = self._download(None)
        self.assertIsNone(path)
        urlretrieve.assert_not_called()

    def test_matching_cached_wheel_skips_the_download(self):
        cached = self.cache / "vulkan" / self.pip_name
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.WHEEL)
        path, urlretrieve = self._download(self._sha(self.WHEEL))
        self.assertEqual(path, cached)
        urlretrieve.assert_not_called()

    def test_stale_cached_wheel_is_downloaded_again(self):
        cached = self.cache / "vulkan" / self.pip_name
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"stale bytes")
        path, urlretrieve = self._download(self._sha(self.WHEEL))
        urlretrieve.assert_called_once()
        self.assertEqual(path.read_bytes(), self.WHEEL)

    def test_cached_wheel_is_used_offline(self):
        cached = self.cache / "vulkan" / self.pip_name
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.WHEEL)
        path, urlretrieve = self._download(None)
        self.assertEqual(path, cached)
        urlretrieve.assert_not_called()

    def test_published_checksum_is_read_from_the_release_sums(self):
        sums = (f"{'a' * 64}  other.whl\n{'b' * 64}  {self.name}\n").encode()
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = sums
        with mock.patch.object(backend_installer.urllib.request, "urlopen",
                               return_value=response) as urlopen:
            self.assertEqual(backend_installer._published_wheel_sha256(self.name), "b" * 64)
            self.assertIsNone(backend_installer._published_wheel_sha256("missing.whl"))
        self.assertTrue(urlopen.call_args.args[0].endswith("/SHA256SUMS.txt"))

    def test_unreachable_checksum_list_yields_none(self):
        with mock.patch.object(backend_installer.urllib.request, "urlopen",
                               side_effect=OSError("offline")):
            self.assertIsNone(backend_installer._published_wheel_sha256(self.name))


class WheelLoadProbeTests(unittest.TestCase):
    def _probe(self, **result):
        run = mock.Mock(return_value=mock.Mock(**result)) if result else mock.Mock(
            side_effect=backend_installer.subprocess.TimeoutExpired("python", 60))
        with mock.patch.object(backend_installer.subprocess, "run", run), \
                mock.patch.object(backend_installer, "log_warning") as warn:
            loaded = backend_installer._wheel_native_libs_load(Path("/venv/bin/pip"))
        return loaded, run, warn

    def test_extension_is_imported_with_the_venv_python(self):
        loaded, run, _ = self._probe(returncode=0, stderr="")
        self.assertTrue(loaded)
        self.assertEqual(run.call_args.args[0], ["/venv/bin/python", "-c", "import _pywhispercpp"])

    def test_missing_library_is_reported(self):
        loaded, _, warn = self._probe(
            returncode=1, stderr="Traceback\nImportError: libvulkan.so.1: cannot open shared object file")
        self.assertFalse(loaded)
        self.assertIn("libvulkan.so.1", warn.call_args.args[0])

    def test_hung_import_counts_as_failure(self):
        self.assertFalse(self._probe()[0])


class VulkanWheelInstallTests(unittest.TestCase):
    def _install(self, wheel, wheel_installs=True, glibc=(2, 42), wheel_loads=True):
        """Run install_pywhispercpp_vulkan; returns (result, download, from_wheel, prepare mocks)."""
        with mock.patch.object(backend_installer, "_glibc_version", return_value=glibc), \
                mock.patch.object(backend_installer, "download_pywhispercpp_wheel",
                                  return_value=wheel) as download, \
                mock.patch.object(backend_installer, "install_pywhispercpp_from_wheel",
                                  return_value=wheel_installs) as from_wheel, \
                mock.patch.object(backend_installer, "_wheel_native_libs_load",
                                  return_value=wheel_loads), \
                mock.patch.object(backend_installer, "install_system_dependencies"), \
                mock.patch.object(backend_installer, "_missing_vulkan_build_tools", return_value=[]), \
                mock.patch.object(backend_installer, "_prepare_pywhispercpp_sources",
                                  return_value=False) as prepare, \
                mock.patch.object(backend_installer, "log_info"), \
                mock.patch.object(backend_installer, "log_warning"):
            result = backend_installer.install_pywhispercpp_vulkan(Path("/venv/bin/pip"))
        return result, download, from_wheel, prepare

    def test_prebuilt_wheel_skips_the_source_build(self):
        wheel = Path("/cache/vulkan/pywhispercpp.whl")
        result, download, from_wheel, prepare = self._install(wheel)
        self.assertTrue(result)
        download.assert_called_once_with("vulkan")
        from_wheel.assert_called_once_with(Path("/venv/bin/pip"), wheel)
        prepare.assert_not_called()

    def test_missing_wheel_builds_from_source(self):
        result, _, from_wheel, prepare = self._install(None)
        self.assertFalse(result)
        from_wheel.assert_not_called()
        prepare.assert_called_once()

    def test_failed_wheel_install_builds_from_source(self):
        _, _, _, prepare = self._install(Path("/cache/vulkan/pywhispercpp.whl"), wheel_installs=False)
        prepare.assert_called_once()

    def test_wheel_that_does_not_load_builds_from_source(self):
        result, _, from_wheel, prepare = self._install(Path("/cache/vulkan/x.whl"), wheel_loads=False)
        self.assertFalse(result)
        from_wheel.assert_called_once()
        prepare.assert_called_once()

    def test_older_or_unknown_glibc_skips_the_wheel(self):
        minimum = backend_installer.VULKAN_WHEEL_MIN_GLIBC
        for glibc in ((minimum[0], minimum[1] - 1), None):
            with self.subTest(glibc=glibc):
                _, download, _, prepare = self._install(Path("/cache/vulkan/pywhispercpp.whl"), glibc=glibc)
                download.assert_not_called()
                prepare.assert_called_once()

    def test_minimum_glibc_uses_the_wheel(self):
        _, download, _, prepare = self._install(
            Path("/cache/vulkan/pywhispercpp.whl"), glibc=backend_installer.VULKAN_WHEEL_MIN_GLIBC)
        download.assert_called_once()
        prepare.assert_not_called()


class GlibcVersionTests(unittest.TestCase):
    def test_parses_glibc_and_rejects_other_libcs(self):
        for libc, expected in ((("glibc", "2.41"), (2, 41)), (("glibc", "2.42.9000"), (2, 42)),
                               (("musl", "1.2.5"), None), (("", ""), None), (("glibc", "x.y"), None)):
            with self.subTest(libc=libc), \
                    mock.patch.object(backend_installer.platform, "libc_ver", return_value=libc):
                self.assertEqual(backend_installer._glibc_version(), expected)


class WheelWorkflowContractTests(unittest.TestCase):
    def test_vulkan_jobs_derive_the_variant_and_run_the_verifier(self):
        workflow = (ROOT / ".github" / "workflows" / "build-wheels.yml").read_text(encoding="utf-8")
        self.assertIn("b.VULKAN_WHEEL_VARIANT", workflow)
        self.assertIn("b.VULKAN_WHEEL_MIN_GLIBC", workflow)
        self.assertIn("b.PYWHISPERCPP_WHEEL_BUILD", workflow)
        self.assertIn("container: debian:trixie", workflow)
        self.assertIn("python .github/scripts/verify_vulkan_wheel.py", workflow)
        self.assertTrue((ROOT / ".github" / "scripts" / "verify_vulkan_wheel.py").is_file())

    def test_every_wheel_build_is_portable(self):
        workflow = (ROOT / ".github" / "workflows" / "build-wheels.yml").read_text(encoding="utf-8")
        builds = workflow.count("pip wheel . --no-deps")
        self.assertEqual(builds, 2)
        self.assertEqual(workflow.count("GGML_NATIVE: 'OFF'"), builds)
        self.assertEqual(workflow.count("GGML_OPENMP: 'OFF'"), builds)
        self.assertEqual(workflow.count("patchelf --set-rpath '$ORIGIN'"), builds)
        self.assertEqual(workflow.count('--build-number "$WHEEL_BUILD"'), builds)
        wheel_uploads = [line for line in workflow.splitlines() if "gh release upload" in line and ".whl" in line]
        self.assertEqual(len(wheel_uploads), 1)
        self.assertNotIn("--clobber", wheel_uploads[0])
        self.assertNotIn("LD_LIBRARY_PATH=", workflow)


class ConfigDefaultsTests(unittest.TestCase):
    def _manager_for(self, cfg_dir, seed_file=None):
        cfg_file = Path(cfg_dir) / "config.json"
        if seed_file is not None:
            cfg_file.write_text(json.dumps(seed_file))
        with mock.patch.object(config_manager, "CONFIG_DIR", Path(cfg_dir)), \
             mock.patch.object(config_manager, "CONFIG_FILE", cfg_file):
            return config_manager.ConfigManager(verbose=False)

    def test_threads_default_is_capped_cpu_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._manager_for(tmp)
            self.assertEqual(cm.get_setting("threads"), min(8, os.cpu_count() or 4))

    def test_word_overrides_default_seeds_product_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._manager_for(tmp)
            self.assertEqual(cm.get_word_overrides(), {"hyper whisper": "hyprwhspr"})

    def test_existing_user_overrides_are_not_clobbered(self):
        # A user with their own overrides keeps exactly theirs; the product seed
        # is not merged in (config load is a top-level dict replace).
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._manager_for(
                tmp,
                seed_file={"$schema": "x", "word_overrides": {"foo": "bar"}},
            )
            self.assertEqual(cm.get_word_overrides(), {"foo": "bar"})

    def test_pywhispercpp_vad_default_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            cm = self._manager_for(tmp)
            self.assertIs(cm.get_setting("pywhispercpp_use_vad"), False)


class ModelValidityTests(unittest.TestCase):
    def test_file_hash_is_sha256_and_empty_for_missing_files(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "model.bin"
            data = bytes(range(256)) * 5000  # spans several read chunks
            f.write_bytes(data)
            self.assertEqual(backend_installer.compute_file_hash(f), hashlib.sha256(data).hexdigest())
            self.assertEqual(backend_installer.compute_file_hash(Path(tmp) / "absent.bin"), '')

    def test_hash_key_is_per_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "ggml-tiny.bin"
            f.write_bytes(b"tiny model bytes")
            h = backend_installer.compute_file_hash(f)
            own = {f"model_hash_{f.name}": h}
            with mock.patch.object(backend_installer, "get_state", side_effect=own.get):
                self.assertTrue(backend_installer.check_model_validity(f))
            # another model's stored hash must not validate this file
            other = {"model_hash_ggml-base.bin": h}
            with mock.patch.object(backend_installer, "get_state", side_effect=other.get):
                self.assertFalse(backend_installer.check_model_validity(f))

    def test_size_floor_without_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "ggml-tiny.bin"
            f.write_bytes(b"x")
            with mock.patch.object(backend_installer, "get_state", return_value=None):
                self.assertFalse(backend_installer.check_model_validity(f))
                with open(f, "wb") as fh:
                    fh.truncate(75_000_000)  # sparse; ~tiny model size
                self.assertTrue(backend_installer.check_model_validity(f))


class PywhisperVadKwargsTests(unittest.TestCase):
    class _FakeModel:
        last_kwargs = None
        raise_on_vad = False

        def __init__(self, **kwargs):
            if type(self).raise_on_vad and "vad" in kwargs:
                raise TypeError("unexpected keyword argument 'vad'")
            type(self).last_kwargs = kwargs

    def setUp(self):
        self._FakeModel.last_kwargs = None
        self._FakeModel.raise_on_vad = False
        fake_mod = types.ModuleType("pywhispercpp.model")
        fake_mod.Model = self._FakeModel
        fake_pkg = types.ModuleType("pywhispercpp")
        fake_pkg.model = fake_mod
        patcher = mock.patch.dict(
            sys.modules, {"pywhispercpp": fake_pkg, "pywhispercpp.model": fake_mod}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _manager(self, use_vad):
        cfg = mock.Mock()
        settings = {
            "sampling_strategy": "beam_search",
            "pywhispercpp_use_vad": use_vad,
        }
        cfg.get_setting.side_effect = lambda key, default=None: settings.get(key, default)
        return whisper_manager.WhisperManager(config_manager=cfg)

    def test_disabled_omits_vad_kwargs(self):
        wm = self._manager(use_vad=False)
        PywhispercppBackend(wm)._create_pywhisper_model("base", 4)
        self.assertNotIn("vad", self._FakeModel.last_kwargs)
        self.assertNotIn("vad_model_path", self._FakeModel.last_kwargs)

    def test_enabled_with_model_file_passes_vad(self):
        with tempfile.TemporaryDirectory() as tmp:
            vad_file = Path(tmp) / backend_installer.VAD_MODEL_FILENAME
            vad_file.write_bytes(b"x")
            with mock.patch.object(
                pywhispercpp_backend, "PYWHISPERCPP_MODELS_DIR", Path(tmp)
            ):
                wm = self._manager(use_vad=True)
                PywhispercppBackend(wm)._create_pywhisper_model("base", 4)
        self.assertIs(self._FakeModel.last_kwargs["vad"], True)
        self.assertEqual(self._FakeModel.last_kwargs["vad_model_path"], str(vad_file))

    def test_failed_download_falls_back_without_vad(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                pywhispercpp_backend, "PYWHISPERCPP_MODELS_DIR", Path(tmp)
            ), mock.patch.object(
                pywhispercpp_backend, "download_vad_model", return_value=False
            ):
                wm = self._manager(use_vad=True)
                PywhispercppBackend(wm)._create_pywhisper_model("base", 4)
        self.assertNotIn("vad", self._FakeModel.last_kwargs)

    def test_stale_pywhispercpp_typeerror_falls_back(self):
        self._FakeModel.raise_on_vad = True
        with tempfile.TemporaryDirectory() as tmp:
            vad_file = Path(tmp) / backend_installer.VAD_MODEL_FILENAME
            vad_file.write_bytes(b"x")
            with mock.patch.object(
                pywhispercpp_backend, "PYWHISPERCPP_MODELS_DIR", Path(tmp)
            ):
                wm = self._manager(use_vad=True)
                PywhispercppBackend(wm)._create_pywhisper_model("base", 4)
        self.assertNotIn("vad", self._FakeModel.last_kwargs)


class PywhispercppSourcePinTests(unittest.TestCase):
    """The clone/update is best-effort (check=False), so the pin assertion is the
    only thing standing between a failed fetch and a silent build of the old commit."""

    def _run(self, src_dir, head_result):
        def fake_run_command(cmd, **kwargs):
            if "rev-parse" in cmd:
                return head_result
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(backend_installer, "PYWHISPERCPP_SRC_DIR", src_dir), \
             mock.patch.object(backend_installer, "run_command", side_effect=fake_run_command):
            return backend_installer._prepare_pywhispercpp_sources()

    def test_head_on_pin_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pywhispercpp-src"
            (src / ".git").mkdir(parents=True)
            head = subprocess.CompletedProcess(
                [], 0, backend_installer.PYWHISPERCPP_PINNED_COMMIT + "\n", ""
            )
            self.assertTrue(self._run(src, head))

    def test_stale_head_fails_instead_of_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pywhispercpp-src"
            (src / ".git").mkdir(parents=True)
            stale = subprocess.CompletedProcess([], 0, "294e1e15f1fa3991aaa8db5f5e9afb97ade5ba5f\n", "")
            self.assertFalse(self._run(src, stale))

    def test_unreadable_head_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pywhispercpp-src"
            (src / ".git").mkdir(parents=True)
            broken = subprocess.CompletedProcess([], 128, "", "not a git repository")
            self.assertFalse(self._run(src, broken))


if __name__ == "__main__":
    unittest.main()
