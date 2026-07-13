import json
import os
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
        for py in ("3.10", "3.11", "3.12", "3.13", "3.14"):
            tag = "cp" + py.replace(".", "")
            base = f"pywhispercpp-{ver}-{tag}-{tag}-linux_x86_64"
            self.assertEqual(
                backend_installer._get_wheel_filename(py, "cuda12", True),
                f"{base}+cuda12.whl",
            )
            self.assertEqual(
                backend_installer._get_wheel_filename(py, "cuda12", False),
                f"{base}.whl",
            )


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


if __name__ == "__main__":
    unittest.main()
