import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "lib" / "src"))

import backend_installer  # noqa: E402
import config_manager  # noqa: E402


class WheelVariantTests(unittest.TestCase):
    def test_no_cuda_returns_none(self):
        self.assertIsNone(backend_installer._get_wheel_variant(None))
        self.assertIsNone(backend_installer._get_wheel_variant(""))

    def test_cuda12_minors_all_map_to_one_wheel(self):
        for v in ("12.0", "12.4", "12.6", "12.9"):
            self.assertEqual(backend_installer._get_wheel_variant(v), "cuda12")

    def test_cuda11_and_13_fall_back_to_source(self):
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


if __name__ == "__main__":
    unittest.main()
