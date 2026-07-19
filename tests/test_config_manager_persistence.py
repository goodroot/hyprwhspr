import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import config_manager


class ConfigManagerPersistenceTests(unittest.TestCase):
    def _manager(self, root: Path):
        with (
            mock.patch.object(config_manager, "CONFIG_DIR", root),
            mock.patch.object(config_manager, "CONFIG_FILE", root / "config.json"),
        ):
            return config_manager.ConfigManager(verbose=False)

    def test_save_atomically_replaces_config_with_sparse_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            manager.set_setting("language", "fr")

            with mock.patch("config_manager.os.replace", wraps=config_manager.os.replace) as replace:
                self.assertTrue(manager.save_config())

            saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, {
                "$schema": manager.SCHEMA_URL,
                "language": "fr",
            })
            source, destination = replace.call_args.args
            self.assertEqual(destination, root / "config.json")
            self.assertEqual(Path(source).parent, root)
            self.assertFalse(list(root.glob(".config.json.*.tmp")))

    def test_failed_serialization_preserves_previous_config_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = self._manager(root)
            config_file = root / "config.json"
            previous = config_file.read_bytes()
            manager.set_setting("not_json_serializable", object())

            self.assertFalse(manager.save_config())

            self.assertEqual(config_file.read_bytes(), previous)
            self.assertFalse(list(root.glob(".config.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
