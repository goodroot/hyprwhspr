import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from src import config_manager


class ConfigSchemaSyncTests(unittest.TestCase):
    """Every setting must exist in both DEFAULT_CONFIG and config.schema.json.

    The config surface is bookkept in two places (plus docs); this pins the
    two machine-readable ones together so new settings can't ship half-registered.
    """

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_dir = Path(temp_dir) / "hyprwhspr"
            with mock.patch.object(config_manager, "CONFIG_DIR", cfg_dir), \
                    mock.patch.object(config_manager, "CONFIG_FILE", cfg_dir / "config.json"):
                manager = config_manager.ConfigManager(verbose=False)
        cls.defaults = set(manager.default_config)
        schema = json.loads(
            (ROOT / "share" / "config.schema.json").read_text(encoding="utf-8")
        )
        cls.schema_keys = set(schema["properties"]) - {"$schema"}

    def test_every_default_has_a_schema_entry(self):
        self.assertEqual(sorted(self.defaults - self.schema_keys), [])

    def test_every_schema_entry_has_a_default(self):
        self.assertEqual(sorted(self.schema_keys - self.defaults), [])


if __name__ == "__main__":
    unittest.main()
