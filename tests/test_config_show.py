import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from src.cli import config as config_cli


class ConfigShowTests(unittest.TestCase):
    def test_show_all_is_clean_complete_json_with_schema(self):
        manager = mock.Mock()
        manager.SCHEMA_URL = "https://example.test/config.schema.json"
        manager.get_all_settings.return_value = {"setting": 42}

        output = io.StringIO()
        with mock.patch.object(
            config_cli,
            "ConfigManager",
            return_value=manager,
        ) as manager_class, redirect_stdout(output):
            config_cli.show_config(show_all=True)

        manager_class.assert_called_once_with(verbose=False)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "$schema": "https://example.test/config.schema.json",
                "setting": 42,
            },
        )


if __name__ == "__main__":
    unittest.main()
