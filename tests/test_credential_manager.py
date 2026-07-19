import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "lib" / "src"


class CredentialManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SRC))
        import credential_manager
        cls.module = importlib.reload(credential_manager)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(SRC))

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.credentials_file = self.directory / "credentials"
        self.path_patches = (
            mock.patch.object(self.module, "CREDENTIALS_DIR", self.directory),
            mock.patch.object(self.module, "CREDENTIALS_FILE", self.credentials_file),
        )
        for patcher in self.path_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.path_patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def _write_old_credentials(self):
        self.credentials_file.write_text('{"old": "secret"}', encoding="utf-8")

    def _assert_old_file_and_no_temps(self):
        self.assertEqual(json.loads(self.credentials_file.read_text()), {"old": "secret"})
        self.assertEqual(list(self.directory.glob(".credentials.*.tmp")), [])

    def test_save_atomically_replaces_file_with_mode_0600(self):
        self._write_old_credentials()

        self.module._save_credentials({"new": "value"})

        self.assertEqual(json.loads(self.credentials_file.read_text()), {"new": "value"})
        self.assertEqual(self.credentials_file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.directory.glob(".credentials.*.tmp")), [])

    def test_failures_preserve_old_file_and_remove_temporary_file(self):
        failure_points = (
            ("json.dump", mock.patch.object(self.module.json, "dump", side_effect=TypeError("serialize"))),
            ("fsync", mock.patch.object(self.module.os, "fsync", side_effect=OSError("sync"))),
            ("chmod", mock.patch.object(self.module.os, "chmod", side_effect=OSError("chmod"))),
            ("replace", mock.patch.object(self.module.os, "replace", side_effect=OSError("replace"))),
        )
        for label, failure in failure_points:
            with self.subTest(label=label):
                self._write_old_credentials()
                with failure, self.assertRaises(Exception):
                    self.module._save_credentials({"new": "value"})
                self._assert_old_file_and_no_temps()

    def test_malformed_json_shapes_are_rejected_safely(self):
        malformed_values = (["secret"], {"provider": 123}, None)
        for value in malformed_values:
            with self.subTest(value=value):
                self.credentials_file.write_text(json.dumps(value), encoding="utf-8")
                self.assertEqual(self.module._load_credentials(), {})
                self.assertIsNone(self.module.get_credential("provider"))
                self.assertEqual(self.module.list_credentials(), {})


if __name__ == "__main__":
    unittest.main()
