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

    def test_inherited_english_prompt_is_migrated_off_whisper_prompt(self):
        """Configs predating sparse saving carry the old default verbatim (#233)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"whisper_prompt": config_manager.ENGLISH_PROMPT, "language": None}),
                encoding="utf-8",
            )

            manager = self._manager(root)

            self.assertEqual(manager.get_setting("whisper_prompt"), "")
            self.assertEqual(manager.get_setting("whisper_prompt_en"), config_manager.ENGLISH_PROMPT)
            saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertNotIn("whisper_prompt", saved)

    def test_custom_hallucination_markers_are_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"hallucination_markers": ["silence"]}), encoding="utf-8"
            )

            self.assertEqual(self._manager(root).get_hallucination_markers(), ["silence"])

    def test_malformed_hallucination_markers_fall_back_instead_of_crashing(self):
        """Resolved during service startup, so a bad value must not kill the daemon."""
        for bad in (None, "silence", 7):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(
                    json.dumps({"hallucination_markers": bad}), encoding="utf-8"
                )

                markers = self._manager(root).get_hallucination_markers()

                self.assertEqual(markers, list(config_manager.DEFAULT_HALLUCINATION_MARKERS), bad)

    def test_a_customised_prompt_survives_the_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"whisper_prompt": "Talk like a pirate."}), encoding="utf-8"
            )

            manager = self._manager(root)

            self.assertEqual(manager.get_setting("whisper_prompt"), "Talk like a pirate.")


if __name__ == "__main__":
    unittest.main()
