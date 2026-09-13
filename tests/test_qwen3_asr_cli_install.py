"""Regression tests for Qwen3-ASR automated-install model selection."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from src.cli import install  # noqa: E402


class QwenModelPreparationTests(unittest.TestCase):
    def test_invalid_explicit_model_is_rejected_before_install(self):
        with mock.patch.object(install, "ConfigManager") as config, \
                mock.patch.object(install, "log_error") as error:
            self.assertFalse(install._prepare_qwen_model("qwen3-asr", "not-a-model"))
        config.assert_not_called()
        self.assertIn("not-a-model", error.call_args.args[0])

    def test_valid_explicit_model_is_persisted_for_payload_installer(self):
        config = mock.Mock()
        config.save_config.return_value = True
        with mock.patch.object(install, "ConfigManager", return_value=config):
            self.assertTrue(install._prepare_qwen_model("qwen3-asr", "0.6b-q8_0"))
        config.set_setting.assert_called_once_with("qwen3_asr_model", "0.6b-q8_0")
        config.save_config.assert_called_once_with()

    def test_save_failure_aborts_before_payload_install(self):
        config = mock.Mock()
        config.save_config.return_value = False
        with mock.patch.object(install, "ConfigManager", return_value=config), \
                mock.patch.object(install, "log_error"):
            self.assertFalse(install._prepare_qwen_model("qwen3-asr", "1.7b-q8_0"))

    def test_other_backends_do_not_touch_config(self):
        with mock.patch.object(install, "ConfigManager") as config:
            self.assertTrue(install._prepare_qwen_model("cpu", "base"))
        config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
