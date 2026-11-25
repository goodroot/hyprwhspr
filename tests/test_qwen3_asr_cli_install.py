"""Regression tests for Qwen3-ASR automated-install model selection."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from src.cli import install, models  # noqa: E402


class QwenModelPreparationTests(unittest.TestCase):
    def test_invalid_explicit_model_is_rejected_before_install(self):
        with mock.patch.object(install, "ConfigManager") as config, \
                mock.patch.object(install, "log_error") as error:
            self.assertFalse(install._prepare_qwen_model("qwen3-asr", "not-a-model"))
        config.assert_not_called()
        self.assertIn("not-a-model", error.call_args.args[0])

    def test_valid_explicit_model_is_persisted_for_payload_installer(self):
        for model_name in ("0.6b-q8_0", "oruk-fourier-2b"):
            with self.subTest(model=model_name):
                config = mock.Mock()
                config.save_config.return_value = True
                with mock.patch.object(install, "ConfigManager", return_value=config):
                    self.assertTrue(install._prepare_qwen_model("qwen3-asr", model_name))
                config.set_setting.assert_called_once_with("qwen3_asr_model", model_name)
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


class QwenModelListTests(unittest.TestCase):
    def test_model_list_marks_default_and_optional_oruk_fourier_license(self):
        config = mock.Mock()
        config.get_setting.side_effect = lambda key, default=None: (
            "qwen3-asr" if key == "transcription_backend" else default
        )
        output = io.StringIO()
        with mock.patch.object(models, "ConfigManager", return_value=config), \
                redirect_stdout(output):
            self.assertTrue(models.model_command("list"))
        text = output.getvalue()
        self.assertIn("1.7b-q8_0 (recommended)", text)
        self.assertIn("0.6b-q8_0", text)
        self.assertIn("oruk-fourier-2b", text)
        self.assertIn("English only", text)
        self.assertIn("CC BY-NC 4.0", text)


if __name__ == "__main__":
    unittest.main()
