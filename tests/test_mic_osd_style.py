import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.style import configured_daemon_style


class MicOSDStyleTests(unittest.TestCase):
    def test_non_daemon_keeps_waveform_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_daemon_style(), "waveform")

    def test_daemon_reads_pill_from_sparse_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_file.write_text(
                json.dumps({"mic_osd_style": "pill"}),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"HYPRWHSPR_MIC_OSD_DAEMON": "1"},
                clear=True,
            ):
                self.assertEqual(
                    configured_daemon_style(config_file),
                    "pill",
                )

    def test_invalid_or_missing_config_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            with mock.patch.dict(
                os.environ,
                {"HYPRWHSPR_MIC_OSD_DAEMON": "1"},
                clear=True,
            ):
                self.assertEqual(
                    configured_daemon_style(config_file),
                    "waveform",
                )

                config_file.write_text(
                    json.dumps({"mic_osd_style": "unknown"}),
                    encoding="utf-8",
                )
                self.assertEqual(
                    configured_daemon_style(config_file),
                    "waveform",
                )


if __name__ == "__main__":
    unittest.main()
