import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.runner import MicOSDRunner
import mic_osd.runner as runner_module


class MicOSDRunnerTests(unittest.TestCase):
    def test_preview_text_is_written_as_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            preview_file = Path(tmp) / "transcript_preview"
            original = runner_module.TRANSCRIPT_PREVIEW_FILE
            runner_module.TRANSCRIPT_PREVIEW_FILE = preview_file
            try:
                text = "cafe 東京"
                MicOSDRunner().set_preview_text(text)

                self.assertEqual(preview_file.read_bytes(), text.encode("utf-8"))
                self.assertEqual(preview_file.read_text(encoding="utf-8"), text)
            finally:
                runner_module.TRANSCRIPT_PREVIEW_FILE = original


if __name__ == "__main__":
    unittest.main()
