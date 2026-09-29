"""CLI record commands: the lagging status file never gates what is sent."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

LIB_SRC = Path(__file__).resolve().parents[1] / "lib" / "src"
if str(LIB_SRC) not in sys.path:
    sys.path.insert(0, str(LIB_SRC))

from cli import record


class RecordCommandTests(unittest.TestCase):
    def run_action(self, action, status):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "recording_control"
            control.write_text("")
            status_file = Path(temp) / "recording_status"
            if status is not None:
                status_file.write_text(status)
            with (
                mock.patch.object(record, "RECORDING_CONTROL_FILE", control),
                mock.patch.object(record, "RECORDING_STATUS_FILE", status_file),
                mock.patch.object(record, "log_success") as success,
                mock.patch.object(record, "log_info") as info,
            ):
                record.record_command(action)
            return control.read_text(), success, info

    def test_stale_status_still_sends_command(self):
        for action, status in (("cancel", None), ("stop", "false"), ("start", "true")):
            with self.subTest(action=action):
                sent, success, info = self.run_action(action, status)
                self.assertEqual(sent, action + "\n")
                success.assert_not_called()
                info.assert_called_once_with(f"{action.capitalize()} sent")

    def test_release_sends_release_and_claims_no_outcome(self):
        # A release may latch or stop; the CLI can't know which.
        for status in ("true", "false"):
            with self.subTest(status=status):
                sent, success, info = self.run_action("release", status)
                self.assertEqual(sent, "release\n")
                success.assert_not_called()
                info.assert_called_once_with("Release sent")

    def test_matching_status_reports_result(self):
        for action, status in (("cancel", "true"), ("stop", "true"), ("start", "false")):
            with self.subTest(action=action):
                sent, success, info = self.run_action(action, status)
                self.assertEqual(sent, action + "\n")
                success.assert_called_once()
                info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
