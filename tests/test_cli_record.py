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
    def run_action(self, action, status, service_starts=True):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "recording_control"
            control.write_text("")
            status_file = Path(temp) / "recording_status"
            if status is not None:
                status_file.write_text(status)

            def service_tick(_seconds):
                # Stand-in for the daemon: an accepted start flips the status file.
                if service_starts and control.read_text().startswith("start"):
                    status_file.write_text("true")

            with (
                mock.patch.object(record, "RECORDING_CONTROL_FILE", control),
                mock.patch.object(record, "RECORDING_STATUS_FILE", status_file),
                mock.patch.object(record, "LONGFORM_STATE_FILE", Path(temp) / "longform_state"),
                mock.patch.object(record, "START_CONFIRM_TIMEOUT", 0.2),
                mock.patch.object(record.time, "sleep", side_effect=service_tick),
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

    def test_start_is_reported_only_once_the_service_confirms(self):
        for action in ("start", "toggle"):
            with self.subTest(action=action):
                sent, success, info = self.run_action(action, "false")
                self.assertEqual(sent, "start\n")
                success.assert_called_once_with("Recording started")
                info.assert_not_called()

                sent, success, info = self.run_action(action, "false", service_starts=False)
                self.assertEqual(sent, "start\n")   # refused: model loading, text still pasting
                success.assert_not_called()
                info.assert_called_once_with(
                    "Start sent. The service did not confirm; check notifications.")

    def test_long_form_start_is_confirmed_by_its_own_state_file(self):
        # Long-form never writes the status file.
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "recording_control"
            control.write_text("")
            state = Path(temp) / "longform_state"
            state.write_text("PAUSED")
            with (
                mock.patch.object(record, "RECORDING_CONTROL_FILE", control),
                mock.patch.object(record, "RECORDING_STATUS_FILE", Path(temp) / "recording_status"),
                mock.patch.object(record, "LONGFORM_STATE_FILE", state),
                mock.patch.object(record, "START_CONFIRM_TIMEOUT", 0.2),
                mock.patch.object(record.time, "sleep", side_effect=lambda _s: state.write_text("RECORDING")),
                mock.patch.object(record, "log_success") as success,
                mock.patch.object(record, "log_info") as info,
            ):
                record.record_command("start")
        success.assert_called_once_with("Recording started")
        info.assert_not_called()

    def test_long_form_recording_counts_for_start_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            control = Path(temp) / "recording_control"
            control.write_text("")
            state = Path(temp) / "longform_state"
            state.write_text("RECORDING")
            for action, expected in (("start", "Start sent"), ("status", "Status: Recording in progress")):
                with (
                    mock.patch.object(record, "RECORDING_CONTROL_FILE", control),
                    mock.patch.object(record, "RECORDING_STATUS_FILE", Path(temp) / "recording_status"),
                    mock.patch.object(record, "LONGFORM_STATE_FILE", state),
                    mock.patch.object(record, "log_success") as success,
                    mock.patch.object(record, "log_info") as info,
                ):
                    record.record_command(action)
                success.assert_not_called()      # nothing started: it was already recording
                info.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()
