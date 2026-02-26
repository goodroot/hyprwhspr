"""Tests for stale runtime state cleanup on startup and shutdown.

Covers the bug where recording_status is left as 'true' after a service
crash/SIGKILL/reboot, causing 'hyprwhspr record toggle' to always send
'stop' instead of 'start' — so recording never begins.

See: https://github.com/goodroot/hyprwhspr/issues/127
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# The module under test uses relative imports from lib/src/,
# so we need both on sys.path.
LIB_DIR = Path(__file__).parent.parent / 'lib'
SRC_DIR = LIB_DIR / 'src'
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def state_dir(tmp_path):
    """Provide a temporary directory that acts as CONFIG_DIR."""
    return tmp_path


@pytest.fixture
def patch_state_files(state_dir):
    """Patch all state file paths to point at our temp directory."""
    patched = {
        'RECORDING_STATUS_FILE': state_dir / 'recording_status',
        'AUDIO_LEVEL_FILE': state_dir / 'audio_level',
        'MIC_ZERO_VOLUME_FILE': state_dir / '.mic_zero_volume',
        'RECOVERY_REQUESTED_FILE': state_dir / 'recovery_requested',
        'RECOVERY_RESULT_FILE': state_dir / 'recovery_result',
        'LONGFORM_STATE_FILE': state_dir / 'longform_state',
        'RECORDING_CONTROL_FILE': state_dir / 'recording_control',
        'LOCK_FILE': state_dir / 'hyprwhspr.lock',
        'LONGFORM_SEGMENTS_DIR': state_dir / 'longform_segments',
    }
    with patch.multiple('paths', **patched):
        # Re-import so the module picks up the patched paths
        import importlib
        import main as main_mod
        importlib.reload(main_mod)
        yield patched, main_mod


def _make_app(main_mod):
    """Create a hyprwhsprApp instance with heavy dependencies mocked out."""
    with patch.object(main_mod, 'ConfigManager') as mock_cfg, \
         patch.object(main_mod, 'AudioCapture'), \
         patch.object(main_mod, 'WhisperManager'), \
         patch.object(main_mod, 'TextInjector'), \
         patch.object(main_mod, 'AudioManager'), \
         patch.object(main_mod, 'AudioDucker'), \
         patch.object(main_mod, 'DeviceMonitor'), \
         patch.object(main_mod, 'GlobalShortcuts'):
        # ConfigManager needs to return sensible defaults
        cfg_instance = MagicMock()
        cfg_instance.get_setting.return_value = 'toggle'
        mock_cfg.return_value = cfg_instance

        app = main_mod.hyprwhsprApp()
        return app


class TestResetStaleState:
    """_reset_stale_state() should remove leftover runtime files."""

    def test_stale_recording_status_cleared(self, patch_state_files):
        paths, main_mod = patch_state_files
        status_file = paths['RECORDING_STATUS_FILE']
        status_file.write_text('true')

        app = _make_app(main_mod)

        assert not status_file.exists(), \
            "recording_status should be removed on startup"

    def test_stale_audio_level_cleared(self, patch_state_files):
        paths, main_mod = patch_state_files
        level_file = paths['AUDIO_LEVEL_FILE']
        level_file.write_text('0.42')

        app = _make_app(main_mod)

        assert not level_file.exists(), \
            "audio_level should be removed on startup"

    def test_stale_mic_zero_volume_cleared(self, patch_state_files):
        paths, main_mod = patch_state_files
        mic_file = paths['MIC_ZERO_VOLUME_FILE']
        mic_file.write_text('1')

        app = _make_app(main_mod)

        assert not mic_file.exists(), \
            "mic_zero_volume should be removed on startup"

    def test_stale_recovery_files_cleared(self, patch_state_files):
        paths, main_mod = patch_state_files
        paths['RECOVERY_REQUESTED_FILE'].write_text('1')
        paths['RECOVERY_RESULT_FILE'].write_text('{"success": true}')

        app = _make_app(main_mod)

        assert not paths['RECOVERY_REQUESTED_FILE'].exists()
        assert not paths['RECOVERY_RESULT_FILE'].exists()

    def test_longform_state_reset_to_idle(self, patch_state_files):
        paths, main_mod = patch_state_files
        lf_file = paths['LONGFORM_STATE_FILE']
        lf_file.write_text('RECORDING')

        app = _make_app(main_mod)

        assert lf_file.read_text().strip() == 'IDLE', \
            "longform_state should be reset to IDLE, not deleted"

    def test_longform_state_left_alone_if_idle(self, patch_state_files):
        paths, main_mod = patch_state_files
        lf_file = paths['LONGFORM_STATE_FILE']
        lf_file.write_text('IDLE')

        app = _make_app(main_mod)

        assert lf_file.read_text().strip() == 'IDLE'

    def test_no_crash_when_files_missing(self, patch_state_files):
        """Startup should not crash if none of the state files exist."""
        paths, main_mod = patch_state_files
        # Don't create any files — all paths point to nonexistent files
        app = _make_app(main_mod)
        # If we get here without an exception, the test passes


class TestToggleAfterRestart:
    """End-to-end: the CLI toggle logic should work after state is reset."""

    def test_toggle_sends_start_after_stale_status_cleared(self, patch_state_files):
        """Simulates the exact bug: stale recording_status = true made toggle
        always send 'stop'. After _reset_stale_state(), toggle should send 'start'."""
        paths, main_mod = patch_state_files
        status_file = paths['RECORDING_STATUS_FILE']

        # Simulate crash: status file left as 'true'
        status_file.write_text('true')

        # Service starts up — stale state cleared
        app = _make_app(main_mod)

        # Now check what the CLI toggle logic would do.
        # It reads RECORDING_STATUS_FILE to decide start vs stop.
        # After reset, the file should not exist → is_recording() returns False
        # → toggle sends 'start'.
        assert not status_file.exists(), \
            "After startup, recording_status must not exist"

        # Replicate the CLI's is_recording() check
        from cli_commands import record_command
        from paths import RECORDING_STATUS_FILE
        # RECORDING_STATUS_FILE is now patched, so this check uses our tmp dir
        assert not RECORDING_STATUS_FILE.exists()


class TestCleanup:
    """_cleanup() should also clear state files on graceful shutdown."""

    def test_cleanup_removes_recording_status(self, patch_state_files):
        paths, main_mod = patch_state_files
        app = _make_app(main_mod)

        # Simulate: recording was active when shutdown happens
        paths['RECORDING_STATUS_FILE'].write_text('true')
        paths['AUDIO_LEVEL_FILE'].write_text('0.5')

        app._cleanup()

        assert not paths['RECORDING_STATUS_FILE'].exists()
        assert not paths['AUDIO_LEVEL_FILE'].exists()
