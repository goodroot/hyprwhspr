"""Centralized path constants for hyprwhspr with XDG Base Directory support"""
from pathlib import Path
import os

# XDG Base Directory specification
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
HOME = Path.home()
XDG_CONFIG_HOME = Path(os.environ.get('XDG_CONFIG_HOME', HOME / '.config'))
XDG_DATA_HOME = Path(os.environ.get('XDG_DATA_HOME', HOME / '.local' / 'share'))

# hyprwhspr directories
CONFIG_DIR = XDG_CONFIG_HOME / 'hyprwhspr'
DATA_DIR = XDG_DATA_HOME / 'hyprwhspr'

# Configuration files
CONFIG_FILE = CONFIG_DIR / 'config.json'

# Runtime state signal files (used for IPC between service and waybar)
RECORDING_STATUS_FILE = CONFIG_DIR / 'recording_status'
RECORDING_CONTROL_FILE = CONFIG_DIR / 'recording_control'
AUDIO_LEVEL_FILE = CONFIG_DIR / 'audio_level'
RECOVERY_REQUESTED_FILE = CONFIG_DIR / 'recovery_requested'
RECOVERY_RESULT_FILE = CONFIG_DIR / 'recovery_result'
MIC_ZERO_VOLUME_FILE = CONFIG_DIR / '.mic_zero_volume'
MIC_OSD_PID_FILE = CONFIG_DIR / 'mic_osd.pid'
SUSPEND_MARKER_FILE = CONFIG_DIR / '.suspend_marker'
LOCK_FILE = CONFIG_DIR / 'hyprwhspr.lock'
VISUALIZER_STATE_FILE = CONFIG_DIR / 'visualizer_state'  # recording|paused|processing|error|success

# Secure credential storage
CREDENTIALS_DIR = DATA_DIR
CREDENTIALS_FILE = CREDENTIALS_DIR / 'credentials'

# Temporary files and models
TEMP_DIR = DATA_DIR / 'temp'

# Long-form recording mode
LONGFORM_STATE_FILE = CONFIG_DIR / 'longform_state'  # IDLE|RECORDING|PAUSED|PROCESSING|ERROR
LONGFORM_SEGMENTS_DIR = TEMP_DIR / 'longform_segments'
