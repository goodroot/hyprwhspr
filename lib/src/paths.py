"""Centralized path constants for hyprwhspr with XDG Base Directory support"""
from pathlib import Path
import os
import tempfile

# XDG Base Directory specification
# https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
HOME = Path.home()
XDG_CONFIG_HOME = Path(os.environ.get('XDG_CONFIG_HOME', HOME / '.config'))
XDG_DATA_HOME = Path(os.environ.get('XDG_DATA_HOME', HOME / '.local' / 'share'))
XDG_RUNTIME_DIR = os.environ.get('XDG_RUNTIME_DIR')

# hyprwhspr directories
# Runtime-dir resolution is mirrored in config/hyprland/hyprwhspr-tray.sh,
# lib/mic_osd/main.py's import fallback, and the contrib GNOME extension —
# change together
CONFIG_DIR = XDG_CONFIG_HOME / 'hyprwhspr'
DATA_DIR = XDG_DATA_HOME / 'hyprwhspr'
if XDG_RUNTIME_DIR:
    RUNTIME_DIR = Path(XDG_RUNTIME_DIR) / 'hyprwhspr'
else:
    RUNTIME_DIR = Path(tempfile.gettempdir()) / f"hyprwhspr-{os.getuid()}"

# Configuration files
CONFIG_FILE = CONFIG_DIR / 'config.json'

# Transient IPC/signal files (tmpfs; wiped at logout/reboot)
RECORDING_STATUS_FILE = RUNTIME_DIR / 'recording_status'
RECORDING_CONTROL_FILE = RUNTIME_DIR / 'recording_control'
SOCKET_FILE = RUNTIME_DIR / 'hyprwhspr.sock'
AUDIO_LEVEL_FILE = RUNTIME_DIR / 'audio_level'
RECOVERY_REQUESTED_FILE = RUNTIME_DIR / 'recovery_requested'
RECOVERY_RESULT_FILE = RUNTIME_DIR / 'recovery_result'
MIC_ZERO_VOLUME_FILE = RUNTIME_DIR / '.mic_zero_volume'
MIC_OSD_PID_FILE = RUNTIME_DIR / 'mic_osd.pid'
LOCK_FILE = RUNTIME_DIR / 'hyprwhspr.lock'
VISUALIZER_STATE_FILE = RUNTIME_DIR / 'visualizer_state'  # recording|paused|processing|error|success
TRANSCRIPT_PREVIEW_FILE = RUNTIME_DIR / 'transcript_preview'
MIC_OSD_LEVEL_FEED_FILE = RUNTIME_DIR / 'mic_osd_level_feed'

# Secure credential storage
CREDENTIALS_DIR = DATA_DIR
CREDENTIALS_FILE = CREDENTIALS_DIR / 'credentials'

# Temporary files and models
TEMP_DIR = DATA_DIR / 'temp'

# Long-form recording mode
LONGFORM_STATE_FILE = RUNTIME_DIR / 'longform_state'  # IDLE|RECORDING|PAUSED|PROCESSING|ERROR
LONGFORM_SEGMENTS_DIR = TEMP_DIR / 'longform_segments'

# Model lifecycle signal file (presence = model is manually unloaded from memory)
MODEL_UNLOADED_FILE = RUNTIME_DIR / 'model_unloaded'
