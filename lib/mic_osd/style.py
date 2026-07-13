"""Mic OSD style selection."""

import json
import os
from pathlib import Path


SUPPORTED_STYLES = ("waveform", "vu_meter", "pill")


def _config_file() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "hyprwhspr" / "config.json"


def configured_daemon_style(config_file: Path | None = None) -> str:
    """Return the configured daemon style with a safe waveform fallback."""
    if os.environ.get("HYPRWHSPR_MIC_OSD_DAEMON") != "1":
        return "waveform"

    path = config_file or _config_file()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return "waveform"

    style = config.get("mic_osd_style", "waveform")
    return style if style in SUPPORTED_STYLES else "waveform"
