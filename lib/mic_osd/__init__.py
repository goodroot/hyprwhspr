"""
mic-osd: A minimal audio visualization OSD for Wayland/Hyprland

Displays real-time microphone input visualization as an overlay,
integrated into hyprwhspr for recording feedback.
"""

__version__ = "0.1.0"

from .runner import MicOSDRunner
from .notification_presenter import NotificationPresenter

__all__ = ["MicOSDRunner", "NotificationPresenter"]
