"""
Visualization modules for mic-osd.
"""

from ..style import configured_daemon_style
from .base import BaseVisualization
from .pill import PillVisualization
from .vu_meter import VUMeterVisualization
from .waveform import WaveformVisualization

_STYLE_CLASSES = {
    "pill": PillVisualization,
    "vu_meter": VUMeterVisualization,
    "waveform": WaveformVisualization,
}

# The daemon starts mic-osd with the existing "waveform" default. Resolve that
# default from config here so the integration stays additive and the standalone
# CLI keeps its existing behavior.
VISUALIZATIONS = {
    "pill": PillVisualization,
    "vu_meter": VUMeterVisualization,
    "waveform": _STYLE_CLASSES[configured_daemon_style()],
}

__all__ = [
    "BaseVisualization",
    "PillVisualization",
    "VUMeterVisualization",
    "WaveformVisualization",
    "VISUALIZATIONS",
]
