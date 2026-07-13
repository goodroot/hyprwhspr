"""
Visualization modules for mic-osd.
"""

from .base import BaseVisualization
from .pill import PillVisualization
from .vu_meter import VUMeterVisualization
from .waveform import WaveformVisualization

VISUALIZATIONS = {
    "pill": PillVisualization,
    "vu_meter": VUMeterVisualization,
    "waveform": WaveformVisualization,
}

__all__ = [
    "BaseVisualization",
    "PillVisualization",
    "VUMeterVisualization",
    "WaveformVisualization",
    "VISUALIZATIONS",
]
