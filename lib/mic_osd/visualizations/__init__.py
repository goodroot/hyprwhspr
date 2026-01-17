"""
Visualization modules for mic-osd.
"""

from .base import BaseVisualization
from .vu_meter import VUMeterVisualization
from .waveform import WaveformVisualization
from .pill import PillVisualization

VISUALIZATIONS = {
    "vu_meter": VUMeterVisualization,
    "waveform": WaveformVisualization,
    "pill": PillVisualization,
}

__all__ = [
    "BaseVisualization",
    "VUMeterVisualization",
    "WaveformVisualization",
    "PillVisualization",
    "VISUALIZATIONS",
]
