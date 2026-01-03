"""
Visualization modules for mic-osd.
"""

from .base import BaseVisualization
from .vu_meter import VUMeterVisualization
from .waveform import WaveformVisualization

VISUALIZATIONS = {
    "vu_meter": VUMeterVisualization,
    "waveform": WaveformVisualization,
}

__all__ = [
    "BaseVisualization",
    "VUMeterVisualization",
    "WaveformVisualization",
    "VISUALIZATIONS",
]
