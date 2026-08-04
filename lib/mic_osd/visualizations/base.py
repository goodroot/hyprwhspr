"""
Base visualization class for mic-osd.
"""

from abc import ABC, abstractmethod
from enum import Enum
import cairo
import math
import numpy as np
import time

from ..theme import theme


class VisualizerState(Enum):
    """States for the visualizer indicator."""
    RECORDING = "recording"      # Pulsing red dot
    PAUSED = "paused"            # Static amber dot
    PROCESSING = "processing"    # Green wave animation
    ERROR = "error"              # Red flash/strobe
    SUCCESS = "success"          # Green pulse + fade


class StateManager:
    """Manages visualizer state transitions and animations."""

    def __init__(self):
        self.current_state = VisualizerState.RECORDING
        self.state_changed_at = time.time()
        self.animation_phase = 0.0

    def set_state(self, new_state: VisualizerState):
        """Set a new state, resetting animation timing."""
        if new_state != self.current_state:
            self.current_state = new_state
            self.state_changed_at = time.time()
            self.animation_phase = 0.0

    def set_state_from_string(self, state_str: str):
        """Set state from string value (for IPC)."""
        state_map = {
            'recording': VisualizerState.RECORDING,
            'paused': VisualizerState.PAUSED,
            'processing': VisualizerState.PROCESSING,
            'error': VisualizerState.ERROR,
            'success': VisualizerState.SUCCESS,
        }
        new_state = state_map.get(state_str.lower(), VisualizerState.RECORDING)
        self.set_state(new_state)

    def update(self, dt: float = 0.016):
        """Update animation phase (called every frame, ~60 FPS)."""
        self.animation_phase += 0.15  # Same rate as existing pulse
        if self.animation_phase > 2 * math.pi:
            self.animation_phase -= 2 * math.pi

    def get_state_color(self) -> tuple:
        """Return the appropriate color for current state from theme."""
        color_map = {
            VisualizerState.RECORDING: theme.recording_dot,
            VisualizerState.PAUSED: theme.paused_dot,
            VisualizerState.PROCESSING: theme.processing_dot,
            VisualizerState.ERROR: theme.error_dot,
            VisualizerState.SUCCESS: theme.success_dot,
        }
        return color_map.get(self.current_state, theme.recording_dot)

    def get_animation_value(self) -> float:
        """Return 0-1 animation parameter based on state type."""
        elapsed = time.time() - self.state_changed_at

        if self.current_state == VisualizerState.RECORDING:
            # Gentle pulse: varies 0.7-1.0
            return 0.7 + 0.3 * math.sin(self.animation_phase)

        elif self.current_state == VisualizerState.PAUSED:
            # Static, no animation
            return 1.0

        elif self.current_state == VisualizerState.PROCESSING:
            # Flowing wave: varies 0.6-1.0
            return 0.6 + 0.4 * math.sin(self.animation_phase)

        elif self.current_state == VisualizerState.ERROR:
            # Fast strobe: 4x speed, varies 0.3-1.0
            return 0.3 + 0.7 * abs(math.sin(self.animation_phase * 4))

        elif self.current_state == VisualizerState.SUCCESS:
            # Pulse then fade out over 1 second
            if elapsed > 1.0:
                return 0.0
            fade = 1.0 - (elapsed / 1.0)
            pulse = 0.7 + 0.3 * math.sin(self.animation_phase)
            return fade * pulse

        return 1.0


class AutoGain:
    """Adaptive input gain for the meters.

    Mic amplitude spans ~30 dB across hardware: a USB condenser with internal
    makeup gain peaks near full scale, while a distant XLR mic through an
    interface preamp can sit two orders of magnitude lower. A fixed multiplier
    tuned for the former renders the latter as a flat line: a distant XLR mic
    on a Focusrite Scarlett peaked at 0.031 per bucket, which the waveform's
    old fixed 4.0 turned into a 12% bar, under its 2px floor.

    Tracks a peak envelope with instant attack and slow release, then scales it
    toward target_peak. min_gain preserves the previous fixed-gain behaviour for
    hot mics, so only quiet inputs are boosted. Room tone is left alone: an
    envelope under noise_floor means nobody is speaking, so no amplification.
    """

    TARGET_PEAK = 0.7
    MAX_GAIN = 80.0
    RELEASE = 0.995  # Per-frame decay; ~3s time constant at 60 FPS
    GATE_FRACTION = 0.2  # Relaxed gate as a share of the envelope

    def __init__(self, min_gain: float, noise_floor: float):
        self.min_gain = min_gain
        self.noise_floor = noise_floor
        self.envelope = 0.0

    def update(self, peak: float) -> float:
        """Feed this frame's peak amplitude, get the gain to apply to it."""
        if peak > self.envelope:
            self.envelope = peak
        else:
            self.envelope *= self.RELEASE

        if self.envelope <= self.noise_floor:
            return self.min_gain
        gain = self.TARGET_PEAK / self.envelope
        return max(self.min_gain, min(self.MAX_GAIN, gain))

    def reset(self):
        """Forget the tracked envelope. The meter only runs while the OSD is
        visible, so the envelope would otherwise carry a previous recording's
        loudest moment into the next one and hold the gain down there."""
        self.envelope = 0.0

    def gate(self, default: float) -> float:
        """Scale a fixed noise gate down for quiet inputs, whose speech would
        otherwise sit entirely beneath it. Silence keeps the full gate, so
        room tone is still held down rather than boosted into the meter."""
        if self.envelope <= self.noise_floor:
            return default
        return min(default, self.envelope * self.GATE_FRACTION)


class BaseVisualization(ABC):
    """
    Abstract base class for audio visualizations.
    
    Subclasses must implement the draw() method to render
    the visualization using Cairo.
    """
    
    def __init__(self):
        self.width = 300
        self.height = 60
        self.audio_level = 0.0
        self.audio_samples = np.zeros(1024)
    
    @property
    def background_color(self):
        """Get background color from theme."""
        bg = theme.background
        if len(bg) == 3:
            return (*bg, 0.95)
        return bg
    
    def update(self, level: float, samples: np.ndarray = None):
        """
        Update visualization with new audio data.
        
        Args:
            level: Audio level (0.0 to 1.0)
            samples: Raw audio samples (optional, for waveform)
        """
        self.audio_level = max(0.0, min(1.0, level))
        if samples is not None:
            self.audio_samples = samples
    
    @abstractmethod
    def draw(self, cr: cairo.Context, width: int, height: int):
        """
        Draw the visualization.
        
        Args:
            cr: Cairo context to draw on
            width: Available width in pixels
            height: Available height in pixels
        """
        pass
    
    def draw_background(self, cr: cairo.Context, width: int, height: int):
        """Draw common background with border."""
        # Fill background
        cr.set_source_rgba(*self.background_color)
        cr.rectangle(0, 0, width, height)
        cr.fill()
        
        # Draw border
        border_color = theme.border
        if border_color:
            if len(border_color) == 3:
                cr.set_source_rgb(*border_color)
            else:
                cr.set_source_rgba(*border_color)
            cr.set_line_width(2)
            cr.rectangle(1, 1, width - 2, height - 2)
            cr.stroke()
