"""
Waveform visualization - shows microphone input as animated vertical bars.
"""

import math
import time
import cairo
import numpy as np
from .base import BaseVisualization, StateManager, VisualizerState
from ..theme import theme


class WaveformVisualization(BaseVisualization):
    """
    A bar-style audio visualization with recording indicator.
    
    Displays vertical bars that rise and fall with the audio input,
    creating a dynamic visual representation of sound.
    
    Colors are loaded from the Omarchy theme.
    """
    
    def __init__(self):
        super().__init__()

        # Bar settings
        self.num_bars = 32
        self.bar_width = 4
        self.bar_gap = 2
        self.min_bar_height = 2

        # Amplification for more visible response
        self.amplification = 4.0

        # Smoothing for bar heights (makes animation smoother)
        self.bar_heights = np.zeros(self.num_bars)
        self.decay_rate = 0.85  # How fast bars fall
        self.rise_rate = 0.5    # How fast bars rise

        # Animation for pulsing dot (legacy, now managed by StateManager)
        self.pulse_phase = 0.0

        # State manager for visualizer states (recording, paused, processing, etc.)
        self.state_manager = StateManager()

        # Elapsed time tracking for long-form mode
        self._recording_start_time = None
        self._elapsed_seconds = 0.0
        self._show_elapsed_time = False
    
    def update(self, level: float, samples: np.ndarray = None):
        """Update with new audio samples."""
        super().update(level, samples)
        
        if samples is not None and len(samples) > 0:
            # Calculate bar heights from audio samples
            # Divide samples into chunks for each bar
            chunk_size = len(samples) // self.num_bars
            if chunk_size > 0:
                new_heights = np.zeros(self.num_bars)
                for i in range(self.num_bars):
                    start = i * chunk_size
                    end = start + chunk_size
                    chunk = samples[start:end]
                    # Use RMS of chunk for smoother visualization
                    rms = np.sqrt(np.mean(chunk ** 2))
                    new_heights[i] = min(1.0, rms * self.amplification)
                
                # Smooth transitions - rise fast, fall slow
                for i in range(self.num_bars):
                    if new_heights[i] > self.bar_heights[i]:
                        # Rising - quick response
                        self.bar_heights[i] = (
                            self.rise_rate * new_heights[i] + 
                            (1 - self.rise_rate) * self.bar_heights[i]
                        )
                    else:
                        # Falling - slow decay
                        self.bar_heights[i] *= self.decay_rate
                        if self.bar_heights[i] < new_heights[i]:
                            self.bar_heights[i] = new_heights[i]
        else:
            # No audio - decay all bars
            self.bar_heights *= self.decay_rate

        # Update state manager animation
        self.state_manager.update()
    
    def draw(self, cr: cairo.Context, width: int, height: int):
        """Draw the bar visualization with recording indicator."""
        padding = 16
        
        # Recording indicator (just the dot) takes up left side
        indicator_width = 30
        bars_start_x = padding + indicator_width
        bars_width = width - bars_start_x - padding
        bars_height = height - (padding * 2)
        center_y = height / 2
        
        # Draw recording indicator (red dot + "Recording...")
        self._draw_recording_indicator(cr, padding, center_y)
        
        # Calculate bar dimensions to fill available space
        actual_num_bars = self.num_bars
        bar_gap = 2
        # Calculate bar width to fill the space: bars_width = num_bars * bar_width + (num_bars - 1) * gap
        bar_width = (bars_width - (actual_num_bars - 1) * bar_gap) / actual_num_bars
        total_bar_width = bar_width + bar_gap
        
        start_x = bars_start_x
        
        # Get colors from theme (fresh on each draw)
        bar_left = theme.bar_left
        bar_right = theme.bar_right
        
        # Draw bars
        for i in range(actual_num_bars):
            # Interpolate color from left to right
            t = i / max(1, actual_num_bars - 1)
            r = bar_left[0] * (1 - t) + bar_right[0] * t
            g = bar_left[1] * (1 - t) + bar_right[1] * t
            b = bar_left[2] * (1 - t) + bar_right[2] * t
            
            # Get bar height
            bar_h = max(self.min_bar_height, self.bar_heights[i] * bars_height)
            
            x = start_x + i * total_bar_width
            
            # Draw bar centered vertically
            bar_top = center_y - bar_h / 2
            
            # Draw glow effect
            cr.set_source_rgba(r, g, b, 0.3)
            cr.rectangle(x - 1, bar_top - 1, bar_width + 2, bar_h + 2)
            cr.fill()
            
            # Draw main bar
            cr.set_source_rgba(r, g, b, 0.9)
            cr.rectangle(x, bar_top, bar_width, bar_h)
            cr.fill()

        # Draw elapsed time (for long-form mode)
        self._draw_elapsed_time(cr, width, height)

    def set_state(self, state_str: str):
        """Set the visualizer state from a string value."""
        self.state_manager.set_state_from_string(state_str)

        # Start/stop elapsed time tracking based on state
        if state_str == 'recording':
            if self._recording_start_time is None:
                self._recording_start_time = time.time()
            self._show_elapsed_time = True
        elif state_str == 'paused':
            # Keep showing elapsed time but don't increment
            if self._recording_start_time is not None:
                self._elapsed_seconds += time.time() - self._recording_start_time
                self._recording_start_time = None
            self._show_elapsed_time = True
        else:
            # Reset elapsed time for other states
            self._recording_start_time = None
            self._elapsed_seconds = 0.0
            self._show_elapsed_time = False

    def set_elapsed_time(self, seconds: float):
        """Set the elapsed time directly (for long-form mode)."""
        self._elapsed_seconds = seconds
        self._show_elapsed_time = True

    def _get_elapsed_seconds(self) -> float:
        """Get current elapsed time in seconds."""
        if self._recording_start_time is not None:
            return self._elapsed_seconds + (time.time() - self._recording_start_time)
        return self._elapsed_seconds

    def _format_elapsed_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"

    def _draw_elapsed_time(self, cr: cairo.Context, width: int, height: int):
        """Draw elapsed time in the bottom-right corner."""
        if not self._show_elapsed_time:
            return

        elapsed = self._get_elapsed_seconds()
        text = self._format_elapsed_time(elapsed)

        # Set font (monospace for consistent width)
        cr.select_font_face(
            "monospace",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        cr.set_font_size(11)

        # Measure text
        extents = cr.text_extents(text)
        text_width = extents.width
        text_height = extents.height

        # Position: bottom-right with padding
        padding = 10
        x = width - text_width - padding
        y = height - padding

        # Draw background (semi-transparent)
        bg_padding = 3
        cr.set_source_rgba(0, 0, 0, 0.4)
        cr.rectangle(
            x - bg_padding,
            y - text_height - bg_padding,
            text_width + bg_padding * 2,
            text_height + bg_padding * 2
        )
        cr.fill()

        # Draw text
        text_color = theme.text
        cr.set_source_rgba(text_color[0], text_color[1], text_color[2], 0.9)
        cr.move_to(x, y)
        cr.show_text(text)

    def _draw_recording_indicator(self, cr: cairo.Context, x: float, center_y: float):
        """Draw the state indicator dot with state-appropriate color and animation."""
        dot_radius = 6
        dot_x = x + dot_radius + 4
        dot_y = center_y

        # Get animation value and color from state manager
        pulse = self.state_manager.get_animation_value()
        dot_color = self.state_manager.get_state_color()

        # Skip drawing if animation has faded out completely (e.g., success state after 2s)
        if pulse <= 0:
            return

        # Draw glow behind dot
        cr.set_source_rgba(
            dot_color[0],
            dot_color[1],
            dot_color[2],
            0.3 * pulse
        )
        cr.arc(dot_x, dot_y, dot_radius + 3, 0, 2 * math.pi)
        cr.fill()

        # Draw main dot
        cr.set_source_rgba(
            dot_color[0],
            dot_color[1],
            dot_color[2],
            pulse
        )
        cr.arc(dot_x, dot_y, dot_radius, 0, 2 * math.pi)
        cr.fill()
