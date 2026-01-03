"""
Waveform visualization - shows microphone input as animated vertical bars.
"""

import math
import cairo
import numpy as np
from .base import BaseVisualization
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
        
        # Animation for pulsing dot
        self.pulse_phase = 0.0
    
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
        
        # Update pulse animation
        self.pulse_phase += 0.15
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
    
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
        
        # Get colors from theme
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
    
    def _draw_recording_indicator(self, cr: cairo.Context, x: float, center_y: float):
        """Draw the red recording dot."""
        dot_radius = 6
        dot_x = x + dot_radius + 4
        dot_y = center_y
        
        # Pulsing effect - varies between 0.6 and 1.0 opacity
        pulse = 0.7 + 0.3 * math.sin(self.pulse_phase)
        
        # Get color from theme
        dot_color = theme.recording_dot
        
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
