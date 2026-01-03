"""
VU Meter visualization - shows microphone input level as a horizontal bar.
"""

import cairo
from .base import BaseVisualization
from ..theme import theme


class VUMeterVisualization(BaseVisualization):
    """
    A simple VU meter that displays audio level as a horizontal bar.
    
    The bar fills from left to right based on the audio input level,
    with smooth decay for a more natural appearance.
    
    Colors are loaded from the Omarchy theme.
    """
    
    def __init__(self):
        super().__init__()
        self.smoothed_level = 0.0
        self.peak_level = 0.0
        self.peak_decay = 0.95  # How fast peak indicator falls
        self.smooth_factor = 0.3  # Smoothing for main bar
    
    def update(self, level: float, samples=None):
        """Update with smoothing and peak hold."""
        super().update(level, samples)
        
        # Smooth the main level
        self.smoothed_level = (
            self.smooth_factor * level + 
            (1 - self.smooth_factor) * self.smoothed_level
        )
        
        # Update peak with decay
        if level > self.peak_level:
            self.peak_level = level
        else:
            self.peak_level *= self.peak_decay
    
    def draw(self, cr: cairo.Context, width: int, height: int):
        """Draw the VU meter."""
        padding = 8
        bar_height = height - (padding * 2)
        bar_width = width - (padding * 2)
        bar_x = padding
        bar_y = padding
        
        # Get colors from theme
        bar_left = theme.bar_left
        bar_right = theme.bar_right
        
        # Draw background track
        cr.set_source_rgba(0.15, 0.15, 0.15, 1.0)
        cr.rectangle(bar_x, bar_y, bar_width, bar_height)
        cr.fill()
        
        # Draw level bar with gradient from theme colors
        level_width = bar_width * self.smoothed_level
        if level_width > 0:
            # Create horizontal gradient using theme colors
            gradient = cairo.LinearGradient(bar_x, 0, bar_x + bar_width, 0)
            gradient.add_color_stop_rgb(0.0, *bar_left)
            gradient.add_color_stop_rgb(1.0, *bar_right)
            
            cr.set_source(gradient)
            cr.rectangle(bar_x, bar_y, level_width, bar_height)
            cr.fill()
        
        # Draw peak indicator
        if self.peak_level > 0.01:
            peak_x = bar_x + (bar_width * self.peak_level) - 2
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.8)
            cr.rectangle(peak_x, bar_y, 3, bar_height)
            cr.fill()
        
        # Draw border
        cr.set_source_rgba(0.3, 0.3, 0.3, 0.5)
        cr.set_line_width(1)
        cr.rectangle(bar_x, bar_y, bar_width, bar_height)
        cr.stroke()
