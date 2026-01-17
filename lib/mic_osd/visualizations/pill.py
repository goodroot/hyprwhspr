"""
Pill visualization - minimal, rounded, translucent audio indicator with waveform bars.
"""

import math
import time

import cairo
import numpy as np

from ..theme import theme
from .base import BaseVisualization, StateManager, VisualizerState


class PillVisualization(BaseVisualization):
    """
    A minimal pill-shaped audio visualization with waveform bars.
    """

    def __init__(self):
        super().__init__()
        self.width = 200
        self.height = 40

        self.num_bars = 24
        self.bar_heights = np.zeros(self.num_bars)
        self.decay_rate = 0.85
        self.rise_rate = 0.5
        self.amplification = 4.0

        self.state_manager = StateManager()

        self._recording_start_time = None
        self._elapsed_seconds = 0.0
        self._show_elapsed_time = False

    def update(self, level: float, samples: np.ndarray = None):
        super().update(level, samples)

        if samples is not None and len(samples) > 0:
            chunk_size = len(samples) // self.num_bars
            if chunk_size > 0:
                new_heights = np.zeros(self.num_bars)
                for i in range(self.num_bars):
                    start = i * chunk_size
                    end = start + chunk_size
                    chunk = samples[start:end]
                    rms = np.sqrt(np.mean(chunk**2))
                    new_heights[i] = min(1.0, rms * self.amplification)

                for i in range(self.num_bars):
                    if new_heights[i] > self.bar_heights[i]:
                        self.bar_heights[i] = (
                            self.rise_rate * new_heights[i]
                            + (1 - self.rise_rate) * self.bar_heights[i]
                        )
                    else:
                        self.bar_heights[i] *= self.decay_rate
                        if self.bar_heights[i] < new_heights[i]:
                            self.bar_heights[i] = new_heights[i]
        else:
            self.bar_heights *= self.decay_rate

        self.state_manager.update()

    def draw(self, cr: cairo.Context, width: int, height: int):
        padding = 8
        radius = height / 2

        dot_radius = 5
        dot_x = padding + dot_radius + 2
        dot_y = height / 2

        bars_start = dot_x + dot_radius + 8
        bars_end = width - padding - 6
        bars_width = bars_end - bars_start
        bars_height = height - (padding * 2) - 4
        center_y = height / 2

        is_processing = self.state_manager.current_state == VisualizerState.PROCESSING
        wave_phase = self.state_manager.animation_phase if is_processing else 0.0

        pulse = self.state_manager.get_animation_value()
        dot_color = self.state_manager.get_state_color()

        if pulse > 0:
            cr.set_source_rgba(dot_color[0], dot_color[1], dot_color[2], 0.15 * pulse)
            cr.arc(dot_x, dot_y, dot_radius + 2, 0, 2 * math.pi)
            cr.fill()

            cr.set_source_rgba(dot_color[0], dot_color[1], dot_color[2], 0.85 * pulse)
            cr.arc(dot_x, dot_y, dot_radius, 0, 2 * math.pi)
            cr.fill()

        bar_gap = 2
        bar_width = (bars_width - (self.num_bars - 1) * bar_gap) / self.num_bars
        total_bar_width = bar_width + bar_gap

        for i in range(self.num_bars):
            normalized_height = self.bar_heights[i]

            if is_processing:
                wave_pos = (i / max(1, self.num_bars - 1)) * 2 * math.pi + wave_phase
                wave_modulation = 0.3 + 0.7 * math.sin(wave_pos)
                base_height = 0.5
                normalized_height = (
                    max(normalized_height, base_height) * wave_modulation
                )
                normalized_height = max(0.0, min(1.0, normalized_height))

            bar_h = max(2, normalized_height * bars_height)
            x = bars_start + i * total_bar_width
            bar_top = center_y - bar_h / 2

            cr.set_source_rgba(0.85, 0.85, 0.9, 0.85)
            self._draw_rounded_rect(cr, x, bar_top, bar_width, bar_h, 1)
            cr.fill()

    def _draw_rounded_rect(
        self,
        cr: cairo.Context,
        x: float,
        y: float,
        width: float,
        height: float,
        radius: float,
    ):
        if width < 2 * radius:
            radius = width / 2
        if height < 2 * radius:
            radius = height / 2
        if radius < 0.5:
            cr.rectangle(x, y, width, height)
            return

        cr.new_path()
        cr.arc(x + radius, y + radius, radius, math.pi, 1.5 * math.pi)
        cr.arc(x + width - radius, y + radius, radius, 1.5 * math.pi, 2 * math.pi)
        cr.arc(x + width - radius, y + height - radius, radius, 0, 0.5 * math.pi)
        cr.arc(x + radius, y + height - radius, radius, 0.5 * math.pi, math.pi)
        cr.close_path()

    def draw_background(self, cr: cairo.Context, width: int, height: int):
        radius = height / 2

        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        cr.set_source_rgba(0.0, 0.0, 0.0, 0.5)
        self._draw_rounded_rect(cr, 0, 0, width, height, radius)
        cr.fill()

        cr.set_source_rgba(0.5, 0.5, 0.55, 0.25)
        cr.set_line_width(1)
        self._draw_rounded_rect(cr, 0.5, 0.5, width - 1, height - 1, radius)
        cr.stroke()

    def set_state(self, state_str: str):
        self.state_manager.set_state_from_string(state_str)

        if state_str == "recording":
            if self._recording_start_time is None:
                self._recording_start_time = time.time()
            self._show_elapsed_time = True
        elif state_str == "paused":
            if self._recording_start_time is not None:
                self._elapsed_seconds += time.time() - self._recording_start_time
                self._recording_start_time = None
            self._show_elapsed_time = True
        else:
            self._recording_start_time = None
            self._elapsed_seconds = 0.0
            self._show_elapsed_time = False

    def set_elapsed_time(self, seconds: float):
        self._elapsed_seconds = seconds
        self._show_elapsed_time = True
