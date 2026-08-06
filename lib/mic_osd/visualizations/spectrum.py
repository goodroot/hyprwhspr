"""CAVA-inspired mirrored spectrum visualization for mic-osd."""

from __future__ import annotations

import math
import time

import cairo
import numpy as np

from .base import BaseVisualization, StateManager, VisualizerState
from ..osd_config import MicOSDConfig
from ..theme import theme


class SpectrumVisualization(BaseVisualization):
    """A compact, mirrored frequency spectrum in a translucent panel.

    The renderer intentionally owns only presentation and animation.  Frequency
    analysis remains in AudioCapture, beside the stream that already owns the
    microphone; this avoids opening a second PipeWire/ALSA capture path as CAVA
    would do.
    """

    show_preview = True
    dynamic_preview = True
    preferred_width = 520
    preferred_height = 86

    NUM_BARS = 96
    PANEL_MARGIN = 6.0
    PANEL_RADIUS = 18.0
    INNER_PADDING_X = 14.0
    INNER_PADDING_Y = 12.0
    BAR_GAP = 1.2
    MIN_BAR_HEIGHT = 1.5

    def __init__(self, osd_config=None):
        super().__init__()
        self.osd_config = osd_config or MicOSDConfig.load()
        self.preferred_width = self.osd_config.spectrum_width
        self.preferred_height = self.osd_config.spectrum_meter_height
        self.num_bars = self.NUM_BARS
        self.bar_heights = np.zeros(self.num_bars, dtype=np.float64)
        self.state_manager = StateManager()
        self._last_update = time.monotonic()

    def apply_config(self, osd_config: MicOSDConfig):
        """Apply presentation/response settings without resetting animation."""
        self.osd_config = osd_config
        self.preferred_width = osd_config.spectrum_width
        self.preferred_height = osd_config.spectrum_meter_height

    @staticmethod
    def _rounded_rect(
        cr: cairo.Context,
        x: float,
        y: float,
        width: float,
        height: float,
        radius: float,
    ):
        radius = min(radius, width / 2.0, height / 2.0)
        cr.new_sub_path()
        cr.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
        cr.arc(
            x + width - radius,
            y + height - radius,
            radius,
            0,
            math.pi / 2,
        )
        cr.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
        cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
        cr.close_path()

    @staticmethod
    def _resample(values, count: int) -> np.ndarray:
        if values is None or len(values) == 0:
            return np.zeros(count, dtype=np.float64)
        values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        if len(values) == count:
            return values
        source = np.linspace(0.0, 1.0, len(values))
        target = np.linspace(0.0, 1.0, count)
        return np.interp(target, source, values)

    def _spectrum_target(self, values: np.ndarray) -> np.ndarray:
        """Map FFT amplitudes through the configured dB response curve."""
        if not np.any(values):
            return np.zeros_like(values)
        decibels = 20.0 * np.log10(np.maximum(values, 1e-7))
        floor = self.osd_config.spectrum_noise_floor_db
        peak = self.osd_config.spectrum_peak_db
        normalized = np.clip((decibels - floor) / (peak - floor), 0.0, 1.0)
        shaped = np.power(normalized, self.osd_config.spectrum_curve)
        return np.clip(shaped * self.osd_config.spectrum_gain, 0.0, 1.0)

    def set_state(self, state_str: str):
        self.state_manager.set_state_from_string(state_str or "recording")

    def update(self, level: float, samples: np.ndarray = None):
        super().update(level, samples)
        now = time.monotonic()
        dt = min(0.08, max(0.001, now - self._last_update))
        self._last_update = now
        self.state_manager.animation_phase = (
            self.state_manager.animation_phase + dt * 5.5
        ) % (2 * math.pi)

        state = self.state_manager.current_state
        positions = np.linspace(0.0, 1.0, self.num_bars)
        if state == VisualizerState.RECORDING:
            target = self._spectrum_target(
                self._resample(samples, self.num_bars)
            )
        elif state == VisualizerState.PROCESSING:
            phase = self.state_manager.animation_phase
            primary = 0.5 + 0.5 * np.sin(
                phase - positions * 4.5 * math.pi
            )
            secondary = 0.5 + 0.5 * np.sin(
                phase * 0.63 + positions * 2.0 * math.pi
            )
            target = 0.12 + 0.62 * (0.78 * primary + 0.22 * secondary)
        elif state == VisualizerState.ERROR:
            pulse = 0.22 + 0.78 * abs(
                math.sin(self.state_manager.animation_phase * 2.8)
            )
            target = np.full(self.num_bars, pulse)
        elif state == VisualizerState.SUCCESS:
            envelope = np.sin(math.pi * positions)
            target = 0.08 + 0.35 * envelope
        else:
            target = np.zeros(self.num_bars)

        rise_seconds = self.osd_config.spectrum_rise_ms / 1000.0
        fall_seconds = self.osd_config.spectrum_fall_ms / 1000.0
        rise = 1.0 if rise_seconds <= 0.0 else 1.0 - math.exp(-dt / rise_seconds)
        fall = 1.0 if fall_seconds <= 0.0 else 1.0 - math.exp(-dt / fall_seconds)
        blend = np.where(target > self.bar_heights, rise, fall)
        self.bar_heights += (target - self.bar_heights) * blend

    def draw_background(self, cr: cairo.Context, width: int, height: int):
        margin = self.PANEL_MARGIN
        panel_w = max(0.0, width - margin * 2.0)
        panel_h = max(0.0, height - margin * 2.0)
        if panel_w <= 0 or panel_h <= 0:
            return

        self._rounded_rect(
            cr,
            margin,
            margin + 2.0,
            panel_w,
            panel_h,
            self.PANEL_RADIUS,
        )
        shadow_alpha = self.osd_config.spectrum_background_opacity * 0.20
        cr.set_source_rgba(0.0, 0.0, 0.0, shadow_alpha)
        cr.fill()

        self._rounded_rect(
            cr,
            margin,
            margin,
            panel_w,
            panel_h,
            self.PANEL_RADIUS,
        )
        background = theme.background
        theme_alpha = background[3] if len(background) > 3 else 1.0
        cr.set_source_rgba(
            background[0],
            background[1],
            background[2],
            self.osd_config.spectrum_background_opacity * theme_alpha,
        )
        cr.fill_preserve()
        border = theme.border
        border_alpha = border[3] if len(border) > 3 else 1.0
        cr.set_source_rgba(
            border[0],
            border[1],
            border[2],
            self.osd_config.spectrum_border_opacity * border_alpha,
        )
        cr.set_line_width(2.0)
        cr.stroke()

    def draw(self, cr: cairo.Context, width: int, height: int):
        left = self.PANEL_MARGIN + self.INNER_PADDING_X
        right = width - self.PANEL_MARGIN - self.INNER_PADDING_X
        top = self.PANEL_MARGIN + self.INNER_PADDING_Y
        bottom = height - self.PANEL_MARGIN - self.INNER_PADDING_Y
        usable_width = max(0.0, right - left)
        usable_height = max(0.0, bottom - top)
        if usable_width <= 0 or usable_height <= 0:
            return

        center_y = top + usable_height / 2.0
        max_bar_height = usable_height * 0.94
        step = usable_width / self.num_bars
        bar_width = max(1.0, step - self.BAR_GAP)

        base, middle, high, peak = theme.spectrum_colors
        gradient = cairo.LinearGradient(0.0, top, 0.0, bottom)
        gradient.add_color_stop_rgba(0.00, peak[0], peak[1], peak[2], 0.98)
        gradient.add_color_stop_rgba(0.15, high[0], high[1], high[2], 0.98)
        gradient.add_color_stop_rgba(0.32, middle[0], middle[1], middle[2], 0.98)
        gradient.add_color_stop_rgba(0.50, base[0], base[1], base[2], 0.98)
        gradient.add_color_stop_rgba(0.68, middle[0], middle[1], middle[2], 0.98)
        gradient.add_color_stop_rgba(0.85, high[0], high[1], high[2], 0.98)
        gradient.add_color_stop_rgba(1.00, peak[0], peak[1], peak[2], 0.98)
        cr.set_source(gradient)

        for index, normalized in enumerate(self.bar_heights):
            bar_height = max(
                self.MIN_BAR_HEIGHT,
                float(np.clip(normalized, 0.0, 1.0)) * max_bar_height,
            )
            x = left + index * step + (step - bar_width) / 2.0
            y = center_y - bar_height / 2.0
            radius = min(bar_width / 2.0, bar_height / 2.0)
            self._rounded_rect(cr, x, y, bar_width, bar_height, radius)

        cr.fill()
