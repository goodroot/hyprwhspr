"""Compact monochrome pill visualization for mic-osd."""

import math
import time

import cairo
import numpy as np

from .base import BaseVisualization, StateManager, VisualizerState


class PillVisualization(BaseVisualization):
    """A compact black pill with white audio bars and state animations."""

    show_preview = True
    preview_mode = "pill"

    PILL_WIDTH = 126
    PILL_HEIGHT = 42
    # Surfaces this tall are reserving space for the preview text above.
    PREVIEW_HEIGHT_THRESHOLD = PILL_HEIGHT + 34
    NUM_BARS = 13
    BAR_WIDTH = 3.0
    BAR_GAP = 3.0
    MIN_HEIGHT = 3.0
    MAX_HEIGHT = 22.0
    NOISE_GATE = 0.006
    INPUT_GAIN = 18.0

    _CHECK_OFFSETS = np.array([
        0.0, 0.0, 0.0, -1.0, 1.5, 4.0, 6.5,
        3.0, -0.5, -4.0, -7.5, 0.0, 0.0,
    ])
    _CHECK_OPACITY = np.array([
        0.0, 0.0, 0.0, 0.45, 0.70, 0.92, 1.0,
        1.0, 1.0, 0.95, 0.72, 0.0, 0.0,
    ])

    def __init__(self):
        super().__init__()
        self.num_bars = self.NUM_BARS
        self.bar_heights = np.zeros(self.num_bars, dtype=np.float64)
        self.state_manager = StateManager()
        self._last_update = time.monotonic()

    @staticmethod
    def _rounded_rect(cr: cairo.Context, x: float, y: float, width: float,
                      height: float, radius: float):
        radius = min(radius, width / 2.0, height / 2.0)
        cr.new_sub_path()
        cr.arc(x + width - radius, y + radius, radius, -math.pi / 2, 0)
        cr.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
        cr.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
        cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
        cr.close_path()

    @staticmethod
    def _resample(values: np.ndarray, count: int) -> np.ndarray:
        if values is None or len(values) == 0:
            return np.zeros(count, dtype=np.float64)
        values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        if len(values) == count:
            return values
        old_x = np.linspace(0.0, 1.0, len(values))
        new_x = np.linspace(0.0, 1.0, count)
        return np.interp(new_x, old_x, values)

    def set_state(self, state_str: str):
        self.state_manager.set_state_from_string(state_str or "recording")

    def update(self, level: float, samples: np.ndarray = None):
        super().update(level, samples)
        now = time.monotonic()
        dt = min(0.05, max(0.001, now - self._last_update))
        self._last_update = now

        self.state_manager.animation_phase = (
            self.state_manager.animation_phase + dt * 7.2
        ) % (2 * math.pi)

        state = self.state_manager.current_state
        positions = np.linspace(0.0, 1.0, self.num_bars)

        if state == VisualizerState.RECORDING:
            audio = self._resample(samples, self.num_bars)
            if not np.any(audio):
                audio[:] = max(0.0, float(level))

            # Gate mic noise out of the RMS feed, then compress so speech fills the pill.
            energy = np.sqrt(np.clip(
                np.maximum(audio - self.NOISE_GATE, 0.0) * self.INPUT_GAIN,
                0.0,
                1.0,
            ))
            center_envelope = 0.62 + 0.38 * np.sin(math.pi * positions)
            target = energy * center_envelope

        elif state == VisualizerState.PROCESSING:
            phase = self.state_manager.animation_phase
            wave = 0.5 + 0.5 * np.sin(phase - positions * 2.6 * math.pi)
            second = 0.5 + 0.5 * np.sin(
                phase * 0.72 + positions * 1.7 * math.pi
            )
            target = 0.16 + 0.70 * (0.72 * wave + 0.28 * second)

        elif state == VisualizerState.ERROR:
            blink = 0.25 + 0.75 * abs(
                math.sin(self.state_manager.animation_phase * 2.5)
            )
            target = np.full(self.num_bars, blink)

        elif state == VisualizerState.PAUSED:
            target = np.zeros(self.num_bars)

        else:  # Success morphs into a checkmark in draw().
            target = np.full(self.num_bars, 0.18)

        rise = 1.0 - math.exp(-dt * 18.0)
        fall = 1.0 - math.exp(-dt * 9.0)
        blend = np.where(target > self.bar_heights, rise, fall)
        self.bar_heights += (target - self.bar_heights) * blend

    def _pill_geometry(self, width: int, height: int):
        pill_w = min(self.PILL_WIDTH, width - 4)
        pill_h = min(self.PILL_HEIGHT, height - 4)
        x = (width - pill_w) / 2.0

        # A taller surface reserves space for live text above the pill. Keep
        # legacy standalone/preview-disabled geometry unchanged at the old size.
        if height >= self.PREVIEW_HEIGHT_THRESHOLD:
            y = height - pill_h - 4.0
        else:
            y = (height - pill_h) / 2.0

        return x, y, pill_w, pill_h

    def _success_fade(self) -> float:
        if self.state_manager.current_state != VisualizerState.SUCCESS:
            return 1.0
        elapsed = time.time() - self.state_manager.state_changed_at
        if elapsed <= 0.72:
            return 1.0
        return max(0.0, 1.0 - (elapsed - 0.72) / 0.28)

    def draw_background(self, cr: cairo.Context, width: int, height: int):
        x, y, pill_w, pill_h = self._pill_geometry(width, height)
        alpha = self._success_fade()

        self._rounded_rect(cr, x, y + 2.0, pill_w, pill_h, pill_h / 2.0)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.30 * alpha)
        cr.fill()

        self._rounded_rect(cr, x, y, pill_w, pill_h, pill_h / 2.0)
        cr.set_source_rgba(0.015, 0.015, 0.018, 0.96 * alpha)
        cr.fill_preserve()
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.075 * alpha)
        cr.set_line_width(1.0)
        cr.stroke()

    def draw(self, cr: cairo.Context, width: int, height: int):
        x, y, pill_w, pill_h = self._pill_geometry(width, height)
        center_x = x + pill_w / 2.0
        center_y = y + pill_h / 2.0
        total_width = (
            self.num_bars * self.BAR_WIDTH
            + (self.num_bars - 1) * self.BAR_GAP
        )
        start_x = center_x - total_width / 2.0

        state = self.state_manager.current_state
        elapsed = time.time() - self.state_manager.state_changed_at
        success_fade = self._success_fade()

        morph = 0.0
        if state == VisualizerState.SUCCESS:
            t = min(1.0, elapsed / 0.22)
            morph = 1.0 - (1.0 - t) ** 3

        for i in range(self.num_bars):
            bar_x = start_x + i * (self.BAR_WIDTH + self.BAR_GAP)
            height_norm = float(np.clip(self.bar_heights[i], 0.0, 1.0))
            bar_h = self.MIN_HEIGHT + height_norm * (
                self.MAX_HEIGHT - self.MIN_HEIGHT
            )
            bar_center_y = center_y
            opacity = 0.94

            if state == VisualizerState.SUCCESS:
                target_h = 5.5
                bar_h = bar_h * (1.0 - morph) + target_h * morph
                bar_center_y += self._CHECK_OFFSETS[i] * morph
                opacity *= (
                    (1.0 - morph) + self._CHECK_OPACITY[i] * morph
                ) * success_fade
            elif state == VisualizerState.ERROR:
                opacity *= 0.55 + 0.45 * abs(
                    math.sin(self.state_manager.animation_phase * 2.5)
                )

            if opacity <= 0.01:
                continue

            bar_y = bar_center_y - bar_h / 2.0
            radius = min(self.BAR_WIDTH / 2.0, bar_h / 2.0)
            self._rounded_rect(
                cr,
                bar_x,
                bar_y,
                self.BAR_WIDTH,
                bar_h,
                radius,
            )
            cr.set_source_rgba(1.0, 1.0, 1.0, opacity)
            cr.fill()
