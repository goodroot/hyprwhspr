"""
TTS audio visualization - state-driven wave animation.

Maps TTS states to mic-osd style visuals:
- generating -> processing (wave animation)
- speaking -> recording (wave)
- success -> pulse + fade
- error -> strobe
"""

import math
import time
from typing import Optional

import numpy as np
import cairo

# Import from mic_osd (sibling package under lib/)
from mic_osd.visualizations.base import BaseVisualization, StateManager, VisualizerState
from mic_osd.theme import theme

try:
    from ..src.paths import TTS_OSD_DURATION_FILE
except ImportError:
    try:
        from src.paths import TTS_OSD_DURATION_FILE
    except ImportError:
        TTS_OSD_DURATION_FILE = None


def _rounded_rect(cr: cairo.Context, x: float, y: float, w: float, h: float, r: float):
    """Draw a rounded rectangle path. Call fill() or stroke() after."""
    if r > w / 2:
        r = w / 2
    if r > h / 2:
        r = h / 2
    cr.move_to(x + r, y)
    cr.line_to(x + w - r, y)
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.line_to(x + w, y + h - r)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.line_to(x + r, y + h)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.line_to(x, y + r)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


class SpeakingVisualization(BaseVisualization):
    """
    TTS audio visualization - wave bars only.

    No audio input; animation driven by state file (generating/speaking/success/error).
    Maps to VisualizerState: generating=processing, speaking=recording.
    """

    HINT_TEXT = "press shortcut again to cancel"
    CORNER_RADIUS = 12
    MARGIN = 8
    INNER_PADDING = 16
    SIDE_TEXT_WIDTH = 52

    def __init__(self):
        super().__init__()
        self.num_bars = 32
        self.bar_width = 4
        self.bar_gap = 2
        self.min_bar_height = 2
        self.bar_heights = np.zeros(self.num_bars)
        self.tau = 0.35
        self._last_update_time = None
        self.state_manager = StateManager()
        self._countdown_duration_sec: Optional[float] = None
        self._countdown_start_time: Optional[float] = None

    def _wave_value(self, wave_pos: float) -> float:
        """Pure sine for processing state."""
        return 0.5 + 0.48 * math.sin(wave_pos)

    def _audio_like_value(self, i: int, t: float) -> float:
        """Random vertical bar heights only - no left-to-right or directional wave.
        Per-bar independent pseudo-random; each bar varies on its own over time."""
        step = int(t * 18)
        step2 = int(t * 13)
        seed = ((i * 7919 + step * 7829) % 10000) / 10000.0
        seed2 = ((i * 6311 + step2 * 6043) % 10000) / 10000.0
        return 0.15 + 0.8 * (0.6 * seed + 0.4 * seed2)

    def update(self, level: float, samples: np.ndarray = None):
        """Update - no audio input; use wave for visual feedback."""
        super().update(level, samples)

        now = time.time()
        dt = 0.016
        if self._last_update_time is not None:
            dt = min(now - self._last_update_time, 0.1)
        self._last_update_time = now

        t = time.time()
        is_recording = self.state_manager.current_state == VisualizerState.RECORDING
        # Snappy response for speaking - bars jump to new values, not smooth wave
        tau = 0.04 if is_recording else self.tau
        alpha = 1.0 - math.exp(-dt / tau)
        for i in range(self.num_bars):
            if is_recording:
                target = self._audio_like_value(i, t)
            else:
                wave_pos = (i / max(1, self.num_bars - 1)) * 2 * math.pi + t * 0.7
                target = self._wave_value(wave_pos)
            self.bar_heights[i] += (target - self.bar_heights[i]) * alpha

        # No smoothing when speaking - keep jagged random bar heights
        smooth_passes = 0 if is_recording else 2
        for _ in range(smooth_passes):
            smoothed = np.copy(self.bar_heights)
            smoothed[0] = 0.7 * self.bar_heights[0] + 0.3 * self.bar_heights[1]
            for i in range(1, self.num_bars - 1):
                smoothed[i] = 0.25 * self.bar_heights[i - 1] + 0.5 * self.bar_heights[i] + 0.25 * self.bar_heights[i + 1]
            smoothed[-1] = 0.3 * self.bar_heights[-2] + 0.7 * self.bar_heights[-1]
            self.bar_heights = smoothed

        self.state_manager.update()

    def set_state(self, state_str: str, duration_sec: Optional[float] = None):
        """Set state from string. Maps generating->processing, speaking->recording.
        duration_sec: optional countdown duration when speaking (from separate file).
        """
        state_map = {
            'generating': VisualizerState.PROCESSING,
            'speaking': VisualizerState.RECORDING,
            'processing': VisualizerState.PROCESSING,
            'recording': VisualizerState.RECORDING,
            'success': VisualizerState.SUCCESS,
            'error': VisualizerState.ERROR,
            'paused': VisualizerState.PAUSED,
        }
        base_state = state_str.strip().lower()
        new_state = state_map.get(base_state, VisualizerState.RECORDING)
        self.state_manager.set_state(new_state)
        if new_state == VisualizerState.RECORDING and duration_sec is not None and duration_sec > 0:
            self._countdown_duration_sec = duration_sec
            self._countdown_start_time = time.time()
        else:
            self._countdown_duration_sec = None
            self._countdown_start_time = None

    def _layout(self, width: int, height: int) -> tuple:
        """Return (padding, bars_width, bars_height, center_y) for content layout."""
        padding = self.MARGIN + self.INNER_PADDING
        bars_width = width - (padding * 2) - self.SIDE_TEXT_WIDTH - 8
        bars_height = height - (padding * 2)
        center_y = padding + bars_height / 2
        return padding, bars_width, bars_height, center_y

    def draw_background(self, cr: cairo.Context, width: int, height: int):
        """Draw background with rounded corners and margin."""
        m = self.MARGIN
        radius = self.CORNER_RADIUS
        w = width - 2 * m
        h = height - 2 * m
        _rounded_rect(cr, m, m, w, h, radius)
        cr.set_source_rgba(*self.background_color)
        cr.fill()
        if border_color := theme.border:
            cr.set_source_rgba(*(border_color if len(border_color) == 4 else (*border_color, 1.0)))
            cr.set_line_width(2)
            _rounded_rect(cr, m, m, w, h, radius)
            cr.stroke()

    def draw(self, cr: cairo.Context, width: int, height: int):
        """Draw the TTS wave visualization and hint/timer text."""
        padding, bars_width, bars_height, center_y = self._layout(width, height)

        bar_width = (bars_width - (self.num_bars - 1) * self.bar_gap) / self.num_bars
        total_bar_width = bar_width + self.bar_gap
        bar_left = theme.bar_left
        bar_right = theme.bar_right

        state = self.state_manager.current_state
        is_processing = state == VisualizerState.PROCESSING
        is_success = state == VisualizerState.SUCCESS
        wave_phase = self.state_manager.animation_phase if is_processing else 0.0
        pulse_value = self.state_manager.get_animation_value() if is_success else 1.0

        for i in range(self.num_bars):
            t = i / max(1, self.num_bars - 1)
            r = bar_left[0] * (1 - t) + bar_right[0] * t
            g = bar_left[1] * (1 - t) + bar_right[1] * t
            b = bar_left[2] * (1 - t) + bar_right[2] * t
            norm_h = self.bar_heights[i]

            if is_processing:
                wave_pos = t * 2 * math.pi + wave_phase
                wave_mod = self._wave_value(wave_pos)
                boosted = max(norm_h, 0.6) * wave_mod
                norm_h = max(0.0, min(1.0, boosted))
                opacity = 0.82 + 0.18 * wave_mod
            elif is_success:
                opacity = pulse_value
            else:
                opacity = 1.0

            bar_h = max(self.min_bar_height, norm_h * bars_height)
            if is_success:
                bar_h *= 0.7 + 0.3 * pulse_value
            x = padding + i * total_bar_width
            bar_top = center_y - bar_h / 2

            cr.set_source_rgba(r, g, b, 0.3 * opacity)
            cr.rectangle(x - 1, bar_top - 1, bar_width + 2, bar_h + 2)
            cr.fill()
            cr.set_source_rgba(r, g, b, 0.9 * opacity)
            cr.rectangle(x, bar_top, bar_width, bar_h)
            cr.fill()

        self._draw_hint_text(cr, width, height, padding, center_y)

    def _draw_hint_text(
        self, cr: cairo.Context, width: int, height: int,
        padding: float, center_y: float
    ):
        """Draw countdown on right (same level as wave) or hint text at bottom."""
        cr.select_font_face(
            "sans",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        text_color = theme.text if hasattr(theme, 'text') else (0.7, 0.74, 0.86, 0.9)
        cr.set_source_rgba(*(text_color if len(text_color) == 4 else (*text_color, 0.9)))

        is_speaking = self.state_manager.current_state == VisualizerState.RECORDING
        countdown_str = None
        duration_sec = self._countdown_duration_sec
        if is_speaking:
            if TTS_OSD_DURATION_FILE and TTS_OSD_DURATION_FILE.exists():
                try:
                    duration_sec = float(TTS_OSD_DURATION_FILE.read_text().strip())
                except (ValueError, OSError):
                    pass
            if duration_sec is not None and duration_sec > 0:
                if self._countdown_start_time is None:
                    self._countdown_start_time = time.time()
                total_duration = float(duration_sec)
                time_elapsed = time.time() - self._countdown_start_time
                time_remaining = total_duration - time_elapsed
                display_seconds = max(0, time_remaining)
                mins = int(display_seconds // 60)
                secs = int(round(display_seconds)) % 60
                countdown_str = f"{mins}:{secs:02d}"

        if countdown_str:
            cr.set_font_size(12)
            extents = cr.text_extents(countdown_str)
            x = width - extents.width - padding
            # Vertically center: baseline y such that text center aligns with center_y
            y = center_y - extents.y_bearing - extents.height / 2
            cr.move_to(x, y)
            cr.show_text(countdown_str)
        else:
            cr.set_font_size(10)
            extents = cr.text_extents(self.HINT_TEXT)
            x = (width - extents.width) / 2
            y = height - self.MARGIN - 8 - extents.y_bearing - extents.height
            cr.move_to(x, y)
            cr.show_text(self.HINT_TEXT)
