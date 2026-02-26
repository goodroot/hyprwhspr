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
import numpy as np
import cairo

# Import from mic_osd (sibling package under lib/)
from mic_osd.visualizations.base import BaseVisualization, StateManager, VisualizerState
from mic_osd.theme import theme


class SpeakingVisualization(BaseVisualization):
    """
    TTS audio visualization - wave bars only.

    No audio input; animation driven by state file (generating/speaking/success/error).
    Maps to VisualizerState: generating=processing, speaking=recording.
    """

    HINT_TEXT = "press shortcut again to cancel"

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

    def _wave_value(self, wave_pos: float) -> float:
        """Pure sine for maximum smoothness (no harmonics)."""
        return 0.5 + 0.48 * math.sin(wave_pos)

    def update(self, level: float, samples: np.ndarray = None):
        """Update - no audio input; use gentle wave for visual feedback."""
        super().update(level, samples)

        now = time.time()
        dt = 0.016
        if self._last_update_time is not None:
            dt = min(now - self._last_update_time, 0.1)
        self._last_update_time = now

        alpha = 1.0 - math.exp(-dt / self.tau)
        t = time.time() * 0.7
        for i in range(self.num_bars):
            wave_pos = (i / max(1, self.num_bars - 1)) * 2 * math.pi + t
            target = self._wave_value(wave_pos)
            self.bar_heights[i] += (target - self.bar_heights[i]) * alpha

        for _ in range(2):
            smoothed = np.copy(self.bar_heights)
            smoothed[0] = 0.7 * self.bar_heights[0] + 0.3 * self.bar_heights[1]
            for i in range(1, self.num_bars - 1):
                smoothed[i] = 0.25 * self.bar_heights[i - 1] + 0.5 * self.bar_heights[i] + 0.25 * self.bar_heights[i + 1]
            smoothed[-1] = 0.3 * self.bar_heights[-2] + 0.7 * self.bar_heights[-1]
            self.bar_heights = smoothed

        self.state_manager.update()

    def set_state(self, state_str: str):
        """Set state from string. Maps generating->processing, speaking->recording."""
        state_map = {
            'generating': VisualizerState.PROCESSING,
            'speaking': VisualizerState.RECORDING,
            'processing': VisualizerState.PROCESSING,
            'recording': VisualizerState.RECORDING,
            'success': VisualizerState.SUCCESS,
            'error': VisualizerState.ERROR,
            'paused': VisualizerState.PAUSED,
        }
        new_state = state_map.get(state_str.lower(), VisualizerState.RECORDING)
        self.state_manager.set_state(new_state)

    def draw(self, cr: cairo.Context, width: int, height: int):
        """Draw the TTS wave visualization and hint text."""
        padding = 16
        hint_height = 18
        bars_height = height - (padding * 2) - hint_height
        center_y = padding + bars_height / 2
        bars_width = width - (padding * 2)

        actual_num_bars = self.num_bars
        bar_gap = 2
        bar_width = (bars_width - (actual_num_bars - 1) * bar_gap) / actual_num_bars
        total_bar_width = bar_width + bar_gap
        start_x = padding

        bar_left = theme.bar_left
        bar_right = theme.bar_right

        is_processing = self.state_manager.current_state == VisualizerState.PROCESSING
        wave_phase = self.state_manager.animation_phase if is_processing else 0.0
        is_success = self.state_manager.current_state == VisualizerState.SUCCESS
        pulse_value = self.state_manager.get_animation_value() if is_success else 1.0

        for i in range(actual_num_bars):
            t = i / max(1, actual_num_bars - 1)
            r = bar_left[0] * (1 - t) + bar_right[0] * t
            g = bar_left[1] * (1 - t) + bar_right[1] * t
            b = bar_left[2] * (1 - t) + bar_right[2] * t
            normalized_height = self.bar_heights[i]

            if is_processing:
                wave_pos = (i / max(1, actual_num_bars - 1)) * 2 * math.pi + wave_phase
                wave_modulation = self._wave_value(wave_pos)
                base_height_boost = 0.6
                boosted = max(normalized_height, base_height_boost)
                normalized_height = max(0.0, min(1.0, boosted * wave_modulation))
                opacity_modulation = 0.82 + 0.18 * wave_modulation
            elif is_success:
                bar_h = max(self.min_bar_height, normalized_height * bars_height)
                pulse_modulation = 0.7 + 0.3 * pulse_value
                bar_h = bar_h * pulse_modulation
                opacity_modulation = pulse_value
            else:
                opacity_modulation = 1.0

            bar_h = max(self.min_bar_height, normalized_height * bars_height)
            x = start_x + i * total_bar_width
            bar_top = center_y - bar_h / 2

            cr.set_source_rgba(r, g, b, 0.3 * opacity_modulation)
            cr.rectangle(x - 1, bar_top - 1, bar_width + 2, bar_h + 2)
            cr.fill()

            cr.set_source_rgba(r, g, b, 0.9 * opacity_modulation)
            cr.rectangle(x, bar_top, bar_width, bar_h)
            cr.fill()

        self._draw_hint_text(cr, width, height)

    def _draw_hint_text(self, cr: cairo.Context, width: int, height: int):
        """Draw 'press shortcut again to cancel' at bottom center."""
        text = self.HINT_TEXT

        cr.select_font_face(
            "sans",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        cr.set_font_size(10)

        extents = cr.text_extents(text)
        text_width = extents.width

        x = (width - text_width) / 2
        y = height - 8

        text_color = theme.text if hasattr(theme, 'text') else (0.7, 0.74, 0.86, 0.9)
        if len(text_color) >= 4:
            cr.set_source_rgba(*text_color)
        else:
            cr.set_source_rgba(*text_color, 0.9)
        cr.move_to(x, y)
        cr.show_text(text)
