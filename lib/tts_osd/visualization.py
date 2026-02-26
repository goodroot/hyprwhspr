"""
Speaking visualization - "Reading..." indicator with state-driven animation.

Maps TTS states to mic-osd style visuals:
- generating -> processing (wave animation)
- speaking -> recording (pulsing dot)
- success -> pulse + fade
- error -> strobe
"""

import math
import numpy as np
import cairo

# Import from mic_osd (sibling package under lib/)
from mic_osd.visualizations.base import BaseVisualization, StateManager, VisualizerState
from mic_osd.theme import theme


class SpeakingVisualization(BaseVisualization):
    """
    TTS "Reading..." visualization - state-driven bars and indicator.

    No audio input; animation driven by state file (generating/speaking/success/error).
    Maps to VisualizerState: generating=processing, speaking=recording.
    """

    def __init__(self):
        super().__init__()
        self.num_bars = 32
        self.bar_width = 4
        self.bar_gap = 2
        self.min_bar_height = 2
        self.amplification = 4.0
        self.bar_heights = np.zeros(self.num_bars)
        self.decay_rate = 0.85
        self.rise_rate = 0.5
        self.state_manager = StateManager()

    def update(self, level: float, samples: np.ndarray = None):
        """Update - no audio input; use gentle wave for visual feedback."""
        super().update(level, samples)

        # Gentle wave animation (no mic input)
        import time
        t = time.time() * 2.0
        for i in range(self.num_bars):
            wave_pos = (i / max(1, self.num_bars - 1)) * 2 * math.pi + t * 0.5
            wave_val = 0.3 + 0.4 * math.sin(wave_pos) + 0.2 * math.sin(wave_pos * 2)
            target = max(0.0, min(1.0, wave_val))
            if target > self.bar_heights[i]:
                self.bar_heights[i] = (
                    self.rise_rate * target + (1 - self.rise_rate) * self.bar_heights[i]
                )
            else:
                self.bar_heights[i] *= self.decay_rate
                if self.bar_heights[i] < target:
                    self.bar_heights[i] = target

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
        """Draw the "Reading..." bar visualization."""
        padding = 16
        indicator_width = 30
        bars_start_x = padding + indicator_width
        bars_width = width - bars_start_x - padding
        bars_height = height - (padding * 2)
        center_y = height / 2

        self._draw_reading_indicator(cr, padding, center_y)

        actual_num_bars = self.num_bars
        bar_gap = 2
        bar_width = (bars_width - (actual_num_bars - 1) * bar_gap) / actual_num_bars
        total_bar_width = bar_width + bar_gap
        start_x = bars_start_x

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
                wave_modulation = 0.3 + 0.7 * math.sin(wave_pos)
                base_height_boost = 0.7
                boosted = max(normalized_height, base_height_boost)
                normalized_height = max(0.0, min(1.0, boosted * wave_modulation))
                opacity_modulation = 0.75 + 0.25 * (0.5 + 0.5 * math.sin(wave_pos))
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

    def _draw_reading_indicator(self, cr: cairo.Context, x: float, center_y: float):
        """Draw the state indicator dot with 'Reading...' label."""
        dot_radius = 6
        dot_x = x + dot_radius + 4
        dot_y = center_y

        pulse = self.state_manager.get_animation_value()
        dot_color = self.state_manager.get_state_color()

        if pulse <= 0:
            return

        cr.set_source_rgba(
            dot_color[0], dot_color[1], dot_color[2], 0.3 * pulse
        )
        cr.arc(dot_x, dot_y, dot_radius + 3, 0, 2 * math.pi)
        cr.fill()

        cr.set_source_rgba(
            dot_color[0], dot_color[1], dot_color[2], pulse
        )
        cr.arc(dot_x, dot_y, dot_radius, 0, 2 * math.pi)
        cr.fill()

        # Draw "Reading..." text
        label = "Reading..."
        cr.select_font_face(
            "sans",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL
        )
        cr.set_font_size(11)
        text_color = theme.text
        cr.set_source_rgba(
            text_color[0], text_color[1], text_color[2],
            text_color[3] if len(text_color) > 3 else 1.0
        )
        cr.move_to(dot_x + dot_radius + 12, center_y + 4)
        cr.show_text(label)
