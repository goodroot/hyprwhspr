"""
GTK4 Layer Shell window for mic-osd.

Creates an overlay window at the bottom center of the screen
for displaying audio visualizations.
"""

from __future__ import annotations

import time

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
import cairo

try:
    gi.require_version('Gtk4LayerShell', '1.0')
    LAYER_SHELL_AVAILABLE = True
except ValueError:
    LAYER_SHELL_AVAILABLE = False

from gi.repository import Gtk, Gdk, GLib
from .theme import theme
from .transcript_preview import PillTranscriptAnimator, PillTranscriptConfig

if LAYER_SHELL_AVAILABLE:
    from gi.repository import Gtk4LayerShell


class OSDWindow(Gtk.Window):
    """
    An overlay window for displaying audio visualizations.
    
    Uses gtk4-layer-shell to create a Wayland layer surface
    that appears above all windows at the bottom of the screen.
    """

    PREVIEW_WORD_LIMIT = 14
    PREVIEW_TIMER_RESERVE = 58
    
    def __init__(
        self,
        visualization,
        width=300,
        height=60,
        transcript_config=None,
    ):
        """
        Initialize the OSD window.
        
        Args:
            visualization: A BaseVisualization instance
            width: Window width in pixels
            height: Window height in pixels
            transcript_config: Validated PillTranscriptConfig for pill previews.
        """
        super().__init__()
        
        self.visualization = visualization
        self._is_pill_preview = getattr(visualization, 'preview_mode', None) == 'pill'
        self._pill_transcript_config = (
            transcript_config
            if transcript_config is not None
            else PillTranscriptConfig.load()
        )
        if self._is_pill_preview and self._pill_transcript_config.enabled:
            height = max(
                height,
                int(
                    getattr(self.visualization, 'PILL_HEIGHT', 42)
                    + self._pill_transcript_config.font_size
                    + self._pill_transcript_config.offset_y
                    + 16
                ),
                # Must clear the pill's own preview-mode threshold.
                getattr(self.visualization, 'PREVIEW_HEIGHT_THRESHOLD', 76) + 1,
            )
            width = max(
                width,
                int(min(800.0, self._pill_transcript_config.max_width + 24.0)),
            )

        self._width = width
        self._height = height
        self._preview_text = ""
        self._visualizer_state = "recording"
        self._pill_transcript_animator = None
        if self._is_pill_preview:
            self._pill_transcript_animator = PillTranscriptAnimator(
                self._pill_transcript_config
            )
        
        # Layer shell MUST be initialized immediately after window creation
        # and BEFORE any other window configuration
        self._setup_layer_shell()
        self._setup_window()
        self._setup_drawing_area()
    
    def _setup_layer_shell(self):
        """Configure layer shell behavior."""
        if not LAYER_SHELL_AVAILABLE:
            return
        
        # Initialize layer shell - MUST be initialized before other calls
        Gtk4LayerShell.init_for_window(self)
        
        # Set namespace for window rules
        Gtk4LayerShell.set_namespace(self, "mic-osd")
        
        # Put on overlay layer (above everything)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        
        # Anchor to bottom only (centers horizontally)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, False)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.TOP, False)
        
        # Margin from bottom
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, 130)
        
        # Don't reserve exclusive space
        Gtk4LayerShell.set_exclusive_zone(self, -1)
        
        # No keyboard input
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
    
    def _setup_window(self):
        """Configure basic window properties."""
        self.set_decorated(False)
        self.set_resizable(False)
        self.set_default_size(self._width, self._height)
        
        # Make window transparent
        self.add_css_class('mic-osd-window')
    
    def _setup_drawing_area(self):
        """Set up the Cairo drawing area."""
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.set_content_width(self._width)
        self.drawing_area.set_content_height(self._height)
        self.drawing_area.set_draw_func(self._on_draw)
        self.set_child(self.drawing_area)
    
    def _on_draw(self, area, cr, width, height):
        """
        Called when the drawing area needs to be redrawn.
        
        Args:
            area: The DrawingArea widget
            cr: Cairo context
            width: Available width
            height: Available height
        """
        # Draw background
        self.visualization.draw_background(cr, width, height)
        
        # Draw the visualization
        self.visualization.draw(cr, width, height)

        if self._is_pill_preview:
            self._draw_pill_preview_text(cr, width, height)
        elif getattr(self.visualization, 'show_preview', True):
            self._draw_preview_text(cr, width, height)
    
    def update(self, level: float, samples=None):
        """
        Update the visualization with new audio data.
        
        Args:
            level: Audio level (0.0 to 1.0)
            samples: Raw audio samples (optional)
        """
        self.visualization.update(level, samples)
        self.drawing_area.queue_draw()

    def set_preview_text(self, text: str):
        """Set compact transcript preview text."""
        self._preview_text = (text or "").rstrip('\r\n')
        if self._pill_transcript_animator is not None:
            self._pill_transcript_animator.set_text(self._preview_text)
        self.drawing_area.queue_draw()

    def set_visualizer_state(self, state: str):
        """Track visualizer state so partial previews only render while recording."""
        previous_state = self._visualizer_state
        self._visualizer_state = (state or "recording").lower()
        if (
            self._pill_transcript_animator is not None
            and previous_state == "recording"
            and self._visualizer_state != "recording"
        ):
            self._pill_transcript_animator.clear()
        self.drawing_area.queue_draw()
    
    def _draw_preview_text(self, cr: cairo.Context, width: int, height: int):
        if not self._preview_text or self._visualizer_state != "recording":
            return

        padding = 14
        max_width = max(0, width - padding * 2 - self.PREVIEW_TIMER_RESERVE)

        cr.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(12)
        text = self._ellipsize(cr, self._recent_preview_text(self._preview_text), max_width)
        if not text:
            return

        extents = cr.text_extents(text)
        text_width = self._text_extent(extents, 'width', 2)
        text_height = self._text_extent(extents, 'height', 3)

        y = height - 10
        bg_padding_x = 7
        bg_padding_y = 4
        bg_x = padding - bg_padding_x
        bg_y = y - text_height - bg_padding_y - 1
        bg_w = min(max_width + bg_padding_x * 2, text_width + bg_padding_x * 2)
        bg_h = text_height + bg_padding_y * 2 + 2

        bg = theme.background
        cr.set_source_rgba(bg[0], bg[1], bg[2], 0.88)
        cr.rectangle(bg_x, bg_y, bg_w, bg_h)
        cr.fill()

        text_color = theme.text
        cr.set_source_rgba(text_color[0], text_color[1], text_color[2], 0.96)
        cr.move_to(padding, y)
        cr.show_text(text)

    def _draw_pill_preview_text(
        self,
        cr: cairo.Context,
        width: int,
        height: int,
    ):
        animator = self._pill_transcript_animator
        config = self._pill_transcript_config
        if (
            animator is None
            or not config.enabled
            or self._visualizer_state != "recording"
        ):
            return

        frame = animator.frame(time.monotonic())
        if not frame.words:
            return

        cr.select_font_face(
            config.font_family,
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL,
        )

        max_width = min(config.max_width, max(0.0, width - 20.0))
        effective_font_size = config.font_size
        cr.set_font_size(effective_font_size)
        previous_texts = frame.previous_words
        current_texts = frame.current_words
        _, previous_total = self._word_layout(cr, previous_texts, width)
        _, current_total = self._word_layout(cr, current_texts, width)

        largest_total = max(previous_total, current_total)
        if largest_total > max_width and largest_total > 0:
            effective_font_size = max(
                8.0,
                effective_font_size * max_width / largest_total,
            )
            cr.set_font_size(effective_font_size)

        previous_texts = tuple(
            self._ellipsize_pill_token(cr, word, max_width)
            for word in previous_texts
        )
        current_texts = tuple(
            self._ellipsize_pill_token(cr, word, max_width)
            for word in current_texts
        )
        previous_positions, _ = self._word_layout(cr, previous_texts, width)
        current_positions, _ = self._word_layout(cr, current_texts, width)

        pill_geometry = getattr(self.visualization, '_pill_geometry', None)
        if callable(pill_geometry):
            _, pill_y, _, _ = pill_geometry(width, height)
        else:
            pill_y = height - 46.0
        baseline = pill_y - config.offset_y

        for word in frame.words:
            if word.alpha <= 0.01:
                continue

            if word.source == "previous":
                if word.index >= len(previous_positions):
                    continue
                x = previous_positions[word.index]
                text = previous_texts[word.index]
            else:
                if word.index >= len(current_positions):
                    continue
                x = current_positions[word.index]
                text = current_texts[word.index]
                if (
                    word.matched_from is not None
                    and word.matched_from < len(previous_positions)
                ):
                    old_x = previous_positions[word.matched_from]
                    x = old_x + (
                        x - old_x
                    ) * word.layout_progress

            y = baseline + word.y_offset

            # A restrained shadow keeps white text readable over bright windows
            # without introducing a visible badge or background rectangle.
            cr.set_source_rgba(0.0, 0.0, 0.0, 0.55 * word.alpha)
            cr.move_to(x, y + 2.0)
            cr.show_text(text)

            cr.set_source_rgba(1.0, 1.0, 1.0, 0.97 * word.alpha)
            cr.move_to(x, y)
            cr.show_text(text)

    def _word_layout(self, cr: cairo.Context, words, width: int):
        if not words:
            return (), 0.0

        # Cairo's ink width for a space is zero. Layout with glyph advances so
        # whitespace and side bearings are preserved between separately drawn
        # words instead of making them visually run together.
        space_width = self._text_advance(cr, " ")
        widths = [self._text_advance(cr, word) for word in words]
        total_width = sum(widths) + space_width * max(0, len(words) - 1)
        x = (width - total_width) / 2.0
        positions = []
        for word_width in widths:
            positions.append(x)
            x += word_width + space_width
        return tuple(positions), total_width

    @staticmethod
    def _bisect_truncate(measure, available: float, length: int) -> int:
        """Longest length in [0, length] fitting `available`."""
        low, high = 0, length
        while low < high:
            mid = (low + high + 1) // 2
            if measure(mid) <= available:
                low = mid
            else:
                high = mid - 1
        return low

    def _ellipsize_pill_token(
        self,
        cr: cairo.Context,
        text: str,
        max_width: float,
    ) -> str:
        if self._text_advance(cr, text) <= max_width:
            return text

        suffix = "…"
        available = max_width - self._text_advance(cr, suffix)
        if available <= 0:
            return ""

        low = self._bisect_truncate(
            lambda n: self._text_advance(cr, text[:n]), available, len(text)
        )
        return text[:low].rstrip() + suffix if low else suffix

    @staticmethod
    def _text_extent(extents, field: str, index: int) -> float:
        if hasattr(extents, field):
            return getattr(extents, field)
        return extents[index]

    def _text_width(self, cr: cairo.Context, text: str) -> float:
        return self._text_extent(cr.text_extents(text), 'width', 2)

    def _text_advance(self, cr: cairo.Context, text: str) -> float:
        extents = cr.text_extents(text)
        advance = self._text_extent(extents, 'x_advance', 4)
        if advance > 0:
            return advance
        return self._text_extent(extents, 'width', 2)

    def _text_height(self, cr: cairo.Context, text: str) -> float:
        return self._text_extent(cr.text_extents(text), 'height', 3)

    def _recent_preview_text(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.PREVIEW_WORD_LIMIT:
            return " ".join(words)
        return "... " + " ".join(words[-self.PREVIEW_WORD_LIMIT:])

    def _ellipsize(self, cr: cairo.Context, text: str, max_width: float) -> str:
        if self._text_width(cr, text) <= max_width:
            return text

        prefix = "... "
        if text.startswith(prefix):
            text = text[len(prefix):]

        available = max_width - self._text_width(cr, prefix)
        if available <= 0:
            return ""

        low = self._bisect_truncate(
            lambda n: self._text_width(cr, text[-n:]) if n else 0.0, available, len(text)
        )

        truncated = text[-low:].lstrip()
        return prefix + truncated if truncated else prefix


def load_css(css_path=None):
    """
    Load CSS styling for the OSD.
    
    Args:
        css_path: Path to CSS file (optional)
    """
    css_provider = Gtk.CssProvider()
    
    default_css = """
    .mic-osd-window {
        background-color: transparent;
    }
    """
    
    if css_path:
        try:
            css_provider.load_from_path(css_path)
        except GLib.Error:
            css_provider.load_from_string(default_css)
    else:
        css_provider.load_from_string(default_css)
    
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
