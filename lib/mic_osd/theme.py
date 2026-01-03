"""
Theme loading for mic-osd.

Reads colors from Omarchy theme files in ~/.config/omarchy/current/theme/
"""

import os
import re
from pathlib import Path


# Default colors (fallback if theme not found)
DEFAULT_COLORS = {
    'background-color': (0.1, 0.1, 0.15, 0.95),
    'border-color': (0.2, 0.8, 1.0),        # Cyan
    'bar-color-left': (0.2, 0.8, 1.0),      # Cyan
    'bar-color-right': (0.0, 1.0, 0.6),     # Green
    'recording-dot': (1.0, 0.2, 0.33),      # Red
    'text-color': (0.8, 0.84, 0.96, 1.0),   # Light gray
}


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert hex color to RGB tuple (0.0-1.0 range).
    
    Args:
        hex_color: Color in #RRGGBB or #RRGGBBAA format
        
    Returns:
        Tuple of (r, g, b) or (r, g, b, a) floats
    """
    hex_color = hex_color.strip().lstrip('#')
    
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    elif len(hex_color) == 8:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0
        return (r, g, b, a)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")


def load_theme() -> dict:
    """
    Load theme colors from Omarchy theme file.
    
    Looks for (in order):
    1. ~/.config/omarchy/current/theme/mic-osd.css
    2. ~/.config/omarchy/current/theme/swayosd.css (fallback)
    
    Returns:
        Dict of color name -> RGB(A) tuple
    """
    colors = DEFAULT_COLORS.copy()
    theme_dir = Path.home() / '.config' / 'omarchy' / 'current' / 'theme'
    
    # Try mic-osd specific theme first
    mic_osd_path = theme_dir / 'mic-osd.css'
    if mic_osd_path.exists():
        try:
            colors.update(parse_css_colors(mic_osd_path))
            return colors
        except Exception:
            pass  # Fall through to swayosd fallback
    
    # Fall back to swayosd.css for consistent OSD styling
    swayosd_path = theme_dir / 'swayosd.css'
    if swayosd_path.exists():
        try:
            swayosd_colors = parse_css_colors(swayosd_path)
            
            # Map swayosd colors to mic-osd colors
            if 'background-color' in swayosd_colors:
                bg = swayosd_colors['background-color']
                # Add alpha if not present
                if len(bg) == 3:
                    bg = (*bg, 0.95)
                colors['background-color'] = bg
            
            if 'border-color' in swayosd_colors:
                colors['border-color'] = swayosd_colors['border-color']
                # Also use border color for bar gradient by default
                colors['bar-color-left'] = swayosd_colors['border-color']
                colors['bar-color-right'] = swayosd_colors['border-color']
            
            if 'progress' in swayosd_colors:
                # Use progress color for bars if available
                colors['bar-color-left'] = swayosd_colors['progress']
                colors['bar-color-right'] = swayosd_colors['progress']
            
        except Exception:
            pass  # Use defaults
    
    return colors


def parse_css_colors(css_path: Path) -> dict:
    """
    Parse @define-color directives from a CSS file.
    
    Args:
        css_path: Path to CSS file
        
    Returns:
        Dict of color name -> RGB(A) tuple
    """
    colors = {}
    
    # Pattern: @define-color name #hexvalue;
    pattern = re.compile(r'@define-color\s+([\w-]+)\s+(#[0-9a-fA-F]{6,8})\s*;')
    
    with open(css_path, 'r') as f:
        content = f.read()
    
    for match in pattern.finditer(content):
        name = match.group(1)
        hex_color = match.group(2)
        try:
            colors[name] = hex_to_rgb(hex_color)
        except ValueError:
            pass  # Skip invalid colors silently
    
    return colors


class Theme:
    """
    Theme singleton for easy access to colors.
    """
    _instance = None
    _colors = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._colors = load_theme()
        return cls._instance
    
    def get(self, name: str, default=None):
        """Get a color by name."""
        return self._colors.get(name, default or DEFAULT_COLORS.get(name))
    
    def reload(self):
        """Reload theme from disk."""
        self._colors = load_theme()
    
    @property
    def background(self):
        return self.get('background-color')
    
    @property
    def border(self):
        return self.get('border-color')
    
    @property
    def bar_left(self):
        return self.get('bar-color-left')
    
    @property
    def bar_right(self):
        return self.get('bar-color-right')
    
    @property
    def recording_dot(self):
        color = self.get('recording-dot')
        # Ensure alpha channel
        if len(color) == 3:
            return (*color, 1.0)
        return color
    
    @property
    def text(self):
        color = self.get('text-color')
        # Ensure alpha channel
        if len(color) == 3:
            return (*color, 1.0)
        return color


# Global theme instance
theme = Theme()


class ThemeWatcher:
    """
    Watches for Omarchy theme changes and reloads the theme.
    
    Watches the 'current' symlink in ~/.config/omarchy/ since theme switching
    changes the symlink target rather than modifying files in place.
    
    Uses GLib.FileMonitor (inotify on Linux) for efficient file watching.
    """
    
    def __init__(self, on_theme_changed=None):
        """
        Initialize the theme watcher.
        
        Args:
            on_theme_changed: Optional callback to invoke after theme reload
        """
        self._monitor = None
        self._on_theme_changed = on_theme_changed
    
    def start(self):
        """Start watching the 'current' symlink for changes."""
        from gi.repository import Gio
        
        # Watch the 'current' symlink itself, not its contents
        # Theme changes update the symlink target, not files within
        current_link = Path.home() / '.config' / 'omarchy' / 'current'
        
        if not current_link.exists():
            return False
        
        try:
            gfile = Gio.File.new_for_path(str(current_link))
            # WATCH_MOUNTS helps catch symlink target changes
            self._monitor = gfile.monitor_file(
                Gio.FileMonitorFlags.WATCH_MOUNTS,
                None
            )
            self._monitor.connect('changed', self._on_symlink_changed)
            return True
        except Exception:
            return False
    
    def stop(self):
        """Stop watching."""
        if self._monitor:
            self._monitor.cancel()
            self._monitor = None
    
    def _on_symlink_changed(self, monitor, file, other_file, event_type):
        """Handle symlink change events."""
        from gi.repository import Gio, GLib
        
        # CHANGED fires when symlink target changes
        if event_type in (Gio.FileMonitorEvent.CHANGED, 
                          Gio.FileMonitorEvent.ATTRIBUTE_CHANGED):
            GLib.idle_add(self._reload_theme)
    
    def _reload_theme(self):
        """Reload theme on main thread."""
        theme.reload()
        if self._on_theme_changed:
            self._on_theme_changed()
        return False  # Don't repeat
