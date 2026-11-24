"""Window backend selection for mic-osd.

Kept free of GTK imports so the choice can be made (and tested) before either
toolkit is loaded: a process can only bind one GTK major version.
"""

import os

GTK4 = 'gtk4'
GTK3 = 'gtk3'


def select_backend(environ=None) -> str:
    """Return 'gtk4' (layer-shell overlay) or 'gtk3' (X11 overlay).

    HYPRWHSPR_MIC_OSD_BACKEND wins, so the runner can start the daemon with the
    backend it probed. Otherwise a Wayland socket means GTK4 and a bare X11
    display means GTK3; with neither, keep the Wayland path (it degrades to a
    plain window, which is what GNOME expects).
    """
    env = os.environ if environ is None else environ
    forced = (env.get('HYPRWHSPR_MIC_OSD_BACKEND') or '').strip().lower()
    if forced == 'x11':
        return GTK3
    if forced in (GTK3, GTK4):
        return forced
    if env.get('WAYLAND_DISPLAY'):
        return GTK4
    if env.get('DISPLAY'):
        return GTK3
    return GTK4


def bottom_center(geometry, width: int, height: int, margin: int):
    """Top-left corner that centers a surface above the bottom edge.

    geometry: anything with x, y, width and height (a Gdk monitor geometry).
    """
    x = geometry.x + (geometry.width - width) // 2
    y = geometry.y + geometry.height - height - margin
    return max(geometry.x, x), max(geometry.y, y)
