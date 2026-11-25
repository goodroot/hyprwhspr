"""Mic-OSD backend choice. No GTK imports: a process binds one GTK version."""

import os

GTK4 = 'gtk4'
GTK3 = 'gtk3'


def x11_session(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get('DISPLAY')) and not env.get('WAYLAND_DISPLAY')


def select_backend(environ=None) -> str:
    """The runner's choice if set, else GTK3 on X11, else GTK4."""
    env = os.environ if environ is None else environ
    forced = env.get('HYPRWHSPR_MIC_OSD_BACKEND')
    if forced in (GTK3, GTK4):
        return forced
    return GTK3 if x11_session(env) else GTK4


def bottom_center(geometry, width: int, height: int, margin: int):
    """Top-left corner for a surface centered `margin` above the bottom edge."""
    x = geometry.x + (geometry.width - width) // 2
    y = geometry.y + geometry.height - height - margin
    return max(geometry.x, x), max(geometry.y, y)
