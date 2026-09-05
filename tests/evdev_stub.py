"""Minimal evdev constants for importing keyboard code without input devices."""

from types import SimpleNamespace


def modifier_codes():
    return SimpleNamespace(
        KEY_LEFTCTRL=29, KEY_RIGHTCTRL=97,
        KEY_LEFTALT=56, KEY_RIGHTALT=100,
        KEY_LEFTSHIFT=42, KEY_RIGHTSHIFT=54,
        KEY_LEFTMETA=125, KEY_RIGHTMETA=126,
    )
