"""
Keyboard layout helpers for hyprwhspr

Compiles the active XKB keymap for the shortcut handler (which physical key
produces 'd'?) and the text injector (would `ydotool type` emit ASCII verbatim?).
"""

import re
import subprocess
import time
from typing import Dict, Optional, Tuple

# X11 keycode = evdev keycode + 8
X11_TO_EVDEV_OFFSET = 8

# The keys carrying printable ASCII on a US keyboard: number row, the three
# letter rows, backslash and space. The rest is layout-independent for us.
_ASCII_KEY_NAMES = frozenset(
    ['TLDE', 'BKSL', 'SPCE']
    + [f'AE{i:02d}' for i in range(1, 13)]
    + [f'AD{i:02d}' for i in range(1, 13)]
    + [f'AC{i:02d}' for i in range(1, 12)]
    + [f'AB{i:02d}' for i in range(1, 11)]
)

# `key <AD01> { [ q, Q ] };`, possibly spread over several lines.
_KEY_RE = re.compile(r'key\s+<(\w+)>\s*\{(.*?)\};', re.S)
# Keysyms are the assigned list: `symbols[1]= [ q, Q ]` in long form, or a bare
# `[ q, Q ]` in compact form. Matching the assignment first keeps the index
# brackets of `type[1]=` / `symbols[Group1]=` out of the result.
_ASSIGNED_SYMS_RE = re.compile(r'=\s*\[([^\]]*)\]')
_BARE_SYMS_RE = re.compile(r'\[([^\]]*)\]')

# A transient failure is retried, but not on every injection: a persistently
# hanging xkbcli would otherwise add its timeout to each one.
_TRANSIENT_RETRY_S = 30.0

_keymap_text_cache: Dict[Tuple[str, str], Optional[str]] = {}
_transient_failures: Dict[Tuple[str, str], float] = {}
_ascii_match_cache: Dict[Tuple[str, str], Optional[bool]] = {}


def compile_keymap(layout: str, variant: str = '', timeout: float = 2.0) -> Optional[str]:
    """Compile an XKB keymap with xkbcli, None when that fails. Settled answers
    are cached (a layout's keymap does not change while we run); a transient
    failure is not, so one bad moment cannot disable layout awareness for good."""
    key = (layout, variant)
    if key in _keymap_text_cache:
        return _keymap_text_cache[key]
    failed_at = _transient_failures.get(key)
    if failed_at is not None and time.monotonic() - failed_at < _TRANSIENT_RETRY_S:
        return None

    cmd = ['xkbcli', 'compile-keymap', '--layout', layout]
    if variant:
        cmd.extend(['--variant', variant])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        _keymap_text_cache[key] = None  # xkbcli is absent; that is settled
        return None
    except Exception:
        _transient_failures[key] = time.monotonic()
        return None  # a timeout or fork failure may not repeat: retry later

    if result.returncode != 0:
        _keymap_text_cache[key] = None  # xkbcli rejects this layout; also settled
        return None
    if not result.stdout.strip():
        _transient_failures[key] = time.monotonic()
        return None

    _keymap_text_cache[key] = result.stdout
    return result.stdout


def _parse_ascii_levels(keymap_text: str) -> Dict[str, Tuple[str, str]]:
    """Map each ASCII-bearing key to its (unshifted, shifted) keysym names."""
    levels = {}
    for name, body in _KEY_RE.findall(keymap_text):
        if name not in _ASCII_KEY_NAMES:
            continue
        match = _ASSIGNED_SYMS_RE.search(body) or _BARE_SYMS_RE.search(body)
        if not match:
            continue
        syms = [sym.strip() for sym in match.group(1).split(',')]
        if syms and syms[0]:
            # A single-level key (space) shifts to itself.
            levels[name] = (syms[0], syms[1] if len(syms) > 1 else syms[0])
    return levels


def ascii_positions_match_us(layout: str, variant: str = '') -> Optional[bool]:
    """True when this layout puts ASCII on the same keys as US QWERTY, on both
    shift levels — the precondition for typing ASCII by keycode (`ydotool type`,
    `wtype -k`), which those tools do from a built-in US table.

    Layouts that merely add levels (pl, ro, lv: AltGr for their own characters)
    pass; anything that moves a key (de's z/y, any Dvorak/Colemak variant,
    us(intl)'s dead keys) does not. None when the keymap won't compile.
    """
    key = (layout, variant)
    if key in _ascii_match_cache:
        return _ascii_match_cache[key]

    match: Optional[bool] = None
    us_keymap = compile_keymap('us')
    active_keymap = compile_keymap(layout, variant)
    if us_keymap and active_keymap:
        us_levels = _parse_ascii_levels(us_keymap)
        # A partly-parsed reference would let a moved key slip through on the
        # keys that did parse, so treat anything short of all 48 as no answer.
        if len(us_levels) == len(_ASCII_KEY_NAMES):
            active_levels = _parse_ascii_levels(active_keymap)
            match = all(
                active_levels.get(name) == syms
                for name, syms in us_levels.items()
            )

    if match is not None:
        _ascii_match_cache[key] = match
    return match


def reset_caches() -> None:
    """Drop cached keymaps (tests; a layout definition cannot change at runtime)."""
    _keymap_text_cache.clear()
    _transient_failures.clear()
    _ascii_match_cache.clear()
