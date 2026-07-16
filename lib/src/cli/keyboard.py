"""
Keyboard device management commands for hyprwhspr
"""

import os
import sys
from typing import Optional

from rich.prompt import Prompt, Confirm
from rich.console import Console
from rich.table import Table

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..global_shortcuts import get_available_keyboards, test_key_accessibility
except ImportError:
    from global_shortcuts import get_available_keyboards, test_key_accessibility

try:
    from ..output_control import (log_info, log_success, log_warning, log_error,
                                  run_command)
except ImportError:
    from output_control import (log_info, log_success, log_warning, log_error,
                                run_command)

from ._shared import SERVICE_NAME
from .systemd import _is_service_running_via_systemd


# ==================== Keyboard Command ====================

def keyboard_command(action: str):
    """Handle keyboard subcommands"""
    if action == 'list':
        list_keyboards()
    elif action == 'test':
        test_keyboard_access()
    elif action == 'configure':
        configure_keyboard_allowlist()
    elif action == 'detect':
        detect_keyboard()
    else:
        log_error(f"Unknown keyboard action: {action}")


# Virtual input devices are created by hyprwhspr/ydotool, not real hardware —
# they must never go in the allowlist. Same tokens list_keyboards() marks.
_VIRTUAL_KEYBOARD_TOKENS = ('hyprwhspr', 'ydotoold', 'uinput')


def _classify_input_devices() -> dict:
    """Map /dev/input/eventN -> {'is_keyboard': bool, 'is_mouse': bool}.

    Uses udev's own ID_INPUT_KEYBOARD / ID_INPUT_MOUSE classification, which is
    far more reliable than counting key capabilities (a fancy mouse can advertise
    keyboard keys). Returns {} if pyudev is unavailable or anything fails;
    callers degrade gracefully.
    """
    try:
        import pyudev
    except (ImportError, ModuleNotFoundError):
        return {}
    result = {}
    try:
        ctx = pyudev.Context()
        for dev in ctx.list_devices(subsystem='input'):
            node = dev.device_node
            if not node or not node.startswith('/dev/input/event'):
                continue
            props = dev.properties
            result[node] = {
                'is_keyboard': props.get('ID_INPUT_KEYBOARD') == '1',
                'is_mouse': props.get('ID_INPUT_MOUSE') == '1',
            }
    except Exception:
        return {}
    return result


def _gather_keyboard_candidates(shortcut) -> list:
    """Deduped keyboard candidates that can emit `shortcut`.

    get_available_keyboards() returns one row per /dev/input/eventN, so the same
    physical keyboard appears several times; we dedup by lowercased name and
    merge udev classification across the device's event nodes. Each item:
    {'name', 'is_keyboard', 'is_mouse', 'is_virtual'}. Sorted real-keyboards
    first, dual-role next, virtual last.
    """
    raw = get_available_keyboards(shortcut)
    classification = _classify_input_devices()
    by_name = {}
    for kb in raw:
        name = kb['name']
        key = name.lower()
        cls = classification.get(kb['path'], {})
        is_virtual = any(tok in key for tok in _VIRTUAL_KEYBOARD_TOKENS)
        entry = by_name.get(key)
        if entry is None:
            by_name[key] = {
                'name': name,
                'is_keyboard': cls.get('is_keyboard', False),
                'is_mouse': cls.get('is_mouse', False),
                'is_virtual': is_virtual,
            }
        else:
            entry['is_keyboard'] = entry['is_keyboard'] or cls.get('is_keyboard', False)
            entry['is_mouse'] = entry['is_mouse'] or cls.get('is_mouse', False)
            entry['is_virtual'] = entry['is_virtual'] or is_virtual

    def _sort_key(c):
        if c['is_virtual']:
            group = 2
        elif c['is_keyboard'] and not c['is_mouse']:
            group = 0
        else:
            group = 1
        return (group, c['name'].lower())

    candidates = sorted(by_name.values(), key=_sort_key)
    return candidates


def _keyboard_preselection(candidates: list, existing_allowlist: list) -> set:
    """Names to preselect.

    - Existing allowlist set -> preselect exactly those names.
    - Otherwise -> preselect pure keyboards (keyboard and not mouse, not virtual).
    - If udev classification was unavailable (nothing classified) -> preselect
      all non-virtual candidates, so the user's real keyboard isn't silently
      dropped (they can deselect mice).
    """
    if existing_allowlist:
        allow_lower = {n.lower() for n in existing_allowlist}
        return {c['name'] for c in candidates if c['name'].lower() in allow_lower}
    classified = any(c['is_keyboard'] or c['is_mouse'] for c in candidates)
    if not classified:
        return {c['name'] for c in candidates if not c['is_virtual']}
    return {c['name'] for c in candidates
            if c['is_keyboard'] and not c['is_mouse'] and not c['is_virtual']}


def _flush_input_buffer():
    """Discard any pending terminal input (best effort).

    During evdev keypress detection the terminal stays in canonical (cooked)
    mode, so the physical keystrokes the user makes are echoed and queued in
    stdin's line buffer in addition to being read from the device. Left there,
    that stray input (e.g. the 'd' typed to enter detect mode, or the keypress
    used for detection) gets consumed by the next Prompt.ask and misread as a
    menu command — most visibly re-triggering detect mode when the user only
    pressed Enter to accept.
    """
    try:
        import termios
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _detect_pressed_keyboard(candidates: list, timeout: float = 5.0):
    """Open candidate devices read-only and return the name of the first to emit
    a key press within `timeout`, or None.

    Read-only (no grab) is safe: under grab_keys=False the service holds no
    exclusive grab, so multiple readers coexist.
    """
    try:
        import time
        import select as _select
        from evdev import InputDevice, ecodes, list_devices as _list_devices
    except Exception:
        return None

    names_lower = {c['name'].lower() for c in candidates}
    fd_to_dev = {}
    opened = []
    try:
        for path in _list_devices():
            try:
                dev = InputDevice(path)
            except Exception:
                continue
            try:
                if dev.name.lower() in names_lower and dev.fd not in fd_to_dev:
                    fd_to_dev[dev.fd] = dev
                    opened.append(dev)
                else:
                    dev.close()
            except Exception:
                try:
                    dev.close()
                except Exception:
                    pass
        if not fd_to_dev:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = _select.select(list(fd_to_dev), [], [], 0.2)
            for fd in ready:
                dev = fd_to_dev.get(fd)
                if dev is None:
                    continue
                try:
                    for event in dev.read():
                        if event.type == ecodes.EV_KEY and event.value == 1:
                            return dev.name
                except Exception:
                    continue
        return None
    finally:
        for dev in opened:
            try:
                dev.close()
            except Exception:
                pass


def _run_keyboard_selection(existing_cfg: Optional[dict] = None):
    """Interactive keyboard-allowlist selection shared by `keyboard configure`
    and `hyprwhspr setup`.

    Returns a list of device names, [] (chose none -> auto-detect), or None
    (skipped / not applicable). Does NOT write config or restart the service.
    """
    existing_cfg = existing_cfg or {}
    shortcut = existing_cfg.get('primary_shortcut') or 'Super+Alt+D'
    existing_allowlist = existing_cfg.get('keyboard_device_names') or []

    print("\n" + "=" * 60)
    print("Keyboard Devices")
    print("=" * 60)
    print(f"\nhyprwhspr listens to specific keyboards to detect your shortcut ({shortcut}).")
    print("Listening to the right keyboards:")
    print("  - keeps the shortcut working after unplug/replug, docking and")
    print("    suspend (enables hotplug re-attach for the listed devices), and")
    print("  - stops mouse-like devices from being grabbed by accident.")
    print("\nThe keyboards marked active below are the recommended set — you can")
    print("accept them as-is or adjust the list.")

    candidates = _gather_keyboard_candidates(shortcut)
    if not candidates:
        log_warning("No accessible keyboard devices found.")
        log_info("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
        return None

    # Seed synthetic rows for allowlist names not currently present, so a re-run
    # doesn't silently prune an unplugged/docked keyboard.
    present_lower = {c['name'].lower() for c in candidates}
    for name in existing_allowlist:
        if name.lower() not in present_lower:
            candidates.append({
                'name': name, 'is_keyboard': True, 'is_mouse': False,
                'is_virtual': False, 'absent': True,
            })

    selected = _keyboard_preselection(candidates, existing_allowlist)
    # Synthetic absent rows are part of the existing allowlist -> keep selected.
    selected.update(c['name'] for c in candidates if c.get('absent'))

    interactive = (sys.stdin.isatty() and sys.stdout.isatty()
                   and not os.environ.get('HYPRWHSPR_NONINTERACTIVE'))
    if not interactive:
        print("\nNon-interactive session; leaving keyboard allowlist unchanged.")
        return None

    _print_keyboard_table(candidates, selected)

    if not selected:
        # Nothing recommended (e.g. all dual-role/virtual) — there's no sensible
        # "use the 0 recommended" question, so go straight to the editor.
        return _edit_keyboard_selection(candidates, selected)

    if Confirm.ask(f"\nListen to the {len(selected)} recommended keyboard(s)?",
                   default=True):
        # Preserve candidate order for deterministic, readable config.
        return [c['name'] for c in candidates if c['name'] in selected]
    return _edit_keyboard_selection(candidates, selected)


def _print_keyboard_table(candidates: list, selected: set):
    """Show the candidate keyboards and which ones will be listened to."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Device")
    table.add_column("Status", no_wrap=True)
    for i, c in enumerate(candidates, 1):
        name = c['name']
        if c.get('absent'):
            status = "[yellow]✓ listen (not connected)[/]"
        elif name in selected:
            status = "[green]✓ listen[/]"
        elif c['is_virtual']:
            status = "[dim]– virtual[/]"
        elif c['is_keyboard'] and c['is_mouse']:
            status = "[dim]– also a mouse[/]"
        elif c['is_mouse']:
            status = "[dim]– mouse[/]"
        else:
            status = "[dim]– off[/]"
        table.add_row(str(i), name, status)
    Console().print(table)


def _edit_keyboard_selection(candidates: list, selected: set):
    """Let the user enter the exact set of keyboards to listen to (by number).

    Replace semantics: the numbers entered become the full set. The default is
    the current recommendation, so a plain Enter keeps it. '0' = listen to none
    (auto-detect / legacy mode). Returns names in candidate order, or [].
    """
    default_nums = ",".join(str(i) for i, c in enumerate(candidates, 1)
                            if c['name'] in selected)
    print("\nEnter the numbers of ALL keyboards to listen to (comma/space-separated).")
    print("  Enter = keep the recommendation · 0 = none (auto-detect)")
    while True:
        raw = Prompt.ask("Numbers", default=default_nums).strip()
        if raw == "" and default_nums == "":
            return []
        if raw == "0":
            return []
        chosen = set()
        for tok in raw.replace(',', ' ').split():
            if not tok.isdigit():
                continue
            idx = int(tok) - 1
            if 0 <= idx < len(candidates):
                chosen.add(candidates[idx]['name'])
        if not chosen:
            log_warning("Enter valid device numbers, or 0 for none.")
            continue
        return [c['name'] for c in candidates if c['name'] in chosen]


def detect_keyboard():
    """`hyprwhspr keyboard detect` — identify which device a keypress comes from.

    Purely informational: it reads devices read-only (no grab, no injection) and
    reports the device a key was pressed on, plus its number in
    `keyboard configure`. Handy when device names are cryptic.
    """
    config = ConfigManager()
    shortcut = config.get_setting("primary_shortcut", "Super+Alt+D")
    candidates = _gather_keyboard_candidates(shortcut)
    if not candidates:
        log_warning("No accessible keyboard devices found.")
        log_info("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
        return

    print("\nPress a key on the keyboard you use for the shortcut (5s timeout)...")
    detected = _detect_pressed_keyboard(candidates)
    # Drop keystrokes echoed into the tty during the detection window so they
    # don't leak into the shell afterwards.
    _flush_input_buffer()
    if not detected:
        log_warning("No keypress detected (timed out).")
        return

    log_success(f"Key detected on: {detected}")
    for i, c in enumerate(candidates, 1):
        if c['name'] == detected:
            log_info(f"That's #{i} in 'hyprwhspr keyboard configure'.")
            break


def configure_keyboard_allowlist():
    """`hyprwhspr keyboard configure` — choose which keyboards hyprwhspr listens
    to, save the allowlist, and restart the service so it takes effect.
    """
    config = ConfigManager()
    existing_cfg = config.get_all_settings()
    choice = _run_keyboard_selection(existing_cfg)
    if choice is None:
        return

    config.set_setting('keyboard_device_names', choice or None)
    config.save_config()
    if choice:
        log_success(f"Saved keyboard allowlist ({len(choice)} device(s)):")
        for name in choice:
            print(f"  - {name}")
    else:
        log_info("Cleared keyboard allowlist — using auto-detection.")

    # The allowlist only takes effect when the service (re)starts.
    if _hyprwhspr_service_active():
        if Confirm.ask("\nRestart hyprwhspr now to apply?", default=True):
            try:
                run_command(['systemctl', '--user', 'restart', SERVICE_NAME], check=False)
                log_success("hyprwhspr restarted — the shortcut now uses the new selection.")
            except Exception as e:
                log_error(f"Could not restart service: {e}")
                log_info("Restart manually: systemctl --user restart hyprwhspr.service")
        else:
            log_info("Restart later to apply: systemctl --user restart hyprwhspr.service")
    else:
        log_info("Start/restart hyprwhspr to apply: "
                 "systemctl --user restart hyprwhspr.service")


def _hyprwhspr_service_active() -> bool:
    """True if the hyprwhspr user service is currently active."""
    return _is_service_running_via_systemd()


def list_keyboards():
    """List available keyboard devices"""
    log_info("Discovering available keyboard devices...")
    
    try:
        # Get current config to show selected device
        config = ConfigManager()
        shortcut = config.get_setting("primary_shortcut", "Super+Alt+D")
        selected_device_name = config.get_setting("selected_device_name", None)
        selected_device_path = config.get_setting("selected_device_path", None)
        keyboard_device_names = config.get_setting("keyboard_device_names", None) or []
        allowlist_lower = [n.lower() for n in keyboard_device_names]
        
        # Get available keyboards
        keyboards = get_available_keyboards(shortcut)
        
        if not keyboards:
            log_warning("No accessible keyboard devices found")
            log_info("Make sure you're in the 'input' group: sudo usermod -aG input $USER")
            return
        
        print("\nAvailable keyboard devices:")
        print("-" * 70)
        
        # Find which device would actually be selected (matching GlobalShortcuts logic)
        selected_device_index = None
        if selected_device_name:
            search_name_lower = selected_device_name.lower()
            for i, kb in enumerate(keyboards):
                kb_name_lower = kb['name'].lower()
                if kb_name_lower == search_name_lower:
                    selected_device_index = i
                    break  # Use first match, same as GlobalShortcuts
        elif selected_device_path:
            for i, kb in enumerate(keyboards):
                if kb['path'] == selected_device_path:
                    selected_device_index = i
                    break
        
        for i, kb in enumerate(keyboards, 1):
            name_lower = kb['name'].lower()
            markers = []
            if (i - 1) == selected_device_index:
                markers.append("SELECTED")
            if allowlist_lower and name_lower in allowlist_lower:
                markers.append("ALLOWED")
            # Virtual devices aren't real hardware — they're created by
            # hyprwhspr itself or by ydotool. Don't put them in your allowlist.
            if ('hyprwhspr' in name_lower
                    or 'ydotoold' in name_lower
                    or 'uinput' in name_lower):
                markers.append("VIRTUAL")
            marker_str = f" [{' '.join(markers)}]" if markers else ""
            print(f"  {i}. {kb['name']}")
            print(f"     Path: {kb['path']}{marker_str}")
        
        print("-" * 70)
        print(f"\nTotal: {len(keyboards)} accessible device(s)")
        
        if selected_device_name:
            print(f"\nCurrently selected by name: '{selected_device_name}'")
        elif selected_device_path:
            print(f"\nCurrently selected by path: {selected_device_path}")
        elif keyboard_device_names:
            print(f"\nAllowlist active (keyboard_device_names), {len(keyboard_device_names)} device(s):")
            for name in keyboard_device_names:
                print(f"  - {name}")
            print("Hotplug detection enabled for listed devices.")
            # Surface allowlist entries that don't match any present device —
            # helps the user catch typos vs. just-unplugged devices.
            present_names = {kb['name'].lower() for kb in keyboards}
            missing = [n for n in keyboard_device_names if n.lower() not in present_names]
            if missing:
                print("\nAllowlist entries not currently present on this system:")
                for name in missing:
                    print(f"  - {name}")
                print("  (These may just be unplugged; if so, they'll be grabbed when plugged in.)")
        else:
            print("\nNo specific device selected — using auto-detection.")
            # Point the user at the allowlist in case auto-detection grabs a
            # mouse or media controller. Use a real device name from this
            # system as the example so it's obvious how to populate the list.
            real_candidates = [kb for kb in keyboards
                               if 'hyprwhspr' not in kb['name'].lower()
                               and 'ydotoold' not in kb['name'].lower()
                               and 'uinput' not in kb['name'].lower()]
            example_name = real_candidates[0]['name'] if real_candidates else "My Keyboard"
            print("\nTo enable keyboard hotplug detection (useful for laptops that dock)")
            print("or to restrict grabbing when auto-detection grabs a mouse-like device,")
            print("set an allowlist in ~/.config/hyprwhspr/config.json:")
            print('  "keyboard_device_names": [')
            print(f'    "{example_name}"')
            print('  ]')
            print("(Also enables hotplug for listed devices plugged in after startup.)")
        
        print("\nOther single-device overrides (take priority over the allowlist):")
        print('  "selected_device_name": "Device Name"')
        print('  "selected_device_path": "/dev/input/eventX"')
        
    except Exception as e:
        log_error(f"Error listing keyboards: {e}")
        import traceback
        traceback.print_exc()


def test_keyboard_access():
    """Test keyboard device accessibility"""
    log_info("Testing keyboard device accessibility...")
    
    try:
        results = test_key_accessibility()
        
        print("\n" + "=" * 70)
        print("Keyboard Device Accessibility Test")
        print("=" * 70)
        
        print(f"\nTotal devices found: {results['total_devices']}")
        print(f"Accessible devices: {len(results['accessible_devices'])}")
        print(f"Inaccessible devices: {len(results['inaccessible_devices'])}")
        
        if results['accessible_devices']:
            print("\n✓ Accessible devices:")
            for dev in results['accessible_devices']:
                print(f"  - {dev['name']}")
                print(f"    Path: {dev['path']}")
        
        if results['inaccessible_devices']:
            print("\n✗ Inaccessible devices:")
            for dev in results['inaccessible_devices']:
                print(f"  - {dev['name']}")
                print(f"    Path: {dev['path']}")
            print("\nNote: Inaccessible devices may be in use by another process")
            print("      (e.g., Espanso, keyd, kmonad) or require permissions")
        
        if not results['accessible_devices']:
            print("\n⚠ No accessible devices found!")
            print("Solutions:")
            print("  1. Add yourself to 'input' group: sudo usermod -aG input $USER")
            print("     (then log out and back in)")
            print("  2. Check if devices are grabbed by other tools:")
            print("     sudo fuser /dev/input/event*")
            print("  3. Consider using 'selected_device_name' in config to avoid conflicts")
        
    except Exception as e:
        log_error(f"Error testing keyboard access: {e}")
        import traceback
        traceback.print_exc()
