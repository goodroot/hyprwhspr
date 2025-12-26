"""
Global shortcuts handler for hyprwhspr
Manages system-wide keyboard shortcuts for dictation control
"""

import sys
import threading
import select
import time
from typing import Callable, Optional, List, Set, Dict
from pathlib import Path

try:
    import evdev
    from evdev import InputDevice, categorize, ecodes, UInput
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-evdev is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pacman -S python-evdev    # system-wide on Arch", file=sys.stderr)
    sys.exit(1)


# Key aliases mapping to evdev KEY_* constants
KEY_ALIASES: dict[str, str] = {
    # Left-side modifiers
    'ctrl': 'KEY_LEFTCTRL', 'control': 'KEY_LEFTCTRL', 'lctrl': 'KEY_LEFTCTRL',
    'alt': 'KEY_LEFTALT', 'lalt': 'KEY_LEFTALT',
    'shift': 'KEY_LEFTSHIFT', 'lshift': 'KEY_LEFTSHIFT',
    'super': 'KEY_LEFTMETA', 'meta': 'KEY_LEFTMETA', 'lsuper': 'KEY_LEFTMETA',
    'win': 'KEY_LEFTMETA', 'windows': 'KEY_LEFTMETA', 'cmd': 'KEY_LEFTMETA',
    
    # Right-side modifiers
    'rctrl': 'KEY_RIGHTCTRL', 'rightctrl': 'KEY_RIGHTCTRL',
    'ralt': 'KEY_RIGHTALT', 'rightalt': 'KEY_RIGHTALT',
    'rshift': 'KEY_RIGHTSHIFT', 'rightshift': 'KEY_RIGHTSHIFT',
    'rsuper': 'KEY_RIGHTMETA', 'rightsuper': 'KEY_RIGHTMETA', 'rmeta': 'KEY_RIGHTMETA',
    
    # Common special keys
    'enter': 'KEY_ENTER', 'return': 'KEY_ENTER',
    'backspace': 'KEY_BACKSPACE', 'bksp': 'KEY_BACKSPACE',
    'tab': 'KEY_TAB',
    'caps': 'KEY_CAPSLOCK', 'capslock': 'KEY_CAPSLOCK',
    'esc': 'KEY_ESC', 'escape': 'KEY_ESC',
    'space': 'KEY_SPACE', 'spacebar': 'KEY_SPACE',
    'delete': 'KEY_DELETE', 'del': 'KEY_DELETE',
    'insert': 'KEY_INSERT', 'ins': 'KEY_INSERT',
    'home': 'KEY_HOME',
    'end': 'KEY_END',
    'pageup': 'KEY_PAGEUP', 'pgup': 'KEY_PAGEUP',
    'pagedown': 'KEY_PAGEDOWN', 'pgdn': 'KEY_PAGEDOWN', 'pgdown': 'KEY_PAGEDOWN',
    
    # Arrow keys
    'up': 'KEY_UP', 'uparrow': 'KEY_UP',
    'down': 'KEY_DOWN', 'downarrow': 'KEY_DOWN',
    'left': 'KEY_LEFT', 'leftarrow': 'KEY_LEFT',
    'right': 'KEY_RIGHT', 'rightarrow': 'KEY_RIGHT',
    
    # Lock keys
    'numlock': 'KEY_NUMLOCK',
    'scrolllock': 'KEY_SCROLLLOCK', 'scroll': 'KEY_SCROLLLOCK',
    
    # Function keys (f1-f24)
    'f1': 'KEY_F1', 'f2': 'KEY_F2', 'f3': 'KEY_F3', 'f4': 'KEY_F4',
    'f5': 'KEY_F5', 'f6': 'KEY_F6', 'f7': 'KEY_F7', 'f8': 'KEY_F8',
    'f9': 'KEY_F9', 'f10': 'KEY_F10', 'f11': 'KEY_F11', 'f12': 'KEY_F12',
    'f13': 'KEY_F13', 'f14': 'KEY_F14', 'f15': 'KEY_F15', 'f16': 'KEY_F16',
    'f17': 'KEY_F17', 'f18': 'KEY_F18', 'f19': 'KEY_F19', 'f20': 'KEY_F20',
    'f21': 'KEY_F21', 'f22': 'KEY_F22', 'f23': 'KEY_F23', 'f24': 'KEY_F24',
    
    # Numpad keys
    'kp0': 'KEY_KP0', 'kp1': 'KEY_KP1', 'kp2': 'KEY_KP2', 'kp3': 'KEY_KP3',
    'kp4': 'KEY_KP4', 'kp5': 'KEY_KP5', 'kp6': 'KEY_KP6', 'kp7': 'KEY_KP7',
    'kp8': 'KEY_KP8', 'kp9': 'KEY_KP9',
    'kpenter': 'KEY_KPENTER', 'kpplus': 'KEY_KPPLUS', 'kpminus': 'KEY_KPMINUS',
    'kpmultiply': 'KEY_KPASTERISK', 'kpdivide': 'KEY_KPSLASH',
    'kpdot': 'KEY_KPDOT', 'kpperiod': 'KEY_KPDOT',
    
    # Media keys
    'mute': 'KEY_MUTE', 'volumemute': 'KEY_MUTE',
    'volumeup': 'KEY_VOLUMEUP', 'volup': 'KEY_VOLUMEUP',
    'volumedown': 'KEY_VOLUMEDOWN', 'voldown': 'KEY_VOLUMEDOWN',
    'play': 'KEY_PLAYPAUSE', 'playpause': 'KEY_PLAYPAUSE',
    'stop': 'KEY_STOPCD', 'mediastop': 'KEY_STOPCD',
    'nextsong': 'KEY_NEXTSONG', 'next': 'KEY_NEXTSONG',
    'previoussong': 'KEY_PREVIOUSSONG', 'prev': 'KEY_PREVIOUSSONG',
    
    # Browser keys (for keyboards with browser control buttons)
    'browser': 'KEY_WWW',
    'browserback': 'KEY_BACK',
    'browserforward': 'KEY_FORWARD',
    'refresh': 'KEY_REFRESH',
    'browsersearch': 'KEY_SEARCH',
    'favorites': 'KEY_BOOKMARKS',
    
    # System keys
    'menu': 'KEY_MENU',
    'print': 'KEY_PRINT', 'printscreen': 'KEY_SYSRQ', 'prtsc': 'KEY_SYSRQ',
    'pause': 'KEY_PAUSE', 'break': 'KEY_PAUSE',

    # Punctuation and symbol keys
    '.': 'KEY_DOT', 'dot': 'KEY_DOT', 'period': 'KEY_DOT',
    ',': 'KEY_COMMA', 'comma': 'KEY_COMMA',
    '/': 'KEY_SLASH', 'slash': 'KEY_SLASH',
    '\\': 'KEY_BACKSLASH', 'backslash': 'KEY_BACKSLASH',
    ';': 'KEY_SEMICOLON', 'semicolon': 'KEY_SEMICOLON',
    "'": 'KEY_APOSTROPHE', 'apostrophe': 'KEY_APOSTROPHE', 'quote': 'KEY_APOSTROPHE',
    '[': 'KEY_LEFTBRACE', 'leftbrace': 'KEY_LEFTBRACE', 'lbrace': 'KEY_LEFTBRACE',
    ']': 'KEY_RIGHTBRACE', 'rightbrace': 'KEY_RIGHTBRACE', 'rbrace': 'KEY_RIGHTBRACE',
    '-': 'KEY_MINUS', 'minus': 'KEY_MINUS', 'dash': 'KEY_MINUS',
    '=': 'KEY_EQUAL', 'equal': 'KEY_EQUAL', 'equals': 'KEY_EQUAL',
    '`': 'KEY_GRAVE', 'grave': 'KEY_GRAVE', 'backtick': 'KEY_GRAVE',

    # Number keys (top row)
    '0': 'KEY_0', '1': 'KEY_1', '2': 'KEY_2', '3': 'KEY_3', '4': 'KEY_4',
    '5': 'KEY_5', '6': 'KEY_6', '7': 'KEY_7', '8': 'KEY_8', '9': 'KEY_9',

    # Letter keys (for completeness - allows lowercase in config)
    'a': 'KEY_A', 'b': 'KEY_B', 'c': 'KEY_C', 'd': 'KEY_D', 'e': 'KEY_E',
    'f': 'KEY_F', 'g': 'KEY_G', 'h': 'KEY_H', 'i': 'KEY_I', 'j': 'KEY_J',
    'k': 'KEY_K', 'l': 'KEY_L', 'm': 'KEY_M', 'n': 'KEY_N', 'o': 'KEY_O',
    'p': 'KEY_P', 'q': 'KEY_Q', 'r': 'KEY_R', 's': 'KEY_S', 't': 'KEY_T',
    'u': 'KEY_U', 'v': 'KEY_V', 'w': 'KEY_W', 'x': 'KEY_X', 'y': 'KEY_Y',
    'z': 'KEY_Z',
}


class GlobalShortcuts:
    """Handles global keyboard shortcuts using evdev for hardware-level capture"""

    def __init__(self, primary_key: str = '<f12>', callback: Optional[Callable] = None, release_callback: Optional[Callable] = None, device_path: Optional[str] = None, grab_keys: bool = True):
        self.primary_key = primary_key
        self.callback = callback
        self.selected_device_path = device_path
        self.release_callback = release_callback
        self.grab_keys = grab_keys

        # Device and event handling
        self.devices = []
        self.device_fds = {}
        self.listener_thread = None
        self.is_running = False
        self.stop_event = threading.Event()

        # Virtual keyboard for re-emitting non-shortcut keys
        self.uinput = None
        self.devices_grabbed = False

        # State tracking
        self.pressed_keys = set()
        self.last_trigger_time = 0
        self.debounce_time = 0.1  # 100ms debounce - shorter for push-to-talk responsiveness
        self.combination_active = False  # Track if full combination is currently active
        self.last_release_time = 0  # Debounce for release events

        # Track which keys are currently being suppressed (part of active shortcut)
        self.suppressed_keys = set()

        # Parse the primary key combination
        self.target_keys = self._parse_key_combination(primary_key)

        # Initialize keyboard devices
        self._discover_keyboards()
        
    def _discover_keyboards(self):
        """Discover and initialize input devices that can emit the configured shortcut"""
        self.devices = []
        self.device_fds = {}
        
        try:
            # Find all input devices
            all_device_paths = evdev.list_devices()
            devices = [evdev.InputDevice(path) for path in all_device_paths]
            
            # If a specific device path is selected, only use that device and skip auto-detection
            if self.selected_device_path:
                selected_device = None
                for device in devices:
                    if device.path == self.selected_device_path:
                        selected_device = device
                    else:
                        # Close devices that don't match the selected path
                        device.close()
                
                if selected_device is None:
                    print(f"[WARN] Selected device {self.selected_device_path} not found!")
                    return
                
                devices = [selected_device]
            
            for device in devices:
                # Require EV_KEY events
                capabilities = device.capabilities()
                if ecodes.EV_KEY not in capabilities:
                    if self.selected_device_path:
                        print(f"[ERROR] Selected device '{device.name}' ({device.path}) does not support keyboard events (EV_KEY)")
                        print("[ERROR] This device cannot be used for keyboard shortcuts")
                        device.close()
                        return
                    device.close()
                    continue
                
                # Check that device can emit ALL keys required for the shortcut
                available_keys = set(capabilities[ecodes.EV_KEY])
                if not self.target_keys.issubset(available_keys):
                    if self.selected_device_path:
                        missing_keys = self.target_keys - available_keys
                        missing_key_names = [self._keycode_to_name(k) for k in missing_keys]
                        print(f"[ERROR] Selected device '{device.name}' ({device.path}) cannot emit all keys required for shortcut '{self.primary_key}'")
                        print(f"[ERROR] Missing keys: {', '.join(missing_key_names)}")
                        print("[ERROR] This device is incompatible with the configured shortcut")
                        device.close()
                        return
                    device.close()
                    continue
                
                # Device can emit all required keys - test accessibility
                # When grab_keys is true, test grab capability; when false, just test read access
                if self.grab_keys:
                    # Retry logic to handle cases where device is temporarily busy (e.g., during service restart)
                    grab_test_success = False
                    for retry in range(2):
                        try:
                            device.grab()
                            device.ungrab()
                            grab_test_success = True
                            break
                        except (OSError, IOError) as e:
                            if retry < 1:  # Don't sleep on last retry
                                # Device might be busy from previous process - wait a bit
                                time.sleep(0.05)
                                continue
                            # Last retry failed - this is a real error
                            if self.selected_device_path:
                                print(f"[ERROR] Cannot access selected device '{device.name}' ({device.path}): {e}")
                                print("[ERROR] This usually means you need root or input group membership")
                                print("[ERROR]   Run: sudo usermod -aG input $USER (then log out and back in)")
                                device.close()
                                return
                            print(f"[WARN] Cannot access device {device.name}: {e}")
                            print("[WARN]   This usually means you need root or input group membership")
                            device.close()
                            break
                    
                    if not grab_test_success:
                        continue
                else:
                    # When grab_keys is false, still test if we can read from the device
                    # This prevents adding inaccessible devices that will error in _event_loop()
                    try:
                        # Try to read capabilities as a basic accessibility test
                        # This will fail if we don't have read permission
                        device.capabilities()
                    except (OSError, IOError) as e:
                        if self.selected_device_path:
                            print(f"[ERROR] Cannot access selected device '{device.name}' ({device.path}): {e}")
                            print("[ERROR] This usually means you need root or input group membership")
                            print("[ERROR]   Run: sudo usermod -aG input $USER (then log out and back in)")
                            device.close()
                            return
                        print(f"[WARN] Cannot access device {device.name}: {e}")
                        print("[WARN]   This usually means you need root or input group membership")
                        device.close()
                        continue
                
                # Device is usable - add it
                self.devices.append(device)
                self.device_fds[device.fd] = device
                
                # If we selected a specific device, we're done
                if self.selected_device_path:
                    break
                        
        except Exception as e:
            print(f"[ERROR] Error discovering devices: {e}")
            import traceback
            traceback.print_exc()
            
        if not self.devices:
            if self.selected_device_path:
                # This shouldn't happen if we handled all cases above, but just in case
                print(f"[ERROR] Selected device {self.selected_device_path} could not be initialized")
            else:
                print("[ERROR] No accessible devices found that can emit the configured shortcut!")
                print("[ERROR] Solutions:")
                print("[ERROR]   1. Add yourself to 'input' group: sudo usermod -aG input $USER")
                print("[ERROR]   2. Disable key grabbing in config (grab_keys: false)")
                print(f"[ERROR]   3. Check that your shortcut '{self.primary_key}' uses keys available on your keyboard")
    
    def _parse_key_combination(self, key_string: str) -> Set[int]:
        """Parse a key combination string into a set of evdev key codes"""
        keys = set()
        key_lower = key_string.lower().strip()
        
        # Remove angle brackets if present
        key_lower = key_lower.replace('<', '').replace('>', '')
        
        # Split into parts for modifier + key combinations
        parts = key_lower.split('+')
        
        for part in parts:
            part = part.strip()
            keycode = self._string_to_keycode(part)
            if keycode is not None:
                keys.add(keycode)
            else:
                print(f"Warning: Could not parse key '{part}' in '{key_string}'")
                
        # Default to F12 if no keys parsed
        if not keys:
            print(f"Warning: Could not parse key combination '{key_string}', defaulting to F12")
            keys.add(ecodes.KEY_F12)
            
        return keys
    
    def _string_to_keycode(self, key_string: str) -> Optional[int]:
        """Convert a human-friendly key string into an evdev keycode.
        
        Tries local aliases first, then falls back to evdev-style KEY_* names.
        This hybrid approach supports both user-friendly names (ctrl, super, etc.)
        and direct evdev key names (KEY_COMMA, KEY_1, etc.).
        
        Returns None if no matching keycode is found.
        """
        original = key_string
        key_string = key_string.lower().strip()
        
        # 1. Try alias mapping first, easy names
        if key_string in KEY_ALIASES:
            key_name = KEY_ALIASES[key_string]
        else:
            # 2. Try as direct evdev KEY_* name
            # Can use any evdev key name directly
            key_name = key_string.upper()
            if not key_name.startswith('KEY_'):
                key_name = f'KEY_{key_name}'
        
        # 3. Look up the keycode in evdev's complete mapping
        code = ecodes.ecodes.get(key_name)

        if code is None:
            print(f"Warning: Unknown key string '{original}' (resolved to '{key_name}')")
            return None
        
        return code
    
    def _keycode_to_name(self, keycode: int) -> str:
        """Convert evdev keycode to human readable name"""
        try:
            key_name = ecodes.KEY[keycode]
            # Handle case where evdev returns a tuple of multiple event codes
            if isinstance(key_name, tuple):
                key_name = key_name[0]
            return key_name.replace('KEY_', '')
        except KeyError:
            return f"KEY_{keycode}"
    
    def _event_loop(self):
        """Main event loop for processing keyboard events"""
        try:
            while not self.stop_event.is_set():
                if not self.devices:
                    time.sleep(0.1)
                    continue
                    
                # Use select to wait for events from any device
                device_fds = [dev.fd for dev in self.devices]
                ready_fds, _, _ = select.select(device_fds, [], [], 0.1)
                
                # Don't log every event batch - too verbose
                # Only log if we're debugging a specific issue
                
                for fd in ready_fds:
                    if fd in self.device_fds:
                        device = self.device_fds[fd]
                        try:
                            # device.read() returns a generator, convert to list
                            events = list(device.read())
                            for event in events:
                                self._process_event(event)
                        except (OSError, IOError) as e:
                            # Device disconnected or error
                            print(f"[ERROR] Lost connection to device: {device.name}: {e}")
                            self._remove_device(device)
                            
        except Exception as e:
            print(f"[ERROR] Error in keyboard event loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always cleanup grabs if thread crashes
            # This prevents permanent keyboard lockout
            # Call unconditionally - _cleanup_key_grabbing() already guards internally
            print("[CLEANUP] Event loop exiting - cleaning up device grabs")
            self._cleanup_key_grabbing()
        
    def _remove_device(self, device: InputDevice):
        """Remove a disconnected device from monitoring"""
        try:
            # Always try to ungrab before closing
            # Don't rely on devices_grabbed flag
            # It ensures cleanup even if flag is cleared early
            try:
                device.ungrab()
            except:
                pass  # Device may already be ungrab or closed
            
            if device in self.devices:
                self.devices.remove(device)
            if device.fd in self.device_fds:
                del self.device_fds[device.fd]
            device.close()
        except:
            pass
    
    # Modifier keys that should never get "stuck" - always pass through releases
    MODIFIER_KEYS = {
        ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL,
        ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT,
        ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT,
        ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA,
    }

    def _process_event(self, event):
        """Process individual keyboard events"""
        if event.type == ecodes.EV_KEY:
            try:
                key_event = categorize(event)
            except KeyError:
                # Unknown key code (e.g., 254) - skip this event
                return
            key_name = self._keycode_to_name(event.code)
            should_suppress = False

            if key_event.keystate == key_event.key_down:
                # Key pressed
                self.pressed_keys.add(event.code)

                # Only suppress if pressing this key COMPLETES the shortcut combination
                # AND no extra modifiers are held (e.g., don't suppress for SUPER+SHIFT+.)
                if event.code in self.target_keys and self.target_keys.issubset(self.pressed_keys):
                    extra_modifiers = (self.pressed_keys - self.target_keys) & self.MODIFIER_KEYS
                    if len(extra_modifiers) == 0 and event.code not in self.MODIFIER_KEYS:
                        # Full shortcut with no extra modifiers - suppress this completing key
                        should_suppress = True
                        self.suppressed_keys.add(event.code)

                self._check_shortcut_combination()

            elif key_event.keystate == key_event.key_up:
                # Key released
                was_combination_active = self.combination_active
                self.pressed_keys.discard(event.code)

                # If this key release completes the combination release, clear suppressed_keys
                # This ensures key releases are re-emitted in auto mode when tap starts recording
                # Prevents keyboard lockout - keys must be re-emitted so system knows they're released
                if was_combination_active and event.code in self.target_keys:
                    # Combination is being released - clear all suppressed target keys so releases are re-emitted
                    for key_code in self.target_keys:
                        if key_code in self.suppressed_keys:
                            self.suppressed_keys.discard(key_code)

                # If this key was suppressed, suppress its release too
                # But NEVER suppress modifier key releases to prevent stuck keys
                if event.code in self.suppressed_keys:
                    self.suppressed_keys.discard(event.code)
                    if event.code not in self.MODIFIER_KEYS:
                        should_suppress = True

                self._check_combination_release(was_combination_active)

            elif key_event.keystate == 2:  # Key repeat
                # Suppress repeats for suppressed keys
                if event.code in self.suppressed_keys:
                    should_suppress = True

            # Re-emit non-suppressed key events to virtual keyboard
            if self.uinput and self.devices_grabbed and not should_suppress:
                try:
                    self.uinput.write(ecodes.EV_KEY, event.code, event.value)
                    self.uinput.syn()
                except Exception as e:
                    print(f"Warning: Failed to re-emit key: {e}")

        elif self.uinput and self.devices_grabbed:
            # Pass through non-key events (like EV_SYN, EV_MSC, etc.)
            try:
                self.uinput.write(event.type, event.code, event.value)
            except:
                pass
    
    def _check_shortcut_combination(self):
        """Check if current pressed keys match target combination"""
        # Check if target keys are pressed
        if not self.target_keys.issubset(self.pressed_keys):
            keys_match = False
        else:
            # Target keys are pressed - but check for unwanted extra modifiers
            # If user presses SUPER+SHIFT+. but shortcut is SUPER+., don't trigger
            extra_keys = self.pressed_keys - self.target_keys
            extra_modifiers = extra_keys & self.MODIFIER_KEYS
            # Only match if no extra modifiers are pressed
            keys_match = len(extra_modifiers) == 0
        
        if keys_match:
            current_time = time.time()
            
            # Only trigger if not already active and debounce time has passed
            if not self.combination_active and (current_time - self.last_trigger_time > self.debounce_time):
                self.last_trigger_time = current_time
                self.combination_active = True
                self._trigger_callback()
        else:
            self.combination_active = False
    
    def _trigger_callback(self):
        """Trigger the callback function"""
        if self.callback:
            try:
                # Run callback in a separate thread to avoid blocking the listener
                callback_thread = threading.Thread(target=self.callback, daemon=True)
                callback_thread.start()
            except Exception as e:
                print(f"[ERROR] Error calling shortcut callback: {e}")
                import traceback
                traceback.print_exc()

    def _check_combination_release(self, was_combination_active: bool):
        """Check if combination was released and trigger release callback"""
        if was_combination_active and not self.target_keys.issubset(self.pressed_keys):
            current_time = time.time()
            
            # Implement debouncing for release events
            if current_time - self.last_release_time > self.debounce_time:
                self.last_release_time = current_time
                self.combination_active = False
                self._trigger_release_callback()
    
    def _trigger_release_callback(self):
        """Trigger the release callback function"""
        if self.release_callback:
            try:
                # Run callback in a separate thread to avoid blocking the listener
                callback_thread = threading.Thread(target=self.release_callback, daemon=True)
                callback_thread.start()
            except Exception as e:
                print(f"[ERROR] Error calling shortcut release callback: {e}")
                import traceback
                traceback.print_exc()
    
    def start(self) -> bool:
        """Start listening for global shortcuts"""
        if self.is_running:
            return True

        # Rediscover keyboards if devices list is empty
        if not self.devices:
            print("Rediscovering keyboard devices...")
            self._discover_keyboards()

        if not self.devices:
            print("No keyboard devices available")
            return False

        try:
            # Set up key grabbing if enabled
            if self.grab_keys:
                if not self._setup_key_grabbing():
                    print("[ERROR] Failed to set up key grabbing - cannot start shortcuts")
                    return False

            self.stop_event.clear()
            self.listener_thread = threading.Thread(target=self._event_loop, daemon=True)
            self.listener_thread.start()
            self.is_running = True

            return True

        except Exception as e:
            print(f"[ERROR] Failed to start global shortcuts: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_key_grabbing()
            return False

    def _setup_key_grabbing(self) -> bool:
        """Set up UInput virtual keyboard and grab physical devices
        
        Returns:
            True if at least one device was grabbed, False otherwise
        """
        try:
            # Create a virtual keyboard that can emit all key events
            # This will re-emit keys that aren't part of our shortcut
            self.uinput = UInput(name="hyprwhspr-virtual-keyboard")

            # Grab all keyboard devices to intercept their events
            # Use retry logic to handle cases where devices are temporarily busy (e.g., during service restart)
            # When a service restarts quickly, the kernel may not have released device grabs yet
            grabbed_count = 0
            for device in self.devices:
                for retry in range(3):  # Increased retries for better recovery
                    try:
                        device.grab()
                        grabbed_count += 1
                        break
                    except (OSError, IOError) as e:
                        if retry < 2:  # Sleep on first two retries
                            # Device might be busy from previous process - wait progressively longer
                            # First retry: 0.1s, second retry: 0.2s
                            time.sleep(0.1 * (retry + 1))
                            continue
                        # Last retry failed - this is a real error
                        print(f"[ERROR] Could not grab {device.name} after {retry + 1} retries: {e}")
                        print(f"[ERROR] Device may still be grabbed by previous process.")
                        print(f"[ERROR] Check what's using it: sudo fuser {device.path}")
                        print(f"[ERROR] If needed, kill the process: sudo fuser -k {device.path} (WARNING: kills all processes using this device)")

            if grabbed_count == 0:
                print("[ERROR] No devices were grabbed! Shortcuts will not work!")
                print("[ERROR] Try running with sudo or check permissions")
                # Clean up UInput since we can't use it
                if self.uinput:
                    try:
                        self.uinput.close()
                    except:
                        pass
                    self.uinput = None
                self.devices_grabbed = False
                return False
            
            self.devices_grabbed = True
            return True

        except Exception as e:
            print(f"[ERROR] Could not set up key grabbing: {e}")
            print("[ERROR] Keys may leak through to applications")
            import traceback
            traceback.print_exc()
            self._cleanup_key_grabbing()
            return False

    def _cleanup_key_grabbing(self):
        """Clean up UInput and ungrab devices"""
        # Ungrab all devices
        if self.devices_grabbed:
            for device in self.devices:
                try:
                    device.ungrab()
                except:
                    pass
            self.devices_grabbed = False
            # Small delay to let kernel release devices before closing
            # This prevents "device busy" errors when service restarts quickly
            time.sleep(0.1)

        # Close UInput
        if self.uinput:
            try:
                self.uinput.close()
            except:
                pass
            self.uinput = None

    def stop(self):
        """Stop listening for global shortcuts"""
        if not self.is_running:
            return

        try:
            self.stop_event.set()

            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1.0)

            # Clean up key grabbing
            self._cleanup_key_grabbing()

            # Close all devices
            for device in self.devices[:]:  # Copy list to avoid modification during iteration
                self._remove_device(device)

            self.is_running = False
            self.pressed_keys.clear()
            self.suppressed_keys.clear()

        except Exception as e:
            print(f"Error stopping global shortcuts: {e}")
    
    def is_active(self) -> bool:
        """Check if global shortcuts are currently active"""
        return self.is_running and self.listener_thread and self.listener_thread.is_alive()
    
    def set_callback(self, callback: Callable):
        """Set the callback function for shortcut activation"""
        self.callback = callback
    
    def update_shortcut(self, new_key: str) -> bool:
        """Update the shortcut key combination"""
        try:
            # Parse the new key combination
            new_target_keys = self._parse_key_combination(new_key)
            
            # Update the configuration
            self.primary_key = new_key
            self.target_keys = new_target_keys
            
            print(f"Updated global shortcut to: {new_key}")
            return True
            
        except Exception as e:
            print(f"Failed to update shortcut: {e}")
            return False
    
    def test_shortcut(self) -> bool:
        """Test if shortcuts are working by temporarily setting a test callback"""
        original_callback = self.callback
        test_triggered = threading.Event()
        
        def test_callback():
            print("Test shortcut triggered!")
            test_triggered.set()
        
        # Set test callback
        self.callback = test_callback
        
        print(f"Press {self.primary_key} within 10 seconds to test...")
        
        # Wait for test trigger
        if test_triggered.wait(timeout=10):
            print("Shortcut test successful!")
            result = True
        else:
            print("ERROR: Shortcut test failed - no trigger detected")
            result = False
        
        # Restore original callback
        self.callback = original_callback
        return result
    
    def get_status(self) -> dict:
        """Get the current status of global shortcuts"""
        return {
            'is_running': self.is_running,
            'is_active': self.is_active(),
            'primary_key': self.primary_key,
            'target_keys': [self._keycode_to_name(k) for k in self.target_keys],
            'pressed_keys': [self._keycode_to_name(k) for k in self.pressed_keys],
            'device_count': len(self.devices)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop()
        except:
            pass

# Utility functions for key handling
def normalize_key_name(key_name: str) -> str:
    """Normalize key names for consistent parsing"""
    return key_name.lower().strip().replace(' ', '')

def _string_to_keycode_standalone(key_string: str) -> Optional[int]:
    """Standalone version of string to keycode conversion for use outside GlobalShortcuts class"""
    key_string = key_string.lower().strip()
    
    # 1. Try alias mapping first, easy names
    if key_string in KEY_ALIASES:
        key_name = KEY_ALIASES[key_string]
    else:
        # 2. Try as direct evdev KEY_* name
        key_name = key_string.upper()
        if not key_name.startswith('KEY_'):
            key_name = f'KEY_{key_name}'
    
    # 3. Look up the keycode in evdev's complete mapping
    code = ecodes.ecodes.get(key_name)
    
    if code is None:
        return None
    
    return code

def _parse_key_combination_standalone(key_string: str) -> Set[int]:
    """Standalone version of key combination parsing for use outside GlobalShortcuts class"""
    keys = set()
    key_lower = key_string.lower().strip()
    
    # Remove angle brackets if present
    key_lower = key_lower.replace('<', '').replace('>', '')
    
    # Split into parts for modifier + key combinations
    parts = key_lower.split('+')
    
    for part in parts:
        part = part.strip()
        keycode = _string_to_keycode_standalone(part)
        if keycode is not None:
            keys.add(keycode)
        else:
            print(f"Warning: Could not parse key '{part}' in '{key_string}'")
            
    # Default to F12 if no keys parsed
    if not keys:
        print(f"Warning: Could not parse key combination '{key_string}', defaulting to F12")
        keys.add(ecodes.KEY_F12)
        
    return keys

def get_available_keyboards(shortcut: Optional[str] = None) -> List[Dict[str, str]]:
    """Get a list of available input devices that can emit the specified shortcut.
    
    If shortcut is None, returns all devices with EV_KEY capabilities.
    If shortcut is provided, only returns devices that can emit all keys in the shortcut.
    """
    keyboards = []
    
    # Parse shortcut if provided
    target_keys = None
    if shortcut:
        target_keys = _parse_key_combination_standalone(shortcut)
    
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        
        for device in devices:
            # Check if device has keyboard capabilities
            capabilities = device.capabilities()
            if ecodes.EV_KEY not in capabilities:
                device.close()
                continue
            
            available_keys = set(capabilities[ecodes.EV_KEY])
            
            # If shortcut is provided, check that device can emit all required keys
            if target_keys and not target_keys.issubset(available_keys):
                device.close()
                continue
            
            try:
                # Test if we can access the device
                device.grab()
                device.ungrab()
                
                keyboards.append({
                    'name': device.name,
                    'path': device.path,
                    'display_name': f"{device.name} ({device.path})"
                })
            except (OSError, IOError):
                # Device not accessible, skip it
                pass
            finally:
                device.close()
                
    except Exception as e:
        print(f"Error getting available keyboards: {e}")
    
    return keyboards


def test_key_accessibility() -> Dict:
    """Test which keyboard devices are accessible"""
    print("Testing keyboard device accessibility...")
    
    results = {
        'accessible_devices': [],
        'inaccessible_devices': [],
        'total_devices': 0
    }
    
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        results['total_devices'] = len(devices)
        
        for device in devices:
            # Check if it's a keyboard
            capabilities = device.capabilities()
            if ecodes.EV_KEY in capabilities:
                try:
                    # Test accessibility
                    device.grab()
                    device.ungrab()
                    results['accessible_devices'].append({
                        'name': device.name,
                        'path': device.path
                    })
                except (OSError, IOError):
                    results['inaccessible_devices'].append({
                        'name': device.name,
                        'path': device.path
                    })
                finally:
                    device.close()
                    
    except Exception as e:
        print(f"Error testing devices: {e}")
    
    print(f"Found {len(results['accessible_devices'])} accessible keyboard devices")
    return results


if __name__ == "__main__":
    # Simple test when run directly
    def test_callback():
        print("Global shortcut activated!")
    
    shortcuts = GlobalShortcuts('F12', test_callback)
    
    if shortcuts.start():
        print("Press F12 to test, or Ctrl+C to exit...")
        try:
            # Keep the program running
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
    
    shortcuts.stop()
