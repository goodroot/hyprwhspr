"""
Global shortcuts handler for hyprwhspr
Manages system-wide keyboard shortcuts for dictation control
"""

import threading
import select
import time
from typing import Callable, Optional, List, Set, Dict
from pathlib import Path
import evdev
from evdev import InputDevice, categorize, ecodes


class GlobalShortcuts:
    """Handles global keyboard shortcuts using evdev for hardware-level capture"""
    
    def __init__(self, primary_key: str = '<f12>', callback: Optional[Callable] = None, device_path: Optional[str] = None):
        self.primary_key = primary_key
        self.callback = callback
        self.selected_device_path = device_path
        
        # Device and event handling
        self.devices = []
        self.device_fds = {}
        self.listener_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        # State tracking
        self.pressed_keys = set()
        self.last_trigger_time = 0
        self.debounce_time = 0.5  # 500ms debounce to prevent double triggers
        
        # Parse the primary key combination
        self.target_keys = self._parse_key_combination(primary_key)
        
        # Initialize keyboard devices
        self._discover_keyboards()
        
        print(f"Global shortcuts initialized with key: {primary_key}")
        print(f"Parsed keys: {[self._keycode_to_name(k) for k in self.target_keys]}")
        print(f"Found {len(self.devices)} keyboard device(s)")
        
    def _discover_keyboards(self):
        """Discover and initialize keyboard input devices"""
        self.devices = []
        self.device_fds = {}
        
        try:
            # Find all input devices
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            
            # If a specific device path is selected, only use that device
            if self.selected_device_path:
                devices = [dev for dev in devices if dev.path == self.selected_device_path]
                if not devices:
                    print(f"Warning: Selected device {self.selected_device_path} not found!")
                    # Fall back to auto-discovery
                    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            
            for device in devices:
                # Check if device has keyboard capabilities
                if self._is_keyboard_device(device):
                    try:
                        # Test if we can grab the device (requires root)
                        device.grab()
                        device.ungrab()
                        
                        self.devices.append(device)
                        self.device_fds[device.fd] = device
                        print(f"Added keyboard device: {device.name} ({device.path})")
                        
                        # If we selected a specific device and found it, we can stop here
                        if self.selected_device_path and device.path == self.selected_device_path:
                            break
                        
                    except (OSError, IOError) as e:
                        print(f"Cannot access device {device.name}: {e}")
                        device.close()
                        
        except Exception as e:
            print(f"Error discovering keyboards: {e}")
            
        if not self.devices:
            print("Warning: No accessible keyboard devices found!")
            print("Make sure the application is running with root privileges.")
    
    def _is_keyboard_device(self, device: InputDevice) -> bool:
        """Check if a device is a keyboard by testing for common keyboard keys"""
        capabilities = device.capabilities()
        
        # Check if device has EV_KEY events
        if ecodes.EV_KEY not in capabilities:
            return False
            
        # Check for common keyboard keys
        keys = capabilities[ecodes.EV_KEY]
        
        # Look for alphabetic keys (a good indicator of a keyboard)
        keyboard_keys = [ecodes.KEY_A, ecodes.KEY_S, ecodes.KEY_D, ecodes.KEY_F]
        
        return any(key in keys for key in keyboard_keys)
    
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
        """Convert a key string to evdev keycode"""
        key_string = key_string.lower().strip()
        
        # Function keys
        function_key_map = {
            'f1': ecodes.KEY_F1, 'f2': ecodes.KEY_F2, 'f3': ecodes.KEY_F3, 'f4': ecodes.KEY_F4,
            'f5': ecodes.KEY_F5, 'f6': ecodes.KEY_F6, 'f7': ecodes.KEY_F7, 'f8': ecodes.KEY_F8,
            'f9': ecodes.KEY_F9, 'f10': ecodes.KEY_F10, 'f11': ecodes.KEY_F11, 'f12': ecodes.KEY_F12
        }
        
        # Modifier keys
        modifier_key_map = {
            'ctrl': ecodes.KEY_LEFTCTRL,
            'control': ecodes.KEY_LEFTCTRL,
            'alt': ecodes.KEY_LEFTALT,
            'rightalt': ecodes.KEY_RIGHTALT,
            'ralt': ecodes.KEY_RIGHTALT,
            'shift': ecodes.KEY_LEFTSHIFT,
            'super': ecodes.KEY_LEFTMETA,
            'windows': ecodes.KEY_LEFTMETA,
            'win': ecodes.KEY_LEFTMETA,
            'meta': ecodes.KEY_LEFTMETA,
            'cmd': ecodes.KEY_LEFTMETA
        }
        
        # Character keys
        char_key_map = {
            'a': ecodes.KEY_A, 'b': ecodes.KEY_B, 'c': ecodes.KEY_C, 'd': ecodes.KEY_D,
            'e': ecodes.KEY_E, 'f': ecodes.KEY_F, 'g': ecodes.KEY_G, 'h': ecodes.KEY_H,
            'i': ecodes.KEY_I, 'j': ecodes.KEY_J, 'k': ecodes.KEY_K, 'l': ecodes.KEY_L,
            'm': ecodes.KEY_M, 'n': ecodes.KEY_N, 'o': ecodes.KEY_O, 'p': ecodes.KEY_P,
            'q': ecodes.KEY_Q, 'r': ecodes.KEY_R, 's': ecodes.KEY_S, 't': ecodes.KEY_T,
            'u': ecodes.KEY_U, 'v': ecodes.KEY_V, 'w': ecodes.KEY_W, 'x': ecodes.KEY_X,
            'y': ecodes.KEY_Y, 'z': ecodes.KEY_Z
        }
        
        # Check function keys first
        if key_string in function_key_map:
            return function_key_map[key_string]
            
        # Check modifier keys
        if key_string in modifier_key_map:
            return modifier_key_map[key_string]
            
        # Check character keys
        if key_string in char_key_map:
            return char_key_map[key_string]
            
        return None
    
    def _keycode_to_name(self, keycode: int) -> str:
        """Convert evdev keycode to human readable name"""
        try:
            return ecodes.KEY[keycode].replace('KEY_', '')
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
                
                for fd in ready_fds:
                    if fd in self.device_fds:
                        device = self.device_fds[fd]
                        try:
                            events = device.read()
                            for event in events:
                                self._process_event(event)
                        except (OSError, IOError):
                            # Device disconnected or error
                            print(f"Lost connection to device: {device.name}")
                            self._remove_device(device)
                            
        except Exception as e:
            print(f"Error in keyboard event loop: {e}")
        
    def _remove_device(self, device: InputDevice):
        """Remove a disconnected device from monitoring"""
        try:
            if device in self.devices:
                self.devices.remove(device)
            if device.fd in self.device_fds:
                del self.device_fds[device.fd]
            device.close()
        except:
            pass
    
    def _process_event(self, event):
        """Process individual keyboard events"""
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            
            if key_event.keystate == key_event.key_down:
                # Key pressed
                self.pressed_keys.add(event.code)
                self._check_shortcut_combination()
                
            elif key_event.keystate == key_event.key_up:
                # Key released
                self.pressed_keys.discard(event.code)
    
    def _check_shortcut_combination(self):
        """Check if current pressed keys match target combination"""
        if self.target_keys.issubset(self.pressed_keys):
            current_time = time.time()
            
            # Implement debouncing
            if current_time - self.last_trigger_time > self.debounce_time:
                self.last_trigger_time = current_time
                self._trigger_callback()
    
    def _trigger_callback(self):
        """Trigger the callback function"""
        if self.callback:
            try:
                print(f"Global shortcut triggered: {self.primary_key}")
                # Run callback in a separate thread to avoid blocking the listener
                callback_thread = threading.Thread(target=self.callback, daemon=True)
                callback_thread.start()
            except Exception as e:
                print(f"Error calling shortcut callback: {e}")
    
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
            self.stop_event.clear()
            self.listener_thread = threading.Thread(target=self._event_loop, daemon=True)
            self.listener_thread.start()
            self.is_running = True
            
            print(f"Global shortcuts started, listening for {self.primary_key}")
            return True
            
        except Exception as e:
            print(f"Failed to start global shortcuts: {e}")
            return False
    
    def stop(self):
        """Stop listening for global shortcuts"""
        if not self.is_running:
            return
            
        try:
            self.stop_event.set()
            
            if self.listener_thread and self.listener_thread.is_alive():
                self.listener_thread.join(timeout=1.0)
            
            # Close all devices
            for device in self.devices[:]:  # Copy list to avoid modification during iteration
                self._remove_device(device)
            
            self.is_running = False
            self.pressed_keys.clear()
            
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

def get_available_keyboards() -> List[Dict[str, str]]:
    """Get a list of available keyboard devices for selection"""
    keyboards = []
    
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        
        for device in devices:
            # Check if device has keyboard capabilities
            capabilities = device.capabilities()
            if ecodes.EV_KEY not in capabilities:
                device.close()
                continue
                
            # Check for common keyboard keys
            keys = capabilities[ecodes.EV_KEY]
            keyboard_keys = [ecodes.KEY_A, ecodes.KEY_S, ecodes.KEY_D, ecodes.KEY_F]
            
            if any(key in keys for key in keyboard_keys):
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
            else:
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
