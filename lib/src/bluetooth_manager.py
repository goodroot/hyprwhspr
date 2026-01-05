"""
Bluetooth audio profile manager for hyprwhspr
Auto-switches Bluetooth headset from A2DP (no mic) to HSP/HFP (with mic) when recording.

Problem: Bluetooth headsets default to A2DP profile (high-quality stereo output),
which has no microphone support. For speech-to-text, we need to switch to
headset-head-unit (HSP/HFP) profile which enables the microphone.

Solution: Automatically switch profile when recording starts, restore when recording stops.
"""

import threading
import time
from typing import Optional, Dict, Tuple

try:
    import pulsectl

    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False


class BluetoothProfileManager:
    """
    Manages Bluetooth audio profile switching for recording.

    Automatically switches connected Bluetooth headsets from A2DP (high-quality audio,
    no mic) to HSP/HFP (headset profile with mic) when recording starts, and restores
    the previous profile when recording stops.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the Bluetooth profile manager.

        Args:
            enabled: Whether auto-switching is enabled
        """
        self.enabled = enabled
        self._pulse: Optional[pulsectl.Pulse] = None
        self._saved_state: Optional[Dict[str, str]] = (
            None  # {card_name: previous_profile}
        )
        self._lock = threading.Lock()

        if not PULSECTL_AVAILABLE:
            print(
                '[BT-MANAGER] pulsectl not available, Bluetooth profile switching disabled'
            )

    def is_available(self) -> bool:
        """Check if Bluetooth profile management is available"""
        return PULSECTL_AVAILABLE and self.enabled

    def set_enabled(self, enabled: bool):
        """Enable or disable Bluetooth profile auto-switching"""
        self.enabled = enabled
        if not enabled:
            # Restore any saved profile when disabling
            self.restore_profile()

    def _get_pulse_connection(self) -> Optional[pulsectl.Pulse]:
        """Get or create a PulseAudio connection"""
        if not PULSECTL_AVAILABLE:
            return None

        try:
            if self._pulse is None:
                self._pulse = pulsectl.Pulse('hyprwhspr-bluetooth')
            return self._pulse
        except Exception as e:
            print(f'[BT-MANAGER] Failed to connect to PulseAudio: {e}')
            return None

    def _close_pulse_connection(self):
        """Close the PulseAudio connection"""
        if self._pulse is not None:
            try:
                self._pulse.close()
            except Exception:
                pass
            self._pulse = None

    def _find_bluetooth_card(self) -> Optional[Tuple[str, str, list]]:
        """
        Find a connected Bluetooth card with headset profile capability.

        Returns:
            Tuple of (card_name, current_profile, available_profiles) or None if not found
        """
        pulse = self._get_pulse_connection()
        if pulse is None:
            return None

        try:
            cards = pulse.card_list()

            for card in cards:
                # Check if this is a Bluetooth card
                if not card.name.startswith('bluez_card.'):
                    continue

                # Check if card is connected (has active profile that's not 'off')
                current_profile = (
                    card.profile_active.name if card.profile_active else None
                )
                if current_profile is None or current_profile == 'off':
                    continue

                # Get available profiles
                available_profiles = [
                    p.name for p in card.profile_list if p.available != 0
                ]

                # Check if headset-head-unit profile is available
                headset_profiles = [
                    p for p in available_profiles if 'headset-head-unit' in p
                ]
                if not headset_profiles:
                    continue

                print(f'[BT-MANAGER] Found Bluetooth card: {card.name}')
                print(f'[BT-MANAGER]   Current profile: {current_profile}')
                print(f'[BT-MANAGER]   Available profiles: {available_profiles}')

                return (card.name, current_profile, available_profiles)

            return None

        except Exception as e:
            print(f'[BT-MANAGER] Error finding Bluetooth card: {e}')
            return None

    def switch_to_headset_profile(self) -> bool:
        """
        Switch Bluetooth card to headset-head-unit profile (enables microphone).

        Saves the current profile for later restoration.

        Returns:
            True if switch was successful or not needed, False on error
        """
        if not self.is_available():
            return True  # Not an error, just disabled

        with self._lock:
            try:
                bt_card = self._find_bluetooth_card()
                if bt_card is None:
                    print(
                        '[BT-MANAGER] No connected Bluetooth card with headset capability found'
                    )
                    return True  # Not an error, just no BT headset

                card_name, current_profile, available_profiles = bt_card

                # Find the headset-head-unit profile
                headset_profile = None
                for profile in available_profiles:
                    if 'headset-head-unit' in profile:
                        headset_profile = profile
                        break

                if headset_profile is None:
                    print('[BT-MANAGER] No headset-head-unit profile available')
                    return True

                # Already on headset profile?
                if current_profile == headset_profile:
                    print('[BT-MANAGER] Already on headset profile')
                    return True

                # Save current profile for restoration
                self._saved_state = {'card_name': card_name, 'profile': current_profile}

                # Switch to headset profile
                pulse = self._get_pulse_connection()
                if pulse is None:
                    return False

                print(
                    f'[BT-MANAGER] Switching {card_name}: {current_profile} -> {headset_profile}'
                )
                pulse.card_profile_set_by_name(card_name, headset_profile)

                # Brief delay for BT stack to stabilize
                time.sleep(0.4)

                print('[BT-MANAGER] Switched to headset profile (microphone enabled)')
                return True

            except Exception as e:
                print(f'[BT-MANAGER] Error switching to headset profile: {e}')
                return False

    def restore_profile(self) -> bool:
        """
        Restore the previously saved Bluetooth profile.

        Returns:
            True if restoration was successful or not needed, False on error
        """
        if not PULSECTL_AVAILABLE:
            return True

        with self._lock:
            if self._saved_state is None:
                return True  # Nothing to restore

            try:
                card_name = self._saved_state.get('card_name')
                previous_profile = self._saved_state.get('profile')

                if not card_name or not previous_profile:
                    self._saved_state = None
                    return True

                pulse = self._get_pulse_connection()
                if pulse is None:
                    return False

                print(f'[BT-MANAGER] Restoring {card_name} -> {previous_profile}')
                pulse.card_profile_set_by_name(card_name, previous_profile)

                self._saved_state = None
                print('[BT-MANAGER] Profile restored')
                return True

            except Exception as e:
                print(f'[BT-MANAGER] Error restoring profile: {e}')
                # Clear saved state even on error to avoid repeated failures
                self._saved_state = None
                return False

    def get_status(self) -> Dict:
        """
        Get current Bluetooth profile manager status.

        Returns:
            Dictionary with status information
        """
        status = {
            'available': PULSECTL_AVAILABLE,
            'enabled': self.enabled,
            'bluetooth_card': None,
            'current_profile': None,
            'headset_profile_available': False,
            'saved_profile': None,
        }

        if not PULSECTL_AVAILABLE:
            return status

        try:
            bt_card = self._find_bluetooth_card()
            if bt_card:
                card_name, current_profile, available_profiles = bt_card
                status['bluetooth_card'] = card_name
                status['current_profile'] = current_profile
                status['headset_profile_available'] = any(
                    'headset-head-unit' in p for p in available_profiles
                )

            if self._saved_state:
                status['saved_profile'] = self._saved_state.get('profile')

        except Exception as e:
            print(f'[BT-MANAGER] Error getting status: {e}')

        return status

    def cleanup(self):
        """Clean up resources - restore profile and close connection"""
        self.restore_profile()
        self._close_pulse_connection()

    def __del__(self):
        """Destructor - ensure cleanup on object destruction"""
        try:
            self.cleanup()
        except Exception:
            pass
