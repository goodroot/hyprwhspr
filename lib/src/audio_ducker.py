"""
Audio ducking for hyprwhspr
Reduces system volume during recording to prevent interference
"""

import threading

try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False


class AudioDucker:
    """Manages audio ducking (volume reduction) during recording"""

    def __init__(self, reduction_percent: float = 70.0):
        """
        Initialize audio ducker.

        Args:
            reduction_percent: How much to reduce volume BY (0-100).
                              70 means reduce to 30% of original volume.
        """
        self._reduction_percent = max(0.0, min(100.0, reduction_percent))
        self._original_volumes = {}  # sink_name -> original volume
        self._lock = threading.Lock()
        self._is_ducked = False

        if not PULSECTL_AVAILABLE:
            print("[AUDIO_DUCKER] pulsectl not available, ducking disabled")

    def duck(self) -> bool:
        """
        Reduce system audio volume.
        Stores original volumes for later restoration.

        Returns:
            True if ducking was applied, False otherwise
        """
        if not PULSECTL_AVAILABLE:
            return False

        with self._lock:
            if self._is_ducked:
                return True  # Already ducked

            try:
                with pulsectl.Pulse('hyprwhspr-ducker') as pulse:
                    # Get all sinks and duck them
                    for sink in pulse.sink_list():
                        # Store original volume (average of channels)
                        original_vol = sum(sink.volume.values) / len(sink.volume.values)
                        self._original_volumes[sink.name] = original_vol

                        # Calculate new volume (reduce BY percentage)
                        multiplier = (100.0 - self._reduction_percent) / 100.0
                        new_vol = original_vol * multiplier

                        # Apply new volume
                        pulse.volume_set_all_chans(sink, new_vol)

                    self._is_ducked = True
                    sink_count = len(self._original_volumes)
                    print(f"[AUDIO_DUCKER] Ducked {sink_count} sink(s) by {self._reduction_percent:.0f}%", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to duck audio: {e}", flush=True)
                self._original_volumes.clear()
                return False

    def restore(self) -> bool:
        """
        Restore system audio to original volume.

        Returns:
            True if restoration was successful, False otherwise
        """
        if not PULSECTL_AVAILABLE:
            return False

        with self._lock:
            if not self._is_ducked:
                return True  # Not ducked, nothing to restore

            try:
                with pulsectl.Pulse('hyprwhspr-ducker') as pulse:
                    # Restore all sinks to original volumes
                    for sink in pulse.sink_list():
                        if sink.name in self._original_volumes:
                            original_vol = self._original_volumes[sink.name]
                            pulse.volume_set_all_chans(sink, original_vol)

                    restored_count = len(self._original_volumes)
                    self._original_volumes.clear()
                    self._is_ducked = False
                    print(f"[AUDIO_DUCKER] Restored {restored_count} sink(s) to original volume", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to restore audio: {e}", flush=True)
                # Clear state anyway to avoid stuck ducking
                self._original_volumes.clear()
                self._is_ducked = False
                return False

    def set_reduction_percent(self, percent: float):
        """Update the reduction percentage"""
        self._reduction_percent = max(0.0, min(100.0, percent))

    @property
    def is_ducked(self) -> bool:
        """Check if audio is currently ducked"""
        with self._lock:
            return self._is_ducked

    @staticmethod
    def is_available() -> bool:
        """Check if audio ducking is available"""
        return PULSECTL_AVAILABLE
