"""
Audio ducking for hyprwhspr
Reduces application playback volume during recording to prevent interference.

Ducking operates on sink inputs (per-application streams), not on the sinks
themselves. Changing a sink's volume moves the master volume of the output
device, which desktop shells (Noctalia, GNOME, swayosd, ...) watch and answer
with a volume OSD on every recording — and it also means a crash while ducked
leaves the user's speaker volume wrong. Per-stream ducking is invisible to
master-volume watchers and leaves the device volume untouched.

Known tradeoff: streams are snapshot once at duck time, so a stream that
STARTS during the recording (notification ping, autoplaying video) plays at
full volume. Covering late arrivals needs a sink-input event subscription;
until then this is the accepted cost of not touching the master volume.
"""

import threading

try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False


# Playback tools hyprwhspr itself uses for start/stop/error pings (see
# audio_manager._play_sound). Their streams must never be ducked, or a ping
# that races the duck snapshot gets caught and restored to a ducked level.
_OWN_PLAYBACK_BINARIES = {'paplay', 'pw-play', 'ffplay', 'aplay'}


class AudioDucker:
    """Manages audio ducking (volume reduction) during recording"""

    def __init__(self, reduction_percent: float = 50.0):
        """
        Initialize audio ducker.

        Args:
            reduction_percent: How much to reduce volume BY (0-100).
                              50 means reduce to 50% of original volume.
        """
        self._reduction_percent = max(0.0, min(100.0, reduction_percent))
        self._original_volumes = {}  # sink_input index -> (identity, original volume)
        self._lock = threading.Lock()
        self._is_ducked = False

        if not PULSECTL_AVAILABLE:
            print("[AUDIO_DUCKER] pulsectl not available, ducking disabled")

    @staticmethod
    def _stream_identity(sink_input) -> tuple:
        """Best-effort identity beyond the numeric index.

        Sink-input indices can be reused (PipeWire recycles object ids), so a
        stream that ends while ducked could hand its index to an unrelated new
        stream. Restore only when the identity still matches, never blindly by
        index.
        """
        props = sink_input.proplist
        return (props.get('application.process.id'),
                props.get('application.name'),
                props.get('application.process.binary'))

    @staticmethod
    def _is_own_stream(sink_input) -> bool:
        """True for streams spawned by hyprwhspr's own sound playback.

        PipeWire-native clients (pw-play) don't set application.process.binary,
        only application.name, so check both.
        """
        props = sink_input.proplist
        binary = (props.get('application.process.binary') or '').lower()
        app_name = (props.get('application.name') or '').lower()
        return binary in _OWN_PLAYBACK_BINARIES or app_name in _OWN_PLAYBACK_BINARIES

    def duck(self) -> bool:
        """
        Reduce playback volume of running application streams.
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
                    multiplier = (100.0 - self._reduction_percent) / 100.0

                    for stream in pulse.sink_input_list():
                        if self._is_own_stream(stream):
                            continue

                        # Store original volume (average of channels)
                        original_vol = sum(stream.volume.values) / len(stream.volume.values)
                        self._original_volumes[stream.index] = (
                            self._stream_identity(stream), original_vol)

                        pulse.volume_set_all_chans(stream, original_vol * multiplier)

                    self._is_ducked = True
                    stream_count = len(self._original_volumes)
                    print(f"[AUDIO_DUCKER] Ducked {stream_count} stream(s) by {self._reduction_percent:.0f}%", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to duck audio: {e}", flush=True)
                self._original_volumes.clear()
                return False

    def restore(self) -> bool:
        """
        Restore application streams to their original volume.
        Streams that ended while ducked are silently skipped.

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
                    restored_count = 0
                    for stream in pulse.sink_input_list():
                        entry = self._original_volumes.get(stream.index)
                        if entry is None:
                            continue
                        identity, original_vol = entry
                        if identity != self._stream_identity(stream):
                            continue  # index was reused by a different stream
                        pulse.volume_set_all_chans(stream, original_vol)
                        restored_count += 1

                    self._original_volumes.clear()
                    self._is_ducked = False
                    print(f"[AUDIO_DUCKER] Restored {restored_count} stream(s) to original volume", flush=True)
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
