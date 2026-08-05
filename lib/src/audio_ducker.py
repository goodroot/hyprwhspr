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

# How far up the process tree to look when matching a stream against a paused
# MPRIS player. Browsers play audio from a child process, not the process that
# owns the MPRIS bus name, so a direct PID match alone misses them.
_PID_ANCESTOR_DEPTH = 4


def _pid_ancestry(pid: int, depth: int = _PID_ANCESTOR_DEPTH) -> list:
    """[pid, parent, grandparent, ...] read from /proc, best-effort."""
    chain = []
    current = pid
    for _ in range(depth):
        if current is None or current <= 1:
            break
        chain.append(current)
        try:
            with open(f'/proc/{current}/status', 'r') as handle:
                for line in handle:
                    if line.startswith('PPid:'):
                        current = int(line.split()[1])
                        break
                else:
                    break
        except (OSError, ValueError):
            break
    return chain


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

    @staticmethod
    def _belongs_to_pids(sink_input, pids: set) -> bool:
        """True if the stream's process is (or descends from) one of `pids`."""
        raw_pid = sink_input.proplist.get('application.process.id')
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            return False
        return any(ancestor in pids for ancestor in _pid_ancestry(pid))

    def duck(self, skip_pids=None) -> bool:
        """
        Reduce playback volume of running application streams.
        Stores original volumes for later restoration.

        Args:
            skip_pids: PIDs whose streams must be left alone (players already
                       paused via MPRIS).

        A paused stream is worth nothing to duck and costs something: if it goes
        away before restore() runs, PulseAudio's stream-restore database keeps the
        ducked volume against that app and hands it back to its next stream. So
        anything corked, or belonging to a player we just paused, is skipped.

        Returns:
            True if ducking was applied, False otherwise
        """
        if not PULSECTL_AVAILABLE:
            return False

        skip_pids = set(skip_pids or ())

        with self._lock:
            if self._is_ducked:
                return True  # Already ducked

            try:
                with pulsectl.Pulse('hyprwhspr-ducker') as pulse:
                    multiplier = (100.0 - self._reduction_percent) / 100.0

                    for stream in pulse.sink_input_list():
                        if self._is_own_stream(stream):
                            continue

                        if getattr(stream, 'corked', False):
                            continue
                        if skip_pids and self._belongs_to_pids(stream, skip_pids):
                            continue

                        # Store original volume (average of channels)
                        original_vol = sum(stream.volume.values) / len(stream.volume.values)
                        self._original_volumes[stream.index] = (
                            self._stream_identity(stream), original_vol)

                        pulse.volume_set_all_chans(stream, original_vol * multiplier)

                    self._is_ducked = True
                    stream_count = len(self._original_volumes)
                    if stream_count > 0:
                        print(f"[AUDIO_DUCKER] Ducked {stream_count} stream(s) by {self._reduction_percent:.0f}%", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to duck audio: {e}", flush=True)
                # Whatever was lowered before the failure still needs restoring,
                # so keep the snapshot and stay "ducked" - dropping it here would
                # leave those streams quiet for good.
                self._is_ducked = bool(self._original_volumes)
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
                    if restored_count > 0:
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
