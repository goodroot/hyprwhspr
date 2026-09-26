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
        self._original_volumes = {}  # sink_input index -> (identity, original, ducked volume)
        # app key -> (original volume, ducked volume) for apps whose ducked
        # stream ended before restore() reached it. Kept until a later restore
        # lands, so a stream left at the ducked level heals on the next cycle.
        self._pending_restores = {}
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
    def _app_key(sink_input) -> tuple:
        """Identity of the application rather than the stream. An app's
        streams come and go (a browser recreates one per tab or video), and
        PulseAudio's stream-restore database hands each new one the app's last
        volume - so a stream that ended while ducked leaves its app ducked."""
        props = sink_input.proplist
        return AudioDucker._key_from(props.get('application.name'),
                                     props.get('application.process.binary'))

    @staticmethod
    def _key_from(name, binary):
        # Streams that name no application can't be told apart, so they never
        # carry a debt - healing one could change an unrelated stream.
        if name is None and binary is None:
            return None
        return (name, binary)

    @staticmethod
    def _stream_volume(sink_input) -> float:
        return sum(sink_input.volume.values) / len(sink_input.volume.values)

    @staticmethod
    def _at_level(volume: float, level: float) -> bool:
        # Relative, so a quiet stream's tiny ducked level doesn't match its
        # neighbours; the floor stays well above PulseAudio's integer rounding.
        return abs(volume - level) <= max(0.002, 0.02 * level)

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
                    settled = set()

                    for stream in pulse.sink_input_list():
                        if self._is_own_stream(stream):
                            continue

                        if getattr(stream, 'corked', False):
                            continue
                        if skip_pids and self._belongs_to_pids(stream, skip_pids):
                            continue

                        # Store original volume (average of channels)
                        original_vol = self._stream_volume(stream)
                        key = self._app_key(stream)
                        pending = self._pending_restores.get(key) if key else None
                        if pending and self._at_level(original_vol, pending[1]):
                            # Still at the level an unfinished earlier duck
                            # left it: that's not this stream's real volume.
                            # Every such stream of the app gets it, so the debt
                            # is only settled once the loop is done.
                            original_vol = pending[0]
                            settled.add(key)
                        ducked_vol = original_vol * multiplier
                        self._original_volumes[stream.index] = (
                            self._stream_identity(stream), original_vol, ducked_vol)

                        pulse.volume_set_all_chans(stream, ducked_vol)

                    # Settled debts now live in this cycle's snapshot. One that
                    # matched no stream stays: a stream at another level only
                    # means the app has other streams or the user changed it.
                    for key in settled:
                        self._pending_restores.pop(key, None)
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

            live = set()
            try:
                with pulsectl.Pulse('hyprwhspr-ducker') as pulse:
                    streams = list(pulse.sink_input_list())
                    restored_count = 0
                    for stream in streams:
                        entry = self._original_volumes.get(stream.index)
                        if entry is None or entry[0] != self._stream_identity(stream):
                            continue  # not ours, or index reused by another stream
                        pulse.volume_set_all_chans(stream, entry[1])
                        live.add(stream.index)
                        restored_count += 1

                    # A ducked stream that ended leaves its app's next stream at
                    # the ducked level (stream-restore). Heal any stream of that
                    # app still sitting exactly there, and remember the rest so a
                    # later cycle heals them.
                    self._carry_unrestored(live)
                    healed = set()
                    for stream in streams:
                        key = self._app_key(stream)
                        if stream.index in live or key not in self._pending_restores:
                            continue
                        original_vol, ducked_vol = self._pending_restores[key]
                        if self._at_level(self._stream_volume(stream), ducked_vol):
                            pulse.volume_set_all_chans(stream, original_vol)
                            healed.add(key)
                            restored_count += 1
                    for key in healed:
                        self._pending_restores.pop(key, None)

                    self._original_volumes.clear()
                    self._is_ducked = False
                    if restored_count > 0:
                        print(f"[AUDIO_DUCKER] Restored {restored_count} stream(s) to original volume", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to restore audio: {e}", flush=True)
                # Whatever wasn't restored stays owed, so a later cycle can
                # still heal it; then clear state to avoid stuck ducking.
                self._carry_unrestored(live)
                self._original_volumes.clear()
                self._is_ducked = False
                return False

    def _carry_unrestored(self, live):
        """Move snapshot entries that weren't restored into the per-app debts,
        keeping the ducked level actually applied (not one recomputed from a
        reduction percent that may have changed since). Call under _lock."""
        for index, (identity, original_vol, ducked_vol) in self._original_volumes.items():
            if index in live:
                continue
            key = self._key_from(identity[1], identity[2])
            if key is not None:
                self._pending_restores[key] = (original_vol, ducked_vol)

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
