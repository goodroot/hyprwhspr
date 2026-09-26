"""Duck application streams without changing the master volume.

New streams are not ducked mid-recording. Replacements inheriting a ducked
volume can heal during restore or the next cycle; recovery is memory-only.
"""

import threading

try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False


# Leave feedback sounds at their intended volume.
_OWN_PLAYBACK_BINARIES = {'paplay', 'pw-play', 'ffplay', 'aplay'}

# Browser audio may run in a child of the MPRIS process.
_PID_ANCESTOR_DEPTH = 4

# Avoid mistaking later user-selected levels for stale ducking.
_PENDING_MAX_CYCLES = 1


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
        """Reduce volume by 0–100 percent; 50 halves it."""
        self._reduction_percent = max(0.0, min(100.0, reduction_percent))
        self._original_volumes = {}  # sink_input index -> (identity, original, ducked volume)
        self._pending_restores = {}  # app key -> {(original, ducked): cycle}
        self._matched_pending = set()  # (app key, volume pair, cycle)
        self._cycle = 0
        self._lock = threading.Lock()
        self._is_ducked = False

        if not PULSECTL_AVAILABLE:
            print("[AUDIO_DUCKER] pulsectl not available, ducking disabled")

    @staticmethod
    def _stream_identity(sink_input) -> tuple:
        """Guard against stream-index reuse."""
        props = sink_input.proplist
        return (props.get('application.process.id'),
                props.get('application.name'),
                props.get('application.process.binary'))

    @staticmethod
    def _app_key(sink_input) -> tuple:
        """Identify replacement streams across process changes."""
        props = sink_input.proplist
        return AudioDucker._key_from(props.get('application.name'),
                                     props.get('application.process.binary'))

    @staticmethod
    def _key_from(name, binary):
        # Anonymous streams cannot be matched safely.
        if name is None and binary is None:
            return None
        return (name, binary)

    @staticmethod
    def _stream_volume(sink_input) -> float:
        return sum(sink_input.volume.values) / len(sink_input.volume.values)

    @staticmethod
    def _at_level(volume: float, level: float) -> bool:
        # Relative tolerance with a floor for PulseAudio rounding.
        return abs(volume - level) <= max(0.002, 0.02 * level)

    @staticmethod
    def _is_own_stream(sink_input) -> bool:
        """Check both properties: pw-play may omit the binary."""
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
        """Duck active streams except corked players and skip_pids descendants.

        Return whether ducking succeeded; partial failures remain restorable.
        """
        if not PULSECTL_AVAILABLE:
            return False

        skip_pids = set(skip_pids or ())

        with self._lock:
            if self._is_ducked:
                return True  # Already ducked

            self._cycle += 1
            self._expire_pending()

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

                        original_vol = self._stream_volume(stream)
                        key = self._app_key(stream)
                        matches = self._pending_matches(key, original_vol)
                        if matches:
                            original_vol = next(iter(matches))[0]
                            self._matched_pending.update(
                                (key, pair, cycle) for pair, cycle in matches.items())
                        ducked_vol = original_vol * multiplier
                        self._original_volumes[stream.index] = (
                            self._stream_identity(stream), original_vol, ducked_vol)

                        pulse.volume_set_all_chans(stream, ducked_vol)

                    self._is_ducked = True
                    stream_count = len(self._original_volumes)
                    if stream_count > 0:
                        print(f"[AUDIO_DUCKER] Ducked {stream_count} stream(s) by {self._reduction_percent:.0f}%", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to duck audio: {e}", flush=True)
                # Keep partial ducking restorable.
                self._is_ducked = bool(self._original_volumes)
                return False

    def restore(self) -> bool:
        """Restore snapshots and heal matching replacements; return success."""
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
                    healed_count = 0
                    for stream in streams:
                        entry = self._original_volumes.get(stream.index)
                        if entry is None or entry[0] != self._stream_identity(stream):
                            continue  # not ours, or index reused by another stream
                        pulse.volume_set_all_chans(stream, entry[1])
                        live.add(stream.index)
                        restored_count += 1

                    # Keep candidates through the full pass for late siblings.
                    self._carry_unrestored(live)
                    healed = set()
                    for stream in streams:
                        key = self._app_key(stream)
                        if stream.index in live or key not in self._pending_restores:
                            continue
                        matches = self._pending_matches(key, self._stream_volume(stream))
                        if matches:
                            pulse.volume_set_all_chans(stream, next(iter(matches))[0])
                            healed.update(
                                (key, pair, cycle) for pair, cycle in matches.items())
                            healed_count += 1
                    for key, pair, cycle in self._matched_pending | healed:
                        pending = self._pending_restores.get(key, {})
                        # A vanished snapshot may have renewed the same pair.
                        if pending.get(pair) == cycle:
                            del pending[pair]
                        if not pending:
                            self._pending_restores.pop(key, None)
                    self._matched_pending.clear()

                    self._original_volumes.clear()
                    self._is_ducked = False
                    if restored_count > 0:
                        print(f"[AUDIO_DUCKER] Restored {restored_count} stream(s) to original volume", flush=True)
                    if healed_count > 0:
                        print(f"[AUDIO_DUCKER] Healed {healed_count} stream(s) left at the ducked volume", flush=True)
                    return True

            except Exception as e:
                print(f"[AUDIO_DUCKER] Failed to restore audio: {e}", flush=True)
                # Retain recovery data, but release the active cycle.
                self._carry_unrestored(live)
                self._matched_pending.clear()
                self._original_volumes.clear()
                self._is_ducked = False
                return False

    def _pending_matches(self, key, volume):
        """Find candidates with one agreed original volume. Call under _lock."""
        matches = {
            pair: cycle for pair, cycle in self._pending_restores.get(key, {}).items()
            if self._at_level(volume, pair[1])
        }
        if len({pair[0] for pair in matches}) != 1:
            return {}
        return matches

    def _expire_pending(self):
        """Expire each volume pair independently. Call under _lock."""
        for key, pending in list(self._pending_restores.items()):
            for pair, cycle in list(pending.items()):
                if self._cycle - cycle > _PENDING_MAX_CYCLES:
                    del pending[pair]
            if not pending:
                del self._pending_restores[key]

    def _carry_unrestored(self, live):
        """Keep unapplied restores at their actual ducked levels. Call under _lock."""
        for index, (identity, original_vol, ducked_vol) in self._original_volumes.items():
            if index in live:
                continue
            key = self._key_from(identity[1], identity[2])
            if key is not None:
                pending = self._pending_restores.setdefault(key, {})
                pending[(original_vol, ducked_vol)] = self._cycle

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
