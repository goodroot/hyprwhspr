"""
Playback suppression for hyprwhspr
Quiets other audio while recording, either by ducking volume or by pausing players.

Two modes:
  duck  - reduce every application stream's volume (the original behaviour)
  pause - pause MPRIS players outright (podcasts resume where they left off), then
          duck whatever is left over: games, calls, anything without MPRIS

main.py talks only to this object, so the mode is a config lookup rather than a
branch at each of the recording lifecycle's restore points.
"""

import time

from audio_ducker import AudioDucker
from media_pauser import MediaPauser


MODE_DUCK = 'duck'
MODE_PAUSE = 'pause'
VALID_MODES = (MODE_DUCK, MODE_PAUSE)

# Players cork their stream shortly after Pause() returns. Give that a moment to
# land so the duck sweep can recognise them as already handled.
PAUSE_SETTLE_SECONDS = 0.15


class PlaybackSuppressor:
    """Coordinates volume ducking and MPRIS pausing for a recording"""

    def __init__(self, reduction_percent: float = 50.0):
        self._ducker = AudioDucker(reduction_percent=reduction_percent)
        self._pauser = MediaPauser()

    def suppress(self, mode: str = MODE_DUCK, reduction_percent=None) -> bool:
        """
        Quiet other audio for the duration of a recording.

        Args:
            mode: 'duck' or 'pause'. Unknown values fall back to 'duck'.
            reduction_percent: Optional live override of the duck amount.

        Returns:
            True if any suppression was applied
        """
        if reduction_percent is not None:
            self._ducker.set_reduction_percent(reduction_percent)

        if mode not in VALID_MODES:
            mode = MODE_DUCK

        if mode != MODE_PAUSE:
            return self._ducker.duck()

        paused_pids = self._pauser.pause_all()
        # Keyed on "did we pause anything", not on the PIDs: when the bus won't
        # name a player's PID, corking is the only thing keeping its stream out
        # of the duck sweep, so that is exactly when the settle matters most.
        if self._pauser.is_paused:
            time.sleep(PAUSE_SETTLE_SECONDS)
        ducked = self._ducker.duck(skip_pids=paused_pids)
        return ducked or self._pauser.is_paused

    def restore(self) -> bool:
        """
        Undo whatever suppression is active. Each half is independent so a failure
        in one can't strand the other.

        Returns:
            True if everything restored cleanly
        """
        ok = True
        try:
            if self._ducker.is_ducked:
                ok = self._ducker.restore() and ok
        except Exception as e:
            print(f"[PLAYBACK] Failed to restore ducked audio: {e}", flush=True)
            ok = False
        try:
            if self._pauser.is_paused:
                ok = self._pauser.resume_all() and ok
        except Exception as e:
            print(f"[PLAYBACK] Failed to resume paused media: {e}", flush=True)
            ok = False
        return ok

    @property
    def is_active(self) -> bool:
        """True if audio is currently ducked and/or media is paused"""
        return self._ducker.is_ducked or self._pauser.is_paused
