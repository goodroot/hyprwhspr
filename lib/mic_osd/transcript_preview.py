"""Config and animation state for the pill live-transcript preview."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional, Sequence, Tuple


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _integer(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class PillTranscriptConfig:
    """Validated settings for the transcript shown above the compact pill."""

    enabled: bool = False
    word_limit: int = 4
    idle_timeout_seconds: float = 1.40

    # Visual tuning is intentionally part of the pill style, not public config.
    enter_seconds: float = 0.20
    exit_seconds: float = 0.20
    stagger_seconds: float = 0.028
    font_family: str = "sans-serif"
    font_size: float = 17.0
    offset_y: float = 7.0
    rise_px: float = 9.0
    max_width: float = 320.0

    @classmethod
    def from_getter(cls, get_setting: Callable[[str, object], object]):
        return cls(
            enabled=bool(
                get_setting("mic_osd_pill_transcript_enabled", cls.enabled)
            ),
            word_limit=_clamp(
                _integer(
                    get_setting(
                        "mic_osd_pill_transcript_word_limit", cls.word_limit
                    ),
                    cls.word_limit,
                ),
                1,
                12,
            ),
            idle_timeout_seconds=_clamp(
                _number(
                    get_setting(
                        "mic_osd_pill_transcript_idle_timeout_ms",
                        cls.idle_timeout_seconds * 1000,
                    ),
                    cls.idle_timeout_seconds * 1000,
                )
                / 1000.0,
                0.0,
                30.0,
            ),
        )

    @classmethod
    def load(cls):
        """Load from the normal hyprwhspr config, falling back safely."""
        try:
            try:
                from ..src.config_manager import ConfigManager
            except (ImportError, ValueError):
                from src.config_manager import ConfigManager
            return cls.from_getter(ConfigManager(verbose=False).get_setting)
        except Exception:
            return cls()


@dataclass(frozen=True)
class RenderWord:
    """One word to paint for the current animation frame."""

    text: str
    source: str
    index: int
    alpha: float
    y_offset: float = 0.0
    matched_from: Optional[int] = None
    layout_progress: float = 1.0


@dataclass(frozen=True)
class TranscriptFrame:
    previous_words: Tuple[str, ...]
    current_words: Tuple[str, ...]
    words: Tuple[RenderWord, ...]


def _ease_out_cubic(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return 1.0 - (1.0 - value) ** 3


def _ease_in_out(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _progress(elapsed: float, duration: float, delay: float = 0.0) -> float:
    if duration <= 0.0:
        return 1.0
    return _clamp((elapsed - delay) / duration, 0.0, 1.0)


def _lcs_matches(
    previous: Sequence[str], current: Sequence[str]
) -> Tuple[Tuple[int, int], ...]:
    """Duplicate-safe longest-common-subsequence matches for tiny word windows."""

    rows = len(previous) + 1
    cols = len(current) + 1
    table = [[0] * cols for _ in range(rows)]

    for old_index in range(len(previous) - 1, -1, -1):
        for new_index in range(len(current) - 1, -1, -1):
            if previous[old_index] == current[new_index]:
                table[old_index][new_index] = (
                    table[old_index + 1][new_index + 1] + 1
                )
            else:
                table[old_index][new_index] = max(
                    table[old_index + 1][new_index],
                    table[old_index][new_index + 1],
                )

    matches = []
    old_index = 0
    new_index = 0
    while old_index < len(previous) and new_index < len(current):
        if previous[old_index] == current[new_index]:
            matches.append((old_index, new_index))
            old_index += 1
            new_index += 1
        elif table[old_index + 1][new_index] >= table[old_index][new_index + 1]:
            old_index += 1
        else:
            new_index += 1
    return tuple(matches)


class PillTranscriptAnimator:
    """Latest-state transcript animation with no event queue.

    Incoming partials replace the target immediately. Character-by-character
    corrections to the last token update in place instead of restarting the
    animation. Word-boundary changes animate, so 200–250 WPM speech remains
    readable while the renderer always converges on the newest four words.
    """

    def __init__(
        self,
        config: PillTranscriptConfig,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self._clock = clock
        self.previous_words: Tuple[str, ...] = ()
        self.current_words: Tuple[str, ...] = ()
        self.matches: Tuple[Tuple[int, int], ...] = ()
        self.transition_started_at = self._clock()
        self.last_activity_at = self.transition_started_at

    def _extract_words(self, text: str) -> Tuple[str, ...]:
        words = tuple((text or "").split())
        return words[-self.config.word_limit :]

    def set_text(self, text: str, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        words = self._extract_words(text)

        if words:
            self.last_activity_at = now

        if words == self.current_words:
            return False

        # Realtime providers often revise only the unfinished final token
        # ("trans" -> "transcript"). Do not restart a 200ms animation for every
        # character; preserve the current transition and replace that token.
        if (
            words
            and self.current_words
            and len(words) == len(self.current_words)
            and words[:-1] == self.current_words[:-1]
        ):
            self.current_words = words
            return True

        self._begin_transition(words, now)
        return True

    def clear(self, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        if not self.current_words:
            return False
        self._begin_transition((), now)
        return True

    def _begin_transition(self, words: Tuple[str, ...], now: float):
        self.previous_words = self.current_words
        self.current_words = words
        self.matches = _lcs_matches(self.previous_words, self.current_words)
        self.transition_started_at = now

    def _expire_if_idle(self, now: float):
        timeout = self.config.idle_timeout_seconds
        if (
            timeout > 0
            and self.current_words
            and (now - self.last_activity_at) >= timeout
        ):
            self._begin_transition((), now)

    def frame(self, now: Optional[float] = None) -> TranscriptFrame:
        now = self._clock() if now is None else now
        self._expire_if_idle(now)

        if not self.config.enabled:
            return TranscriptFrame((), (), ())

        elapsed = max(0.0, now - self.transition_started_at)
        stable_by_new = {new: old for old, new in self.matches}
        stable_old = {old for old, _ in self.matches}
        rendered = []

        # Removed words leave first, rising away from the pill. They are never
        # queued; a newer transcript immediately replaces this transition.
        for old_index, word in enumerate(self.previous_words):
            if old_index in stable_old:
                continue
            reverse_index = len(self.previous_words) - 1 - old_index
            delay = reverse_index * self.config.stagger_seconds * 0.55
            progress = _ease_in_out(
                _progress(elapsed, self.config.exit_seconds, delay)
            )
            rendered.append(
                RenderWord(
                    word,
                    "previous",
                    old_index,
                    1.0 - progress,
                    y_offset=-self.config.rise_px * progress,
                )
            )

        for new_index, word in enumerate(self.current_words):
            matched_from = stable_by_new.get(new_index)
            if matched_from is not None:
                layout_progress = _ease_out_cubic(
                    _progress(elapsed, self.config.enter_seconds)
                )
                rendered.append(
                    RenderWord(
                        word,
                        "current",
                        new_index,
                        1.0,
                        matched_from=matched_from,
                        layout_progress=layout_progress,
                    )
                )
                continue

            delay = new_index * self.config.stagger_seconds
            progress = _ease_out_cubic(
                _progress(elapsed, self.config.enter_seconds, delay)
            )
            rendered.append(
                RenderWord(
                    word,
                    "current",
                    new_index,
                    progress,
                    y_offset=self.config.rise_px * (1.0 - progress),
                )
            )

        return TranscriptFrame(
            self.previous_words,
            self.current_words,
            tuple(rendered),
        )
