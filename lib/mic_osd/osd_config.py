"""Validated user-facing layout and response settings for the mic OSD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class MicOSDConfig:
    """Settings shared by the layer-shell window and spectrum renderer."""

    spectrum_width: int = 520
    spectrum_meter_height: int = 86
    spectrum_background_opacity: float = 0.46
    spectrum_border_opacity: float = 0.30

    # FFT amplitudes are converted to dB, mapped between floor and peak, then
    # multiplied by gain.  These three settings cover quiet microphones,
    # unusually hot interfaces, and different room noise floors without
    # changing capture gain or the audio sent to transcription.
    spectrum_gain: float = 1.35
    spectrum_noise_floor_db: float = -72.0
    spectrum_peak_db: float = -24.0
    spectrum_curve: float = 1.10
    spectrum_rise_ms: float = 35.0
    spectrum_fall_ms: float = 150.0

    transcript_font_family: str = "sans-serif"
    transcript_font_size: float = 18.0
    transcript_max_lines: int = 6
    transcript_line_spacing: float = 1.25
    transcript_opacity: float = 0.96
    bottom_margin: int = 130

    @classmethod
    def from_getter(cls, get_setting: Callable[[str, object], object]):
        defaults = cls()
        floor_db = _clamp(
            _number(
                get_setting(
                    "mic_osd_spectrum_noise_floor_db",
                    defaults.spectrum_noise_floor_db,
                ),
                defaults.spectrum_noise_floor_db,
            ),
            -120.0,
            -20.0,
        )
        peak_db = _clamp(
            _number(
                get_setting(
                    "mic_osd_spectrum_peak_db", defaults.spectrum_peak_db
                ),
                defaults.spectrum_peak_db,
            ),
            -80.0,
            0.0,
        )
        # Keep a useful dB range even if a hand-edited config reverses it.
        peak_db = max(peak_db, floor_db + 6.0)

        family = get_setting(
            "mic_osd_transcript_font_family",
            defaults.transcript_font_family,
        )
        if not isinstance(family, str) or not family.strip():
            family = defaults.transcript_font_family

        return cls(
            spectrum_width=_clamp(
                _integer(
                    get_setting(
                        "mic_osd_spectrum_width", defaults.spectrum_width
                    ),
                    defaults.spectrum_width,
                ),
                280,
                1600,
            ),
            spectrum_meter_height=_clamp(
                _integer(
                    get_setting(
                        "mic_osd_spectrum_meter_height",
                        defaults.spectrum_meter_height,
                    ),
                    defaults.spectrum_meter_height,
                ),
                44,
                360,
            ),
            spectrum_background_opacity=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_background_opacity",
                        defaults.spectrum_background_opacity,
                    ),
                    defaults.spectrum_background_opacity,
                ),
                0.0,
                1.0,
            ),
            spectrum_border_opacity=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_border_opacity",
                        defaults.spectrum_border_opacity,
                    ),
                    defaults.spectrum_border_opacity,
                ),
                0.0,
                1.0,
            ),
            spectrum_gain=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_gain", defaults.spectrum_gain
                    ),
                    defaults.spectrum_gain,
                ),
                0.05,
                20.0,
            ),
            spectrum_noise_floor_db=floor_db,
            spectrum_peak_db=peak_db,
            spectrum_curve=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_curve", defaults.spectrum_curve
                    ),
                    defaults.spectrum_curve,
                ),
                0.25,
                4.0,
            ),
            spectrum_rise_ms=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_rise_ms",
                        defaults.spectrum_rise_ms,
                    ),
                    defaults.spectrum_rise_ms,
                ),
                0.0,
                1000.0,
            ),
            spectrum_fall_ms=_clamp(
                _number(
                    get_setting(
                        "mic_osd_spectrum_fall_ms",
                        defaults.spectrum_fall_ms,
                    ),
                    defaults.spectrum_fall_ms,
                ),
                0.0,
                2000.0,
            ),
            transcript_font_family=family.strip(),
            transcript_font_size=_clamp(
                _number(
                    get_setting(
                        "mic_osd_transcript_font_size",
                        defaults.transcript_font_size,
                    ),
                    defaults.transcript_font_size,
                ),
                8.0,
                72.0,
            ),
            transcript_max_lines=_clamp(
                _integer(
                    get_setting(
                        "mic_osd_transcript_max_lines",
                        defaults.transcript_max_lines,
                    ),
                    defaults.transcript_max_lines,
                ),
                1,
                30,
            ),
            transcript_line_spacing=_clamp(
                _number(
                    get_setting(
                        "mic_osd_transcript_line_spacing",
                        defaults.transcript_line_spacing,
                    ),
                    defaults.transcript_line_spacing,
                ),
                1.0,
                3.0,
            ),
            transcript_opacity=_clamp(
                _number(
                    get_setting(
                        "mic_osd_transcript_opacity",
                        defaults.transcript_opacity,
                    ),
                    defaults.transcript_opacity,
                ),
                0.0,
                1.0,
            ),
            bottom_margin=_clamp(
                _integer(
                    get_setting(
                        "mic_osd_bottom_margin", defaults.bottom_margin
                    ),
                    defaults.bottom_margin,
                ),
                0,
                2000,
            ),
        )

    @classmethod
    def load(cls):
        """Load settings from hyprwhspr's normal configuration."""
        try:
            try:
                from ..src.config_manager import ConfigManager
            except (ImportError, ValueError):
                from src.config_manager import ConfigManager
            return cls.from_getter(ConfigManager(verbose=False).get_setting)
        except Exception:
            return cls()
