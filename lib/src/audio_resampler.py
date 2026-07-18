"""Strict, shared audio resampling helpers."""

import numpy as np


class ResamplingError(RuntimeError):
    """Raised when audio cannot be safely converted to the requested rate."""


def resample_audio(audio, source_rate: int, target_rate: int) -> np.ndarray:
    """Return mono float32 audio at ``target_rate`` with an exact sample count.

    Callers must handle :class:`ResamplingError`; returning the input after a
    failed conversion would make its samples carry an incorrect rate label.
    """
    try:
        source_rate = int(source_rate)
        target_rate = int(target_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample rates must be positive integers") from exc
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive integers")

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise ValueError(f"expected mono audio, got shape {samples.shape}")
    if samples.size == 0:
        raise ValueError("cannot resample empty audio")
    if not np.issubdtype(samples.dtype, np.number):
        raise ValueError("audio samples must be numeric")

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio contains non-finite samples")
    if source_rate == target_rate:
        return samples

    expected_length = max(1, round(len(samples) * target_rate / source_rate))
    try:
        import soxr
        converted = np.asarray(
            soxr.resample(samples, source_rate, target_rate, quality="HQ"),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as exc:
        raise ResamplingError(
            f"failed to resample audio {source_rate}Hz -> {target_rate}Hz: {exc}"
        ) from exc

    # libsoxr can differ by one sample for some ratios. Enforce the timing
    # contract expected by providers and WAV consumers.
    if len(converted) > expected_length:
        converted = converted[:expected_length]
    elif len(converted) < expected_length:
        converted = np.pad(converted, (0, expected_length - len(converted)))
    return np.ascontiguousarray(converted, dtype=np.float32)
