"""Decode audio files into the in-memory format used by transcription backends."""

from pathlib import Path


SUPPORTED_AUDIO_SUFFIXES = {'.mp3', '.wav'}


class AudioFileError(ValueError):
    """Raised when an input file cannot be decoded for transcription."""


def decode_audio_file(path):
    """Return ``(mono_float32_samples, sample_rate)`` for a WAV or MP3 file."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise AudioFileError(f"Input file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise AudioFileError(f"Input path is not a file: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise AudioFileError(
            f"Unsupported audio format '{audio_path.suffix or '(none)'}'; "
            "supported formats are .wav and .mp3"
        )

    try:
        import soundfile as sf
    except ImportError as exc:
        raise AudioFileError(
            'Audio file decoding requires soundfile; run: hyprwhspr setup'
        ) from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise AudioFileError(
            'Audio file decoding requires numpy; run: hyprwhspr setup'
        ) from exc
    try:
        samples, sample_rate = sf.read(
            str(audio_path), dtype='float32', always_2d=True
        )
    except Exception as exc:
        raise AudioFileError(f"Could not decode audio file '{audio_path}': {exc}") from exc

    if sample_rate <= 0 or samples.size == 0:
        raise AudioFileError(f"Audio file contains no samples: {audio_path}")
    if samples.shape[1] > 1:
        samples = np.mean(samples, axis=1, dtype=np.float32)
    else:
        samples = samples[:, 0]
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        raise AudioFileError(f"Audio file contains invalid samples: {audio_path}")
    return samples, int(sample_rate)
