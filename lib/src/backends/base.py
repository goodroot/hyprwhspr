"""
Base class for transcription backends.

Each backend owns its model/connection state and implements initialize /
transcribe / reinitialize / unload / cleanup. Shared manager state (config,
ready flag, current model name, last-use timestamp) is proxied to the owning
WhisperManager so moved code keeps reading and writing the same fields the
facade and main.py observe.
"""

import sys
import wave
from io import BytesIO
from typing import Optional

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)


class TranscriptionBackend:
    """Base class for transcription backends owned by WhisperManager."""

    name = ''                 # canonical backend key, e.g. 'onnx-asr'
    is_local = True           # False: rest-api, realtime-ws (no model lock / unload no-op)
    reinit_on_idle = False    # long-idle (>30 min) reinit before transcribing
    reinit_on_resume = False  # reinit after suspend/resume recovery

    def __init__(self, manager):
        self._manager = manager

    # ------------------------------------------------------------------
    # Shared manager state (write-through proxies)
    # ------------------------------------------------------------------

    @property
    def config(self):
        return self._manager.config

    @property
    def temp_dir(self):
        return self._manager.temp_dir

    @property
    def ready(self):
        return self._manager.ready

    @ready.setter
    def ready(self, value):
        self._manager.ready = value

    @property
    def current_model(self):
        return self._manager.current_model

    @current_model.setter
    def current_model(self, value):
        self._manager.current_model = value

    @property
    def _last_use_time(self):
        return self._manager._last_use_time

    @_last_use_time.setter
    def _last_use_time(self, value):
        self._manager._last_use_time = value

    # ------------------------------------------------------------------
    # Backend lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Load the model / establish the connection. Returns success."""
        raise NotImplementedError

    def transcribe(self, audio_data: 'np.ndarray', sample_rate: int = 16000,
                   language_override: Optional[str] = None) -> str:
        """Transcribe validated audio. Returns the transcript ('' on failure)."""
        raise NotImplementedError

    def reinitialize(self) -> bool:
        """Refresh stale model/connection state (idle or suspend/resume)."""
        return True

    def unload(self) -> None:
        """Release the loaded model to free memory (local backends)."""

    def cleanup(self) -> None:
        """Release resources at shutdown."""

    @property
    def is_loaded(self) -> bool:
        """Whether a model/connection is currently held."""
        return False

    # ------------------------------------------------------------------
    # Shared audio helpers (used by multiple backends; names preserved
    # so code moved from WhisperManager keeps working verbatim)
    # ------------------------------------------------------------------

    def _resample_audio(self, audio_data: 'np.ndarray', source_rate: int, target_rate: int) -> 'np.ndarray':
        """Resample float32 mono audio when a backend requires a fixed sample rate."""
        if source_rate == target_rate:
            return audio_data
        try:
            from math import gcd
            from scipy import signal

            divisor = gcd(int(source_rate), int(target_rate))
            resampled = signal.resample_poly(
                audio_data,
                up=int(target_rate) // divisor,
                down=int(source_rate) // divisor,
            )
            return resampled.astype(np.float32, copy=False)
        except Exception as e:
            print(f"[WARN] Failed to resample audio {source_rate}Hz -> {target_rate}Hz: {e}", flush=True)
            return audio_data

    def _numpy_to_wav_bytes(self, audio_data: 'np.ndarray', sample_rate: int = 16000) -> bytes:
        """
        Convert numpy audio array to WAV format bytes (in-memory)

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data

        Returns:
            WAV file as bytes
        """
        try:
            # Ensure mono
            if audio_data.ndim != 1:
                raise ValueError(f'Expected mono audio array, got shape {audio_data.shape}')

            # Convert float32 to int16 for WAV format
            if audio_data.dtype == np.float32:
                # Ensure float never expands (possible in some mic contexts)
                audio_clipped = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)

            # Create WAV file in memory
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            return wav_buffer.getvalue()

        except Exception as e:
            print(f'ERROR: Failed to convert audio to WAV: {e}')
            raise
