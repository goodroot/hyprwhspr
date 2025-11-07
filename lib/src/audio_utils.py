"""
Audio utility functions for hyprwhspr
Shared utilities for audio file handling
"""

import numpy as np
import wave
from pathlib import Path


def save_audio_to_wav(audio_data: np.ndarray, filename: str,
                      sample_rate: int = 16000, channels: int = 1) -> None:
    """
    Save audio data to a WAV file

    Args:
        audio_data: NumPy array of audio samples (float32 or int16)
        filename: Output WAV file path
        sample_rate: Sample rate in Hz (default: 16000)
        channels: Number of audio channels (default: 1)
    """
    try:
        # Ensure the output directory exists and is writable
        output_dir = Path(filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test write permission
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()

        # Convert float32 to int16 for WAV format
        if audio_data.dtype == np.float32:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)

        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        print(f"Audio saved to {filename}")

    except PermissionError as e:
        print(f"ERROR: Cannot write to directory: {output_dir}")
        raise PermissionError(f"Cannot write to directory: {output_dir}") from e
    except OSError as e:
        print(f"ERROR: Failed to create output directory: {e}")
        raise OSError(f"Failed to create output directory: {e}") from e
    except Exception as e:
        print(f"ERROR: Failed to save audio to {filename}: {e}")
        raise
