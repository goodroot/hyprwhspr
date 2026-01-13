"""
Segment manager for long-form recording mode.

Handles saving, loading, and managing audio segments on disk for crash-safe
long-form recordings with pause/resume support.
"""

import os
import time
import wave
import uuid
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    from .paths import LONGFORM_SEGMENTS_DIR
except ImportError:
    from paths import LONGFORM_SEGMENTS_DIR


class SegmentManager:
    """Manages long-form recording segments with disk persistence."""

    SAMPLE_RATE = 16000  # Whisper expects 16kHz
    CHANNELS = 1  # Mono

    def __init__(self, max_size_mb: int = 500):
        """
        Initialize the segment manager.

        Args:
            max_size_mb: Maximum total size of segment storage in MB
        """
        self.segments_dir = LONGFORM_SEGMENTS_DIR
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.session_id = None
        self.segments: List[Path] = []
        self.segment_durations: List[float] = []  # Duration in seconds for each segment
        self.session_start_time = None

    def start_session(self):
        """Start a new recording session."""
        self.session_id = str(uuid.uuid4())[:8]
        self.segments = []
        self.segment_durations = []
        self.session_start_time = time.time()

        # Ensure directory exists
        self.segments_dir.mkdir(parents=True, exist_ok=True)

    def save_segment(self, audio_data: np.ndarray) -> Optional[Path]:
        """
        Save an audio segment to disk.

        Args:
            audio_data: Numpy array of audio samples (float32, 16kHz mono)

        Returns:
            Path to saved segment file, or None if failed
        """
        if self.session_id is None:
            self.start_session()

        if audio_data is None or len(audio_data) == 0:
            return None

        # Generate segment filename
        segment_index = len(self.segments)
        timestamp = int(time.time())
        filename = f"{self.session_id}_{segment_index:03d}_{timestamp}.wav"
        filepath = self.segments_dir / filename

        try:
            # Convert float32 to int16 for WAV format
            if audio_data.dtype == np.float32:
                audio_clipped = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)

            # Write WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())

            # Track segment
            self.segments.append(filepath)
            duration = len(audio_data) / self.SAMPLE_RATE
            self.segment_durations.append(duration)

            print(f"[SEGMENT] Saved segment {segment_index}: {filepath.name} ({duration:.1f}s)")
            return filepath

        except Exception as e:
            print(f"[SEGMENT] Failed to save segment: {e}")
            return None

    def load_segment(self, filepath: Path) -> Optional[np.ndarray]:
        """
        Load an audio segment from disk.

        Args:
            filepath: Path to segment WAV file

        Returns:
            Numpy array of audio samples (float32), or None if failed
        """
        try:
            with wave.open(str(filepath), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0
                return audio_float32
        except Exception as e:
            print(f"[SEGMENT] Failed to load segment {filepath}: {e}")
            return None

    def concatenate_all(self) -> Optional[np.ndarray]:
        """
        Load and concatenate all segments for current session.

        Returns:
            Concatenated audio data as numpy array, or None if no segments
        """
        if not self.segments:
            return None

        all_audio = []
        for filepath in self.segments:
            audio = self.load_segment(filepath)
            if audio is not None:
                all_audio.append(audio)

        if not all_audio:
            return None

        return np.concatenate(all_audio)

    def get_total_size(self) -> int:
        """Return total bytes of all segments in the directory."""
        total = 0
        try:
            if self.segments_dir.exists():
                for f in self.segments_dir.iterdir():
                    if f.is_file() and f.suffix == '.wav':
                        total += f.stat().st_size
        except Exception:
            pass
        return total

    def get_session_size(self) -> int:
        """Return total bytes of segments in current session."""
        total = 0
        for filepath in self.segments:
            try:
                if filepath.exists():
                    total += filepath.stat().st_size
            except Exception:
                pass
        return total

    def get_elapsed_seconds(self) -> float:
        """Calculate total recording time across all segments."""
        return sum(self.segment_durations)

    def cleanup_oldest(self) -> bool:
        """
        Remove oldest segments if over size limit.

        Returns:
            True if any segments were removed
        """
        removed = False
        try:
            while self.get_total_size() > self.max_size_bytes:
                # Find oldest segment file (not in current session)
                oldest = None
                oldest_time = float('inf')

                for f in self.segments_dir.iterdir():
                    if f.is_file() and f.suffix == '.wav':
                        # Skip files in current session
                        if self.session_id and f.name.startswith(self.session_id):
                            continue
                        try:
                            mtime = f.stat().st_mtime
                            if mtime < oldest_time:
                                oldest_time = mtime
                                oldest = f
                        except Exception:
                            pass

                if oldest is None:
                    break  # No more old segments to remove

                print(f"[SEGMENT] Removing old segment to free space: {oldest.name}")
                oldest.unlink()
                removed = True

        except Exception as e:
            print(f"[SEGMENT] Error during cleanup: {e}")

        return removed

    def clear_session(self):
        """Delete all segments for current session."""
        for filepath in self.segments:
            try:
                if filepath.exists():
                    filepath.unlink()
                    print(f"[SEGMENT] Deleted segment: {filepath.name}")
            except Exception as e:
                print(f"[SEGMENT] Failed to delete {filepath}: {e}")

        self.segments = []
        self.segment_durations = []
        self.session_id = None
        self.session_start_time = None

    def clear_all(self):
        """Delete all segments in the directory."""
        try:
            if self.segments_dir.exists():
                for f in self.segments_dir.iterdir():
                    if f.is_file() and f.suffix == '.wav':
                        f.unlink()
                        print(f"[SEGMENT] Deleted: {f.name}")
        except Exception as e:
            print(f"[SEGMENT] Error clearing all segments: {e}")

        self.segments = []
        self.segment_durations = []
        self.session_id = None
        self.session_start_time = None

    def has_segments(self) -> bool:
        """Return True if current session has any segments."""
        return len(self.segments) > 0

    def get_segment_count(self) -> int:
        """Return number of segments in current session."""
        return len(self.segments)
