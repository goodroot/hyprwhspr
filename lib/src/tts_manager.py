"""
TTS Manager for hyprwhspr - Pocket TTS text-to-speech synthesis.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

try:
    from .paths import TEMP_DIR
    from .audio_manager import AudioManager
except ImportError:
    from paths import TEMP_DIR
    from audio_manager import AudioManager

# Pocket TTS built-in voices (English only)
POCKET_TTS_VOICES = [
    'alba', 'marius', 'javert', 'jean', 'fantine',
    'cosette', 'eponine', 'azelma'
]


def _log_tts(msg: str) -> None:
    """Print TTS-related log to stderr (visible when speak runs as subprocess)."""
    print(f"[TTS] {msg}", file=sys.stderr, flush=True)


class TTSManager:
    """Manages Pocket TTS synthesis and playback."""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self._model = None
        self._voice_state_cache = {}
        self._output_path = TEMP_DIR / 'tts_output.wav'

        if self.config_manager:
            self.voice = self.config_manager.get_setting('tts_voice', 'alba')
            self.volume = float(self.config_manager.get_setting('tts_volume', 1.0))
            self.volume = max(0.1, min(1.0, self.volume))
        else:
            self.voice = 'alba'
            self.volume = 1.0

    def get_text_from_input(
        self,
        use_primary: bool = True,
        use_clipboard: bool = True,
        explicit_text: Optional[str] = None
    ) -> str:
        """
        Get text from primary selection, clipboard, or explicit argument.

        Order: explicit_text > primary selection > clipboard
        """
        if explicit_text and str(explicit_text).strip():
            return str(explicit_text).strip()

        text = ''
        if use_primary:
            try:
                result = subprocess.run(
                    ['wl-paste', '-p'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout:
                    text = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        if not text and use_clipboard:
            try:
                result = subprocess.run(
                    ['wl-paste'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout:
                    text = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

        return text or ''

    def _ensure_model(self):
        """Load Pocket TTS model (cached)."""
        if self._model is None:
            t0 = time.perf_counter()
            from pocket_tts import TTSModel
            self._model = TTSModel.load_model()
            elapsed = time.perf_counter() - t0
            _log_tts(f"Model load: {elapsed:.2f}s")

    def _get_voice_state(self, voice: Optional[str] = None):
        """Get voice state for synthesis (cached per voice)."""
        voice = voice or self.voice
        if voice not in self._voice_state_cache:
            self._ensure_model()
            t0 = time.perf_counter()
            self._voice_state_cache[voice] = self._model.get_state_for_audio_prompt(voice)
            elapsed = time.perf_counter() - t0
            _log_tts(f"Voice load ({voice}): {elapsed:.2f}s")
        return self._voice_state_cache[voice]

    def _get_streaming_player_cmd(self, volume: float) -> Optional[list]:
        """Return command for a player that reads raw PCM s16le from stdin, or None."""
        vol = max(0.1, min(1.0, float(volume)))
        sr = str(self._model.sample_rate)
        # Prefer PipeWire/PulseAudio players (native on Wayland) - they tend to handle
        # stdin streaming better than ffplay/aplay which can exit early (broken pipe)
        try:
            r = subprocess.run(['which', 'pw-play'], capture_output=True, timeout=2)
            if r.returncode == 0:
                return [
                    'pw-play', '-a', '-',
                    '--rate', sr, '--channels', '1', '--format', 's16',
                    '--volume', str(vol),
                ]
        except Exception:
            pass
        try:
            r = subprocess.run(['which', 'paplay'], capture_output=True, timeout=2)
            if r.returncode == 0:
                return [
                    'paplay', '--raw', '--format=s16le', f'--rate={sr}',
                    '--channels=1', '-',
                ]
        except Exception:
            pass
        try:
            r = subprocess.run(['which', 'aplay'], capture_output=True, timeout=2)
            if r.returncode == 0:
                return ['aplay', '-q', '-t', 'raw', '-f', 'S16_LE', '-r', sr, '-c', '1', '-']
        except Exception:
            pass
        try:
            r = subprocess.run(['which', 'ffplay'], capture_output=True, timeout=2)
            if r.returncode == 0:
                return [
                    'ffplay', '-nodisp', '-autoexit', '-loglevel', 'error',
                    '-f', 's16le', '-ar', sr, '-ac', '1',
                    '-volume', str(int(vol * 100)), '-i', 'pipe:0'
                ]
        except Exception:
            pass
        return None

    def synthesize_and_play_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        volume: Optional[float] = None,
        on_playback_started: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Synthesize text and play audio as it streams (starts playback before synthesis completes).

        on_playback_started: Optional callback invoked when the first audio chunk is sent (playback begins).

        Returns True on success, False on failure.
        """
        if not text or not text.strip():
            return False

        vol = volume if volume is not None else self.volume
        vol = max(0.1, min(1.0, float(vol)))

        t0_start = time.perf_counter()
        try:
            self._ensure_model()
            voice_state = self._get_voice_state(voice)

            cmd = self._get_streaming_player_cmd(vol)
            if not cmd:
                _log_tts("No streaming player (pw-play/paplay/aplay/ffplay) found, falling back to non-streaming")
                wav = self.synthesize(text, voice)
                if wav and self.config_manager:
                    am = AudioManager(self.config_manager)
                    return bool(am.play_file(wav, volume=vol, blocking=True))
                return False

            first_chunk_logged = False
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                import numpy as np
                for chunk in self._model.generate_audio_stream(voice_state, text):
                    if not first_chunk_logged:
                        first_chunk_logged = True
                        elapsed = time.perf_counter() - t0_start
                        _log_tts(f"First audio chunk: {elapsed:.2f}s from start (playback begins)")
                        if on_playback_started:
                            try:
                                on_playback_started()
                            except Exception:
                                pass
                    if hasattr(chunk, 'numpy'):
                        arr = chunk.cpu().numpy() if hasattr(chunk, 'cpu') else chunk.numpy()
                    else:
                        arr = chunk
                    arr = np.asarray(arr)
                    if arr.dtype.kind == 'f':
                        arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                    try:
                        proc.stdin.write(arr.tobytes())
                        proc.stdin.flush()
                    except BrokenPipeError:
                        _log_tts("Player exited early (broken pipe) - falling back to non-streaming")
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass
                        proc.wait()
                        raise BrokenPipeError("Streaming player exited early")
            finally:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                proc.wait()

            return proc.returncode == 0
        except BrokenPipeError:
            _log_tts("Streaming failed, retrying with non-streaming (synthesize then play)")
            wav = self.synthesize(text, voice)
            if wav and self.config_manager:
                am = AudioManager(self.config_manager)
                return bool(am.play_file(wav, volume=vol, blocking=True))
            return False
        except Exception as e:
            _log_tts(f"Synthesis/playback failed: {e}")
            return False

    def synthesize(self, text: str, voice: Optional[str] = None) -> Optional[Path]:
        """
        Synthesize text to WAV file (non-streaming; use synthesize_and_play_streaming for low latency).

        Args:
            text: Text to speak
            voice: Voice name (e.g. 'alba'). Uses config default if None.

        Returns:
            Path to WAV file, or None on failure
        """
        if not text or not text.strip():
            return None

        try:
            self._ensure_model()
            voice_state = self._get_voice_state(voice)
            t0 = time.perf_counter()
            audio = self._model.generate_audio(voice_state, text)
            elapsed = time.perf_counter() - t0
            _log_tts(f"Full synthesis: {elapsed:.2f}s")

            import scipy.io.wavfile
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(
                str(self._output_path),
                self._model.sample_rate,
                audio.numpy()
            )
            return self._output_path
        except Exception as e:
            _log_tts(f"Synthesis failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Pocket TTS is available (installed and importable)."""
        try:
            import pocket_tts  # noqa: F401
            return True
        except ImportError:
            return False
