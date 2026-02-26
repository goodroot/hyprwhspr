"""
TTS Manager for hyprwhspr - Pocket TTS text-to-speech synthesis.
"""

import subprocess
from pathlib import Path
from typing import Optional

try:
    from .paths import TEMP_DIR
except ImportError:
    from paths import TEMP_DIR

# Pocket TTS built-in voices (English only)
POCKET_TTS_VOICES = [
    'alba', 'marius', 'javert', 'jean', 'fantine',
    'cosette', 'eponine', 'azelma'
]


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
            from pocket_tts import TTSModel
            self._model = TTSModel.load_model()

    def _get_voice_state(self, voice: Optional[str] = None):
        """Get voice state for synthesis (cached per voice)."""
        voice = voice or self.voice
        if voice not in self._voice_state_cache:
            self._ensure_model()
            self._voice_state_cache[voice] = self._model.get_state_for_audio_prompt(voice)
        return self._voice_state_cache[voice]

    def synthesize(self, text: str, voice: Optional[str] = None) -> Optional[Path]:
        """
        Synthesize text to WAV file.

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
            audio = self._model.generate_audio(voice_state, text)

            import scipy.io.wavfile
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(
                str(self._output_path),
                self._model.sample_rate,
                audio.numpy()
            )
            return self._output_path
        except Exception as e:
            print(f"TTS synthesis failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if Pocket TTS is available (installed and importable)."""
        try:
            import pocket_tts  # noqa: F401
            return True
        except ImportError:
            return False
