"""
Parakeet manager for hyprwhspr
ONNX-based backend using onnx-asr library
"""

import os
import time
import numpy as np
import threading
from pathlib import Path
from typing import Optional

try:
    from .config_manager import ConfigManager
    from .stt_backend import STTBackend
    from .audio_utils import save_audio_to_wav
except ImportError:
    from config_manager import ConfigManager
    from stt_backend import STTBackend
    from audio_utils import save_audio_to_wav


class ParakeetManager(STTBackend):
    """Manages Parakeet TDT v3 transcription with ONNX Runtime"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager

        # Parakeet configuration
        self.current_model = self.config.get_setting('parakeet_model', 'nemo-parakeet-tdt-0.6b-v3')
        self.custom_model_path = self.config.get_setting('parakeet_model_path', None)

        # Backend-specific attributes
        self._onnx_model = None
        self.temp_dir = None

        # Thread safety for model operations
        self._model_lock = threading.Lock()

        # State
        self.ready = False

    def initialize(self) -> bool:
        """Initialize the Parakeet manager and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()

            try:
                import onnx_asr

                print(f"[onnx-asr] Initializing model: {self.current_model}")
                print(f"[onnx-asr] Note: First-time download is ~3.2 GB")

                # Load model (handles download if needed)
                model_name = self.current_model
                if self.config.get_setting('parakeet_use_quantized', False):
                    model_name += "-int8"
                    print(f"[onnx-asr] Using quantized model: {model_name}")

                if self.custom_model_path:
                    self._onnx_model = onnx_asr.load_model(self.custom_model_path)
                else:
                    self._onnx_model = onnx_asr.load_model(model_name)

                print("[onnx-asr] Model loaded successfully", flush=True)
                print(f"[BACKEND] Using onnx-asr (ONNX Runtime) for Parakeet TDT v3", flush=True)

                self.ready = True
                return True

            except ImportError as e:
                print(f"[onnx-asr] Import failed: {e}")
                print("ERROR: onnx-asr not installed. Run: pip install onnx-asr>=0.7.0")
                return False
            except Exception as e:
                print(f"[onnx-asr] Initialization failed: {e}")
                print(f"ERROR: Parakeet model not found")
                print(f"Please download from: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx")
                print(f"Expected files: .onnx, .tokenizer.json, vocab.txt")
                return False

        except Exception as e:
            print(f"ERROR: Failed to initialize Parakeet manager: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if Parakeet is ready for transcription"""
        return self.ready

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using Parakeet ONNX model

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data

        Returns:
            Transcribed text string
        """
        if not self.ready:
            raise RuntimeError("Parakeet manager not initialized")

        print("[TRANSCRIBE] Using onnx-asr backend", flush=True)

        # Check if we have valid audio data
        if audio_data is None:
            print("No audio data provided to transcribe")
            return ""

        if len(audio_data) == 0:
            print("Empty audio data provided to transcribe")
            return ""

        # Check if audio is too short (less than 0.1 seconds)
        min_samples = int(sample_rate * 0.1)  # 0.1 seconds minimum
        if len(audio_data) < min_samples:
            print(f"Audio too short: {len(audio_data)} samples (minimum {min_samples})")
            return ""

        temp_filename = None
        try:
            # Parakeet expects file path, so we need to save to temp file
            temp_filename = self.temp_dir / f"parakeet_temp_{time.time()}.wav"

            # Save audio to WAV file using shared utility
            save_audio_to_wav(audio_data, str(temp_filename), sample_rate=sample_rate)

            # Transcribe using ONNX model
            transcription = self._onnx_model.recognize(str(temp_filename))

            return transcription.strip() if transcription else ""

        except Exception as e:
            print(f"ERROR: onnx-asr transcription failed: {e}")
            return ""
        finally:
            # Always clean up temp file
            if temp_filename and temp_filename.exists():
                try:
                    temp_filename.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete temp file {temp_filename}: {e}")

    def set_threads(self, num_threads: int) -> bool:
        """
        Update thread count for ONNX Runtime
        Note: ONNX Runtime may not support dynamic thread changes
        """
        with self._model_lock:
            try:
                # Store setting for future model loads
                self.config.set_setting('threads', int(num_threads))
                print(f"[onnx-asr] Thread setting saved: {int(num_threads)}")
                print(f"[onnx-asr] Note: Reload model to apply thread changes")
                return True
            except Exception as e:
                print(f"ERROR: Failed to set threads: {e}")
                return False

    def set_model(self, model_name: str) -> bool:
        """
        Change the Parakeet model

        Args:
            model_name: Name or path of the model

        Returns:
            True if successful, False otherwise
        """
        with self._model_lock:
            try:
                print(f"[onnx-asr] Switching to model: {model_name}")

                import onnx_asr

                # Apply quantized suffix if enabled
                effective_model_name = model_name
                if self.config.get_setting('parakeet_use_quantized', False):
                    effective_model_name += "-int8"
                    print(f"[onnx-asr] Using quantized model: {effective_model_name}")

                # Clean up existing model
                self._onnx_model = None

                # Load new model
                self._onnx_model = onnx_asr.load_model(effective_model_name)

                # Update state
                self.current_model = model_name
                self.config.set_setting('parakeet_model', model_name)
                print(f"[onnx-asr] Successfully switched to model: {model_name}")

                return True

            except Exception as e:
                print(f"ERROR: Failed to set model {model_name}: {e}")
                self.ready = False
                return False

    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.current_model

    def get_available_models(self) -> list:
        """Get list of available Parakeet models"""
        models_dir = Path.home() / '.local' / 'share' / 'pywhispercpp' / 'models' / 'parakeet'
        available_models = []

        if not models_dir.exists():
            return available_models

        # Look for complete Parakeet model sets
        # Parakeet TDT v3 uses encoder-model.onnx, decoder_joint-model.onnx, and vocab.txt
        encoder_onnx = models_dir / "encoder-model.onnx"
        decoder_onnx = models_dir / "decoder_joint-model.onnx"
        vocab_file = models_dir / "vocab.txt"

        if encoder_onnx.exists() and decoder_onnx.exists() and vocab_file.exists():
            # Use a fixed model name since onnx-asr expects specific structure
            available_models.append("nemo-parakeet-tdt-0.6b-v3")

        return sorted(available_models)

    def get_backend_info(self) -> str:
        """Get information about the current backend"""
        return f"onnx-asr (ONNX Runtime, model: {self.current_model})"
