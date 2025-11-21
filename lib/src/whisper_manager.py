"""
Whisper manager for hyprwhspr
PyWhisperCPP-only backend (in-process, model kept hot)
"""

import os
import shutil
import threading
import wave
from io import BytesIO
from typing import Optional

import numpy as np
import requests

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager


class WhisperManager:
    """Manages whisper transcription with dual backend support"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager

        # Whisper configuration
        self.current_model = self.config.get_setting('model', 'base')
        # Backend-specific attributes (pywhispercpp only)
        self._pywhisper_model = None
        self.temp_dir = None

        # Thread safety for model operations
        self._model_lock = threading.Lock()

        # State
        self.ready = False

    def initialize(self) -> bool:
        """Initialize the whisper manager and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()

            # Check which backend is configured
            backend = self.config.get_setting('transcription_backend', 'local')

            # Configure remote API backend and return early
            if backend == 'remote':
                # Validate REST configuration
                endpoint_url = self.config.get_setting('rest_endpoint_url')

                if not endpoint_url:
                    print('ERROR: REST backend selected but rest_endpoint_url not configured')
                    return False

                if not endpoint_url.startswith('https://') and not endpoint_url.startswith('http://'):
                    print(f'WARNING: REST endpoint URL should start with https:// or http://: {endpoint_url}')

                # Validate timeout is reasonable
                timeout = self.config.get_setting('rest_timeout', 30)
                if timeout < 1 or timeout > 300:
                    print(f'WARNING: REST timeout should be between 1-300 seconds, got {timeout}')

                print(f'[BACKEND] Using remote REST API: {endpoint_url}')
                print(f'[REST] Timeout configured: {timeout}s')

                # Mark as ready for REST mode
                self.ready = True
                return True

            # Initialize local pywhispercpp backend
            # Detect GPU backend for logging
            gpu_backend = self._detect_gpu_backend()

            try:
                from pywhispercpp.model import Model

                print(f"[pywhispercpp] Initializing model: {self.current_model}")
                print(f"[pywhispercpp] Detected GPU backend: {gpu_backend}")

                self._pywhisper_model = Model(
                    model=self.current_model,
                    n_threads=self.config.get_setting('threads', 4),
                    redirect_whispercpp_logs_to=None
                )

                print("[pywhispercpp] Model loaded successfully", flush=True)
                print(f"[BACKEND] Using pywhispercpp (in-process) with {gpu_backend} acceleration", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                self.ready = True
                return True

            except ImportError as e:
                print(f"[pywhispercpp] Import failed: {e}")
                return False
            except Exception as e:
                print(f"[pywhispercpp] Initialization failed: {e}")
                return False

        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False

    def _detect_gpu_backend(self) -> str:
        """Detect available GPU backend for logging purposes"""
        # Check NVIDIA CUDA
        if shutil.which('nvidia-smi'):
            return "CUDA (NVIDIA)"
        # Check AMD ROCm
        if shutil.which('rocm-smi') or os.path.exists('/opt/rocm'):
            return "ROCm (AMD)"
        # Check Vulkan
        if shutil.which('vulkaninfo'):
            return "Vulkan"
        return "CPU"

    def _numpy_to_wav_bytes(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        Convert numpy audio array to WAV format bytes (in-memory)

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data

        Returns:
            WAV file as bytes
        """
        try:
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

    def _transcribe_rest(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using remote REST API endpoint

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data

        Returns:
            Transcribed text string
        """
        try:

            # Get REST endpoint configuration
            endpoint_url = self.config.get_setting('rest_endpoint_url')
            api_key = self.config.get_setting('rest_api_key')
            timeout = self.config.get_setting('rest_timeout', 30)

            if not endpoint_url:
                raise ValueError('REST endpoint URL not configured')

            print(f'[TRANSCRIBE] Using remote REST API: {endpoint_url}', flush=True)

            # Convert audio to WAV format
            wav_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
            print(
                f'[REST] Converted audio to WAV format ({len(wav_bytes)} bytes)',
                flush=True,
            )

            # Prepare the request
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}

            headers = {'Accept': 'application/json'}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            # Add language parameter if configured
            data = {}
            language = self.config.get_setting('language', None)
            if language:
                data['language'] = language

            # Send the request
            print(f'[REST] Sending request to {endpoint_url}...', flush=True)
            response = requests.post(endpoint_url, files=files, data=data, headers=headers, timeout=timeout)

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f'REST API returned status {response.status_code}'
                try:
                    error_detail = response.json()
                    error_msg += f': {error_detail}'
                except:
                    error_msg += f': {response.text[:200]}'
                print(f'ERROR: {error_msg}')
                return ''

            # Parse the response
            result = response.json()

            # Try common response formats
            transcription = ''
            if 'text' in result:
                transcription = result['text']
            elif 'transcription' in result:
                transcription = result['transcription']
            elif 'result' in result:
                transcription = result['result']
            else:
                print(f'ERROR: Unexpected response format: {result}')
                return ''

            print(
                f'[REST] Transcription received ({len(transcription)} chars)',
                flush=True,
            )
            return transcription.strip()

        except requests.exceptions.Timeout:
            print(f'ERROR: REST API request timed out after {timeout}s')
            return ''
        except requests.exceptions.ConnectionError as e:
            print(f'ERROR: Failed to connect to REST API: {e}')
            return ''
        except requests.exceptions.RequestException as e:
            print(f'ERROR: REST API request failed: {e}')
            return ''
        except Exception as e:
            print(f'ERROR: REST transcription failed: {e}')
            return ''

    def is_ready(self) -> bool:
        """Check if whisper is ready for transcription"""
        return self.ready

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using whisper

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data

        Returns:
            Transcribed text string
        """
        # Check that manager is ready regardless of backend
        if not self.ready:
            raise RuntimeError('Whisper manager not initialized')

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

        # Route to appropriate backend
        backend = self.config.get_setting('transcription_backend', 'local')

        if backend == 'remote':
            # Use REST API transcription
            return self._transcribe_rest(audio_data, sample_rate)

        print('[TRANSCRIBE] Using pywhispercpp backend', flush=True)
        import sys

        sys.stdout.flush()
        sys.stderr.flush()

        try:
            # Get language setting from config (None = auto-detect)
            language = self.config.get_setting('language', None)

            # Transcribe with language parameter if specified
            if language:
                segments = self._pywhisper_model.transcribe(audio_data, language=language)
            else:
                segments = self._pywhisper_model.transcribe(audio_data)

            return ' '.join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"ERROR: pywhispercpp transcription failed: {e}")
            return ""

    def _validate_model_file(self, model_name: str) -> bool:
        """Validate that model file exists and is not corrupted"""
        from pathlib import Path
        models_dir = Path.home() / '.local' / 'share' / 'pywhispercpp' / 'models'

        # Check for both multilingual and English-only versions
        model_files = [
            models_dir / f"ggml-{model_name}.bin",
            models_dir / f"ggml-{model_name}.en.bin"
        ]

        for model_file in model_files:
            if model_file.exists():
                # Basic size check (>10MB for any valid model)
                if model_file.stat().st_size > 10000000:
                    return True

        return False

    def _cleanup_model(self) -> None:
        """Safely cleanup existing model instance - GPU-safe approach"""
        if self._pywhisper_model:
            try:
                # Conservative cleanup for GPU compatibility
                # Just clear the reference and let Python handle cleanup
                # Aggressive cleanup (del + gc.collect) corrupts CUDA contexts
                self._pywhisper_model = None
                print("[pywhispercpp] Model reference cleared (GPU-safe cleanup)")
            except Exception as e:
                print(f"Warning: Failed to cleanup model reference: {e}")

    def set_threads(self, num_threads: int) -> bool:
        """Update the number of threads used by the backend."""
        with self._model_lock:
            try:
                # Try dynamic update if the backend supports it
                if self._pywhisper_model and hasattr(self._pywhisper_model, 'set_n_threads'):
                    try:
                        self._pywhisper_model.set_n_threads(int(num_threads))
                        self.config.set_setting('threads', int(num_threads))
                        print(f"[pywhispercpp] Threads updated to {int(num_threads)}")
                        return True
                    except Exception:
                        pass

                # Fallback: reload model with new thread count
                print(f"[pywhispercpp] Reloading model with threads={int(num_threads)}")

                # Clean up existing model (GPU-safe)
                self._cleanup_model()

                # Load model with new thread count
                from pywhispercpp.model import Model
                self._pywhisper_model = Model(
                    model=self.current_model,
                    n_threads=int(num_threads),
                    redirect_whispercpp_logs_to=None
                )

                # Only persist to config if successful
                self.config.set_setting('threads', int(num_threads))
                print(f"[pywhispercpp] Model reloaded with threads={int(num_threads)}")
                return True

            except Exception as e:
                print(f"ERROR: Failed to set threads: {e}")
                self.ready = False
                return False

    def set_model(self, model_name: str) -> bool:
        """
        Change the whisper model

        Args:
            model_name: Name of the model (e.g., 'base', 'small')

        Returns:
            True if successful, False otherwise
        """
        with self._model_lock:
            try:
                # Validate model file exists before attempting to load
                if not self._validate_model_file(model_name):
                    print(f"ERROR: Model file not found or corrupted: {model_name}")
                    print("Please download the model to ~/.local/share/pywhispercpp/models/")
                    return False

                print(f"[pywhispercpp] Switching to model: {model_name}")

                # Clean up existing model (GPU-safe)
                self._cleanup_model()

                # Load new model
                from pywhispercpp.model import Model
                self._pywhisper_model = Model(
                    model=model_name,
                    n_threads=self.config.get_setting('threads', 4),
                    redirect_whispercpp_logs_to=None
                )

                # Only update state if model loading succeeded
                self.current_model = model_name
                self.config.set_setting('model', model_name)
                print(f"[pywhispercpp] Successfully switched to model: {model_name}")

                return True

            except Exception as e:
                print(f"ERROR: Failed to set model {model_name}: {e}")
                self.ready = False
                return False

    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.current_model

    def get_available_models(self) -> list:
        """Get list of available whisper models"""
        from pathlib import Path
        models_dir = Path.home() / '.local' / 'share' / 'pywhispercpp' / 'models'
        available_models = []

        # Look for the supported model files
        supported_models = ['tiny', 'base', 'small', 'medium', 'large']

        for model in supported_models:
            # Check for both multilingual and English-only versions
            # Prefer multilingual for better language auto-detection
            model_files = [
                models_dir / f"ggml-{model}.bin",      # Multilingual (preferred)
                models_dir / f"ggml-{model}.en.bin"   # English-only (fallback)
            ]

            for model_file in model_files:
                if model_file.exists():
                    # Add model name with suffix if it's English-only
                    if model_file.name.endswith('.en.bin'):
                        model_name = f"{model}.en"
                    else:
                        model_name = model

                    if model_name not in available_models:
                        available_models.append(model_name)
                    break  # Don't add both versions of same model

        return sorted(available_models)

    def get_backend_info(self) -> str:
        """Get information about the current backend"""
        return f"pywhispercpp (in-process, model: {self.current_model})"
