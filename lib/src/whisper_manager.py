"""
Whisper manager for hyprwhspr
Facade over the transcription backend classes in backends/ — owns shared
state (config, ready flag, model lock, last-use time) and dispatches to the
configured backend instance.
"""

import threading
import time
from typing import Optional, Callable

try:
    from .dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend

try:
    from .backend_installer import PYWHISPERCPP_MODELS_DIR
except ImportError:
    from backend_installer import PYWHISPERCPP_MODELS_DIR

try:
    from .backends import BACKENDS, PywhispercppBackend
except ImportError:
    from backends import BACKENDS, PywhispercppBackend


class WhisperManager:
    """Facade that manages transcription through per-backend classes"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager

        # Whisper configuration - only set for local backend
        # Will be properly initialized in initialize() based on backend
        self.current_model = None
        self.temp_dir = None
        
        # Realtime partial-preview callback; owned by the manager (not the
        # realtime backend) so it survives backend re-creation on resume
        self._realtime_partial_callback = None

        # Thread safety for model operations
        self._model_lock = threading.Lock()

        # State
        self.ready = False

        # Track last successful transcription time for suspend/resume detection
        self._last_use_time = 0.0

        # Set when model is deliberately unloaded via unload_model() to free GPU resources
        self._model_manually_unloaded = False

        # Active TranscriptionBackend instance, created by initialize()
        self._backend = None

    def initialize(self) -> bool:
        """Initialize the configured transcription backend and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()

            backend = self._current_backend_name()

            # pywhispercpp hardware variants (cpu/nvidia/vulkan) and unknown
            # names fall back to the default pywhispercpp backend
            backend_cls = BACKENDS.get(backend, PywhispercppBackend)
            self._backend = backend_cls(self)
            return self._backend.initialize()

        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False

    def get_realtime_streaming_callback(self) -> Optional[Callable]:
        """
        Get the streaming callback for realtime-ws backend.

        Returns:
            Callback function if realtime-ws backend is active, None otherwise
        """
        if self._backend is not None and self._backend.name == 'realtime-ws':
            return self._backend.get_streaming_callback()
        return None

    def set_realtime_partial_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for realtime partial transcript previews."""
        self._realtime_partial_callback = callback
        if self._backend is not None and self._backend.name == 'realtime-ws':
            self._backend.apply_partial_callback(callback)

    def _current_backend_name(self) -> str:
        """Normalized name of the configured transcription backend."""
        return normalize_backend(
            self.config.get_setting('transcription_backend', 'pywhispercpp'))

    def update_realtime_language(self, language: Optional[str]) -> None:
        """Apply a language override to a connected realtime client (no-op otherwise)."""
        if (self._backend is not None and self._backend.name == 'realtime-ws'
                and self._backend.is_loaded):
            self._backend.update_language(language)

    def close_realtime_connection(self, reason: str = '') -> None:
        """Close the realtime WebSocket if one is active (no-op otherwise)."""
        if (self._backend is not None and self._backend.name == 'realtime-ws'
                and self._backend.is_loaded):
            note = f' ({reason})' if reason else ''
            print(f'[CLEANUP] Closing realtime WebSocket{note}', flush=True)
            self._backend.close()

    def reinitialize_after_resume(self, only_if_idle: bool = False) -> bool:
        """Recover backend state after suspend/resume or audio recovery.

        With only_if_idle, reinitialize only when a loaded local model has been
        unused long enough (>30 min) that a suspend likely invalidated its
        CUDA/GPU context; otherwise reinitialize unconditionally per backend.

        Returns:
            True if the backend is healthy (or needed no action), False otherwise
        """
        if self._backend is None or not self._backend.reinit_on_resume:
            return True

        if only_if_idle:
            if not (self._backend.is_local and self._backend.is_loaded):
                return True
            idle = time.monotonic() - self._last_use_time
            if not (idle > 1800 and self._last_use_time > 0):
                return True
            print(f"[RECOVERY] Reinitializing {self._current_backend_name()} model after audio recovery (suspend/resume detected)", flush=True)

        return self._backend.reinitialize()

    def is_ready(self) -> bool:
        """Check if whisper is ready for transcription"""
        return self.ready

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, language_override: Optional[str] = None) -> str:
        """
        Transcribe audio data using whisper

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            language_override: Optional language code to override config language (e.g., 'it', 'en', 'fr')

        Returns:
            Transcribed text string
        """
        # Check that manager is ready regardless of backend
        if not self.ready:
            raise RuntimeError('Whisper manager not initialized')

        # Check if we have valid audio data
        if audio_data is None:
            print("No audio data provided to transcribe", flush=True)
            return ""

        if len(audio_data) == 0:
            print("Empty audio data provided to transcribe", flush=True)
            return ""

        # Validate audio data format and content
        try:
            # Ensure it's a numpy array
            if not isinstance(audio_data, np.ndarray):
                print(f"Invalid audio data type: {type(audio_data)}, expected numpy.ndarray", flush=True)
                return ""
            
            # Check shape (should be 1D)
            if audio_data.ndim != 1:
                print(f"Invalid audio data shape: {audio_data.shape}, expected 1D array", flush=True)
                # Try to flatten if 2D with single channel
                if audio_data.ndim == 2 and audio_data.shape[1] == 1:
                    audio_data = audio_data.flatten()
                else:
                    return ""
            
            # Check dtype (should be float32)
            if audio_data.dtype != np.float32:
                print(f"Converting audio data from {audio_data.dtype} to float32", flush=True)
                audio_data = audio_data.astype(np.float32)
            
            # Ensure contiguous in memory (required by whisper C++ code)
            if not audio_data.flags['C_CONTIGUOUS']:
                audio_data = np.ascontiguousarray(audio_data, dtype=np.float32)
            
            # Check for NaN or inf values (invalid audio)
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                print("Audio data contains NaN or inf values - invalid", flush=True)
                return ""
            
            # Check if audio is all zeros (silence)
            if np.all(audio_data == 0.0):
                print("Audio data is all zeros (silence) - skipping transcription", flush=True)
                return ""

            # Check if audio is too short (less than 0.1 seconds)
            min_samples = int(sample_rate * 0.1)  # 0.1 seconds minimum
            if len(audio_data) < min_samples:
                print(f"Audio too short: {len(audio_data)} samples (minimum {min_samples})", flush=True)
                return ""
            
            # Check audio level (RMS) - if too quiet, might be invalid
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-6:  # Extremely quiet
                print(f"Audio level too low (RMS: {rms:.2e}) - likely invalid", flush=True)
                return ""
                
        except Exception as e:
            print(f"[ERROR] Audio data validation failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return ""

        # Route to the configured backend. Remote backends (rest-api,
        # realtime-ws) manage their own connection state and need no model lock.
        backend = self._current_backend_name()

        if backend == 'rest-api':
            if self._backend is None:
                # Stateless backend; usable even if initialize() never ran
                self._backend = BACKENDS['rest-api'](self)
            return self._backend.transcribe(audio_data, sample_rate, language_override=language_override)

        if backend == 'realtime-ws':
            # Audio was already streamed via callback during capture;
            # this just commits and waits for the result
            if self._backend is None:
                print('[REALTIME] Backend not initialized', flush=True)
                return ""
            return self._backend.transcribe(audio_data, sample_rate, language_override=language_override)

        if self._backend is None:
            print('[ERROR] No transcription backend initialized', flush=True)
            return ""

        # Use model lock to prevent concurrent transcription calls and
        # crashes from concurrent access to the loaded model
        with self._model_lock:
            if not self._ensure_backend_fresh_locked(backend):
                return ""
            return self._backend.transcribe(audio_data, sample_rate, language_override=language_override)

    def _ensure_backend_fresh_locked(self, backend: str) -> bool:
        """Reinitialize a long-idle local model (call under _model_lock).

        If the model hasn't been used in 30+ minutes a suspend/resume likely
        invalidated its GPU context; refresh it before transcribing. Runs
        inside the lock so concurrent threads can't reinitialize twice.

        Returns:
            False if a needed reinitialization failed, True otherwise
        """
        time_since_last_use = time.monotonic() - self._last_use_time
        if not (time_since_last_use > 1800 and self._last_use_time > 0):
            return True

        if self._backend is None or not self._backend.reinit_on_idle:
            return True
        reinit = self._backend.reinitialize

        print(f"[MODEL] Long idle detected - reinitializing {backend} model (suspend/resume likely)", flush=True)
        if not reinit():
            print("[MODEL] Reinitialization failed, transcription may fail", flush=True)
            return False
        return True

    def cleanup(self) -> None:
        """Public cleanup method to clean up all resources"""
        if self._backend is not None:
            self._backend.cleanup()

    def unload_model(self) -> bool:
        """
        Unload the model from memory (including GPU VRAM) to free resources for other applications.

        The service continues running with all shortcuts active; recording will be blocked
        until reload_model() is called.

        Returns:
            True if a model was unloaded, False if backend has no local model to unload.
        """
        backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
        if backend in ('rest-api', 'realtime-ws'):
            print("[MODEL] Unload not applicable for non-local backend", flush=True)
            return False

        with self._model_lock:
            try:
                if self._backend is not None:
                    self._backend.unload()

                # Trigger Python GC so C++ destructors and ONNX sessions release immediately
                import gc
                gc.collect()

                # Free cached CUDA allocations if torch is present
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("[MODEL] CUDA cache cleared", flush=True)
                except ImportError:
                    pass

                self.ready = False
                self._model_manually_unloaded = True
                print("[MODEL] Model unloaded from memory — GPU resources freed", flush=True)
                return True

            except Exception as e:
                print(f"[MODEL] ERROR: Failed to unload model: {e}", flush=True)
                return False

    def reload_model(self) -> bool:
        """
        Reload the model into memory after unload_model() was called.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        print("[MODEL] Reloading model...", flush=True)
        result = self.initialize()
        if result:
            # Clear the flag only after initialize() fully completes so that
            # _start_recording()'s guard stays active until the model is ready.
            with self._model_lock:
                self._model_manually_unloaded = False
            print("[MODEL] Model reloaded successfully", flush=True)
        else:
            print("[MODEL] ERROR: Failed to reload model", flush=True)
        return result

    def set_threads(self, num_threads: int) -> bool:
        """Update the number of threads used by the backend."""
        set_threads = getattr(self._backend, 'set_threads', None)
        if set_threads is None:
            print("ERROR: Backend does not support changing threads")
            return False
        with self._model_lock:
            return set_threads(int(num_threads))

    def set_model(self, model_name: str) -> bool:
        """
        Change the whisper model

        Args:
            model_name: Name of the model (e.g., 'base', 'small')

        Returns:
            True if successful, False otherwise
        """
        set_model = getattr(self._backend, 'set_model', None)
        if set_model is None:
            print("ERROR: Cannot change model when using REST API backend - switch to pywhispercpp")
            print("Model selection is handled by the REST API endpoint")
            return False
        with self._model_lock:
            return set_model(model_name)

    def get_current_model(self) -> str:
        """Get the current model name ('' for backends without a local model)"""
        return self.current_model or ''

    def get_available_models(self) -> list:
        """Get list of available whisper models"""
        models_dir = PYWHISPERCPP_MODELS_DIR
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
        backend = self._current_backend_name()

        if backend == 'rest-api':
            endpoint_url = self.config.get_setting('rest_endpoint_url', 'not configured')
            return f"REST API ({endpoint_url})"
        if backend not in BACKENDS:
            # pywhispercpp hardware variants (cpu/nvidia/vulkan)
            backend = 'pywhispercpp'
        if backend == 'pywhispercpp':
            return f"pywhispercpp (in-process, model: {self.current_model})"
        return f"{backend} (model: {self.current_model})" if self.current_model else backend
