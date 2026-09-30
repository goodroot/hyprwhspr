"""
pywhispercpp transcription backend (whisper.cpp in-process).

Default local backend: keeps a whisper.cpp model hot in memory, with
optional native Silero VAD. The cpu/nvidia/vulkan backend names all
resolve to this class; the build variant decides the compute device.
"""

import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Optional

try:
    from ..service_log import log
except ImportError:
    from service_log import log

try:
    from ..dependencies import require_package
    from ..text_script import join_segments
except ImportError:
    from dependencies import require_package
    from text_script import join_segments

np = require_package('numpy')

try:
    from ..backend_utils import normalize_backend, vulkaninfo_has_hardware_gpu
    from ..backend_installer import PYWHISPERCPP_MODELS_DIR, VAD_MODEL_FILENAME, download_vad_model
except ImportError:
    from backend_utils import normalize_backend, vulkaninfo_has_hardware_gpu
    from backend_installer import PYWHISPERCPP_MODELS_DIR, VAD_MODEL_FILENAME, download_vad_model

from .base import TranscriptionBackend


class PywhispercppBackend(TranscriptionBackend):
    """whisper.cpp backend; GPU context needs a refresh after long idle/resume."""

    name = 'pywhispercpp'
    reinit_on_idle = True
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        # Backend-specific attributes (pywhispercpp only)
        self._pywhisper_model = None

    def initialize(self) -> bool:
        """Initialize local pywhispercpp backend"""
        self.current_model = self.config.get_setting('model', 'base')

        # Detect GPU backend for logging
        gpu_backend = self._detect_gpu_backend()

        try:
            # Validate model file exists before attempting to load
            if not self._model_files(self.current_model):
                model_file = PYWHISPERCPP_MODELS_DIR / f"ggml-{self.current_model}.bin"
                log(f"[ERROR] Model file not found: {model_file}")
                log(f"[ERROR] Download with: hyprwhspr model download {self.current_model}")
                return False

            self._pywhisper_model = self._create_pywhisper_model(
                self.current_model,
                self.config.get_setting('threads', 4)
            )

            log(f"[BACKEND] pywhispercpp ({gpu_backend}) - model: {self.current_model}")
            self.ready = True
            # Record initialization time for suspend/resume detection
            self._last_use_time = time.monotonic()
            return True

        except ImportError as e:
            log("")
            log("ERROR: pywhispercpp is not installed or incompatible.")
            log(f"Import error: {e}")
            log("Run: hyprwhspr setup to configure a backend.")

            log("")
            return False
        except Exception as e:
            log(f"[ERROR] pywhispercpp initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_pywhisper_model(self, model_name: str, n_threads: int):
        """Construct a pywhispercpp Model with configured sampling and optional native VAD"""
        # Try modern layout first
        try:
            from pywhispercpp.model import Model
        except ImportError:
            # Fallback for flat layout (or older versions)
            from pywhispercpp import Model

        strategy_int = 1 if self.config.get_setting('sampling_strategy', 'beam_search') == 'beam_search' else 0
        kwargs = {
            'model': self._model_arg(model_name),
            'n_threads': n_threads,
            'params_sampling_strategy': strategy_int,
            'redirect_whispercpp_logs_to': None,
        }

        if self.config.get_setting('pywhispercpp_use_vad', False):
            vad_path = self._resolve_vad_model()
            if vad_path is not None:
                kwargs['vad'] = True
                kwargs['vad_model_path'] = str(vad_path)

        try:
            return Model(**kwargs)
        except TypeError:
            if 'vad' not in kwargs:
                raise
            # Installed pywhispercpp predates VAD support (< 1.5.0)
            log("[BACKEND] WARNING: installed pywhispercpp has no VAD support - re-run 'hyprwhspr setup' to upgrade; continuing without VAD")
            del kwargs['vad'], kwargs['vad_model_path']
            return Model(**kwargs)

    def _resolve_vad_model(self):
        """Path to the Silero VAD model, auto-downloading if missing; None on failure"""
        vad_file = PYWHISPERCPP_MODELS_DIR / VAD_MODEL_FILENAME
        if not vad_file.exists():
            log("[BACKEND] Downloading Silero VAD model (~1MB)")
            if not download_vad_model():
                log("[BACKEND] WARNING: VAD model download failed - continuing without VAD")
                return None
        return vad_file

    def _detect_gpu_backend(self) -> str:
        """Detect available GPU backend for logging purposes
        
        First checks which pywhispercpp package is installed via pacman,
        then verifies hardware availability. Falls back to hardware-only
        detection if package detection fails.
        """
        import subprocess

        # First, check which pywhispercpp package is actually installed
        # This tells us what the package supports, not just what hardware exists
        try:
            # Check each package variant individually
            # pacman -Q returns non-zero if package not found, so we use check=False
            packages_to_check = [
                'python-pywhispercpp-cuda',
                'python-pywhispercpp-rocm',
                'python-pywhispercpp-cpu'
            ]
            
            installed_package = None
            for pkg_name in packages_to_check:
                try:
                    result = subprocess.run(
                        ['pacman', '-Q', pkg_name],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False
                    )
                    if result.returncode == 0:
                        installed_package = pkg_name
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    # pacman not available or error - will fall back to hardware detection
                    break
            
            # Handle detected package
            if installed_package == 'python-pywhispercpp-cuda':
                # CUDA package installed - verify hardware
                if shutil.which('nvidia-smi'):
                    try:
                        hw_check = subprocess.run(
                            ['nvidia-smi', '-L'],
                            capture_output=True,
                            timeout=2
                        )
                        if hw_check.returncode == 0:
                            return "CUDA (NVIDIA)"
                        else:
                            return "CUDA (NVIDIA) - hardware unavailable"
                    except (subprocess.TimeoutExpired, Exception):
                        return "CUDA (NVIDIA) - hardware check failed"
                else:
                    return "CUDA (NVIDIA) - hardware not detected"
            
            elif installed_package == 'python-pywhispercpp-rocm':
                # ROCm package installed - verify hardware
                if shutil.which('rocm-smi') or os.path.exists('/opt/rocm'):
                    try:
                        hw_check = subprocess.run(
                            ['rocm-smi', '--showproductname'],
                            capture_output=True,
                            timeout=2
                        )
                        if hw_check.returncode == 0:
                            return "ROCm (AMD)"
                        else:
                            return "ROCm (AMD) - hardware unavailable"
                    except (subprocess.TimeoutExpired, Exception):
                        return "ROCm (AMD) - hardware check failed"
                else:
                    return "ROCm (AMD) - hardware not detected"
            
            elif installed_package == 'python-pywhispercpp-cpu':
                # CPU package installed - always CPU, no hardware check needed
                return "CPU"
                
        except Exception:
            # Package detection failed - fall back to config/hardware detection
            pass

        # Check config for pip-installed variant
        backend_config = self.config.get_setting('transcription_backend', 'pywhispercpp')
        if backend_config == 'nvidia':
            if shutil.which('nvidia-smi'):
                try:
                    result = subprocess.run(['nvidia-smi', '-L'],
                                           capture_output=True,
                                           timeout=2)
                    if result.returncode == 0:
                        return "CUDA (NVIDIA)"
                except (subprocess.TimeoutExpired, Exception):
                    pass
        elif backend_config in ['amd', 'vulkan']:
            # Check for Vulkan first (new default for AMD/Intel)
            if shutil.which('vulkaninfo'):
                try:
                    result = subprocess.run(['vulkaninfo', '--summary'],
                                           capture_output=True,
                                           timeout=2)
                    if result and result.returncode == 0 and result.stdout:
                        summary = result.stdout.decode('utf-8', errors='replace') if isinstance(result.stdout, bytes) else result.stdout
                        if vulkaninfo_has_hardware_gpu(summary):
                            return "Vulkan (AMD/Intel)"
                except (subprocess.TimeoutExpired, Exception):
                    pass
            # Fallback: check for ROCm (backward compatibility)
            if shutil.which('rocm-smi') or os.path.exists('/opt/rocm'):
                try:
                    result = subprocess.run(['rocm-smi', '--showproductname'],
                                           capture_output=True,
                                           timeout=2)
                    if result.returncode == 0:
                        return "ROCm (AMD)"
                except (subprocess.TimeoutExpired, Exception):
                    pass
        elif backend_config == 'cpu':
            return "CPU"

        # Fallback: hardware-only detection (for non-Arch systems or when pacman unavailable)
        # Check NVIDIA CUDA - verify it actually works
        if shutil.which('nvidia-smi'):
            try:
                result = subprocess.run(['nvidia-smi', '-L'],
                                       capture_output=True,
                                       timeout=2)
                if result.returncode == 0:
                    return "CUDA (NVIDIA) - package unknown"
            except (subprocess.TimeoutExpired, Exception):
                pass

        # Check AMD ROCm - verify it actually works
        if shutil.which('rocm-smi') or os.path.exists('/opt/rocm'):
            try:
                result = subprocess.run(['rocm-smi', '--showproductname'],
                                       capture_output=True,
                                       timeout=2)
                if result.returncode == 0:
                    return "ROCm (AMD) - package unknown"
            except (subprocess.TimeoutExpired, Exception):
                pass

        # Check Vulkan
        if shutil.which('vulkaninfo'):
            return "Vulkan - package unknown"

        return "CPU"

    @contextmanager
    def _intercept_progress_logs(self):
        """Context manager to intercept and enhance progress messages from pywhispercpp"""
        model_name = self.current_model or 'unknown'
        context_str = f"[pywhispercpp/{model_name}]"
        
        # Create a custom file-like object to intercept writes
        class ProgressInterceptor:
            def __init__(self, original_stream, context):
                self.original_stream = original_stream
                self.context = context
                self.buffer = ''
            
            def write(self, text):
                # Check if this is a progress message
                if 'Progress:' in text:
                    # Extract percentage and spacing if present
                    match = re.search(r'Progress:(\s+)(\d+)%', text)
                    if match:
                        spacing = match.group(1)  # Preserve original spacing
                        percent = match.group(2)
                        # Write enhanced message preserving original spacing
                        # Preserve newline if present in original text
                        has_newline = text.endswith('\n')
                        enhanced = f"{self.context} Progress:{spacing}{percent}%"
                        if has_newline:
                            self.original_stream.write(enhanced + '\n')
                        else:
                            self.original_stream.write(enhanced)
                        self.original_stream.flush()
                    else:
                        # Try without spacing (just in case)
                        match = re.search(r'Progress:\s*(\d+)%', text)
                        if match:
                            percent = match.group(1)
                            has_newline = text.endswith('\n')
                            enhanced = f"{self.context} Progress: {percent:>3}%"
                            if has_newline:
                                self.original_stream.write(enhanced + '\n')
                            else:
                                self.original_stream.write(enhanced)
                            self.original_stream.flush()
                        else:
                            # Just add context to any progress-related line
                            has_newline = text.endswith('\n')
                            enhanced = f"{self.context} {text.strip()}"
                            if has_newline:
                                self.original_stream.write(enhanced + '\n')
                            else:
                                self.original_stream.write(enhanced)
                            self.original_stream.flush()
                else:
                    # Pass through other messages unchanged
                    self.original_stream.write(text)
                    self.original_stream.flush()
            
            def flush(self):
                self.original_stream.flush()
            
            def __getattr__(self, name):
                # Delegate all other attributes to the original stream
                return getattr(self.original_stream, name)
        
        # Intercept both stdout and stderr (whisper.cpp may use either)
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_interceptor = ProgressInterceptor(original_stdout, context_str)
        stderr_interceptor = ProgressInterceptor(original_stderr, context_str)
        sys.stdout = stdout_interceptor
        sys.stderr = stderr_interceptor
        
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
                   language_override: Optional[str] = None) -> str:
        """Transcribe using the local pywhispercpp model."""
        try:
            audio_data = self._resample_audio(audio_data, sample_rate, 16000)

            # Use language_override if provided, otherwise get from config (None = auto-detect)
            language = language_override if language_override is not None else self.config.get_setting('language', None)

            # pywhispercpp doesn't auto-detect language when no language kwarg is passed —
            # it keeps whisper.cpp's compiled-in default of "en". Call auto_detect_language()
            # explicitly so a null config value behaves as documented.
            detected_prob = None
            if not language:
                try:
                    (detected, prob), _ = self._pywhisper_model.auto_detect_language(audio_data)
                    language = detected
                    detected_prob = prob
                except Exception as e:
                    log(f'[WARN] language auto-detect failed: {e}; falling back to en')
                    language = 'en'

            whisper_prompt, prompt_source = self.resolve_whisper_prompt(language)
            if detected_prob is not None:
                log(f'[LANG] auto-detected: {language} (p={detected_prob:.2f}), prompt={prompt_source}')

            task = self.config.get_setting('task', 'transcribe')

            # Intercept progress logs and enhance them
            with self._intercept_progress_logs():
                # Build transcribe kwargs with available values
                transcribe_kwargs = {'language': language}
                if task == 'translate':
                    transcribe_kwargs['translate'] = True
                if whisper_prompt:
                    transcribe_kwargs['initial_prompt'] = whisper_prompt
                if self.config.get_setting('sampling_strategy', 'beam_search') == 'beam_search':
                    transcribe_kwargs['beam_search'] = {
                        'beam_size': self.config.get_setting('beam_size', 5),
                        'patience': -1.0,
                    }

                segments = self._pywhisper_model.transcribe(audio_data, **transcribe_kwargs)

            result = join_segments(seg.text for seg in segments)

            # Update last use time on successful transcription
            self._last_use_time = time.monotonic()

            return result
        except Exception as e:
            log(f"[ERROR] pywhispercpp transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    @staticmethod
    def _model_files(model_name: str):
        """Installed files for model_name: multilingual first, then English-only."""
        names = (f"ggml-{model_name}.bin", f"ggml-{model_name}.en.bin")
        return [PYWHISPERCPP_MODELS_DIR / n for n in names if (PYWHISPERCPP_MODELS_DIR / n).is_file()]

    def _model_arg(self, model_name: str) -> str:
        """What to hand pywhispercpp's Model(): a file path whenever one fits.

        pywhispercpp downloads any bare name it can't find. With only the .en
        file installed that meant fetching the multilingual model at every
        service start. English dictation can use the .en file; any other
        language still needs the multilingual download.
        """
        exact = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.bin"
        if exact.is_file():
            return str(exact)
        english = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.en.bin"
        language = (self.config.get_setting('language', None) or '').lower()
        if english.is_file() and language.startswith('en'):
            log(f"[BACKEND] Using English-only {english.name} for model '{model_name}'")
            return str(english)
        return model_name

    def _validate_model_file(self, model_name: str) -> bool:
        """Validate that model file exists and is not corrupted"""
        # Basic size check (>10MB for any valid model)
        return any(f.stat().st_size > 10000000 for f in self._model_files(model_name))

    def _cleanup_model(self) -> None:
        """Safely cleanup existing model instance - GPU-safe approach"""
        if self._pywhisper_model:
            try:
                # Conservative cleanup for GPU compatibility
                # Just clear the reference and let Python handle cleanup
                # Aggressive cleanup (del + gc.collect) corrupts CUDA contexts
                self._pywhisper_model = None
            except Exception as e:
                log(f"[WARN] Failed to cleanup model reference: {e}")

    def reinitialize(self) -> bool:
        """
        Reinitialize the pywhispercpp model.
        This is needed after suspend/resume when CUDA contexts become invalid.
        
        Returns:
            True if reinitialization successful, False otherwise
        """
        if not self._pywhisper_model:
            # Model not loaded, just initialize normally
            return self.initialize()
        
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        backend = normalize_backend(backend)

        # Only reinitialize for pywhispercpp and its variants (cpu, nvidia, vulkan/amd)
        pywhispercpp_variants = ['pywhispercpp', 'cpu', 'nvidia', 'amd', 'vulkan']
        if backend not in pywhispercpp_variants:
            return True
        
        log("[MODEL] Reinitializing whisper model (suspend/resume detected)")
        
        try:
            # Save current model name and thread count
            model_name = self.current_model
            threads = self.config.get_setting('threads', 4)
            
            # Clean up old model
            self._cleanup_model()
            
            # Small delay to let CUDA context fully release
            time.sleep(0.1)
            
            # Reload model
            self._pywhisper_model = self._create_pywhisper_model(model_name, threads)
            
            self.ready = True
            # Update last use time to mark successful reinitialization
            self._last_use_time = time.monotonic()
            log("[MODEL] Model reinitialized successfully")
            return True
            
        except Exception as e:
            log(f"[MODEL] ERROR: Failed to reinitialize model: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False
            return False

    def set_threads(self, num_threads: int) -> bool:
        """Update the model's thread count (called under the manager's model lock)."""
        try:
            # Try dynamic update if the backend supports it
            if self._pywhisper_model and hasattr(self._pywhisper_model, 'set_n_threads'):
                try:
                    self._pywhisper_model.set_n_threads(int(num_threads))
                    self.config.set_setting('threads', int(num_threads))
                    return True
                except Exception:
                    pass

            # Fallback: reload model with new thread count

            # Clean up existing model (GPU-safe)
            self._cleanup_model()

            # Load model with new thread count
            self._pywhisper_model = self._create_pywhisper_model(
                self.current_model, int(num_threads)
            )

            # Only persist to config if successful
            self.config.set_setting('threads', int(num_threads))
            return True

        except Exception as e:
            log(f"ERROR: Failed to set threads: {e}")
            self.ready = False
            return False

    def set_model(self, model_name: str) -> bool:
        """Load a different whisper model (called under the manager's model lock)."""
        try:
            # Validate model file exists before attempting to load
            if not self._validate_model_file(model_name):
                log(f"ERROR: Model file not found or corrupted: {model_name}")
                log(f"Please download the model to {PYWHISPERCPP_MODELS_DIR}/")
                return False

            # Clean up existing model (GPU-safe)
            self._cleanup_model()

            # Load new model
            self._pywhisper_model = self._create_pywhisper_model(
                model_name, self.config.get_setting('threads', 4)
            )

            # Only update state if model loading succeeded
            self.current_model = model_name
            self.config.set_setting('model', model_name)

            return True

        except Exception as e:
            log(f"ERROR: Failed to set model {model_name}: {e}")
            self.ready = False
            return False

    def unload(self) -> None:
        self._cleanup_model()

    @property
    def is_loaded(self) -> bool:
        return self._pywhisper_model is not None
