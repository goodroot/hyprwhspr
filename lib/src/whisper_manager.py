"""
Whisper manager for hyprwhspr
PyWhisperCPP-only backend (in-process, model kept hot)
"""

import os
import shutil
import sys
import contextlib
import threading
import time
import types
import wave
import re
import io
from contextlib import contextmanager, redirect_stderr
from io import BytesIO
from typing import Optional, Callable

try:
    from .dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')
requests = require_package('requests')

try:
    from .config_manager import ConfigManager
    from .credential_manager import get_credential
    from .provider_registry import get_provider
except ImportError:
    from config_manager import ConfigManager
    from credential_manager import get_credential
    from provider_registry import get_provider

try:
    from .backend_utils import normalize_backend, vulkaninfo_has_hardware_gpu
except ImportError:
    from backend_utils import normalize_backend, vulkaninfo_has_hardware_gpu

try:
    from .backend_installer import PYWHISPERCPP_MODELS_DIR, VAD_MODEL_FILENAME, download_vad_model
except ImportError:
    from backend_installer import PYWHISPERCPP_MODELS_DIR, VAD_MODEL_FILENAME, download_vad_model

try:
    from .backends import BACKENDS
except ImportError:
    from backends import BACKENDS


class WhisperManager:
    """Manages whisper transcription with dual backend support"""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager

        # Whisper configuration - only set for local backend
        # Will be properly initialized in initialize() based on backend
        self.current_model = None
        # Backend-specific attributes (pywhispercpp only)
        self._pywhisper_model = None
        self.temp_dir = None
        
        # Realtime WebSocket client
        self._realtime_client = None
        self._realtime_streaming_callback = None
        self._realtime_partial_callback = None
        # Connection parameters used for reconnect-on-demand.
        # (Stored in-memory only; do not log API keys.)
        self._realtime_connect_params = None

        # faster-whisper model (CTranslate2)
        self._faster_whisper_model = None

        # Cohere Transcribe model (transformers)
        self._cohere_model = None
        self._cohere_processor = None
        self._cohere_compile_done = False  # True after first torch.compile run; suppression no longer needed

        # Thread safety for model operations
        self._model_lock = threading.Lock()

        # State
        self.ready = False

        # Track last successful transcription time for suspend/resume detection
        self._last_use_time = 0.0

        # Set when model is deliberately unloaded via unload_model() to free GPU resources
        self._model_manually_unloaded = False

        # Active TranscriptionBackend instance (None while the configured
        # backend still uses the legacy in-manager code paths)
        self._backend = None

    def initialize(self) -> bool:
        """Initialize the configured transcription backend and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()

            backend = self._current_backend_name()

            backend_cls = BACKENDS.get(backend)
            if backend_cls is not None:
                self._backend = backend_cls(self)
                return self._backend.initialize()

            # Not yet extracted into a backend class: legacy in-manager init
            self._backend = None
            init_backend = {
                'faster-whisper': self._init_faster_whisper,
                'cohere-transcribe': self._init_cohere_transcribe,
                'realtime-ws': self._init_realtime_ws,
            }.get(backend, self._init_pywhispercpp)
            return init_backend()

        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False

    def _init_faster_whisper(self) -> bool:
        """Configure faster-whisper backend (CTranslate2, CUDA INT8)"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print('ERROR: faster-whisper not installed. Run: hyprwhspr setup and select faster-whisper', flush=True)
            return False

        model_name = self.config.get_setting('faster_whisper_model', 'base')
        device = self.config.get_setting('faster_whisper_device', 'auto')
        compute_type = self.config.get_setting('faster_whisper_compute_type', 'auto')

        # Resolve 'auto' device
        if device == 'auto':
            device = 'cpu'  # default
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    # GPU visible — verify compute libraries are actually loadable.
                    # CTranslate2 loads libcublas lazily; a missing library won't
                    # surface until the first encode(), so we probe upfront.
                    # Check both: system libcublas.so.12 AND pip-installed nvidia-cublas-cu12
                    # (CTranslate2 ≥ 4.0 searches Python package dirs for nvidia libs).
                    cuda_libs_ok = False
                    try:
                        import ctypes
                        ctypes.CDLL('libcublas.so.12')
                        cuda_libs_ok = True
                    except OSError:
                        pass
                    if not cuda_libs_ok:
                        try:
                            import importlib.util
                            if importlib.util.find_spec('nvidia.cublas') is not None:
                                cuda_libs_ok = True
                        except Exception:
                            pass
                    if cuda_libs_ok:
                        device = 'cuda'
                        # Preload pip-installed nvidia CUDA libs with RTLD_GLOBAL so
                        # CTranslate2 can find them at inference time.
                        # LD_LIBRARY_PATH alone is unreliable because the dynamic linker
                        # may cache search paths at process startup. Preloading by full
                        # path registers the library under its SONAME in the linker table,
                        # so any subsequent dlopen("libcublas.so.12") finds it already loaded.
                        try:
                            import ctypes as _ctypes
                            import glob as _glob
                            import site as _site
                            _site_dirs = []
                            try:
                                _site_dirs.extend(_site.getsitepackages())
                            except Exception:
                                pass
                            try:
                                _site_dirs.append(_site.getusersitepackages())
                            except Exception:
                                pass
                            _preloaded = []
                            for _sd in _site_dirs:
                                for _pkg, _soname in [('cublas', 'libcublas.so.12'), ('cudnn', 'libcudnn.so.9')]:
                                    _lib_dir = os.path.join(_sd, 'nvidia', _pkg, 'lib')
                                    if not os.path.isdir(_lib_dir):
                                        continue
                                    # Also add to LD_LIBRARY_PATH as belt-and-suspenders
                                    _ld = os.environ.get('LD_LIBRARY_PATH', '')
                                    if _lib_dir not in _ld:
                                        os.environ['LD_LIBRARY_PATH'] = f"{_lib_dir}:{_ld}" if _ld else _lib_dir
                                    # Preload: try exact soname, then versioned variants
                                    _candidates = (
                                        [os.path.join(_lib_dir, _soname)]
                                        + sorted(_glob.glob(os.path.join(_lib_dir, _soname + '.*')))
                                    )
                                    for _lib_path in _candidates:
                                        if os.path.exists(_lib_path):
                                            try:
                                                _ctypes.CDLL(_lib_path, _ctypes.RTLD_GLOBAL)
                                                _preloaded.append(os.path.basename(_lib_path))
                                                break
                                            except OSError:
                                                continue
                            if _preloaded:
                                print(f'[BACKEND] Preloaded CUDA libs: {", ".join(_preloaded)}', flush=True)
                        except Exception as _e:
                            print(f'[WARN] Could not preload CUDA libs: {_e}', flush=True)
                    else:
                        print('[WARN] NVIDIA GPU detected but CUDA libraries not found.', flush=True)
                        print('[WARN] Re-run: hyprwhspr setup (select faster-whisper) to install CUDA libs.', flush=True)
                        print('[WARN] Falling back to CPU.', flush=True)
            except Exception:
                pass

        # Resolve 'auto' compute_type
        if compute_type == 'auto':
            compute_type = 'int8' if device == 'cuda' else 'float32'

        try:
            print(f'[BACKEND] Loading faster-whisper model: {model_name} (device={device}, compute_type={compute_type})', flush=True)
            self._faster_whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f'[BACKEND] faster-whisper ready (model={model_name}, device={device}, compute_type={compute_type})', flush=True)
        except Exception as e:
            print(f'ERROR: Failed to load faster-whisper model: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return False

        self.current_model = model_name
        self.ready = True
        self._last_use_time = time.monotonic()
        return True

    def _init_cohere_transcribe(self) -> bool:
        """Configure Cohere Transcribe backend (transformers, CUDA/CPU)"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        except ImportError:
            print('ERROR: cohere-transcribe dependencies not installed. Run: hyprwhspr setup and select cohere-transcribe', flush=True)
            return False

        model_id = 'CohereLabs/cohere-transcribe-03-2026'
        device_setting = self.config.get_setting('cohere_transcribe_device', 'auto')
        dtype_setting = self.config.get_setting('cohere_transcribe_dtype', 'bfloat16')

        # Resolve device
        if device_setting == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_setting

        # Resolve dtype — bfloat16 is required for GPU: float16 overflows at the
        # -1e9 attention mask fill value (float16 max ~65504). bfloat16 shares
        # float32's exponent range so handles it correctly at ~4-5 GB VRAM.
        if device == 'cuda' and dtype_setting in ('float16', 'bfloat16'):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Get HuggingFace token for gated model access
        hf_token = None
        try:
            hf_token = get_credential('huggingface') or None
        except Exception:
            pass

        try:
            import contextlib, os as _os
            print(f'[BACKEND] Loading Cohere Transcribe model (device={device}, dtype={torch_dtype})', flush=True)
            # Cohere's trust_remote_code path prints large ANSI-laden blobs that
            # journald records as "[NNK blob data]". Redirect during from_pretrained.
            with open(_os.devnull, 'w') as _devnull, \
                    contextlib.redirect_stdout(_devnull), \
                    contextlib.redirect_stderr(_devnull):
                self._cohere_processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True, token=hf_token,
                    local_files_only=True)
                self._cohere_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=torch_dtype,
                    token=hf_token,
                    local_files_only=True,
                ).to(device)
            self._cohere_model.eval()
            print(f'[BACKEND] Cohere Transcribe ready (device={device}, dtype={torch_dtype})', flush=True)
        except Exception as e:
            print(f'ERROR: Failed to load Cohere Transcribe model: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return False

        self.current_model = model_id
        self.ready = True
        self._last_use_time = time.monotonic()
        return True

    def _init_realtime_ws(self) -> bool:
        """Configure the Realtime WebSocket backend and connect the client"""
        # Validate WebSocket configuration
        provider_id = self.config.get_setting('websocket_provider')
        model_id = self.config.get_setting('websocket_model')

        if not provider_id:
            print('ERROR: Realtime WebSocket backend selected but websocket_provider not configured')
            return False

        if not model_id:
            print('ERROR: Realtime WebSocket backend selected but websocket_model not configured')
            return False

        # Get API key from credential manager
        api_key = get_credential(provider_id)
        if not api_key:
            print(f'ERROR: Provider {provider_id} configured but API key not found in credential store')
            return False

        # Select appropriate client based on provider
        if provider_id == 'google':
            # Use Gemini Live API client
            try:
                from .gemini_realtime_client import GeminiRealtimeClient
            except ImportError:
                from gemini_realtime_client import GeminiRealtimeClient

            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            self._realtime_client = GeminiRealtimeClient(mode=realtime_mode)

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                provider = get_provider(provider_id)
                if provider and 'websocket_endpoint' in provider:
                    websocket_url = provider['websocket_endpoint']
                else:
                    websocket_url = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'

            # Build instructions
            instructions_parts = []
            whisper_prompt = self.config.get_setting('whisper_prompt', None)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

            language = self.config.get_setting('language', None)
            if language:
                instructions_parts.append(f"Transcribe in {language} language.")

            instructions = ' '.join(instructions_parts) if instructions_parts else None

            # Set language
            self._realtime_client.language = language

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect (API key goes in URL query param, handled by client)
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': instructions,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('ERROR: Failed to connect to Gemini Live API')
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio directly to Gemini; client resamples if needed."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[GEMINI] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        elif provider_id == 'elevenlabs':
            # Use ElevenLabs-specific client (Scribe v2 Realtime)
            try:
                from .elevenlabs_realtime_client import ElevenLabsRealtimeClient
            except ImportError:
                from elevenlabs_realtime_client import ElevenLabsRealtimeClient

            self._realtime_client = ElevenLabsRealtimeClient()

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                provider = get_provider(provider_id)
                if provider and 'websocket_endpoint' in provider:
                    websocket_url = provider['websocket_endpoint']
                else:
                    websocket_url = 'wss://api.elevenlabs.io/v1/speech-to-text/realtime'

            # Set language (used at connection time via query params)
            language = self.config.get_setting('language', None)
            self._realtime_client.language = language

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect (ElevenLabs doesn't use instructions)
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': None,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, None):
                print('ERROR: Failed to connect to ElevenLabs Realtime WebSocket')
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio directly to ElevenLabs; client resamples if needed."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[ELEVENLABS] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        else:
            # Use OpenAI-compatible client (default)
            try:
                from .realtime_client import RealtimeClient
            except ImportError:
                from realtime_client import RealtimeClient

            # Initialize RealtimeClient with mode
            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            if provider_id == 'openai' and model_id == 'gpt-realtime-whisper' and realtime_mode != 'transcribe':
                print('ERROR: gpt-realtime-whisper is supported only with realtime_mode="transcribe"', flush=True)
                return False
            self._realtime_client = RealtimeClient(mode=realtime_mode)

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                # For custom providers, websocket_url must be explicitly set
                if provider_id == 'custom':
                    print('ERROR: Custom realtime backend requires websocket_url to be configured')
                    return False

                # For known providers, derive from provider registry
                try:
                    websocket_url = self._get_websocket_url(provider_id, model_id, realtime_mode)
                except Exception as e:
                    print(f'ERROR: Failed to derive WebSocket URL: {e}')
                    return False

            # Build instructions from whisper_prompt and language
            instructions_parts = []
            whisper_prompt = self.config.get_setting('whisper_prompt', None)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

            language = self.config.get_setting('language', None)
            if language:
                instructions_parts.append(f"Transcribe in {language} language.")

            instructions = ' '.join(instructions_parts) if instructions_parts else None

            # Set language in realtime client (for session.update)
            self._realtime_client.language = language

            delay = self.config.get_setting('realtime_transcription_delay', 'low')
            self._realtime_client.set_transcription_delay(delay)
            if self._is_realtime_whisper_preview_enabled(provider_id, model_id, realtime_mode):
                self._realtime_client.set_partial_transcript_callback(self._realtime_partial_callback)
            else:
                self._realtime_client.set_partial_transcript_callback(None)
                self._clear_realtime_partial_preview()

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': instructions,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('ERROR: Failed to connect to Realtime WebSocket')
                # Clean up failed client
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio to realtime client; client handles resampling/queueing."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[REALTIME] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        print(f'[BACKEND] Using Realtime WebSocket: {websocket_url}')
        print(f'[REALTIME] Model: {model_id}, Provider: {provider_id}')

        # Explicitly set to None to avoid confusion with top-level model setting
        self.current_model = None
        self.ready = True
        return True

    def _init_pywhispercpp(self) -> bool:
        """Initialize local pywhispercpp backend"""
        self.current_model = self.config.get_setting('model', 'base')

        # Detect GPU backend for logging
        gpu_backend = self._detect_gpu_backend()

        try:
            # Validate model file exists before attempting to load
            models_dir = PYWHISPERCPP_MODELS_DIR
            model_file = models_dir / f"ggml-{self.current_model}.bin"

            if not model_file.exists():
                # Try English-only variant
                if not self.current_model.endswith('.en'):
                    model_file = models_dir / f"ggml-{self.current_model}.en.bin"

            if not model_file.exists():
                print(f"[ERROR] Model file not found: {model_file}")
                print(f"[ERROR] Download with: hyprwhspr model download {self.current_model}")
                return False

            self._pywhisper_model = self._create_pywhisper_model(
                self.current_model,
                self.config.get_setting('threads', 4)
            )

            print(f"[BACKEND] pywhispercpp ({gpu_backend}) - model: {self.current_model}")
            self.ready = True
            # Record initialization time for suspend/resume detection
            self._last_use_time = time.monotonic()
            return True

        except ImportError as e:
            print("")
            print("ERROR: pywhispercpp is not installed or incompatible.")
            print(f"Import error: {e}")
            print("Run: hyprwhspr setup to configure a backend.")

            print("")
            return False
        except Exception as e:
            print(f"[ERROR] pywhispercpp initialization failed: {e}")
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
            'model': model_name,
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
            print("[BACKEND] WARNING: installed pywhispercpp has no VAD support - re-run 'hyprwhspr setup' to upgrade; continuing without VAD", flush=True)
            del kwargs['vad'], kwargs['vad_model_path']
            return Model(**kwargs)

    def _resolve_vad_model(self):
        """Path to the Silero VAD model, auto-downloading if missing; None on failure"""
        vad_file = PYWHISPERCPP_MODELS_DIR / VAD_MODEL_FILENAME
        if not vad_file.exists():
            print("[BACKEND] Downloading Silero VAD model (~1MB)", flush=True)
            if not download_vad_model():
                print("[BACKEND] WARNING: VAD model download failed - continuing without VAD", flush=True)
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

    def _get_websocket_url(self, provider_id: str, model_id: str, mode: str = 'transcribe') -> str:
        """
        Get WebSocket URL for a provider and model.
        
        Args:
            provider_id: Provider identifier (e.g., 'openai')
            model_id: Model identifier (e.g., 'gpt-realtime-whisper')
            mode: 'transcribe' or 'converse'
        
        Returns:
            WebSocket URL with appropriate query parameters
        """
        provider = get_provider(provider_id)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_id}")
        
        # Check if provider has explicit websocket_endpoint
        if 'websocket_endpoint' in provider:
            base_url = provider['websocket_endpoint']
        else:
            # Derive from HTTP endpoint
            endpoint = provider.get('endpoint', '')
            if not endpoint:
                raise ValueError(f"Provider {provider_id} has no endpoint or websocket_endpoint")
            
            # Transform: https:// -> wss://, replace /audio/transcriptions -> /realtime
            base_url = endpoint.replace('https://', 'wss://').replace('http://', 'ws://')
            if '/audio/transcriptions' in base_url:
                base_url = base_url.replace('/audio/transcriptions', '/realtime')
            elif '/transcriptions' in base_url:
                base_url = base_url.replace('/transcriptions', '/realtime')
        
        # Build query parameters based on mode
        if mode == 'transcribe':
            # Transcription mode uses intent=transcription
            return f"{base_url}?intent=transcription"
        else:
            # Converse mode uses model parameter
            return f"{base_url}?model={model_id}"

    def _transcribe_realtime(self, _audio_data: np.ndarray, _sample_rate: int = 16000, language_override: Optional[str] = None) -> str:
        """
        Transcribe audio using Realtime WebSocket backend.
        
        Note: For realtime-ws backend, audio should be streamed during capture
        via the streaming callback. This method handles the commit and wait.
        
        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data (should be 16000)
            language_override: Optional language code to override config language
        
        Returns:
            Transcribed text string
        """
        if not self._realtime_client:
            print('[REALTIME] Client not initialized')
            return ""
        
        if not self._realtime_client.connected:
            print('[REALTIME] Client not connected')
            return ""
        
        try:
            # Update language if override provided.
            # Some clients (e.g. Gemini) bake language into the setup message at
            # connect time and cannot update it after audio has been streamed —
            # doing so would trigger a reconnect and silently drop the audio.
            if language_override is not None:
                if getattr(self._realtime_client, 'supports_mid_session_language_update', True):
                    self._realtime_client.update_language(language_override)
                else:
                    print(
                        f'[REALTIME] Provider does not support mid-session language override '
                        f'(requested: {language_override}); change will take effect on next session',
                        flush=True,
                    )
            
            # Get timeout from config
            timeout = self.config.get_setting('realtime_timeout', 30)
            
            # Commit and get text (audio was already streamed via callback)
            transcription = self._realtime_client.commit_and_get_text(timeout=timeout)
            
            return transcription.strip()
            
        except Exception as e:
            print(f'[REALTIME] Transcription failed: {e}')
            return ""

    def _transcribe_faster_whisper(self, audio_data: np.ndarray, sample_rate: int = 16000,
                                    language_override: Optional[str] = None) -> str:
        """Transcribe using faster-whisper (CTranslate2)."""
        if self._faster_whisper_model is None:
            print('[ERROR] faster-whisper model not initialized', flush=True)
            return ''
        audio_data = self._resample_audio(audio_data, sample_rate, 16000)

        language = language_override if language_override is not None else self.config.get_setting('language', None)
        whisper_prompt = (self.config.get_setting(f'whisper_prompt_{language}', None) if language else None) or self.config.get_setting('whisper_prompt', None)
        vad_filter = self.config.get_setting('faster_whisper_vad_filter', True)
        task = self.config.get_setting('task', 'transcribe')

        try:
            transcribe_kwargs = {
                'vad_filter': vad_filter,
                'beam_size': self.config.get_setting('beam_size', 5),
                'task': task,
            }
            if language:
                transcribe_kwargs['language'] = language
            if whisper_prompt:
                transcribe_kwargs['initial_prompt'] = whisper_prompt

            segments, _ = self._faster_whisper_model.transcribe(audio_data, **transcribe_kwargs)
            result = ' '.join(seg.text for seg in segments).strip()
            self._last_use_time = time.monotonic()
            return result
        except RuntimeError as e:
            if 'cannot be loaded' in str(e) or 'not found' in str(e):
                print(f'[WARN] CUDA library unavailable ({e}), falling back to CPU', flush=True)
                if self._reinitialize_faster_whisper(force_cpu=True):
                    try:
                        segments, _ = self._faster_whisper_model.transcribe(audio_data, **transcribe_kwargs)
                        result = ' '.join(seg.text for seg in segments).strip()
                        self._last_use_time = time.monotonic()
                        return result
                    except Exception as retry_e:
                        print(f'[ERROR] faster-whisper CPU fallback failed: {retry_e}', flush=True)
                return ''
            print(f'[ERROR] faster-whisper transcription failed: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return ''
        except Exception as e:
            print(f'[ERROR] faster-whisper transcription failed: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return ''

    def _reinitialize_faster_whisper(self, force_cpu: bool = False) -> bool:
        """Reinitialize faster-whisper model after suspend/resume or CUDA library failure."""
        try:
            from faster_whisper import WhisperModel
            model_name = self.config.get_setting('faster_whisper_model', 'base')
            if force_cpu:
                device = 'cpu'
                compute_type = 'float32'
                print('[MODEL] Reinitializing faster-whisper on CPU (CUDA libraries unavailable)', flush=True)
            else:
                device = self.config.get_setting('faster_whisper_device', 'auto')
                compute_type = self.config.get_setting('faster_whisper_compute_type', 'auto')
                if device == 'auto':
                    device = 'cpu'
                    try:
                        import ctranslate2
                        if ctranslate2.get_cuda_device_count() > 0:
                            cuda_libs_ok = False
                            try:
                                import ctypes
                                ctypes.CDLL('libcublas.so.12')
                                cuda_libs_ok = True
                            except OSError:
                                pass
                            if not cuda_libs_ok:
                                try:
                                    import importlib.util
                                    if importlib.util.find_spec('nvidia.cublas') is not None:
                                        cuda_libs_ok = True
                                except Exception:
                                    pass
                            if cuda_libs_ok:
                                device = 'cuda'
                                try:
                                    import ctypes as _ctypes
                                    import glob as _glob
                                    import site as _site
                                    _site_dirs = []
                                    try:
                                        _site_dirs.extend(_site.getsitepackages())
                                    except Exception:
                                        pass
                                    try:
                                        _site_dirs.append(_site.getusersitepackages())
                                    except Exception:
                                        pass
                                    for _sd in _site_dirs:
                                        for _pkg, _soname in [('cublas', 'libcublas.so.12'), ('cudnn', 'libcudnn.so.9')]:
                                            _lib_dir = os.path.join(_sd, 'nvidia', _pkg, 'lib')
                                            if not os.path.isdir(_lib_dir):
                                                continue
                                            _ld = os.environ.get('LD_LIBRARY_PATH', '')
                                            if _lib_dir not in _ld:
                                                os.environ['LD_LIBRARY_PATH'] = f"{_lib_dir}:{_ld}" if _ld else _lib_dir
                                            _candidates = (
                                                [os.path.join(_lib_dir, _soname)]
                                                + sorted(_glob.glob(os.path.join(_lib_dir, _soname + '.*')))
                                            )
                                            for _lib_path in _candidates:
                                                if os.path.exists(_lib_path):
                                                    try:
                                                        _ctypes.CDLL(_lib_path, _ctypes.RTLD_GLOBAL)
                                                        break
                                                    except OSError:
                                                        continue
                                except Exception:
                                    pass
                    except Exception:
                        pass
                if compute_type == 'auto':
                    compute_type = 'int8' if device == 'cuda' else 'float32'
            self._faster_whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
            self._last_use_time = time.monotonic()
            return True
        except Exception as e:
            print(f'[ERROR] faster-whisper reinitialization failed: {e}', flush=True)
            return False

    def _transcribe_cohere_transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
                                      language_override: Optional[str] = None) -> str:
        """Transcribe using Cohere Transcribe (transformers)."""
        if self._cohere_model is None or self._cohere_processor is None:
            print('[ERROR] Cohere Transcribe model not initialized', flush=True)
            return ''

        language = language_override if language_override is not None else self.config.get_setting('language', None)
        if not language:
            language = 'en'

        use_compile = self.config.get_setting('cohere_transcribe_compile', False)

        try:
            # On the first torch.compile call, triton spawns a C compiler subprocess
            # whose output (warnings, notes) leaks to journald. Suppress at fd level
            # for that one call; afterwards the kernel is cached and nothing is emitted.
            needs_suppress = use_compile and not self._cohere_compile_done
            if needs_suppress:
                import os as _os, warnings as _warnings
                _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
                _old_stderr = _os.dup(2)
                _os.dup2(_devnull_fd, 2)
            try:
                if needs_suppress:
                    import warnings as _warnings
                    with _warnings.catch_warnings():
                        _warnings.simplefilter('ignore')
                        texts = self._cohere_model.transcribe(
                            processor=self._cohere_processor,
                            audio_arrays=[audio_data],
                            sample_rates=[sample_rate],
                            language=language,
                            compile=use_compile,
                        )
                else:
                    texts = self._cohere_model.transcribe(
                        processor=self._cohere_processor,
                        audio_arrays=[audio_data],
                        sample_rates=[sample_rate],
                        language=language,
                        compile=use_compile,
                    )
            finally:
                if needs_suppress:
                    _os.dup2(_old_stderr, 2)
                    _os.close(_old_stderr)
                    _os.close(_devnull_fd)
                    self._cohere_compile_done = True
            result = texts[0].strip() if texts else ''
            self._last_use_time = time.monotonic()
            return result
        except Exception as e:
            print(f'[ERROR] Cohere Transcribe transcription failed: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return ''

    def _reinitialize_cohere_transcribe(self) -> bool:
        """Reinitialize Cohere Transcribe model after suspend/resume."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            model_id = 'CohereLabs/cohere-transcribe-03-2026'
            device_setting = self.config.get_setting('cohere_transcribe_device', 'auto')
            dtype_setting = self.config.get_setting('cohere_transcribe_dtype', 'bfloat16')

            device = 'cuda' if (device_setting == 'auto' and torch.cuda.is_available()) else device_setting if device_setting != 'auto' else 'cpu'
            torch_dtype = torch.bfloat16 if (device == 'cuda' and dtype_setting in ('float16', 'bfloat16')) else torch.float32

            hf_token = None
            try:
                hf_token = get_credential('huggingface') or None
            except Exception:
                pass

            print(f'[MODEL] Reinitializing Cohere Transcribe (device={device})', flush=True)
            import contextlib, os as _os
            with open(_os.devnull, 'w') as _devnull, \
                    contextlib.redirect_stdout(_devnull), \
                    contextlib.redirect_stderr(_devnull):
                self._cohere_processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True, token=hf_token,
                    local_files_only=True)
                self._cohere_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    dtype=torch_dtype,
                    token=hf_token,
                    local_files_only=True,
                ).to(device)
            self._cohere_model.eval()
            self._last_use_time = time.monotonic()
            return True
        except Exception as e:
            print(f'[ERROR] Cohere Transcribe reinitialization failed: {e}', flush=True)
            return False

    def get_realtime_streaming_callback(self) -> Optional[Callable]:
        """
        Get the streaming callback for realtime-ws backend.
        
        Returns:
            Callback function if realtime-ws backend is active, None otherwise
        """
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        backend = normalize_backend(backend)
        
        if backend == 'realtime-ws' and self._realtime_client:
            # If the server closed the socket while idle, reconnect on-demand here
            # (before we start capturing audio) to avoid dropping the first chunks.
            if not self._realtime_client.connected:
                if not self._reconnect_realtime_client():
                    return None

            # Clear server buffer before starting new recording
            self._realtime_client.clear_audio_buffer()
            return self._realtime_streaming_callback
        return None

    def _resample_audio(self, audio_data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
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

    def set_realtime_partial_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set callback for realtime partial transcript previews."""
        self._realtime_partial_callback = callback
        if self._realtime_client and hasattr(self._realtime_client, 'set_partial_transcript_callback'):
            provider_id = self.config.get_setting('websocket_provider')
            model_id = self.config.get_setting('websocket_model')
            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            if self._is_realtime_whisper_preview_enabled(provider_id, model_id, realtime_mode):
                self._realtime_client.set_partial_transcript_callback(callback)
            else:
                self._realtime_client.set_partial_transcript_callback(None)
                self._clear_realtime_partial_preview()

    def _is_realtime_whisper_preview_enabled(self, provider_id: str, model_id: str, realtime_mode: str) -> bool:
        return (
            self.config.get_setting('mic_osd_enabled', True)
            and provider_id == 'openai'
            and model_id == 'gpt-realtime-whisper'
            and realtime_mode == 'transcribe'
            and self._realtime_partial_callback is not None
        )

    def _clear_realtime_partial_preview(self) -> None:
        if not self._realtime_partial_callback:
            return
        try:
            self._realtime_partial_callback("")
        except Exception as e:
            print(f'[REALTIME] Failed to clear partial transcript preview: {e}', flush=True)

    def _reconnect_realtime_client(self) -> bool:
        """Reconnect realtime client using stored connect params."""
        if not self._realtime_client:
            return False

        params = self._realtime_connect_params or {}
        websocket_url = params.get('websocket_url')
        api_key = params.get('api_key')
        model_id = params.get('model_id')
        instructions = params.get('instructions')

        if not (websocket_url and api_key and model_id):
            print('[REALTIME] Missing connection parameters; cannot reconnect', flush=True)
            return False

        try:
            # Best-effort: close any stale connection first
            try:
                self._realtime_client.close()
            except Exception:
                pass

            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('[REALTIME] Reconnect failed', flush=True)
                return False

            print('[REALTIME] Reconnected on-demand', flush=True)
            return True
        except Exception as e:
            print(f'[REALTIME] Reconnect failed: {e}', flush=True)
            return False

    def _current_backend_name(self) -> str:
        """Normalized name of the configured transcription backend."""
        return normalize_backend(
            self.config.get_setting('transcription_backend', 'pywhispercpp'))

    def update_realtime_language(self, language: Optional[str]) -> None:
        """Apply a language override to a connected realtime client (no-op otherwise)."""
        if self._current_backend_name() == 'realtime-ws' and self._realtime_client:
            self._realtime_client.update_language(language)

    def close_realtime_connection(self, reason: str = '') -> None:
        """Close the realtime WebSocket if one is active (no-op otherwise)."""
        if self._current_backend_name() == 'realtime-ws' and self._realtime_client:
            note = f' ({reason})' if reason else ''
            print(f'[CLEANUP] Closing realtime WebSocket{note}', flush=True)
            self._cleanup_realtime_client()

    def reinitialize_after_resume(self, only_if_idle: bool = False) -> bool:
        """Recover backend state after suspend/resume or audio recovery.

        With only_if_idle, reinitialize only when a loaded local model has been
        unused long enough (>30 min) that a suspend likely invalidated its
        CUDA/GPU context; otherwise reinitialize unconditionally per backend.

        Returns:
            True if the backend is healthy (or needed no action), False otherwise
        """
        backend = self._current_backend_name()
        pywhispercpp_variants = ('pywhispercpp', 'cpu', 'nvidia', 'amd', 'vulkan')

        if only_if_idle:
            if self._backend is not None:
                loaded = (self._backend.reinit_on_resume and self._backend.is_local
                          and self._backend.is_loaded)
            elif backend in pywhispercpp_variants:
                loaded = self._pywhisper_model is not None
            elif backend == 'faster-whisper':
                loaded = self._faster_whisper_model is not None
            else:
                return True
            if not loaded:
                return True
            idle = time.monotonic() - self._last_use_time
            if not (idle > 1800 and self._last_use_time > 0):
                return True
            print(f"[RECOVERY] Reinitializing {backend} model after audio recovery (suspend/resume detected)", flush=True)

        if self._backend is not None:
            if not self._backend.reinit_on_resume:
                return True
            return self._backend.reinitialize()
        if backend in pywhispercpp_variants:
            return self._reinitialize_model()
        if backend == 'faster-whisper':
            return self._reinitialize_faster_whisper()
        if backend == 'realtime-ws':
            return self.initialize()
        # Stateless backends (rest-api, onnx-asr in-process without GPU context
        # concerns) need no reinitialization
        return True

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
            if self._backend is not None:
                return self._backend.transcribe(audio_data, sample_rate, language_override=language_override)
            return self._transcribe_realtime(audio_data, sample_rate, language_override=language_override)

        if self._backend is not None:
            transcribe_backend = self._backend.transcribe
        else:
            transcribe_backend = {
                'faster-whisper': self._transcribe_faster_whisper,
                'cohere-transcribe': self._transcribe_cohere_transcribe,
            }.get(backend, self._transcribe_pywhispercpp)

        # Use model lock to prevent concurrent transcription calls and
        # crashes from concurrent access to the loaded model
        with self._model_lock:
            if not self._ensure_backend_fresh_locked(backend):
                return ""
            return transcribe_backend(audio_data, sample_rate, language_override=language_override)

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

        if self._backend is not None:
            if not self._backend.reinit_on_idle:
                return True
            reinit = self._backend.reinitialize
        elif backend == 'faster-whisper':
            reinit = self._reinitialize_faster_whisper
        elif backend == 'cohere-transcribe':
            reinit = self._reinitialize_cohere_transcribe
        else:
            reinit = self._reinitialize_model

        print(f"[MODEL] Long idle detected - reinitializing {backend} model (suspend/resume likely)", flush=True)
        if not reinit():
            print("[MODEL] Reinitialization failed, transcription may fail", flush=True)
            return False
        return True

    def _transcribe_pywhispercpp(self, audio_data: np.ndarray, sample_rate: int = 16000,
                                 language_override: Optional[str] = None) -> str:
        """Transcribe using the local pywhispercpp model."""
        try:
            audio_data = self._resample_audio(audio_data, sample_rate, 16000)

            # Use language_override if provided, otherwise get from config (None = auto-detect)
            language = language_override if language_override is not None else self.config.get_setting('language', None)

            # pywhispercpp doesn't auto-detect language when no language kwarg is passed —
            # it keeps whisper.cpp's compiled-in default of "en". Call auto_detect_language()
            # explicitly so a null config value behaves as documented.
            if not language:
                try:
                    (detected, prob), _ = self._pywhisper_model.auto_detect_language(audio_data)
                    language = detected
                    print(f'[LANG] auto-detected: {detected} (p={prob:.2f})', flush=True)
                except Exception as e:
                    print(f'[WARN] language auto-detect failed: {e}; falling back to en', flush=True)
                    language = 'en'

            whisper_prompt = (self.config.get_setting(f'whisper_prompt_{language}', None) if language else None) or self.config.get_setting('whisper_prompt', None)

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

            result = ' '.join(seg.text for seg in segments).strip()

            # Update last use time on successful transcription
            self._last_use_time = time.monotonic()

            return result
        except Exception as e:
            print(f"[ERROR] pywhispercpp transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _validate_model_file(self, model_name: str) -> bool:
        """Validate that model file exists and is not corrupted"""
        models_dir = PYWHISPERCPP_MODELS_DIR

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
            except Exception as e:
                print(f"[WARN] Failed to cleanup model reference: {e}")
    
    def _reinitialize_model(self) -> bool:
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
        
        print("[MODEL] Reinitializing whisper model (suspend/resume detected)", flush=True)
        
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
            print("[MODEL] Model reinitialized successfully", flush=True)
            return True
            
        except Exception as e:
            print(f"[MODEL] ERROR: Failed to reinitialize model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.ready = False
            return False
    
    def _cleanup_realtime_client(self) -> None:
        """Cleanup Realtime WebSocket client"""
        if self._realtime_client:
            try:
                self._realtime_client.close()
                self._realtime_client = None
                self._realtime_streaming_callback = None
            except Exception as e:
                print(f"[WARN] Failed to cleanup realtime client: {e}")
    
    def cleanup(self) -> None:
        """Public cleanup method to clean up all resources"""
        # Cleanup realtime WebSocket client if active
        self._cleanup_realtime_client()

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
                self._cleanup_model()
                self._faster_whisper_model = None
                self._cohere_model = None
                self._cohere_processor = None

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
        with self._model_lock:
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
        # Check if using REST API backend - model changes don't apply
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        
        # Backward compatibility
        if backend == 'local':
            backend = 'pywhispercpp'
        elif backend == 'remote':
            backend = 'rest-api'
        
        if backend == 'rest-api':
            print("ERROR: Cannot change model when using REST API backend - switch to pywhispercpp")
            print("Model selection is handled by the REST API endpoint")
            return False

        with self._model_lock:
            try:
                # Validate model file exists before attempting to load
                if not self._validate_model_file(model_name):
                    print(f"ERROR: Model file not found or corrupted: {model_name}")
                    print(f"Please download the model to {PYWHISPERCPP_MODELS_DIR}/")
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
                print(f"ERROR: Failed to set model {model_name}: {e}")
                self.ready = False
                return False

    def get_current_model(self) -> str:
        """Get the current model name"""
        # For REST API backends, return empty string or None
        if self.current_model is None:
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            
            # Backward compatibility
            if backend == 'local':
                backend = 'pywhispercpp'
            elif backend == 'remote':
                backend = 'rest-api'
            
            if backend == 'rest-api':
                return ''
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
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        
        # Backward compatibility
        if backend == 'local':
            backend = 'pywhispercpp'
        elif backend == 'remote':
            backend = 'rest-api'
        
        if backend == 'rest-api':
            endpoint_url = self.config.get_setting('rest_endpoint_url', 'not configured')
            return f"REST API ({endpoint_url})"
        else:
            return f"pywhispercpp (in-process, model: {self.current_model})"
