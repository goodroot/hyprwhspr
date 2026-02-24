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
    from .backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend


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
        # Connection parameters used for reconnect-on-demand.
        # (Stored in-memory only; do not log API keys.)
        self._realtime_connect_params = None

        # ONNX-ASR model (CPU-optimized)
        self._onnx_asr_model = None

        # faster-whisper model (CTranslate2)
        self._faster_whisper_model = None

        # Thread safety for model operations
        self._model_lock = threading.Lock()

        # State
        self.ready = False
        
        # Track last successful transcription time for suspend/resume detection
        self._last_use_time = 0.0

    def initialize(self) -> bool:
        """Initialize the whisper manager and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()

            # Check which backend is configured
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            backend = normalize_backend(backend)  # Backward compatibility

            # Configure ONNX-ASR backend (CPU or GPU-optimized)
            if backend == 'onnx-asr':
                try:
                    import onnx_asr
                except ImportError:
                    print('ERROR: onnx-asr not installed. Run: hyprwhspr setup')
                    print('ERROR: Select option [1] ONNX Parakeet to install')
                    return False

                # Suppress ONNX Runtime verbose error logging
                # Errors about missing CUDA libraries are expected and will fall back to CPU
                import logging
                from io import StringIO
                
                # Set ONNX Runtime log level to suppress warnings/errors
                os.environ['ORT_LOGGING_LEVEL'] = '4'  # 4 = FATAL (suppress ERROR/WARNING/INFO)
                
                # Detect GPU availability at runtime (but don't claim it if libraries aren't available)
                use_gpu = False
                try:
                    import onnxruntime
                    # Suppress ONNX Runtime Python logging
                    logging.getLogger('onnxruntime').setLevel(logging.CRITICAL)
                    
                    # Check if providers are listed (but they may not actually work)
                    available_providers = onnxruntime.get_available_providers()
                    if 'CUDAExecutionProvider' in available_providers or 'TensorrtExecutionProvider' in available_providers:
                        # Note: We'll let onnx-asr try to use GPU, but it will fall back to CPU
                        # if libraries aren't available. We won't claim GPU support upfront.
                        use_gpu = True
                except Exception:
                    pass

                model_name = self.config.get_setting('onnx_asr_model', 'nemo-parakeet-tdt-0.6b-v3')
                quantization = self.config.get_setting('onnx_asr_quantization', 'int8')
                use_vad = self.config.get_setting('onnx_asr_use_vad', True)

                print(f'[BACKEND] Loading onnx-asr model: {model_name} ({"GPU" if use_gpu else "CPU"})', flush=True)

                try:
                    # Load model with optional quantization
                    # onnx-asr automatically uses GPU providers if available
                    # Suppress stderr during model loading to avoid CUDA library error spam
                    # These errors are harmless - ONNX Runtime will fall back to CPU automatically
                    with redirect_stderr(StringIO()):
                        if quantization:
                            self._onnx_asr_model = onnx_asr.load_model(model_name, quantization=quantization)
                        else:
                            self._onnx_asr_model = onnx_asr.load_model(model_name)

                    # Add VAD for long audio handling (>30s)
                    if use_vad:
                        print('[BACKEND] Loading Silero VAD for long audio support', flush=True)
                        vad = onnx_asr.load_vad('silero')
                        self._onnx_asr_model = self._onnx_asr_model.with_vad(vad)

                    print(f'[BACKEND] onnx-asr ready (model={model_name}, quantization={quantization}, vad={use_vad}, gpu={use_gpu})', flush=True)

                except Exception as e:
                    print(f'ERROR: Failed to load onnx-asr model: {e}', flush=True)
                    import traceback
                    traceback.print_exc()
                    return False

                # onnx-asr doesn't use current_model in the same way
                self.current_model = None
                self.ready = True
                return True

            # Configure faster-whisper backend (CTranslate2, CUDA INT8)
            if backend == 'faster-whisper':
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

            # Configure Realtime WebSocket backend
            if backend == 'realtime-ws':
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
                if provider_id == 'elevenlabs':
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
                    
                    # ElevenLabs uses 16kHz audio - no resampling needed!
                    def _send_direct(audio_chunk: np.ndarray):
                        """Send audio directly to ElevenLabs (16kHz, no resampling)"""
                        try:
                            self._realtime_client.append_audio(audio_chunk)
                        except Exception as e:
                            print(f'[ELEVENLABS] Streaming error: {e}', flush=True)
                    
                    self._realtime_streaming_callback = _send_direct
                
                else:
                    # Use OpenAI-compatible client (default)
                    try:
                        from .realtime_client import RealtimeClient
                    except ImportError:
                        from realtime_client import RealtimeClient
                    
                    # Initialize RealtimeClient with mode
                    realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
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
                    
                    # OpenAI requires 24kHz audio - resample from 16kHz
                    def _send_direct(audio_chunk: np.ndarray):
                        """Send audio to realtime client (16kHz; client handles resampling/queueing)."""
                        try:
                            self._realtime_client.append_audio(audio_chunk)
                        except Exception as e:
                            print(f'[REALTIME] Streaming error: {e}', flush=True)
                    
                    self._realtime_streaming_callback = _send_direct
                
                print(f'[BACKEND] Using Realtime WebSocket: {websocket_url}')
                print(f'[REALTIME] Model: {model_id}, Provider: {provider_id}')
                
                # Explicitly set to None to avoid confusion with top-level model setting
                self.current_model = None
                self.ready = True
                return True

            # Configure REST API backend and return early
            if backend == 'rest-api':
                # Attempt migration of API key if needed (backup in case config was loaded before migration)
                self.config.migrate_api_key_to_credential_manager()
                
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

                print(f'[BACKEND] Using REST API: {endpoint_url}')
                print(f'[REST] Timeout configured: {timeout}s')

                # Log user-defined config objects (sanitized)
                rest_headers = self.config.get_setting('rest_headers', {})
                rest_body = self.config.get_setting('rest_body', {})
                
                # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
                api_key = None
                provider_id = self.config.get_setting('rest_api_provider')
                if provider_id:
                    api_key = get_credential(provider_id)
                    if api_key:
                        print(f'[REST] API key configured (via credential manager, provider: {provider_id})')
                    else:
                        print(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
                else:
                    # Backward compatibility: check for old rest_api_key in config
                    api_key = self.config.get_setting('rest_api_key')
                    if api_key:
                        print('[REST] API key configured (via rest_api_key - deprecated, consider migrating)')

                if rest_headers and isinstance(rest_headers, dict):
                    header_count = len([k for k in rest_headers.keys() if rest_headers.get(k) is not None])
                    if header_count > 0:
                        print(f'[REST] Custom headers configured ({header_count} keys)')

                if rest_body and isinstance(rest_body, dict):
                    body_count = len([k for k in rest_body.keys() if rest_body.get(k) is not None])
                    if body_count > 0:
                        print(f'[REST] Custom body fields configured ({body_count} fields)')

                language = self.config.get_setting('language', None)
                if language:
                    print(f'[REST] Language hint: {language}')

                # Explicitly set to None to avoid confusion with top-level model setting
                self.current_model = None
                self.ready = True
                return True

            # Initialize local pywhispercpp backend
            self.current_model = self.config.get_setting('model', 'base')
            
            # Detect GPU backend for logging
            gpu_backend = self._detect_gpu_backend()

            try:
                # Try modern layout first
                try:
                    from pywhispercpp.model import Model
                except ImportError:
                    # Fallback for flat layout (or older versions)
                    from pywhispercpp import Model

                # Validate model file exists before attempting to load
                from pathlib import Path
                models_dir = Path.home() / '.local' / 'share' / 'pywhispercpp' / 'models'
                model_file = models_dir / f"ggml-{self.current_model}.bin"
                
                if not model_file.exists():
                    # Try English-only variant
                    if not self.current_model.endswith('.en'):
                        model_file = models_dir / f"ggml-{self.current_model}.en.bin"
                
                if not model_file.exists():
                    print(f"[ERROR] Model file not found: {model_file}")
                    print(f"[ERROR] Download with: hyprwhspr model download {self.current_model}")
                    return False

                self._pywhisper_model = Model(
                    model=self.current_model,
                    n_threads=self.config.get_setting('threads', 4),
                    redirect_whispercpp_logs_to=None
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

        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False

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
                    if result and result.returncode == 0:
                        output = result.stdout.lower() if result.stdout else ''
                        if 'llvmpipe' not in output and 'software' not in output:
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

    def _get_websocket_url(self, provider_id: str, model_id: str, mode: str = 'transcribe') -> str:
        """
        Get WebSocket URL for a provider and model.
        
        Args:
            provider_id: Provider identifier (e.g., 'openai')
            model_id: Model identifier (e.g., 'gpt-realtime-mini')
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

    def _transcribe_rest(self, audio_data: np.ndarray, sample_rate: int = 16000, language_override: Optional[str] = None) -> str:
        """
        Transcribe audio using remote REST API endpoint

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            language_override: Optional language code to override config language

        Returns:
            Transcribed text string
        """
        try:

            # Get REST endpoint configuration
            endpoint_url = self.config.get_setting('rest_endpoint_url')
            
            # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
            api_key = None
            provider_id = self.config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if not api_key:
                    print(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
            else:
                # Backward compatibility: check for old rest_api_key in config
                api_key = self.config.get_setting('rest_api_key')
            
            timeout = self.config.get_setting('rest_timeout', 30)
            rest_headers = self.config.get_setting('rest_headers', {})
            rest_body = self.config.get_setting('rest_body', {})

            if not isinstance(rest_headers, dict):
                print('WARNING: rest_headers must be an object/dict; ignoring invalid value')
                rest_headers = {}

            if not isinstance(rest_body, dict):
                print('WARNING: rest_body must be an object/dict; ignoring invalid value')
                rest_body = {}

            extra_headers = {}
            for key, value in rest_headers.items():
                if value is None:
                    continue
                try:
                    extra_headers[str(key)] = str(value)
                except Exception:
                    print(f'WARNING: Skipping non-serializable rest_headers entry: {key}')

            extra_body = {}
            for key, value in rest_body.items():
                if value is None:
                    continue
                try:
                    key_str = str(key)
                except Exception:
                    print(f'WARNING: Skipping rest_body entry with non-stringable key: {key}')
                    continue

                if isinstance(value, (dict, list, tuple, set)):
                    print(f'WARNING: rest_body values must be scalar (key: {key_str}); skipping entry')
                    continue

                extra_body[key_str] = value

            # Fill prompt from config if not provided
            if 'prompt' not in extra_body:
                whisper_prompt = self.config.get_setting('whisper_prompt', None)
                if whisper_prompt:
                    extra_body['prompt'] = whisper_prompt

            if not endpoint_url:
                raise ValueError('REST endpoint URL not configured')

            # Detect backend type and extract model info
            # Note: We need to check rest_body before it's processed into extra_body
            # to get the model info early for logging
            backend_name = None
            model_info = None
            
            # Check if this is parakeet backend
            if endpoint_url in ('http://127.0.0.1:8080/transcribe', 'http://localhost:8080/transcribe'):
                backend_name = 'parakeet-tdt-0.6b-v3'
            else:
                # Generic REST API - use endpoint URL
                backend_name = endpoint_url
            
            # Extract model information from rest_body if available (before processing)
            if isinstance(rest_body, dict):
                model_info = rest_body.get('model')
            
            # Format the log message
            if backend_name == 'parakeet-tdt-0.6b-v3':
                log_msg = f'[REST API] {backend_name}'
            elif model_info:
                log_msg = f'[REST API] {backend_name} - model: {model_info}'
            else:
                log_msg = f'[REST API] {backend_name}'
            
            print(log_msg, flush=True)

            # Convert audio to WAV format
            wav_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
            audio_duration = len(audio_data) / sample_rate
            print(
                f'[REST] Audio: {audio_duration:.2f}s @ {sample_rate}Hz, {len(wav_bytes)} bytes',
                flush=True,
            )

            # Prepare the request
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}

            headers = {'Accept': 'application/json'}
            headers.update(extra_headers)
            if api_key:
                header_names = {key.lower() for key in headers.keys()}
                if 'authorization' not in header_names:
                    headers['Authorization'] = f'Bearer {api_key}'

            # Add language parameter if configured
            data = extra_body.copy()
            # Use language_override if provided, otherwise get from config
            language = language_override if language_override is not None else self.config.get_setting('language', None)
            if language and 'language' not in data:
                data['language'] = language

            # Log request parameters for debugging
            if data:
                # Sanitize - don't log full prompt, just keys
                param_summary = ', '.join(f'{k}={v[:20] + "..." if isinstance(v, str) and len(v) > 20 else v}' for k, v in data.items())
                print(f'[REST] Request params: {param_summary}', flush=True)

            # Send the request
            print(f'[REST] Sending request to {endpoint_url}...', flush=True)
            start_time = time.time()
            response = requests.post(endpoint_url, files=files, data=data, headers=headers, timeout=timeout)
            response_time = time.time() - start_time
            print(f'[REST] Response received in {response_time:.2f}s (status: {response.status_code})', flush=True)

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f'REST API returned status {response.status_code}'
                try:
                    error_detail = response.json()
                    error_msg += f': {error_detail}'
                except Exception:
                    error_msg += f': {response.text[:200]}'
                print(f'ERROR: {error_msg}')
                return ''

            # Parse the response
            try:
                result = response.json()
            except Exception as json_err:
                # Show raw response for debugging
                raw_body = response.text[:500] if response.text else '(empty)'
                print(f'ERROR: Failed to parse JSON response: {json_err}')
                print(f'[REST] Raw response body: {raw_body}')
                print(f'[REST] Content-Type: {response.headers.get("Content-Type", "not set")}')
                return ''

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
            # Update language if override provided
            if language_override is not None:
                self._realtime_client.update_language(language_override)
            
            # Get timeout from config
            timeout = self.config.get_setting('realtime_timeout', 30)
            
            # Commit and get text (audio was already streamed via callback)
            transcription = self._realtime_client.commit_and_get_text(timeout=timeout)
            
            return transcription.strip()
            
        except Exception as e:
            print(f'[REALTIME] Transcription failed: {e}')
            return ""

    def _transcribe_onnx_asr(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using onnx-asr backend (CPU-optimized).

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data (should be 16000)

        Returns:
            Transcribed text string
        """
        if not self._onnx_asr_model:
            print('[ONNX-ASR] Model not loaded')
            return ""

        try:
            audio_duration = len(audio_data) / sample_rate
            print(f'[ONNX-ASR] Transcribing {audio_duration:.2f}s of audio', flush=True)

            # onnx-asr accepts numpy arrays directly (float32)
            # It handles resampling internally if needed
            start_time = time.time()
            result = self._onnx_asr_model.recognize(audio_data)
            elapsed = time.time() - start_time

            # When VAD is enabled, recognize() returns a generator of segments
            if isinstance(result, types.GeneratorType):
                # Collect all segments and combine their text
                segments = list(result)
                if not segments:
                    transcription = ""
                else:
                    # Extract text from each segment
                    segment_texts = []
                    for seg in segments:
                        if hasattr(seg, 'text'):
                            segment_texts.append(seg.text)
                        elif isinstance(seg, str):
                            segment_texts.append(seg)
                        else:
                            # Fallback: try to get text representation
                            segment_texts.append(str(seg))
                    transcription = ' '.join(segment_texts)
            else:
                # No VAD - direct result (string or object with .text attribute)
                if hasattr(result, 'text'):
                    transcription = result.text
                elif isinstance(result, str):
                    transcription = result
                else:
                    transcription = str(result)

            print(f'[ONNX-ASR] Transcription completed in {elapsed:.2f}s', flush=True)

            return transcription.strip()

        except Exception as e:
            print(f'[ONNX-ASR] Transcription failed: {e}', flush=True)
            import traceback
            traceback.print_exc()
            return ""

    def _transcribe_faster_whisper(self, audio_data: np.ndarray, sample_rate: int = 16000,
                                    language_override: Optional[str] = None) -> str:
        """Transcribe using faster-whisper (CTranslate2)."""
        if self._faster_whisper_model is None:
            print('[ERROR] faster-whisper model not initialized', flush=True)
            return ''

        language = language_override if language_override is not None else self.config.get_setting('language', None)
        whisper_prompt = self.config.get_setting('whisper_prompt', None)
        vad_filter = self.config.get_setting('faster_whisper_vad_filter', True)

        with self._model_lock:
            current_time = time.monotonic()
            time_since_last_use = current_time - self._last_use_time
            if time_since_last_use > 1800 and self._last_use_time > 0:
                print('[MODEL] Long idle detected - reinitializing faster-whisper model', flush=True)
                if not self._reinitialize_faster_whisper():
                    return ''

            try:
                transcribe_kwargs = {
                    'vad_filter': vad_filter,
                    'beam_size': 5,
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

        # Route to appropriate backend
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        
        # Backward compatibility
        if backend == 'local':
            backend = 'pywhispercpp'
        elif backend == 'remote':
            backend = 'rest-api'

        if backend == 'rest-api':
            # Use REST API transcription
            # Debug: ensure we're not accidentally using pywhispercpp
            if self._pywhisper_model is not None:
                print(f"[WARN] REST API backend selected but pywhispercpp model is loaded - this should not happen", flush=True)
            return self._transcribe_rest(audio_data, sample_rate, language_override=language_override)

        if backend == 'realtime-ws':
            # Use Realtime WebSocket transcription
            # Note: Audio was already streamed via callback during capture
            # This just commits and waits for the result
            return self._transcribe_realtime(audio_data, sample_rate, language_override=language_override)

        if backend == 'onnx-asr':
            # Use ONNX-ASR transcription (CPU-optimized)
            return self._transcribe_onnx_asr(audio_data, sample_rate)

        if backend == 'faster-whisper':
            return self._transcribe_faster_whisper(audio_data, sample_rate, language_override=language_override)

        # Use model lock to prevent concurrent transcription calls
        # This prevents crashes from concurrent access to the whisper model
        with self._model_lock:
            # Check if model needs reinitialization (suspend/resume detection)
            # This check is inside the lock to prevent race conditions where
            # multiple threads detect idle period and try to reinitialize simultaneously
            current_time = time.monotonic()
            time_since_last_use = current_time - self._last_use_time

            # If model hasn't been used in 30+ minutes, likely suspend/resume occurred
            # Reinitialize to refresh CUDA context before using stale model
            if time_since_last_use > 1800 and self._last_use_time > 0:
                print("[MODEL] Long idle period detected - reinitializing model (suspend/resume likely)", flush=True)
                if not self._reinitialize_model():
                    print("[MODEL] Reinitialization failed, transcription may fail", flush=True)
                    return ""

            try:
                # Use language_override if provided, otherwise get from config (None = auto-detect)
                language = language_override if language_override is not None else self.config.get_setting('language', None)
                whisper_prompt = self.config.get_setting('whisper_prompt', None)

                # Intercept progress logs and enhance them
                with self._intercept_progress_logs():
                    # Build transcribe kwargs with available values
                    transcribe_kwargs = {}
                    if language:
                        transcribe_kwargs['language'] = language
                    if whisper_prompt:
                        transcribe_kwargs['initial_prompt'] = whisper_prompt

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
            try:
                from pywhispercpp.model import Model
            except ImportError:
                from pywhispercpp import Model
            
            self._pywhisper_model = Model(
                model=model_name,
                n_threads=threads,
                redirect_whispercpp_logs_to=None
            )
            
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
                # Try modern layout first
                try:
                    from pywhispercpp.model import Model
                except ImportError:
                    # Fallback for flat layout (or older versions)
                    from pywhispercpp import Model
                self._pywhisper_model = Model(
                    model=self.current_model,
                    n_threads=int(num_threads),
                    redirect_whispercpp_logs_to=None
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
                    print("Please download the model to ~/.local/share/pywhispercpp/models/")
                    return False

                # Clean up existing model (GPU-safe)
                self._cleanup_model()

                # Load new model
                # Try modern layout first
                try:
                    from pywhispercpp.model import Model
                except ImportError:
                    # Fallback for flat layout (or older versions)
                    from pywhispercpp import Model
                self._pywhisper_model = Model(
                    model=model_name,
                    n_threads=self.config.get_setting('threads', 4),
                    redirect_whispercpp_logs_to=None
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
