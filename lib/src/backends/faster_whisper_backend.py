"""
faster-whisper transcription backend (CTranslate2).

Loads a CTranslate2 Whisper model with CUDA INT8 when available, falling
back to CPU when CUDA libraries are missing or fail at inference time.
"""

import os
import time
from typing import Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

from .base import TranscriptionBackend


class FasterWhisperBackend(TranscriptionBackend):
    """CTranslate2 backend; GPU context needs a refresh after long idle/resume."""

    name = 'faster-whisper'
    reinit_on_idle = True
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        # faster-whisper model (CTranslate2)
        self._faster_whisper_model = None

    def initialize(self) -> bool:
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

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
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
                if self.reinitialize(force_cpu=True):
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

    def reinitialize(self, force_cpu: bool = False) -> bool:
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

    def unload(self) -> None:
        self._faster_whisper_model = None

    @property
    def is_loaded(self) -> bool:
        return self._faster_whisper_model is not None
