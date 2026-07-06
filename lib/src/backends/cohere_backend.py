"""
Cohere Transcribe transcription backend (transformers).

Runs CohereLabs/cohere-transcribe via HuggingFace transformers with
trust_remote_code, on CUDA (bfloat16) or CPU (float32).
"""

import time
from typing import Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

from .base import TranscriptionBackend


class CohereBackend(TranscriptionBackend):
    """Transformers backend; CUDA context needs a refresh after long idle."""

    name = 'cohere-transcribe'
    reinit_on_idle = True

    def __init__(self, manager):
        super().__init__(manager)
        # Cohere Transcribe model (transformers)
        self._cohere_model = None
        self._cohere_processor = None
        self._cohere_compile_done = False  # True after first torch.compile run; suppression no longer needed

    def initialize(self) -> bool:
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

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
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

    def reinitialize(self) -> bool:
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

    def unload(self) -> None:
        self._cohere_model = None
        self._cohere_processor = None

    @property
    def is_loaded(self) -> bool:
        return self._cohere_model is not None
