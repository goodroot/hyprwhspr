"""Experimental persistent-context Parakeet.cpp backend; native loading is lazy."""

import contextlib
import ctypes
import os
import numpy as np

from .base import TranscriptionBackend, log
try:
    from .. import parakeet_cpp_runtime as runtime
except ImportError:
    import parakeet_cpp_runtime as runtime


class ParakeetCppBackend(TranscriptionBackend):
    name = 'parakeet-cpp'
    loads_in_background = True
    reinit_on_idle = True
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        self._library = None
        self._context = None
        self.device = None

    def initialize(self):
        # WhisperManager serializes initialize/transcribe/unload/cleanup with its
        # model lock. Never download or mutate process-wide environment here.
        if self.is_loaded:
            return True
        self.ready = False
        try:
            self.device = runtime.resolve_device(self.config, probe=False)
            path = runtime.library_path(self.device)
            if not path.is_file() or not runtime.model_installed():
                raise RuntimeError('runtime or pinned model missing')
            self._library = runtime.bind_library(path)
            self._context = self._library.parakeet_capi_load(os.fsencode(runtime.model_path()))
            if not self._context:
                raise RuntimeError('native model initialization failed (see native diagnostic)')
            self.current_model = runtime.MODEL_ID
            self.ready = True
            log(f'[PARAKEET-CPP] Ready: {runtime.RELEASE}, {self.device}, {runtime.MODEL_ID}')
            return True
        except Exception as exc:
            self.unload()
            log(f'[PARAKEET-CPP] Initialization failed: {exc}. Run hyprwhspr setup '
                '(Parakeet → Parakeet.cpp), select reinstall.')
            return False

    def transcribe(self, audio_data, sample_rate=16000, language_override=None):
        if not self.is_loaded:
            log('[PARAKEET-CPP] Model not loaded; run hyprwhspr setup and reinstall Parakeet.cpp')
            return ''
        output = None
        try:
            audio = np.asarray(audio_data, dtype=np.float32)
            if audio.ndim != 1:
                raise ValueError('Expected mono audio')
            if not audio.size:
                return ''
            if sample_rate != 16000:
                audio = self._resample_audio(audio, sample_rate, 16000)
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if audio.size > 2147483647 or not np.isfinite(audio).all():
                raise ValueError('Audio length or samples are invalid')
            output = self._library.parakeet_capi_transcribe_pcm(
                self._context, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                audio.size, 16000, 0)  # architecture-default greedy decoder
            if not output:
                error = self._library.parakeet_capi_last_error(self._context)
                raise RuntimeError((error or b'Native inference failed').decode('utf-8', errors='replace'))
            return ctypes.string_at(output).decode('utf-8').strip()
        except Exception as exc:
            log(f'[PARAKEET-CPP] Transcription failed: {exc}')
            return ''
        finally:
            if output:
                self._library.parakeet_capi_free_string(output)

    def _model_lock(self):
        # Resume recovery and shutdown call in without the manager's lock; freeing
        # the native context mid-inference is a use-after-free. RLock: re-entrant.
        return getattr(self._manager, '_model_lock', None) or contextlib.nullcontext()

    def unload(self):
        with self._model_lock():
            context, self._context = self._context, None
            self.ready = False
            if context:
                self._library.parakeet_capi_free(context)

    def reinitialize(self):
        with self._model_lock():
            self.unload()
            return self.initialize()

    def cleanup(self):
        self.unload()

    @property
    def is_loaded(self):
        return self._context is not None and bool(self._context)
