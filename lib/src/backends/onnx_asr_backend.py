"""
ONNX-ASR transcription backend (CPU-optimized, optional GPU).

Runs nemo-parakeet (or another onnx-asr model) in-process via ONNX Runtime,
with optional Silero VAD routing for long recordings.
"""

import os
import time
import types
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

# Set at import, on the main thread: initialize() runs on a background thread
# once the service is live, and writing os.environ there races other threads.
os.environ['ORT_LOGGING_LEVEL'] = '4'  # 4 = FATAL (suppress ERROR/WARNING/INFO)
# A first-start download would otherwise flood the journal with progress bars
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

from .base import TranscriptionBackend


def _orukeet():
    """Import the stdlib-only Orukeet helpers lazily (package or flat layout)."""
    try:
        from .. import orukeet
    except ImportError:
        import orukeet
    return orukeet


class OnnxAsrBackend(TranscriptionBackend):
    """In-process ONNX Runtime backend (no GPU-context reinit concerns)."""

    name = 'onnx-asr'
    loads_in_background = True

    def __init__(self, manager):
        super().__init__(manager)
        # ONNX-ASR model (CPU-optimized)
        self._onnx_asr_model = None
        self._onnx_asr_vad_model = None

    def initialize(self) -> bool:
        """Configure ONNX-ASR backend (CPU or GPU-optimized)"""
        try:
            import onnx_asr
        except ImportError:
            log('ERROR: onnx-asr not installed. Run: hyprwhspr setup')
            log('ERROR: Select option [1] ONNX Parakeet to install')
            return False

        # Suppress ONNX Runtime verbose error logging
        # Errors about missing CUDA libraries are expected and will fall back to CPU
        import logging


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

        try:
            from ..onnx_model import load_model, effective_quantization
        except ImportError:
            from onnx_model import load_model, effective_quantization
        model_name = self.config.get_setting('onnx_asr_model', 'nemo-parakeet-tdt-0.6b-v3')
        quantization = effective_quantization(
            model_name, self.config.get_setting('onnx_asr_quantization', 'int8'))
        use_vad = self.config.get_setting('onnx_asr_use_vad', True)
        vad_min_duration = self._get_onnx_asr_vad_min_duration()

        log(f'[BACKEND] Loading onnx-asr model: {model_name} ({"GPU" if use_gpu else "CPU"})')
        if model_name == 'orukeet' and _orukeet().cache_state() != 'verified':
            # Download progress is hidden with the CUDA noise below, so say why it is slow.
            log('[BACKEND] Fetching or verifying Orukeet (~672 MB); first start takes a while')

        try:
            # onnx-asr uses GPU providers when available and falls back to CPU.
            # No stderr redirect: this runs on a background thread while the
            # service is live, and swapping sys.stderr would swallow other
            # threads' output. ORT's own noise is silenced by the log level above.
            self._onnx_asr_model, self._onnx_asr_vad_model = load_model(
                model_name, quantization, use_vad
            )

            vad_info = f', vad_min_duration={vad_min_duration}s' if use_vad else ''
            log(f'[BACKEND] onnx-asr ready (model={model_name}, quantization={quantization}, vad={use_vad}{vad_info}, gpu={use_gpu})')

        except Exception as e:
            log(f'ERROR: Failed to load onnx-asr model: {e}')
            if isinstance(e, _orukeet().ChecksumError):
                log("Repair the cache with 'hyprwhspr model download', then restart.")
            import traceback
            traceback.print_exc()
            return False

        # onnx-asr doesn't use current_model in the same way
        self.current_model = None
        self.ready = True
        return True

    def _get_onnx_asr_vad_min_duration(self) -> float:
        """Return the duration threshold for routing ONNX-ASR audio through VAD."""
        try:
            threshold = float(self.config.get_setting('onnx_asr_vad_min_duration', 30))
        except (TypeError, ValueError):
            threshold = 30.0
        return max(0.0, threshold)

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
                   language_override: Optional[str] = None) -> str:
        """
        Transcribe audio using onnx-asr backend (CPU-optimized).

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data (should be 16000)

        Returns:
            Transcribed text string
        """
        if not self._onnx_asr_model:
            log('[ONNX-ASR] Model not loaded')
            return ""

        try:
            audio_duration = len(audio_data) / sample_rate
            log(f'[ONNX-ASR] Transcribing {audio_duration:.2f}s of audio')

            # onnx-asr accepts numpy arrays directly (float32)
            # It handles resampling internally if needed
            vad_min_duration = self._get_onnx_asr_vad_min_duration()
            use_vad_model = (
                self._onnx_asr_vad_model is not None
                and audio_duration >= vad_min_duration
            )
            model = self._onnx_asr_vad_model if use_vad_model else self._onnx_asr_model
            if self._onnx_asr_vad_model is not None:
                mode = 'vad' if use_vad_model else 'direct'
                log(f'[ONNX-ASR] Mode: {mode} (vad_min_duration={vad_min_duration}s)')
            start_time = time.time()
            result = model.recognize(audio_data, sample_rate=sample_rate)
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
                    transcription = join_segments(segment_texts)
            else:
                # No VAD - direct result (string or object with .text attribute)
                if hasattr(result, 'text'):
                    transcription = result.text
                elif isinstance(result, str):
                    transcription = result
                else:
                    transcription = str(result)

            log(f'[ONNX-ASR] Transcription completed in {elapsed:.2f}s')

            return transcription.strip()

        except Exception as e:
            log(f'[ONNX-ASR] Transcription failed: {e}')
            import traceback
            traceback.print_exc()
            return ""

    def unload(self) -> None:
        self._onnx_asr_model = None
        self._onnx_asr_vad_model = None

    @property
    def is_loaded(self) -> bool:
        return self._onnx_asr_model is not None
