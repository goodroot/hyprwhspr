"""
ONNX-ASR transcription backend (CPU-optimized, optional GPU).

Runs nemo-parakeet (or another onnx-asr model) in-process via ONNX Runtime,
with optional Silero VAD routing for long recordings.
"""

import os
import time
import types
from contextlib import redirect_stderr
from typing import Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

from .base import TranscriptionBackend


class OnnxAsrBackend(TranscriptionBackend):
    """In-process ONNX Runtime backend (no GPU-context reinit concerns)."""

    name = 'onnx-asr'

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
        vad_min_duration = self._get_onnx_asr_vad_min_duration()

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

            self._onnx_asr_vad_model = None
            # Add VAD for long audio handling without putting short
            # dictations through an aggressive speech-boundary trimmer.
            if use_vad:
                print('[BACKEND] Loading Silero VAD for long audio support', flush=True)
                vad = onnx_asr.load_vad('silero')
                self._onnx_asr_vad_model = self._onnx_asr_model.with_vad(vad)

            vad_info = f', vad_min_duration={vad_min_duration}s' if use_vad else ''
            print(f'[BACKEND] onnx-asr ready (model={model_name}, quantization={quantization}, vad={use_vad}{vad_info}, gpu={use_gpu})', flush=True)

        except Exception as e:
            print(f'ERROR: Failed to load onnx-asr model: {e}', flush=True)
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
            print('[ONNX-ASR] Model not loaded')
            return ""

        try:
            audio_duration = len(audio_data) / sample_rate
            print(f'[ONNX-ASR] Transcribing {audio_duration:.2f}s of audio', flush=True)

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
                print(f'[ONNX-ASR] Mode: {mode} (vad_min_duration={vad_min_duration}s)', flush=True)
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

    def unload(self) -> None:
        self._onnx_asr_model = None
        self._onnx_asr_vad_model = None

    @property
    def is_loaded(self) -> bool:
        return self._onnx_asr_model is not None
