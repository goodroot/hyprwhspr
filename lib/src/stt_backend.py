"""
Abstract base class for speech-to-text backends
Provides a common interface for different STT engines (Whisper, Parakeet, etc.)
"""

from abc import ABC, abstractmethod
import numpy as np


class STTBackend(ABC):
    """Abstract base class for speech-to-text backends"""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend and load models"""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text"""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is ready for transcription"""
        pass

    @abstractmethod
    def set_threads(self, num_threads: int) -> bool:
        """Update thread count for processing"""
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        pass

    @abstractmethod
    def get_current_model(self) -> str:
        """Get the current model name"""
        pass

    @abstractmethod
    def get_available_models(self) -> list:
        """Get list of available models"""
        pass

    @abstractmethod
    def get_backend_info(self) -> str:
        """Get information about the backend"""
        pass
