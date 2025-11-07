"""
Factory for creating STT backend instances
"""

try:
    from .config_manager import ConfigManager
    from .stt_backend import STTBackend
    from .whisper_manager import WhisperManager
    from .parakeet_manager import ParakeetManager
except ImportError:
    from config_manager import ConfigManager
    from stt_backend import STTBackend
    from whisper_manager import WhisperManager
    from parakeet_manager import ParakeetManager


class STTBackendFactory:
    """Factory for creating STT backend instances"""

    @staticmethod
    def create(config_manager: ConfigManager) -> STTBackend:
        """
        Create an STT backend based on configuration

        Args:
            config_manager: Configuration manager instance

        Returns:
            STT backend instance (WhisperManager or ParakeetManager)
        """
        backend_name = config_manager.get_setting('stt_backend', 'whisper')

        if backend_name == 'parakeet':
            print(f"[FACTORY] Creating Parakeet backend")
            return ParakeetManager(config_manager)
        elif backend_name == 'whisper':
            print(f"[FACTORY] Creating Whisper backend")
            return WhisperManager(config_manager)
        else:
            print(f"[FACTORY] Unknown backend '{backend_name}', defaulting to Whisper")
            return WhisperManager(config_manager)
