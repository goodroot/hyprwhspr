"""
Factory for creating STT backend instances
"""

try:
    from .config_manager import ConfigManager
    from .stt_backend import STTBackend
    from .whisper_manager import WhisperManager
    from .parakeet_manager import ParakeetManager
    from .logger import log_info, log_error
except ImportError:
    from config_manager import ConfigManager
    from stt_backend import STTBackend
    from whisper_manager import WhisperManager
    from parakeet_manager import ParakeetManager
    from logger import log_info, log_error


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
            log_info("Creating Parakeet backend", "FACTORY")
            return ParakeetManager(config_manager)
        elif backend_name == 'whisper':
            log_info("Creating Whisper backend", "FACTORY")
            return WhisperManager(config_manager)
        else:
            log_error(f"Unknown backend '{backend_name}', defaulting to Whisper", "FACTORY")
            return WhisperManager(config_manager)
