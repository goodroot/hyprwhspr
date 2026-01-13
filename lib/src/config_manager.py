"""
Configuration manager for hyprwhspr
Handles loading, saving, and managing application settings
"""

import json
from pathlib import Path
from typing import Any, Dict

try:
    from .paths import CONFIG_DIR, CONFIG_FILE, TEMP_DIR
except ImportError:
    from paths import CONFIG_DIR, CONFIG_FILE, TEMP_DIR


class ConfigManager:
    """Manages application configuration and settings"""
    
    def __init__(self):
        # Default configuration values - minimal set for hyprwhspr
        self.default_config = {
            'primary_shortcut': 'SUPER+ALT+D',
            'secondary_shortcut': None,  # Optional secondary hotkey for language-specific recording (e.g., "SUPER+ALT+I")
            'secondary_language': None,  # Language code for secondary shortcut (e.g., "it", "en", "fr", etc.)
            'recording_mode': 'toggle',  # 'toggle' | 'push_to_talk' | 'auto' (hybrid tap/hold)
            'grab_keys': False,     # Exclusive keyboard grab (false = safer, true = suppress shortcut from other apps)
            'use_hypr_bindings': False,  # Use Hyprland compositor bindings instead of evdev (disables GlobalShortcuts)
            # Audio device persistence (for reliable device matching across reboots)
            'audio_device_id': None,        # PortAudio device index (can change on reboot)
            'audio_device_name': None,      # Human-readable device name (more stable)
            'audio_device_vendor_id': None, # USB vendor ID (most stable, from udev)
            'audio_device_model_id': None,  # USB model ID (most stable, from udev)
            'model': 'base',
            'threads': 4,           # Thread count for whisper processing
            'language': None,       # Language code for transcription (None = auto-detect, or 'en', 'nl', 'fr', etc.)
            'word_overrides': {},  # Dictionary of word replacements: {"original": "replacement"}
            'whisper_prompt': 'Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules.',
            'clipboard_behavior': False,  # Boolean: true = clear clipboard after delay, false = keep (current behavior)
            'clipboard_clear_delay': 5.0,  # Float: seconds to wait before clearing clipboard (only used if clipboard_behavior is true)
            # Values: "super" | "ctrl_shift" | "ctrl"
            # Default "ctrl_shift" for flexible unix-y primitive
            'paste_mode': 'ctrl_shift',
            # Back-compat for older configs (used only if paste_mode is absent):
            'shift_paste': True,  # true = Ctrl+Shift+V, false = Ctrl+V
            # Transcription backend settings
            'transcription_backend': 'pywhispercpp',  # "pywhispercpp" (or "cpu"/"nvidia"/"vulkan"/"amd") or "rest-api"
            'rest_endpoint_url': None,         # Full HTTP or HTTPS URL for remote transcription
            'rest_api_provider': None,          # Provider identifier for credential lookup (e.g., 'openai', 'groq', 'custom')
            'rest_api_key': None,              # DEPRECATED: Optional API key for authentication (kept for backward compatibility)
            'rest_headers': {},                # Additional HTTP headers for remote transcription
            'rest_body': {},                   # Additional body fields for remote transcription
            'rest_timeout': 30,                # Request timeout in seconds
            'rest_audio_format': 'wav',        # Audio format for remote transcription
            # WebSocket realtime backend settings
            'websocket_provider': None,        # Provider identifier for credential lookup (e.g., 'openai')
            'websocket_model': None,           # Model identifier (e.g., 'gpt-realtime-mini-2025-12-15')
            'websocket_url': None,             # Optional: explicit WebSocket URL (auto-derived if None)
            'realtime_timeout': 30,            # Completion timeout (seconds)
            'realtime_buffer_max_seconds': 5,  # Max buffer before dropping chunks
            'realtime_mode': 'transcribe',      # 'transcribe' (speech-to-text) or 'converse' (voice-to-AI)
            # ONNX-ASR backend settings (CPU-optimized)
            'onnx_asr_model': 'nemo-parakeet-tdt-0.6b-v3',  # Best balance of speed and quality for CPU (includes punctuation)
            'onnx_asr_quantization': 'int8',             # INT8 quantization for CPU performance (or None for fp32)
            'onnx_asr_use_vad': True,                    # Use VAD for long recordings (>30s)
            # Visual feedback settings
            'mic_osd_enabled': True,             # Show microphone visualization overlay during recording
            'mute_detection': True,              # Enable mute detection to cancel recording when mic is muted
            # Audio ducking settings
            'audio_ducking': False,              # Reduce system volume during recording
            'audio_ducking_percent': 50,         # How much to reduce BY (50 = reduce to 50% of original)
            # Post-paste behavior
            'auto_submit': False,                # Send Enter key after pasting text (for chat/search inputs)
            # Long-form recording mode settings
            'long_form_submit_shortcut': None,   # Shortcut to submit long-form recording (e.g., "Super+Return")
            'long_form_temp_limit_mb': 500,      # Max temp storage in MB for long-form segments
            'long_form_auto_save_interval': 300  # Auto-save interval in seconds (default: 5 minutes)
        }
        
        # Set up config directory and file path
        self.config_dir = CONFIG_DIR
        self.config_file = CONFIG_FILE
        
        # Current configuration (starts with defaults)
        self.config = self.default_config.copy()
        
        # Ensure config directory exists
        self._ensure_config_dir()
        
        # Load existing configuration
        self._load_config()
    
    def _ensure_config_dir(self):
        """Ensure the configuration directory exists"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            try:
                from .logger import log_warning
                log_warning(f"Could not create config directory: {e}", "CONFIG")
            except ImportError:
                print(f"Warning: Could not create config directory: {e}")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    
                # Migrate old push_to_talk config to recording_mode (before merging with defaults)
                # Check the original loaded_config, not self.config (which has defaults merged)
                migration_occurred = False
                if 'push_to_talk' in loaded_config and 'recording_mode' not in loaded_config:
                    if loaded_config['push_to_talk']:
                        loaded_config['recording_mode'] = 'push_to_talk'
                    else:
                        loaded_config['recording_mode'] = 'toggle'
                    # Remove old push_to_talk key from loaded config
                    del loaded_config['push_to_talk']
                    migration_occurred = True

                # Migrate old audio_device config key to audio_device_id
                if 'audio_device' in loaded_config and 'audio_device_id' not in loaded_config:
                    loaded_config['audio_device_id'] = loaded_config['audio_device']
                    del loaded_config['audio_device']
                    migration_occurred = True

                # Merge loaded config with defaults (preserving any new default keys)
                self.config.update(loaded_config)
                
                # Attempt automatic migration of API key if needed
                self.migrate_api_key_to_credential_manager()
                
                # Save migrated config if migration occurred
                if migration_occurred:
                    self.save_config()
                    print("Migrated 'push_to_talk' config to 'recording_mode'")
                
                print(f"Configuration loaded from {self.config_file}")
            else:
                print("No existing configuration found, using defaults")
                # Save default configuration
                self.save_config()
                
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            print("Using default configuration")
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"Error: Could not save configuration: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting"""
        return self.config.get(key, default)
    
    def set_setting(self, key: str, value: Any):
        """Set a configuration setting"""
        self.config[key] = value
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings"""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self.default_config.copy()
        print("Configuration reset to defaults")
    
    def get_temp_directory(self) -> Path:
        """Get the temporary directory for audio files"""
        # Use user-writable temp directory instead of system installation directory
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return TEMP_DIR
    
    def get_word_overrides(self) -> Dict[str, str]:
        """Get the word overrides dictionary"""
        return self.config.get('word_overrides', {}).copy()
    
    def add_word_override(self, original: str, replacement: str):
        """Add or update a word override"""
        if 'word_overrides' not in self.config:
            self.config['word_overrides'] = {}
        self.config['word_overrides'][original.lower().strip()] = replacement.strip()
    
    def remove_word_override(self, original: str):
        """Remove a word override"""
        if 'word_overrides' in self.config:
            self.config['word_overrides'].pop(original.lower().strip(), None)
    
    def clear_word_overrides(self):
        """Clear all word overrides"""
        self.config['word_overrides'] = {}
    
    def migrate_api_key_to_credential_manager(self) -> bool:
        """
        Migrate API key from config.json to credential manager.
        
        This function attempts to migrate existing rest_api_key from config
        to the secure credential manager. It tries to identify the provider
        from the endpoint URL or API key prefix, defaulting to 'custom' if
        identification fails.
        
        Returns:
            True if migration was performed, False if no migration was needed
        """
        # Check if migration is needed
        api_key = self.config.get('rest_api_key')
        provider_id = self.config.get('rest_api_provider')
        
        # No migration needed if:
        # - No API key in config, OR
        # - Provider already set (already migrated)
        if not api_key or provider_id:
            return False
        
        # Import here to avoid circular dependencies
        try:
            from .credential_manager import save_credential
            from .provider_registry import PROVIDERS
        except ImportError:
            from credential_manager import save_credential
            from provider_registry import PROVIDERS
        
        # Try to identify provider from endpoint URL
        endpoint_url = self.config.get('rest_endpoint_url', '')
        identified_provider = None
        
        # Check known provider endpoints
        for provider_id_check, provider_data in PROVIDERS.items():
            if provider_data.get('endpoint') == endpoint_url:
                identified_provider = provider_id_check
                break
        
        # If not identified by endpoint, try API key prefix
        if not identified_provider:
            for provider_id_check, provider_data in PROVIDERS.items():
                prefix = provider_data.get('api_key_prefix')
                if prefix and api_key.startswith(prefix):
                    identified_provider = provider_id_check
                    break
        
        # Default to 'custom' if we can't identify
        if not identified_provider:
            identified_provider = 'custom'
        
        # Save API key to credential manager
        if save_credential(identified_provider, api_key):
            # Update config: set provider, remove API key
            self.config['rest_api_provider'] = identified_provider
            self.config['rest_api_key'] = None  # Set to None instead of deleting for backward compat
            self.save_config()
            
            try:
                from .logger import log_info
                log_info(f"Migrated API key to credential manager (provider: {identified_provider})", "CONFIG")
            except ImportError:
                print(f"Migrated API key to credential manager (provider: {identified_provider})")
            
            return True
        else:
            # Failed to save credential, keep old config
            try:
                from .logger import log_warning
                log_warning("Failed to migrate API key to credential manager", "CONFIG")
            except ImportError:
                print("Warning: Failed to migrate API key to credential manager")
            return False
