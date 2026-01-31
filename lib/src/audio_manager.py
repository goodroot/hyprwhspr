"""
Audio feedback manager for hyprwhspr
Handles audio feedback for dictation start/stop events
"""

import os
import subprocess
import threading
from pathlib import Path
from typing import Optional


class AudioManager:
    """Handles audio feedback for recording events"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        
        # Initialize settings from config if available
        if self.config_manager:
            self.enabled = self.config_manager.get_setting('audio_feedback', False)  # Default to disabled
            # General audio volume (fallback for _play_sound when volume=None)
            self.volume = self.config_manager.get_setting('audio_volume', 0.5)  # Default 50% volume
            self.start_volume = self.config_manager.get_setting('start_sound_volume', 1.0)
            self.stop_volume = self.config_manager.get_setting('stop_sound_volume', 1.0)
            self.error_volume = self.config_manager.get_setting('error_sound_volume', 0.5)
            self.start_sound_path = self.config_manager.get_setting('start_sound_path', None)
            self.stop_sound_path = self.config_manager.get_setting('stop_sound_path', None)
            self.error_sound_path = self.config_manager.get_setting('error_sound_path', None)
        else:
            self.enabled = False  # Default to disabled
            self.volume = 0.5  
            self.start_volume = 1.0
            self.stop_volume = 1.0
            self.error_volume = 0.5
            self.start_sound_path = None
            self.stop_sound_path = None
            self.error_sound_path = None

        # Validate volumes
        self.volume = self._validate_volume(self.volume)
        self.start_volume = self._validate_volume(self.start_volume)
        self.stop_volume = self._validate_volume(self.stop_volume)
        self.error_volume = self._validate_volume(self.error_volume)
        
        # Audio file paths - use custom paths if specified, otherwise fall back to defaults
        # Check HYPRWHSPR_ROOT env var first, then fall back to hardcoded path
        install_dir = Path(os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr'))
        self.assets_dir = install_dir / "share" / "assets"
        
        # Fallback to relative paths if installation directory doesn't exist
        if not self.assets_dir.exists():
            # Try parent directories for development
            self.assets_dir = Path(__file__).parent.parent.parent / "share" / "assets"
            # Final fallback for development
            if not self.assets_dir.exists():
                self.assets_dir = Path(__file__).parent.parent / "assets"
        
        # Resolve sound paths (custom path -> relative to assets -> default)
        self.start_sound = self._resolve_sound_path(self.start_sound_path, "ping-up.ogg")
        self.stop_sound = self._resolve_sound_path(self.stop_sound_path, "ping-down.ogg")
        self.error_sound = self._resolve_sound_path(self.error_sound_path, "ping-error.ogg")

        # Check if audio files exist
        self.start_sound_available = self.start_sound.exists()
        self.stop_sound_available = self.stop_sound.exists()
        self.error_sound_available = self.error_sound.exists()

        if not self.start_sound_available or not self.stop_sound_available or not self.error_sound_available:
            print(f"⚠️  Audio feedback files not found:")
            print(f"   Start sound: {'✓' if self.start_sound_available else '✗'} {self.start_sound}")
            print(f"   Stop sound: {'✓' if self.stop_sound_available else '✗'} {self.stop_sound}")
            print(f"   Error sound: {'✓' if self.error_sound_available else '✗'} {self.error_sound}")
            if self.start_sound_path or self.stop_sound_path or self.error_sound_path:
                print(f"   Custom paths specified: start='{self.start_sound_path}', stop='{self.stop_sound_path}', error='{self.error_sound_path}'")
            print(f"   Default assets directory: {self.assets_dir}")
    
    def _validate_volume(self, volume: float) -> float:
        """Validate and clamp volume to reasonable bounds"""
        try:
            volume = float(volume)
        except (ValueError, TypeError):
            volume = 0.3

        # Clamp between 0.1 (10%) and 1.0 (100%)
        volume = max(0.1, min(volume, 1.0))
        return volume

    def _resolve_sound_path(self, custom_path: Optional[str], default_filename: str) -> Path:
        """
        Resolve a sound file path with fallback logic.

        Tries: custom_path as-is -> custom_path relative to assets -> default
        """
        if custom_path:
            # Try as absolute path first
            path = Path(custom_path)
            if path.exists():
                return path
            # Try relative to assets directory
            path = self.assets_dir / custom_path
            if path.exists():
                return path
        # Fall back to default
        return self.assets_dir / default_filename

    def _play_sound(self, sound_file: Path, volume: float = None) -> bool:
        """
        Play an audio file with volume control
        
        Args:
            sound_file: Path to the audio file
            volume: Volume level (0.1 to 1.0), uses instance volume if None
            
        Returns:
            True if successful, False otherwise
        """
        if not sound_file.exists():
            print(f"Audio file not found: {sound_file}")
            return False
        
        if volume is None:
            volume = self.volume
        
        try:
            # Try using ffplay (most reliable, supports volume control and all formats)
            if self._is_tool_available('ffplay'):
                return self._play_with_ffplay(sound_file, volume)

            # Fallback to paplay (PulseAudio/PipeWire, supports OGG)
            elif self._is_tool_available('paplay'):
                return self._play_with_paplay(sound_file)

            # Fallback to pw-play (native PipeWire, supports OGG)
            elif self._is_tool_available('pw-play'):
                return self._play_with_pwplay(sound_file)

            # Fallback to aplay (ALSA) - only for WAV files, not OGG
            elif self._is_tool_available('aplay'):
                if sound_file.suffix.lower() in ['.wav', '.wave']:
                    return self._play_with_aplay(sound_file)
                else:
                    print(f"aplay does not support {sound_file.suffix} files (only WAV). Install ffplay or paplay for OGG support.")
                    return False

            else:
                print("No audio playback tools available (ffplay, paplay, pw-play, or aplay)")
                return False
                
        except Exception as e:
            print(f"Failed to play audio: {e}")
            return False
    
    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if a command-line tool is available"""
        try:
            result = subprocess.run(['which', tool_name], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _run_audio_command(self, cmd: list, tool_name: str) -> bool:
        """Run an audio command in a background thread"""
        try:
            def play_audio():
                subprocess.run(cmd, capture_output=True, timeout=5)

            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return True
        except Exception as e:
            print(f"{tool_name} failed: {e}")
            return False

    def _play_with_ffplay(self, sound_file: Path, volume: float) -> bool:
        """Play audio with ffplay (supports volume control)"""
        # Convert volume from 0.1-1.0 to ffplay's -volume (0-100)
        ffplay_volume = int(volume * 100)
        cmd = [
            'ffplay',
            '-nodisp',  # No display window
            '-autoexit',  # Exit after playing
            '-volume', str(ffplay_volume),
            '-loglevel', 'error',  # Minimal logging
            str(sound_file)
        ]
        return self._run_audio_command(cmd, 'ffplay')

    def _play_with_aplay(self, sound_file: Path) -> bool:
        """Play audio with aplay (ALSA, no volume control)"""
        return self._run_audio_command(['aplay', '-q', str(sound_file)], 'aplay')

    def _play_with_paplay(self, sound_file: Path) -> bool:
        """Play audio with paplay (PulseAudio/PipeWire, no volume control)"""
        return self._run_audio_command(['paplay', str(sound_file)], 'paplay')

    def _play_with_pwplay(self, sound_file: Path) -> bool:
        """Play audio with pw-play (native PipeWire, no volume control)"""
        return self._run_audio_command(['pw-play', str(sound_file)], 'pw-play')
    
    def play_start_sound(self) -> bool:
        """Play the recording start sound"""
        if not self.enabled or not self.start_sound_available:
            return False
        
        result = self._play_sound(self.start_sound, self.start_volume)
        if not result:
            print(f"Failed to play start sound: {self.start_sound}")
        return result
    
    def play_stop_sound(self) -> bool:
        """Play the recording stop sound"""
        if not self.enabled or not self.stop_sound_available:
            return False

        result = self._play_sound(self.stop_sound, self.stop_volume)
        if not result:
            print(f"Failed to play stop sound: {self.stop_sound}")
        return result

    def play_error_sound(self) -> bool:
        """Play the error sound (e.g., for blank audio or failed transcription)"""
        if not self.enabled or not self.error_sound_available:
            return False

        result = self._play_sound(self.error_sound, self.error_volume)
        if not result:
            print(f"Failed to play error sound: {self.error_sound}")
        return result

    def set_audio_feedback(self, enabled: bool):
        """Enable or disable audio feedback"""
        self.enabled = bool(enabled)
        if self.config_manager:
            self.config_manager.set_setting('audio_feedback', self.enabled)
        print(f"Audio feedback {'enabled' if self.enabled else 'disabled'}")
    
    def set_audio_volume(self, volume: float):
        """Set the audio volume (0.1 to 1.0)"""
        self.volume = self._validate_volume(volume)
        if self.config_manager:
            self.config_manager.set_setting('audio_volume', self.volume)
        print(f"Audio volume set to {self.volume:.1%}")
    
    def set_start_sound_volume(self, volume: float):
        """Set the start sound volume (0.1 to 1.0)"""
        self.start_volume = self._validate_volume(volume)
        if self.config_manager:
            self.config_manager.set_setting('start_sound_volume', self.start_volume)
        print(f"Start sound volume set to {self.start_volume:.1%}")
    
    def set_stop_sound_volume(self, volume: float):
        """Set the stop sound volume (0.1 to 1.0)"""
        self.stop_volume = self._validate_volume(volume)
        if self.config_manager:
            self.config_manager.set_setting('stop_sound_volume', self.stop_volume)
        print(f"Stop sound volume set to {self.stop_volume:.1%}")
    
    def set_start_sound_path(self, sound_path: str):
        """Set custom path for start sound file"""
        if sound_path:
            # Try the path as-is first (for absolute paths)
            path = Path(sound_path)
            if path.exists():
                self.start_sound_path = str(path)
                self.start_sound = path
                self.start_sound_available = True
                if self.config_manager:
                    self.config_manager.set_setting('start_sound_path', self.start_sound_path)
                print(f"Start sound path set to: {self.start_sound_path}")
            else:
                # Try relative to assets directory (for relative paths like "ping-up.ogg")
                path = self.assets_dir / sound_path
                if path.exists():
                    self.start_sound_path = sound_path  # Keep relative path in config
                    self.start_sound = path
                    self.start_sound_available = True
                    if self.config_manager:
                        self.config_manager.set_setting('start_sound_path', self.start_sound_path)
                    print(f"Start sound path set to: {self.start_sound_path}")
                else:
                    print(f"Start sound file not found: {sound_path}")
        else:
            # Reset to default
            self.start_sound_path = None
            self.start_sound = self.assets_dir / "ping-up.ogg"
            self.start_sound_available = self.start_sound.exists()
            if self.config_manager:
                self.config_manager.set_setting('start_sound_path', None)
            print("Start sound reset to default")
    
    def set_stop_sound_path(self, sound_path: str):
        """Set custom path for stop sound file"""
        if sound_path:
            # Try the path as-is first (for absolute paths)
            path = Path(sound_path)
            if path.exists():
                self.stop_sound_path = str(path)
                self.stop_sound = path
                self.stop_sound_available = True
                if self.config_manager:
                    self.config_manager.set_setting('stop_sound_path', self.stop_sound_path)
                print(f"Stop sound path set to: {self.stop_sound_path}")
            else:
                # Try relative to assets directory (for relative paths like "ping-down.ogg")
                path = self.assets_dir / sound_path
                if path.exists():
                    self.stop_sound_path = sound_path  # Keep relative path in config
                    self.stop_sound = path
                    self.stop_sound_available = True
                    if self.config_manager:
                        self.config_manager.set_setting('stop_sound_path', self.stop_sound_path)
                    print(f"Stop sound path set to: {self.stop_sound_path}")
                else:
                    print(f"Stop sound file not found: {sound_path}")
        else:
            # Reset to default
            self.stop_sound_path = None
            self.stop_sound = self.assets_dir / "ping-down.ogg"
            self.stop_sound_available = self.stop_sound.exists()
            if self.config_manager:
                self.config_manager.set_setting('stop_sound_path', None)
            print("Stop sound reset to default")
    
    def get_status(self) -> dict:
        """Get the status of the audio manager"""
        return {
            'enabled': self.enabled,
            'start_volume': self.start_volume,
            'stop_volume': self.stop_volume,
            'error_volume': self.error_volume,
            'start_sound_available': self.start_sound_available,
            'stop_sound_available': self.stop_sound_available,
            'error_sound_available': self.error_sound_available,
            'start_sound_path': str(self.start_sound),
            'stop_sound_path': str(self.stop_sound),
            'error_sound_path': str(self.error_sound),
            'start_sound_custom': self.start_sound_path,
            'stop_sound_custom': self.stop_sound_path,
            'error_sound_custom': self.error_sound_path,
            'default_assets_dir': str(self.assets_dir),
        }
