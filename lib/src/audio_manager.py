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
            self.start_volume = self.config_manager.get_setting('start_sound_volume', 0.3)  # Default 30% volume
            self.stop_volume = self.config_manager.get_setting('stop_sound_volume', 0.3)  # Default 30% volume
            self.start_sound_path = self.config_manager.get_setting('start_sound_path', None)  # Custom start sound path
            self.stop_sound_path = self.config_manager.get_setting('stop_sound_path', None)  # Custom stop sound path
        else:
            self.enabled = False  # Default to disabled
            self.start_volume = 0.3
            self.stop_volume = 0.3
            self.start_sound_path = None
            self.stop_sound_path = None
        
        # Validate volumes
        self.start_volume = self._validate_volume(self.start_volume)
        self.stop_volume = self._validate_volume(self.stop_volume)
        
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
        
        # Start sound path resolution
        if self.start_sound_path:
            # Try the path as-is first (for absolute paths)
            start_path = Path(self.start_sound_path)
            if start_path.exists():
                self.start_sound = start_path
            else:
                # Try relative to assets directory (for relative paths like "ping-up.ogg")
                start_path = self.assets_dir / self.start_sound_path
                if start_path.exists():
                    self.start_sound = start_path
                else:
                    # Fall back to default
                    self.start_sound = self.assets_dir / "ping-up.ogg"
        else:
            self.start_sound = self.assets_dir / "ping-up.ogg"
        
        # Stop sound path resolution
        if self.stop_sound_path:
            # Try the path as-is first (for absolute paths)
            stop_path = Path(self.stop_sound_path)
            if stop_path.exists():
                self.stop_sound = stop_path
            else:
                # Try relative to assets directory (for relative paths like "ping-down.ogg")
                stop_path = self.assets_dir / self.stop_sound_path
                if stop_path.exists():
                    self.stop_sound = stop_path
                else:
                    # Fall back to default
                    self.stop_sound = self.assets_dir / "ping-down.ogg"
        else:
            self.stop_sound = self.assets_dir / "ping-down.ogg"
        
        # Check if audio files exist
        self.start_sound_available = self.start_sound.exists()
        self.stop_sound_available = self.stop_sound.exists()
        
        if not self.start_sound_available or not self.stop_sound_available:
            print(f"⚠️  Audio feedback files not found:")
            print(f"   Start sound: {'✓' if self.start_sound_available else '✗'} {self.start_sound}")
            print(f"   Stop sound: {'✓' if self.stop_sound_available else '✗'} {self.stop_sound}")
            if self.start_sound_path or self.stop_sound_path:
                print(f"   Custom paths specified: start='{self.start_sound_path}', stop='{self.stop_sound_path}'")
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
            # Try using ffplay (most reliable, supports volume control)
            if self._is_ffplay_available():
                return self._play_with_ffplay(sound_file, volume)
            
            # Fallback to aplay (ALSA, no volume control)
            elif self._is_aplay_available():
                return self._play_with_aplay(sound_file)
            
            # Fallback to paplay (PulseAudio, no volume control)
            elif self._is_paplay_available():
                return self._play_with_paplay(sound_file)
            
            else:
                print("No audio playback tools available (ffplay, aplay, or paplay)")
                return False
                
        except Exception as e:
            print(f"Failed to play audio: {e}")
            return False
    
    def _is_ffplay_available(self) -> bool:
        """Check if ffplay is available"""
        try:
            result = subprocess.run(['which', 'ffplay'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _is_aplay_available(self) -> bool:
        """Check if aplay is available"""
        try:
            result = subprocess.run(['which', 'aplay'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _is_paplay_available(self) -> bool:
        """Check if paplay is available"""
        try:
            result = subprocess.run(['which', 'paplay'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _play_with_ffplay(self, sound_file: Path, volume: float) -> bool:
        """Play audio with ffplay (supports volume control)"""
        try:
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
            
            # Run in background thread to avoid blocking
            def play_audio():
                subprocess.run(cmd, capture_output=True, timeout=5)
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return True
            
        except Exception as e:
            print(f"ffplay failed: {e}")
            return False
    
    def _play_with_aplay(self, sound_file: Path) -> bool:
        """Play audio with aplay (ALSA, no volume control)"""
        try:
            cmd = ['aplay', '-q', str(sound_file)]
            
            def play_audio():
                subprocess.run(cmd, capture_output=True, timeout=5)
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return True
            
        except Exception as e:
            print(f"aplay failed: {e}")
            return False
    
    def _play_with_paplay(self, sound_file: Path) -> bool:
        """Play audio with paplay (PulseAudio, no volume control)"""
        try:
            cmd = ['paplay', str(sound_file)]
            
            def play_audio():
                subprocess.run(cmd, capture_output=True, timeout=5)
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
            return True
            
        except Exception as e:
            print(f"paplay failed: {e}")
            return False
    
    def play_start_sound(self) -> bool:
        """Play the recording start sound"""
        print(f"🔊 play_start_sound called - enabled: {self.enabled}, available: {self.start_sound_available}")
        if not self.enabled or not self.start_sound_available:
            print(f"❌ Start sound blocked - enabled: {self.enabled}, available: {self.start_sound_available}")
            return False
        
        print(f"🔊 Playing start sound: {self.start_sound} (volume: {self.start_volume})")
        result = self._play_sound(self.start_sound, self.start_volume)
        print(f"🔊 Start sound result: {'Success' if result else 'Failed'}")
        return result
    
    def play_stop_sound(self) -> bool:
        """Play the recording stop sound"""
        print(f"🔊 play_stop_sound called - enabled: {self.enabled}, available: {self.stop_sound_available}")
        if not self.enabled or not self.stop_sound_available:
            print(f"❌ Stop sound blocked - enabled: {self.enabled}, available: {self.stop_sound_available}")
            return False
        
        print(f"🔊 Playing stop sound: {self.stop_sound} (volume: {self.stop_volume})")
        result = self._play_sound(self.stop_sound, self.stop_volume)
        print(f"🔊 Stop sound result: {'Success' if result else 'Failed'}")
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
            'start_sound_available': self.start_sound_available,
            'stop_sound_available': self.stop_sound_available,
            'start_sound_path': str(self.start_sound),
            'stop_sound_path': str(self.stop_sound),
            'start_sound_custom': self.start_sound_path,
            'stop_sound_custom': self.stop_sound_path,
            'default_assets_dir': str(self.assets_dir),
        }
