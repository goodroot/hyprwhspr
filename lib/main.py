#!/usr/bin/env python3
"""
hyprwhspr - Voice dictation application for Hyprland (Headless Mode)
Fast, reliable speech-to-text with instant text injection
"""

import sys
import time
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from config_manager import ConfigManager
from audio_capture import AudioCapture
from whisper_manager import WhisperManager
from text_injector import TextInjector
from global_shortcuts import GlobalShortcuts
from audio_manager import AudioManager

class hyprwhsprApp:
    """Main application class for hyprwhspr voice dictation (Headless Mode)"""

    def __init__(self):
        # Initialize core components
        self.config = ConfigManager()

        # Initialize audio capture with configured device
        audio_device_id = self.config.get_setting('audio_device', None)
        self.audio_capture = AudioCapture(device_id=audio_device_id)

        # Initialize audio feedback manager
        self.audio_manager = AudioManager(self.config)

        self.whisper_manager = WhisperManager()
        self.text_injector = TextInjector(self.config)
        self.global_shortcuts = None

        # Application state
        self.is_recording = False
        self.is_processing = False
        self.current_transcription = ""

        # Set up global shortcuts (needed for headless operation)
        self._setup_global_shortcuts()

    def _setup_global_shortcuts(self):
        """Initialize global keyboard shortcuts"""
        try:
            shortcut_key = self.config.get_setting('primary_shortcut', 'Super+Alt+D')
            self.global_shortcuts = GlobalShortcuts(shortcut_key, self._on_shortcut_triggered)
            print(f"🎯 Global shortcut configured: {shortcut_key}")
        except Exception as e:
            print(f"❌ Failed to initialize global shortcuts: {e}")
            self.global_shortcuts = None

    def _on_shortcut_triggered(self):
        """Handle global shortcut trigger"""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start voice recording"""
        if self.is_recording:
            return

        try:
            print("🎤 Starting recording...")
            self.is_recording = True
            
            # Write recording status to file for tray script
            self._write_recording_status(True)
            
            # Play start sound
            self.audio_manager.play_start_sound()
            
            # Start audio capture
            self.audio_capture.start_recording()
            
            print("✅ Recording started - speak now!")
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.is_recording = False
            self._write_recording_status(False)

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        if not self.is_recording:
            return

        try:
            print("🛑 Stopping recording...")
            self.is_recording = False
            
            # Write recording status to file for tray script
            self._write_recording_status(False)
            
            # Stop audio capture
            audio_data = self.audio_capture.stop_recording()
            
            # Play stop sound
            self.audio_manager.play_stop_sound()
            
            if audio_data is not None:
                self._process_audio(audio_data)
            else:
                print("⚠️ No audio data captured")
                
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")

    def _process_audio(self, audio_data):
        """Process captured audio through Whisper"""
        if self.is_processing:
            return

        try:
            self.is_processing = True
            print("🧠 Processing audio with Whisper...")
            
            # Transcribe audio
            transcription = self.whisper_manager.transcribe_audio(audio_data)
            
            if transcription and transcription.strip():
                self.current_transcription = transcription.strip()
                print(f"📝 Transcription: {self.current_transcription}")
                
                # Inject text
                self._inject_text(self.current_transcription)
            else:
                print("⚠️ No transcription generated")
                
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
        finally:
            self.is_processing = False

    def _inject_text(self, text):
        """Inject transcribed text into active application"""
        try:
            print(f"⌨️ Injecting text: {text}")
            self.text_injector.inject_text(text)
            print("✅ Text injection completed")
        except Exception as e:
            print(f"❌ Text injection failed: {e}")

    def _write_recording_status(self, is_recording):
        """Write recording status to file for tray script"""
        try:
            status_file = Path.home() / '.config' / 'hyprwhspr' / 'recording_status'
            status_file.parent.mkdir(parents=True, exist_ok=True)
            
            if is_recording:
                with open(status_file, 'w') as f:
                    f.write('true')
            else:
                # Remove the file when not recording to avoid stale state
                if status_file.exists():
                    status_file.unlink()
        except Exception as e:
            print(f"⚠️ Failed to write recording status: {e}")

    def run(self):
        """Start the application"""
        print("🚀 Starting hyprwhspr...")

        # Initialize whisper manager
        if not self.whisper_manager.initialize():
            print("❌ Failed to initialize Whisper. Please ensure whisper.cpp is built.")
            print("Run the build scripts first.")
            return False

        print("✅ hyprwhspr initialized successfully")
        print("🎤 Listening for global shortcuts...")
        
        # Start global shortcuts
        if self.global_shortcuts:
            self.global_shortcuts.start()
        
        try:
            # Keep the application running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down hyprwhspr...")
            self._cleanup()
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            self._cleanup()
            return False
        
        return True

    def _cleanup(self):
        """Clean up resources when shutting down"""
        try:
            # Stop global shortcuts
            if self.global_shortcuts:
                self.global_shortcuts.stop()

            # Stop audio capture
            if self.is_recording:
                self.audio_capture.stop_recording()

            # Save configuration
            self.config.save_config()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


def main():
    """Main entry point"""
    print("🎤 hyprwhspr")
    print("🚀 Starting hyprwhspr...")
    
    try:
        app = hyprwhsprApp()
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopping hyprwhspr...")
        app._cleanup()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
