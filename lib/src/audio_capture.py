"""
Audio capture module for hyprwhspr
Handles real-time audio capture for speech recognition
"""

import sys
import wave
import threading
import time
from typing import Optional, Callable
from io import BytesIO

try:
    import sounddevice as sd
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-sounddevice is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pacman -S python-sounddevice    # system-wide on Arch", file=sys.stderr)
    sys.exit(1)

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: python-numpy is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pacman -S python-numpy    # system-wide on Arch", file=sys.stderr)
    sys.exit(1)


class AudioCapture:
    """Handles audio recording and real-time level monitoring"""
    
    def __init__(self, device_id=None):
        # Audio configuration - whisper.cpp prefers 16kHz mono
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.dtype = np.float32
        
        # Device configuration
        self.preferred_device_id = device_id
        
        # Recording state
        self.is_recording = False
        self.is_monitoring = False
        self.audio_data = []
        self.current_level = 0.0
        
        # Threading
        self.record_thread = None
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Callbacks
        self.level_callback = None
        
        # Audio stream
        self.stream = None
        
        # Initialize sounddevice
        self._initialize_sounddevice()
    
    def _initialize_sounddevice(self):
        """Initialize sounddevice and check for available devices"""
        try:
            # Set default settings
            sd.default.samplerate = self.sample_rate
            sd.default.channels = self.channels
            sd.default.dtype = self.dtype
            
            # Set the preferred device if specified
            if self.preferred_device_id is not None:
                try:
                    # Validate that the device exists and has input channels
                    device_info = sd.query_devices(device=self.preferred_device_id, kind='input')
                    if device_info['max_input_channels'] > 0:
                        sd.default.device[0] = self.preferred_device_id
                        print(f"Using configured audio device: {device_info['name']} (ID: {self.preferred_device_id})")
                    else:
                        print(f"⚠ Configured device {self.preferred_device_id} has no input channels, using default")
                        self.preferred_device_id = None
                except Exception as e:
                    print(f"⚠ Configured audio device {self.preferred_device_id} not available: {e}")
                    self.preferred_device_id = None
            
            # If no specific device was configured or it failed, use system default
            if self.preferred_device_id is None:
                self._set_system_default_device()
            
            # Get device information
            try:
                # Get current input device info
                current_device_id = sd.default.device[0] if sd.default.device[0] is not None else sd.default.device
                device_info = sd.query_devices(device=current_device_id, kind='input')
                
                # Store device info for later use
                self.device_info = device_info
                self.device_id = current_device_id
                
            except Exception as e:
                print(f"⚠ Could not query device details: {e}")
                self.device_info = None
                self.device_id = None
            
        except Exception as e:
            print(f"ERROR: Failed to initialize sounddevice: {e}")
            self.device_info = None
            self.device_id = None
    
    def _set_system_default_device(self):
        """Set system default device when no specific device is configured"""
        try:
            # Use system default - no need to list all devices
            pass
        except Exception as e:
            print(f"⚠ Could not query audio devices: {e}")
    
    @staticmethod
    def get_available_input_devices():
        """Get list of available input devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    host_api_info = sd.query_hostapis(device['hostapi'])
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'host_api': host_api_info['name'],
                        'display_name': f"{device['name']} ({host_api_info['name']})"
                    })
            
            return input_devices
            
        except Exception as e:
            print(f"Error getting input devices: {e}")
            return []
    
    def get_current_device_info(self):
        """Get information about the currently selected device"""
        try:
            if self.device_info:
                return {
                    'id': self.device_id,
                    'name': self.device_info['name'],
                    'channels': self.device_info['max_input_channels'],
                    'sample_rate': self.device_info['default_samplerate']
                }
            return None
        except:
            return None
    
    def set_device(self, device_id):
        """Set the audio input device"""
        try:
            if device_id is None:
                # Reset to system default
                self.preferred_device_id = None
                sd.default.device[0] = None
            else:
                # Validate device exists and has input channels
                device_info = sd.query_devices(device=device_id, kind='input')
                if device_info['max_input_channels'] > 0:
                    self.preferred_device_id = device_id
                    sd.default.device[0] = device_id
                    self.device_info = device_info
                    self.device_id = device_id
                    print(f"Audio device changed to: {device_info['name']} (ID: {device_id})")
                    return True
                else:
                    print(f"Device {device_id} has no input channels")
                    return False
                    
        except Exception as e:
            print(f"Error setting audio device: {e}")
            return False
    
    def _find_system_input_device(self):
        """Try to find the system's configured input device"""
        try:
            # Try to get the system's default input device using pactl (PulseAudio)
            import subprocess
            try:
                result = subprocess.run(['pactl', 'get-default-source'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    default_source = result.stdout.strip()
                    print(f"PulseAudio default source: {default_source}")
                    
                    # Try to match this with sounddevice devices
                    devices = sd.query_devices()
                    for device_idx, device in enumerate(devices):
                        if (device['max_input_channels'] > 0 and 
                            (default_source.lower() in device['name'].lower() or 
                             any(keyword in device['name'].lower() 
                                 for keyword in ['blue', 'microphone', 'usb', 'webcam']))):
                            return device_idx
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass  # pactl not available or failed
            
            # Fallback: Look for devices that are commonly preferred
            devices = sd.query_devices()
            
            # Priority order: USB microphones, USB audio, built-in audio
            device_priorities = [
                ['blue', 'microphone'],  # Blue Microphones (your preferred device)
                ['usb', 'audio'],        # Other USB audio devices
                ['webcam', 'usb'],       # USB webcams with audio
                ['analog']               # Built-in analog audio
            ]
            
            for priority_keywords in device_priorities:
                for device_idx, device in enumerate(devices):
                    if (device['max_input_channels'] > 0 and
                        all(keyword in device['name'].lower() for keyword in priority_keywords)):
                        return device_idx
            
            return None
            
        except Exception as e:
            print(f"Error finding system input device: {e}")
            return None
    
    def _find_pulseaudio_input_device(self):
        """Find a PulseAudio input device"""
        try:
            host_apis = sd.query_hostapis()
            pulseaudio_idx = None
            
            for idx, api in enumerate(host_apis):
                if 'pulse' in api['name'].lower():
                    pulseaudio_idx = idx
                    break
            
            if pulseaudio_idx is not None:
                devices = sd.query_devices()
                for device_idx, device in enumerate(devices):
                    if (device['hostapi'] == pulseaudio_idx and 
                        device['max_input_channels'] > 0):
                        return device_idx
            
            return None
            
        except Exception as e:
            print(f"Error finding PulseAudio device: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if audio capture is available"""
        try:
            # Test if we can query devices
            sd.query_devices()
            return True
        except Exception:
            return False
    
    def start_recording(self) -> bool:
        """Start recording audio"""
        if not self.is_available():
            raise RuntimeError("Audio capture not available")
        
        if self.is_recording:
            return True
        
        try:
            # Clear previous audio data
            with self.lock:
                self.audio_data = []
                self.is_recording = True
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return the recorded audio data"""
        if not self.is_recording:
            return None
        
        # Signal to stop recording
        with self.lock:
            self.is_recording = False
        
        # Wait for recording thread to finish
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2.0)
        
        # Clean up stream
        self._cleanup_stream()
        
        # Return recorded data
        with self.lock:
            if self.audio_data:
                # Concatenate all audio chunks
                audio_array = np.concatenate(self.audio_data, axis=0)
                duration = len(audio_array) / self.sample_rate
                if duration < 0.5:
                    print(f"[WARN] Recording very short ({duration:.2f}s), may not have captured audio")
                return audio_array
            else:
                return None
    
    def _record_audio(self):
        """Internal method to record audio in a separate thread"""
        try:
            chunk_count = 0
            # Callback function for sounddevice
            def audio_callback(indata, frames, time_info, status):
                nonlocal chunk_count
                if status:
                    print(f"[WARN] Audio callback status: {status}")
                
                with self.lock:
                    if self.is_recording:
                        # Store the audio data (indata is already numpy array)
                        audio_chunk = indata[:, 0]  # Get mono channel
                        
                        # Update current audio level for monitoring
                        self.current_level = np.sqrt(np.mean(audio_chunk**2))
                        
                        # Store audio data
                        self.audio_data.append(audio_chunk.copy())
                        chunk_count += 1
            
            # Determine device to use for recording
            device_to_use = self.preferred_device_id if self.preferred_device_id is not None else None
            
            # Start audio stream with callback - explicitly specify device
            with sd.InputStream(
                device=device_to_use,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                # Keep recording while is_recording is True
                while self.is_recording:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"[ERROR] Error in recording thread: {e}")
            import traceback
            traceback.print_exc()
    
    def start_monitoring(self, level_callback: Optional[Callable[[float], None]] = None):
        """Start monitoring audio levels without recording"""
        if self.is_monitoring:
            return
            
        if not self.is_available():
            print("Audio capture not available for monitoring")
            return
            
        self.level_callback = level_callback
        self.is_monitoring = True
        
        try:
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_audio, daemon=True)
            self.monitor_thread.start()
            
        except Exception as e:
            print(f"Failed to start audio monitoring: {e}")
            self.is_monitoring = False
    
    def stop_monitoring(self):
        """Stop monitoring audio levels"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_audio(self):
        """Internal method to monitor audio levels"""
        try:
            # Callback function for monitoring
            def monitor_callback(indata, frames, time_info, status):
                if status:
                    print(f"Monitor callback status: {status}")
                
                if self.is_monitoring and not self.is_recording:
                    # Calculate RMS level
                    audio_chunk = indata[:, 0]  # Get mono channel
                    level = np.sqrt(np.mean(audio_chunk**2))
                    self.current_level = level
                    
                    # Call callback if provided
                    if self.level_callback:
                        self.level_callback(level)
            
            # Start monitoring stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=monitor_callback
            ):
                # Keep monitoring while is_monitoring is True and not recording
                while self.is_monitoring:
                    if self.is_recording:
                        # If recording, just use the current level from recording
                        if self.level_callback:
                            self.level_callback(self.current_level)
                    
                    time.sleep(0.05)  # ~20Hz update rate
                
        except Exception as e:
            print(f"Error in monitoring thread: {e}")
        finally:
            print("Audio monitoring thread finished")
    
    def get_audio_level(self) -> float:
        """Get the current audio level (0.0 to 1.0)"""
        return min(1.0, self.current_level * 10)  # Scale for better visualization
    
    def _cleanup_stream(self):
        """Clean up the audio stream"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            print(f"Error cleaning up audio stream: {e}")
    
    def list_devices(self):
        """List available audio input devices"""
        if not self.is_available():
            print("sounddevice not available")
            return
            
        print("Available audio input devices:")
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Input device
                    print(f"  Device {i}: {device['name']} "
                          f"(Channels: {device['max_input_channels']}, "
                          f"Sample Rate: {device['default_samplerate']})")
        except Exception as e:
            print(f"Error querying devices: {e}")
    
    def save_audio_to_wav(self, audio_data: np.ndarray, filename: str):
        """Save audio data to a WAV file"""
        try:
            # Convert float32 to int16 for WAV format
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            print(f"Audio saved to {filename}")
            
        except Exception as e:
            print(f"ERROR: Failed to save audio: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.is_recording:
                self.stop_recording()
            if self.is_monitoring:
                self.stop_monitoring()
        except:
            pass  # Ignore errors during cleanup
