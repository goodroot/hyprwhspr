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
    from .dependencies import require_package
except ImportError:
    from dependencies import require_package

sd = require_package('sounddevice')
np = require_package('numpy')


class AudioCapture:
    """Handles audio recording and real-time level monitoring"""
    
    def __init__(self, device_id=None, config_manager=None):
        # Audio configuration - whisper.cpp prefers 16kHz mono
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.dtype = np.float32

        # Device configuration
        self.preferred_device_id = device_id
        self.config = config_manager  # For accessing device name fallback
        
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
        self.streaming_callback = None  # For realtime streaming backends
        
        # Audio stream
        self.stream = None
        
        # Recovery state tracking
        self.recovery_in_progress = False
        self.recovery_lock = threading.Lock()
        self.last_callback_monotonic = 0.0  # Timestamp of last callback
        self.frames_since_start = 0  # Frame count for success criteria
        self.recovery_start_time = 0.0  # When recovery started
        
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
            device_found = False
            if self.preferred_device_id is not None:
                try:
                    # Validate that the device exists and has input channels
                    device_info = sd.query_devices(device=self.preferred_device_id, kind='input')
                    if device_info['max_input_channels'] > 0:
                        sd.default.device[0] = self.preferred_device_id
                        print(f"Using configured audio device: {device_info['name']} (ID: {self.preferred_device_id})")
                        device_found = True
                    else:
                        print(f"⚠ Configured device {self.preferred_device_id} has no input channels")
                        self.preferred_device_id = None
                except Exception as e:
                    print(f"⚠ Configured audio device ID {self.preferred_device_id} not available: {e}")
                    # Try fallback to system default
                    pulse_default_id = self._get_pulse_default_source_device_id()
                    if pulse_default_id is not None:
                        try:
                            device_info = sd.query_devices(device=pulse_default_id, kind='input')
                            sd.default.device[0] = pulse_default_id
                            self.preferred_device_id = pulse_default_id
                            device_found = True
                            print(f"[FALLBACK] Using system default: {device_info['name']} (ID: {pulse_default_id})")
                            self._notify_device_fallback(device_info['name'])
                        except Exception:
                            pass

                    if not device_found:
                        self.preferred_device_id = None

            # If device ID failed, try to find by name (more stable across reboots)
            if not device_found and self.config:
                configured_name = self.config.get_setting('audio_device_name')
                if configured_name:
                    print(f"Searching for device by name: {configured_name}")
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0 and configured_name in device['name']:
                            self.preferred_device_id = i
                            sd.default.device[0] = i
                            print(f"Found device by name: {device['name']} (ID: {i})")
                            device_found = True
                            break

                    # If name search failed, try fallback to system default
                    if not device_found:
                        pulse_default_id = self._get_pulse_default_source_device_id()
                        if pulse_default_id is not None:
                            try:
                                device_info = sd.query_devices(device=pulse_default_id, kind='input')
                                sd.default.device[0] = pulse_default_id
                                self.preferred_device_id = pulse_default_id
                                device_found = True
                                print(f"[FALLBACK] Using system default: {device_info['name']} (ID: {pulse_default_id})")
                                self._notify_device_fallback(device_info['name'])
                            except Exception:
                                pass

            # If no specific device was configured or it failed, use system default
            if not device_found:
                if self.preferred_device_id is None:
                    print("Using system default audio device")
                self._set_system_default_device()
            
            # Get device information
            try:
                # Extract input device ID (sd.default.device is tuple (input, output) or None)
                if sd.default.device is None:
                    current_device_id = None
                elif isinstance(sd.default.device, tuple):
                    current_device_id = sd.default.device[0]
                else:
                    # Legacy: single integer
                    current_device_id = sd.default.device
                
                if current_device_id is not None:
                    device_info = sd.query_devices(device=current_device_id, kind='input')
                    self.device_info = device_info
                    self.device_id = current_device_id
                else:
                    self.device_info = None
                    self.device_id = None
                
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
            # Ensure we have a valid default input device
            # sd.default.device is tuple (input, output) or None
            if sd.default.device is None or (isinstance(sd.default.device, tuple) and sd.default.device[0] is None):
                # Find first available input device
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        if sd.default.device is None:
                            sd.default.device = (i, None)
                        else:
                            sd.default.device = (i, sd.default.device[1] if isinstance(sd.default.device, tuple) else None)
                        break
        except Exception as e:
            print(f"⚠ Could not set system default device: {e}")
    
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
                # Reset to system default - re-initialize to get fresh default
                self.preferred_device_id = None
                self._initialize_sounddevice()
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
    
    def _get_pulse_default_source_device_id(self) -> Optional[int]:
        """Get PortAudio device ID for PulseAudio/PipeWire default source"""
        import subprocess
        try:
            result = subprocess.run(
                ['pactl', 'get-default-source'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return None

            pulse_source_name = result.stdout.strip()
            print(f"[PULSE] System default source: {pulse_source_name}")

            # Match pulse source name to PortAudio device
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    # PulseAudio source names appear in PortAudio device names
                    # Example: pulse source "alsa_input.usb-Blue_Microphones" matches
                    # PortAudio device name "Blue Microphones: USB Audio (hw:2,0)"
                    device_name = device['name'].lower()
                    source_name = pulse_source_name.lower()

                    # Direct match (source name in device name)
                    if source_name in device_name:
                        print(f"[PULSE] Matched device {idx}: {device['name']}")
                        return idx

                    # Partial match (extract model from source name)
                    # Example: "alsa_input.usb-Blue_Microphones" → "blue_microphones"
                    if 'alsa_input' in source_name:
                        model_part = source_name.split('alsa_input.')[-1]
                        # Remove USB- prefix if present
                        model_part = model_part.replace('usb-', '').replace('_', ' ')
                        if model_part in device_name:
                            print(f"[PULSE] Matched device {idx} via model: {device['name']}")
                            return idx

            # Fallback: return first PulseAudio device with input channels
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0 and 'pulse' in device['name'].lower():
                    print(f"[PULSE] Fallback to first PulseAudio device {idx}: {device['name']}")
                    return idx

            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"[PULSE] Could not query default source: {e}")
            return None

    def _notify_device_fallback(self, device_name: str):
        """Notify user that device fell back to system default"""
        try:
            import subprocess
            subprocess.run(
                ["notify-send", "-u", "normal", "hyprwhspr",
                 f"Configured microphone unavailable - using system default:\n{device_name}"],
                timeout=2, check=False, capture_output=True
            )
        except Exception:
            pass  # Best effort notification
    
    def is_available(self) -> bool:
        """Check if audio capture is available"""
        try:
            # Test if we can query devices
            sd.query_devices()
            return True
        except Exception:
            return False
    
    def start_recording(self, streaming_callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Start recording audio
        
        Args:
            streaming_callback: Optional callback function to receive audio chunks in real-time
                                (for streaming backends like WebSocket)
        """
        if not self.is_available():
            raise RuntimeError("Audio capture not available")
        
        if self.is_recording:
            return True
        
        # Validate device ID still exists (works for configured and system default)
        if self.device_id is not None:
            try:
                sd.query_devices(device=self.device_id, kind='input')
            except Exception:
                print(f"[INFO] Device ID {self.device_id} no longer available, re-initializing")
                self.preferred_device_id = None
                self.device_id = None
                self.device_info = None
                self._initialize_sounddevice()
                # Verify re-initialization succeeded
                if self.device_id is None:
                    print("[WARN] Re-initialization failed - no device available")
        
        # Safety: Clean up any leftover stream before starting
        if self.stream is not None:
            try:
                self._cleanup_stream()
            except Exception:
                pass  # Ignore cleanup errors
        
        try:
            # Clear previous audio data
            with self.lock:
                self.audio_data = []
                self.is_recording = True
                self.streaming_callback = streaming_callback
                # Reset callback health tracking
                self.frames_since_start = 0
                self.last_callback_monotonic = 0.0
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            # Ensure cleanup on failure
            try:
                self._cleanup_stream()
            except Exception:
                pass
            with self.lock:
                self.is_recording = False
            return False
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return the recorded audio data"""
        if not self.is_recording:
            return None
        
        # Signal to stop recording
        with self.lock:
            self.is_recording = False
        
        # Wait for recording thread to finish (it handles cleanup in finally block)
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=3.0)  # Increased from 2.0s to 3.0s

            # Check if thread actually exited
            if self.record_thread.is_alive():
                # Only warn if this is a normal stop (not during recovery)
                # During recovery, it's expected that the thread may not exit cleanly when device is dead
                if not (hasattr(self, 'recovery_in_progress') and self.recovery_in_progress):
                    print("[WARN] Recording thread did not exit cleanly after 3 seconds", flush=True)

        # Thread's finally block handles cleanup - verify it completed
        # Do NOT cleanup here to avoid deadlock (callback may still hold lock)
        with self.lock:
            if self.stream is not None:
                print("[WARN] Stream still exists after thread exit - this should not happen", flush=True)
                # Force clear the reference, but don't try to stop/close
                # (that would deadlock if callback thread is still waiting for lock)
                self.stream = None
        
        # Return recorded data
        with self.lock:
            if self.audio_data:
                try:
                    # Concatenate all audio chunks
                    audio_array = np.concatenate(self.audio_data, axis=0)
                    
                    # Ensure it's 1D (flatten if needed)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.flatten()
                    
                    # Ensure float32 dtype
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)
                    
                    # Ensure contiguous in memory
                    if not audio_array.flags['C_CONTIGUOUS']:
                        audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
                    
                    # Validate no NaN/inf
                    if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
                        print(f"[ERROR] Audio data contains invalid values (NaN/inf) - dropping", flush=True)
                        return None
                    
                    duration = len(audio_array) / self.sample_rate
                    if duration < 0.5:
                        print(f"[WARN] Recording very short ({duration:.2f}s), may not have captured audio", flush=True)
                    
                    return audio_array
                except Exception as e:
                    print(f"[ERROR] Failed to process audio data: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    return None
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
                    # Update callback health tracking (for recovery success criteria)
                    self.last_callback_monotonic = time.monotonic()
                    self.frames_since_start += 1
                    
                    if self.is_recording:
                        # Store the audio data (indata is already numpy array)
                        audio_chunk = indata[:, 0]  # Get mono channel
                        
                        # Update current audio level for monitoring
                        self.current_level = np.sqrt(np.mean(audio_chunk**2))
                        
                        # Store audio data
                        self.audio_data.append(audio_chunk.copy())
                        
                        # Call streaming callback if set (for realtime backends)
                        if self.streaming_callback:
                            try:
                                self.streaming_callback(audio_chunk.copy())
                            except Exception as e:
                                print(f"[WARN] Streaming callback error: {e}")
                        
                        chunk_count += 1
            
            # Determine device to use for recording (use validated device_id)
            device_to_use = self.device_id
            
            # Create and own the stream handle (for recovery)
            self.stream = sd.InputStream(
                device=device_to_use,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=audio_callback
            )
            
            # Start the stream explicitly
            self.stream.start()
            
                # Keep recording while is_recording is True
            try:
                while self.is_recording:
                    time.sleep(0.1)
            finally:
                # Clean up stream on exit (recording thread owns this cleanup)
                stream = None
                with self.lock:
                    stream = self.stream
                    if stream is not None:
                        self.stream = None  # Clear reference immediately
                
                # Clean up outside lock
                if stream is not None:
                    try:
                        stream.stop()
                    except (AttributeError, RuntimeError, Exception):
                        pass  # Stream might already be stopped
                    try:
                        stream.close()
                    except (AttributeError, RuntimeError, Exception):
                        pass  # Stream might already be closed
                    
        except Exception as e:
            # Always log the error message
            print(f"[ERROR] Error in recording thread: {e}", flush=True)

            # Only print traceback for unexpected errors (not common device/stream errors)
            # This reduces log noise during startup/recovery when device isn't ready yet
            error_msg = str(e).lower()
            if not ('device' in error_msg or 'stream' in error_msg or 'portaudio' in error_msg):
                # Unexpected error - print full traceback for debugging
                import traceback
                traceback.print_exc()
        finally:
            # Ensure stream is cleaned up even on exception during stream creation
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                with self.lock:
                    self.stream = None
    
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
        """Clean up the audio stream (idempotent - safe to call multiple times)"""
        stream = None
        with self.lock:
            stream = self.stream
            if stream is not None:
                self.stream = None  # Clear reference immediately to prevent double cleanup
        
        # Clean up stream outside lock to avoid deadlocks
        if stream is not None:
            try:
                stream.stop()
            except (AttributeError, RuntimeError, Exception):
                pass  # Stream might already be stopped or invalid
            try:
                stream.close()
            except (AttributeError, RuntimeError, Exception):
                pass  # Stream might already be closed or invalid
    
    def is_recovery_successful(self) -> bool:
        """
        Check if recovery was successful using objective callback-based criteria.

        Recovery is successful if:
        - At least 2 callbacks received (reduced from 3 for faster recovery)
        - last_callback_monotonic updated within 2.0s after recovery start (increased from 0.8s)

        More lenient criteria to handle:
        - Post-suspend CPU throttling
        - Slow audio driver reinitialization
        - USB device enumeration delays
        """
        if self.recovery_start_time == 0.0:
            return False

        now = time.monotonic()
        time_since_recovery = now - self.recovery_start_time

        # Read shared state with lock held to avoid data race
        with self.lock:
            frames_count = self.frames_since_start
            last_callback_time = self.last_callback_monotonic

        # Check timing: callback must have been received within 2.0s of recovery start
        # (increased from 0.8s to handle slow post-suspend recovery)
        if time_since_recovery > 2.0:
            # Too much time has passed, check if we got callbacks
            if frames_count >= 2 and last_callback_time > self.recovery_start_time:
                # Got callbacks and they were recent enough
                return True
            return False

        # Still within timeout window, check if we're getting callbacks
        # Need at least 2 callbacks (reduced from 3) and they must be recent
        if frames_count >= 2:
            # Check if last callback was recent (within last 0.5s)
            # (increased from 0.2s to handle CPU throttling)
            if now - last_callback_time < 0.5:
                return True

        return False
    
    def recover_audio_capture(self, reason: str, streaming_callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Recover audio capture by tearing down and rebuilding the stream.
        
        This is the single entry point for recovery. It:
        1. Hard stops current capture (Step A)
        2. Re-enumerates devices and rebinds defaults (Step B)
        3. Recreates capture (Step C)
        
        Args:
            reason: Reason for recovery (for logging)
            streaming_callback: Optional callback to restore after recovery
            
        Returns:
            True if recovery successful, False otherwise
        """
        # Check if recovery is already in progress (serialization)
        with self.recovery_lock:
            if self.recovery_in_progress:
                print(f"[RECOVERY] Recovery already in progress, skipping")
                return False
            
            # Set recovery in progress
            self.recovery_in_progress = True
        
        try:
            print(f"[RECOVERY] Starting audio capture recovery (reason: {reason})", flush=True)
            
            # Step A: Hard stop current capture
            print("[RECOVERY] Step A: Stopping current capture")
            with self.lock:
                was_recording = self.is_recording
                self.is_recording = False
            
            # Stop and close stream if it exists
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    print(f"[RECOVERY] Error stopping stream: {e}")
                finally:
                    self.stream = None
            
            # Join record thread with timeout
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=2.0)
                if self.record_thread.is_alive():
                    # Thread is stuck (likely in stream.stop() or stream.close() waiting for dead device)
                    # This is expected during recovery of a truly dead/unplugged device
                    print("[RECOVERY] Warning: Recording thread stuck in cleanup (device unresponsive)", flush=True)

                    # Try waiting longer before giving up
                    print("[RECOVERY] Waiting additional 3 seconds for thread cleanup...", flush=True)
                    self.record_thread.join(timeout=3.0)

                    if self.record_thread.is_alive():
                        # Still stuck after 5 total seconds - this is a zombie thread
                        # Abandon it and proceed with recovery anyway
                        # The zombie thread will eventually time out and die on its own
                        print("[RECOVERY] Thread still stuck - abandoning zombie thread and proceeding with recovery", flush=True)
                        print("[RECOVERY] Note: If issues persist, restart the service", flush=True)
                        self.record_thread = None  # Abandon reference to stuck thread
            
            # Reset tracking
            with self.lock:
                self.frames_since_start = 0
                self.last_callback_monotonic = 0.0
                self.audio_data = []
            
            # Step B: Re-enumerate devices and rebind defaults
            print("[RECOVERY] Step B: Re-enumerating devices and rebinding defaults", flush=True)
            try:
                # Force PortAudio refresh (lightest reset)
                sd.query_devices()
                
                # Re-run initialization to set sd.default.* and validate device
                self._initialize_sounddevice()
                
                # If preferred device ID is now invalid, existing logic already fell back
                print(f"[RECOVERY] Device re-initialized: {self.device_id}", flush=True)
            except Exception as e:
                print(f"[RECOVERY] ERROR: Failed to re-enumerate devices: {e}", flush=True)
                return False
            
            # Step C: Recreate capture
            print("[RECOVERY] Step C: Recreating capture")
            self.recovery_start_time = time.monotonic()
            
            # If we were recording, skip the test recording to avoid model conflicts
            # The caller will restart recording after recovery, so we just need to verify device works
            if was_recording:
                print("[RECOVERY] Was recording - skipping test recording to avoid model conflicts")
                print("[RECOVERY] Device re-initialized successfully, ready for recording restart")
                # Device enumeration already succeeded in Step B, so recovery is successful
                return True
            
            # Restore streaming callback if provided (only for test recording when not was_recording)
            if streaming_callback:
                self.streaming_callback = streaming_callback
            
            # Start recording again (spawns fresh thread with fresh stream)
            # This is needed to verify the device works, even if we weren't recording before
            try:
                if not self.start_recording(streaming_callback):
                    print("[RECOVERY] ERROR: Failed to start recording after recovery")
                    return False
            except Exception as e:
                print(f"[RECOVERY] ERROR: Exception starting recording: {e}")
                return False
            
            # Wait a bit for callbacks to start
            time.sleep(0.5)
            
            # Check if recovery was successful
            if self.is_recovery_successful():
                print("[RECOVERY] Recovery successful - callbacks received")
                # Stop the test recording (we just verified device works)
                with self.lock:
                    self.is_recording = False
                # Stop the stream
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
                # Join thread
                if self.record_thread and self.record_thread.is_alive():
                    self.record_thread.join(timeout=1.0)
                return True
            else:
                print("[RECOVERY] Recovery failed - please re-attach your mic usb")
                # Stop the failed recording attempt
                if self.is_recording:
                    with self.lock:
                        self.is_recording = False
                if self.stream:
                    try:
                        self.stream.stop()
                        self.stream.close()
                    except:
                        pass
                    self.stream = None
                return False
                
        except Exception as e:
            print(f"[RECOVERY] ERROR: Exception during recovery: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clear recovery in progress flag
            with self.recovery_lock:
                self.recovery_in_progress = False
    
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
