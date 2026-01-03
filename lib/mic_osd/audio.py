"""
Audio monitoring module for mic-osd.

Captures microphone input in real-time and provides level/sample data
for visualization.
"""

import threading
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioMonitor:
    """
    Real-time microphone audio monitor.
    
    Uses sounddevice to capture audio from the default microphone
    and provides peak levels and raw samples for visualization.
    """
    
    def __init__(self, callback=None, samplerate=44100, blocksize=1024):
        """
        Initialize the audio monitor.
        
        Args:
            callback: Function called with (peak_level, samples) on each audio block
            samplerate: Audio sample rate in Hz
            blocksize: Number of samples per callback
        """
        if sd is None:
            raise ImportError("sounddevice is required for audio monitoring")
        
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stream = None
        self.running = False
        
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.samples = np.zeros(blocksize)
        
        self._lock = threading.Lock()
    
    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block."""
        
        # Get mono samples
        samples = indata[:, 0].copy()
        
        # Calculate levels
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2)))
        
        with self._lock:
            self.peak_level = peak
            self.rms_level = rms
            self.samples = samples
        
        # Call user callback
        if self.callback:
            self.callback(peak, samples)
    
    def get_default_device(self):
        """Get info about the default input device."""
        try:
            return sd.query_devices(kind='input')
        except Exception as e:
            return {"name": f"Error: {e}"}
    
    def list_devices(self):
        """List all available audio input devices."""
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'samplerate': dev['default_samplerate']
                })
        return devices
    
    def start(self, device=None):
        """
        Start monitoring the microphone.
        
        Args:
            device: Device index or name (None = default)
        """
        if self.running:
            return
        
        try:
            self.stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
        except Exception as e:
            raise RuntimeError(f"Failed to start audio monitoring: {e}")
    
    def stop(self):
        """Stop monitoring."""
        if not self.running:
            return
        
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
        
        # Reset levels
        with self._lock:
            self.peak_level = 0.0
            self.rms_level = 0.0
            self.samples = np.zeros(self.blocksize)
    
    def get_level(self):
        """Get current peak level (thread-safe)."""
        with self._lock:
            return self.peak_level
    
    def get_samples(self):
        """Get current samples (thread-safe)."""
        with self._lock:
            return self.samples.copy()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
