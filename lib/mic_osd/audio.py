"""
Audio monitoring module for mic-osd.

Provides level/sample data for visualization from one of two sources:
- FeedLevelSource: reads frames the main hyprwhspr process streams from its
  own capture stream, so the meter tracks the device actually being recorded.
- AudioMonitor: opens a system-default input stream directly (fallback for
  standalone use, or when no feed is being written).
"""

import os
import time
import threading
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


class FeedLevelSource:
    """Reads the runtime feed file MicOSDRunner writes (space-separated floats:
    level then bucket RMS values). Same get_level()/get_samples() surface as
    AudioMonitor, but opens no audio stream of its own."""

    STALE_AFTER_SECONDS = 1.0

    def __init__(self, path):
        self.path = str(path)
        self._mtime_ns = None
        self._level = 0.0
        self._samples = np.zeros(0)

    @classmethod
    def available(cls, path, max_age=STALE_AFTER_SECONDS):
        """True when a feed file exists and was written recently."""
        try:
            return (time.time() - os.stat(str(path)).st_mtime) <= max_age
        except OSError:
            return False

    def start(self, device=None):
        pass

    def stop(self):
        self._mtime_ns = None
        self._level = 0.0
        self._samples = np.zeros(0)

    def _refresh(self):
        try:
            stat = os.stat(self.path)
        except OSError:
            # Feed gone — decay to silence
            self._level = 0.0
            self._samples = np.zeros(0)
            return
        if time.time() - stat.st_mtime > self.STALE_AFTER_SECONDS:
            self._level = 0.0
            self._samples = np.zeros(0)
            return
        if stat.st_mtime_ns == self._mtime_ns:
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                values = [float(part) for part in f.read().split()]
        except (OSError, ValueError):
            return  # Keep last good frame
        if not values:
            return
        self._mtime_ns = stat.st_mtime_ns
        self._level = values[0]
        self._samples = np.array(values[1:], dtype=np.float64)

    def get_level(self):
        self._refresh()
        return self._level

    def get_samples(self):
        self._refresh()
        return self._samples


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

        # Use the device's native rate; a hardcoded 44100 fails with
        # PaErrorCode -9997 on 48 kHz-only hardware (issue #205).
        samplerate = self.samplerate
        try:
            default_sr = sd.query_devices(device, kind='input').get('default_samplerate')
            if default_sr:
                samplerate = int(default_sr)
        except Exception:
            pass

        try:
            self.stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=samplerate,
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
