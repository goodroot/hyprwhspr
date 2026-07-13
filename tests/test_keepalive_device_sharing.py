"""Keepalive must never open a raw ALSA hw device.

A raw `hw:N,M` device is exclusive: holding it open between recordings blocks
every other app from the mic. `_is_multiplexed_audio_server` gates the
keepalive on device shareability, but a system-wide PipeWire/Pulse socket must
not mark a raw-ALSA device as shareable — the explicit device-selection paths
all resolve to such devices. (Found while fixing issue #205.)
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import audio_capture


class MultiplexedServerDetectionTests(unittest.TestCase):
    def _capture_with_device(self, device_name, host_api_name):
        capture = object.__new__(audio_capture.AudioCapture)
        capture.device_id = 0
        fake_sd = types.SimpleNamespace(
            query_devices=lambda device_id: {
                "name": device_name,
                "hostapi": 0,
            },
            query_hostapis=lambda hostapi: {"name": host_api_name},
        )
        return capture, fake_sd

    def test_raw_alsa_hw_device_is_not_shareable_even_with_pulse_socket(self):
        capture, fake_sd = self._capture_with_device(
            "Logitech StreamCam: USB Audio (hw:3,0)", "ALSA"
        )
        with mock.patch.object(audio_capture, "sd", fake_sd), \
                mock.patch("os.path.exists", return_value=True):
            # A Pulse socket exists system-wide, but the bound device is raw hw:
            self.assertFalse(capture._is_multiplexed_audio_server())

    def test_alsa_default_virtual_device_is_shareable_when_socket_present(self):
        capture, fake_sd = self._capture_with_device("default", "ALSA")
        with mock.patch.object(audio_capture, "sd", fake_sd), \
                mock.patch("os.path.exists", return_value=True):
            self.assertTrue(capture._is_multiplexed_audio_server())

    def test_alsa_default_virtual_device_not_shareable_without_socket(self):
        capture, fake_sd = self._capture_with_device("default", "ALSA")
        with mock.patch.object(audio_capture, "sd", fake_sd), \
                mock.patch("os.path.exists", return_value=False):
            self.assertFalse(capture._is_multiplexed_audio_server())

    def test_pipewire_named_device_is_shareable(self):
        capture, fake_sd = self._capture_with_device("pipewire", "ALSA")
        with mock.patch.object(audio_capture, "sd", fake_sd), \
                mock.patch("os.path.exists", return_value=False):
            self.assertTrue(capture._is_multiplexed_audio_server())

    def test_pulse_host_api_is_shareable(self):
        capture, fake_sd = self._capture_with_device(
            "Some Mic (hw:3,0)", "PulseAudio"
        )
        with mock.patch.object(audio_capture, "sd", fake_sd), \
                mock.patch("os.path.exists", return_value=False):
            # Routed through the Pulse host API despite the hw-style name.
            self.assertTrue(capture._is_multiplexed_audio_server())

    def test_no_device_is_not_shareable(self):
        capture = object.__new__(audio_capture.AudioCapture)
        capture.device_id = None
        self.assertFalse(capture._is_multiplexed_audio_server())


if __name__ == "__main__":
    unittest.main()
