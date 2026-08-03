"""A pinned microphone must stay inside the PipeWire/PulseAudio graph (#234).

PortAudio lists raw ALSA entries ("... (hw:1,0)") ahead of the "pulse"
aggregate, so name matching used to pin the hardware and bypass everything the
sound server applies (EasyEffects, noise suppression, virtual sources).
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class FakeCompleted:
    def __init__(self, stdout, returncode=0):
        self.returncode = returncode
        self.stdout = stdout


class FakeStream:
    def __init__(self, owner, device, callback=None):
        self.owner = owner
        self.device = device
        self.callback = callback

    def start(self):
        self.owner.started_devices.append(self.device)

    def stop(self):
        pass

    def close(self):
        pass


class FakeSoundDevice(types.ModuleType):
    """Device list as PortAudio reports it on a PipeWire box: raw ALSA first."""

    def __init__(self, devices=None):
        super().__init__("sounddevice")
        self.default = types.SimpleNamespace(
            samplerate=None,
            channels=None,
            dtype=None,
            device=[None, None],
        )
        self.devices = devices if devices is not None else [
            {
                "name": "HDA Intel PCH: ALC257 Analog (hw:0,0)",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Elgato Wave XLR: USB Audio (hw:1,0)",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "pulse",
                "max_input_channels": 32,
                "default_samplerate": 44100,
                "hostapi": 0,
            },
        ]
        self.started_devices = []

    def query_devices(self, device=None, kind=None):
        if device is None:
            return list(self.devices)
        return self.devices[device]

    def query_hostapis(self, hostapi):
        return {"name": "ALSA"}

    def InputStream(self, device=None, samplerate=None, channels=None,
                    dtype=None, blocksize=None, callback=None):
        return FakeStream(self, device, callback=callback)


class FakeNumpy(types.ModuleType):
    def __init__(self):
        super().__init__("numpy")
        self.float32 = "float32"
        self.ndarray = object


ELGATO_SOURCE = "alsa_input.usb-Elgato_Systems_Elgato_Wave_XLR_ABC123-00.analog-stereo"
BUILTIN_SOURCE = "alsa_input.pci-0000_00_1f.3.analog-stereo"


class SoundServerRoutingTests(unittest.TestCase):
    def _load_audio_capture(self, fake_sd):
        self._saved_sounddevice = sys.modules.get("sounddevice")
        self._saved_numpy = sys.modules.get("numpy")
        sys.modules["sounddevice"] = fake_sd
        sys.modules["numpy"] = FakeNumpy()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))
        import audio_capture

        self._saved_audio_capture = audio_capture
        return importlib.reload(audio_capture)

    def setUp(self):
        self._saved_sounddevice = None
        self._saved_numpy = None
        self._saved_audio_capture = None
        self._saved_pulse_source = os.environ.pop("PULSE_SOURCE", None)
        self.addCleanup(self._restore_env)
        self.addCleanup(self._restore_modules)

    def _restore_env(self):
        os.environ.pop("PULSE_SOURCE", None)
        if self._saved_pulse_source is not None:
            os.environ["PULSE_SOURCE"] = self._saved_pulse_source

    def _restore_modules(self):
        if self._saved_sounddevice is not None:
            sys.modules["sounddevice"] = self._saved_sounddevice
        else:
            sys.modules.pop("sounddevice", None)
        if self._saved_numpy is not None:
            sys.modules["numpy"] = self._saved_numpy
        else:
            sys.modules.pop("numpy", None)
        if self._saved_audio_capture is not None and self._saved_sounddevice is not None:
            importlib.reload(self._saved_audio_capture)

    def _fake_pactl(self, sources=(ELGATO_SOURCE, BUILTIN_SOURCE),
                    default=BUILTIN_SOURCE, missing=False):
        """Stand in for pactl, dispatching on argv."""
        def run(args, *rest, **kwargs):
            if missing:
                raise FileNotFoundError("pactl")
            if "list" in args:
                lines = [
                    f"{idx}\t{name}\tmodule-alsa-card\ts16le 2ch 48000Hz\tSUSPENDED"
                    for idx, name in enumerate(sources)
                ]
                # Monitors appear in this list and must never be selected.
                lines.append("99\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\t"
                             "module-alsa-card\ts16le 2ch 48000Hz\tSUSPENDED")
                return FakeCompleted("\n".join(lines) + "\n")
            return FakeCompleted(f"{default}\n")

        return run

    def _build(self, module, config, pactl=None):
        with mock.patch("subprocess.run", side_effect=pactl or self._fake_pactl()):
            return module.AudioCapture(
                device_id=config.get_setting("audio_device_id"),
                config_manager=config,
            )

    # --- the #234 repro -------------------------------------------------

    def test_configured_name_routes_through_pulse_not_raw_alsa(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))

        self.assertEqual(capture.device_id, 2, "should bind the pulse aggregate, not hw:1,0")
        self.assertEqual(os.environ.get("PULSE_SOURCE"), ELGATO_SOURCE)

    def test_configured_source_name_via_device_id_routes_through_pulse(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_id": ELGATO_SOURCE}))

        self.assertEqual(capture.device_id, 2)
        self.assertEqual(os.environ.get("PULSE_SOURCE"), ELGATO_SOURCE)

    def test_virtual_source_with_no_portaudio_entry_resolves(self):
        """EasyEffects' source has no PortAudio device; it used to match nothing."""
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        pactl = self._fake_pactl(sources=(ELGATO_SOURCE, "easyeffects_source"))

        capture = self._build(
            module, FakeConfig({"audio_device_name": "easyeffects_source"}), pactl=pactl
        )

        self.assertEqual(capture.device_id, 2)
        self.assertEqual(os.environ.get("PULSE_SOURCE"), "easyeffects_source")

    # --- opting out and failing soft ------------------------------------

    def test_hw_prefix_opts_into_raw_alsa(self):
        """"hw:1,0" is a substring of the PortAudio name, so legacy matching finds it."""
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "hw:1,0"}))

        self.assertEqual(capture.device_id, 1)
        self.assertEqual(fake_sd.devices[1]["name"], "Elgato Wave XLR: USB Audio (hw:1,0)")
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_without_pulse_device_falls_back_to_raw_alsa_with_warning(self):
        fake_sd = FakeSoundDevice(devices=[
            {
                "name": "Elgato Wave XLR: USB Audio (hw:1,0)",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ])
        module = self._load_audio_capture(fake_sd)

        with mock.patch("os.path.exists", return_value=True), \
                mock.patch("builtins.print") as fake_print:
            capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))

        self.assertEqual(capture.device_id, 0)
        self.assertNotIn("PULSE_SOURCE", os.environ)
        printed = " ".join(str(call.args[0]) for call in fake_print.call_args_list if call.args)
        self.assertIn("raw ALSA hardware", printed)

    def test_no_warning_without_a_sound_server(self):
        """Pure ALSA: raw capture is correct and the only option, so stay quiet."""
        fake_sd = FakeSoundDevice(devices=[
            {
                "name": "Elgato Wave XLR: USB Audio (hw:1,0)",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ])
        module = self._load_audio_capture(fake_sd)

        with mock.patch("os.path.exists", return_value=False), \
                mock.patch("builtins.print") as fake_print:
            capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))

        self.assertEqual(capture.device_id, 0)
        printed = " ".join(str(call.args[0]) for call in fake_print.call_args_list if call.args)
        self.assertNotIn("raw ALSA hardware", printed)

    def test_hardware_named_pulse_is_not_mistaken_for_the_aggregate(self):
        """A Sony Pulse 3D headset is raw ALSA and ignores PULSE_SOURCE."""
        fake_sd = FakeSoundDevice(devices=[
            {
                "name": "Pulse 3D Wireless Headset: USB Audio (hw:2,0)",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Elgato Wave XLR: USB Audio (hw:1,0)",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ])
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))

        self.assertEqual(capture.device_id, 1, "must not bind the headset")
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_missing_pactl_keeps_legacy_matching(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(
            module, FakeConfig({"audio_device_name": "Elgato"}),
            pactl=self._fake_pactl(missing=True),
        )

        self.assertEqual(capture.device_id, 1, "no sound server: legacy hw: match stands")
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_ambiguous_sources_do_not_guess(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        pactl = self._fake_pactl(
            sources=(
                "alsa_input.usb-Elgato_Wave_XLR_ABC123-00.analog-stereo",
                "alsa_input.usb-Elgato_Wave_XLR_ABC123-00.iec958-stereo",
            ),
            default=BUILTIN_SOURCE,  # neither candidate; no tie-break available
        )

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}), pactl=pactl)

        self.assertEqual(capture.device_id, 1, "ambiguous: fall through to legacy matching")
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_ambiguous_sources_broken_by_server_default(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        chosen = "alsa_input.usb-Elgato_Wave_XLR_ABC123-00.iec958-stereo"
        pactl = self._fake_pactl(
            sources=("alsa_input.usb-Elgato_Wave_XLR_ABC123-00.analog-stereo", chosen),
            default=chosen,
        )

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}), pactl=pactl)

        self.assertEqual(capture.device_id, 2)
        self.assertEqual(os.environ.get("PULSE_SOURCE"), chosen)

    def test_monitor_sources_are_never_selected(self):
        """A monitor mirrors an output; pinning capture to one is never right."""
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        pactl = self._fake_pactl(sources=(ELGATO_SOURCE,))

        with mock.patch("subprocess.run", side_effect=pactl):
            listed = module.AudioCapture._list_pulse_sources()
            self.assertNotIn(monitor, listed)

        self._build(module, FakeConfig({"audio_device_name": monitor}), pactl=pactl)
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_pin_survives_sound_server_restart(self):
        """Recovery re-initialises while the server restarts and pactl is mute.

        Falling through to legacy matching there would rebind the raw ALSA
        device the pin exists to avoid — and hold the card exclusively.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))
        self.assertEqual(capture.device_id, 2)

        with mock.patch("subprocess.run", side_effect=self._fake_pactl(missing=True)):
            capture._initialize_sounddevice()

        self.assertEqual(capture.device_id, 2, "must not degrade to hw:1,0")
        self.assertEqual(os.environ.get("PULSE_SOURCE"), ELGATO_SOURCE)

    def test_unplugged_mic_still_falls_back(self):
        """A reachable server that no longer lists the mic is a real loss."""
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))
        self.assertEqual(os.environ.get("PULSE_SOURCE"), ELGATO_SOURCE)

        with mock.patch("subprocess.run", side_effect=self._fake_pactl(sources=(BUILTIN_SOURCE,))):
            capture._initialize_sounddevice()

        self.assertNotIn("PULSE_SOURCE", os.environ,
                         "gone is gone; don't pin a source the server no longer has")

    # --- #195 regression guards -----------------------------------------

    def test_integer_device_id_passes_through_untouched(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_id": 1}))

        self.assertEqual(capture.device_id, 1)
        self.assertNotIn("PULSE_SOURCE", os.environ)

    def test_stale_pulse_source_cleared_on_reresolve(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        capture = self._build(module, FakeConfig({"audio_device_name": "Elgato"}))
        self.assertEqual(os.environ.get("PULSE_SOURCE"), ELGATO_SOURCE)

        # Re-resolve against a config that no longer routes through the server.
        capture.config = FakeConfig({"audio_device_name": "hw:1,0"})
        fake_sd.devices[1]["name"] = "hw:1,0"
        with mock.patch("subprocess.run", side_effect=self._fake_pactl()):
            capture._initialize_sounddevice()

        self.assertNotIn("PULSE_SOURCE", os.environ)


if __name__ == "__main__":
    unittest.main()
