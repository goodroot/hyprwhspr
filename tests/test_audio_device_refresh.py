import importlib
import subprocess
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
    def __init__(self, source):
        self.returncode = 0
        self.stdout = f"{source}\n"


class FakeStream:
    def __init__(self, owner, device, callback=None):
        self.owner = owner
        self.device = device
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.owner.started_devices.append(self.device)
        self.started = True
        if self.owner.fail_once_for_device == self.device:
            self.owner.fail_once_for_device = None
            raise RuntimeError("Unanticipated host error")
        if self.owner.on_stream_start:
            self.owner.on_stream_start()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeSoundDevice(types.ModuleType):
    NO_FAIL_DEVICE = object()

    def __init__(self):
        super().__init__("sounddevice")
        self.default = types.SimpleNamespace(
            samplerate=None,
            channels=None,
            dtype=None,
            device=[None, None],
        )
        self.devices = [
            {
                "name": "alsa_input.old mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "alsa_input.new mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Preferred USB Mic",
                "max_input_channels": 1,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
            {
                "name": "Output Only",
                "max_input_channels": 0,
                "default_samplerate": 48000,
                "hostapi": 0,
            },
        ]
        self.streams = []
        self.started_devices = []
        self.opened_samplerates = []
        self.fail_once_for_device = self.NO_FAIL_DEVICE
        self.on_stream_start = None

    def query_devices(self, device=None, kind=None):
        if device is None:
            return list(self.devices)
        return self.devices[device]

    def query_hostapis(self, hostapi):
        return {"name": "PulseAudio"}

    def InputStream(self, device=None, samplerate=None, channels=None,
                    dtype=None, blocksize=None, callback=None):
        self.opened_samplerates.append(samplerate)
        stream = FakeStream(self, device, callback=callback)
        self.streams.append(stream)
        return stream


class FakeNumpy(types.ModuleType):
    def __init__(self):
        super().__init__("numpy")
        self.float32 = "float32"
        self.ndarray = object

    def concatenate(self, arrays, axis=0):
        result = []
        for array in arrays:
            result.extend(array)
        return result

    def sqrt(self, value):
        return value ** 0.5

    def mean(self, values):
        return sum(values) / len(values)

    def any(self, values):
        return any(values)

    def isnan(self, values):
        return [False for _ in values]

    def isinf(self, values):
        return [False for _ in values]

    def ascontiguousarray(self, values, dtype=None):
        return values


class FakeStreamingCallback:
    def __init__(self):
        self.input_sample_rate = None

    def __call__(self, audio_chunk):
        pass

    def set_input_sample_rate(self, sample_rate):
        self.input_sample_rate = sample_rate


class AudioDeviceRefreshTests(unittest.TestCase):
    def _load_audio_capture(self, fake_sd):
        self._saved_audio_capture = sys.modules.get("audio_capture")
        self._saved_sounddevice = sys.modules.get("sounddevice")
        self._saved_numpy = sys.modules.get("numpy")
        sys.modules["sounddevice"] = fake_sd
        sys.modules["numpy"] = FakeNumpy()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))
        import audio_capture
        return importlib.reload(audio_capture)

    def tearDown(self):
        if hasattr(self, "_saved_sounddevice"):
            if self._saved_sounddevice is None:
                sys.modules.pop("sounddevice", None)
            else:
                sys.modules["sounddevice"] = self._saved_sounddevice
        if hasattr(self, "_saved_numpy"):
            if self._saved_numpy is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = self._saved_numpy
        if hasattr(self, "_saved_audio_capture"):
            if self._saved_audio_capture is None:
                sys.modules.pop("audio_capture", None)
            else:
                sys.modules["audio_capture"] = self._saved_audio_capture
                # _load_audio_capture reloaded this module in place against the
                # fake sounddevice/numpy, rebinding its module-level `sd`/`np`.
                # Restoring the sys.modules reference alone doesn't undo that,
                # so reload once more against the now-restored real modules to
                # avoid leaking the fakes into other tests' imports.
                if self._saved_sounddevice is not None and self._saved_numpy is not None:
                    importlib.reload(self._saved_audio_capture)

    def _run_sources(self, sources, listed=()):
        """Fake `pactl`, one default-source answer per get-default-source call.

        Dispatches on argv: `list short sources` answers from `listed` (empty by
        default, so device selection skips sound-server routing) while
        `get-default-source` walks `sources` in order.
        """
        source_iter = iter(sources)

        def run(args, *rest, **kwargs):
            if 'list' in args:
                return FakeCompleted('\n'.join(
                    f"{idx}\t{name}\tmodule\ts16le 2ch 48000Hz\tRUNNING"
                    for idx, name in enumerate(listed)
                ))
            return FakeCompleted(next(source_iter))

        return run

    def test_system_default_init_does_not_make_default_preferred(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        self.assertIsNone(capture.preferred_device_id)
        self.assertEqual(capture.device_id, 0)

    def test_init_uses_resolved_device_default_sample_rate(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.sample_rate, 48000)
        self.assertEqual(fake_sd.default.samplerate, 48000)
        self.assertIn(48000, fake_sd.opened_samplerates)

    def test_streaming_callback_gets_refreshed_record_start_sample_rate(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices[1]["default_samplerate"] = 44100
        module = self._load_audio_capture(fake_sd)
        callback = FakeStreamingCallback()

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording(streaming_callback=callback))
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.device_id, 1)
        self.assertEqual(capture.sample_rate, 44100)
        self.assertEqual(callback.input_sample_rate, 44100)
        self.assertIn(44100, fake_sd.opened_samplerates)

    def test_record_start_reresolves_default_even_when_old_device_exists(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.device_id, 1)
        self.assertIn(1, fake_sd.started_devices)
        self.assertIsNone(capture.preferred_device_id)

    def test_explicit_device_id_does_not_follow_default_change(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.new mic"])):
            capture = module.AudioCapture(device_id=0, config_manager=FakeConfig({"audio_device_id": 0, "stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(capture.device_id, 0)
        self.assertIn(0, fake_sd.started_devices)

    def test_configured_name_fallback_does_not_become_preferred(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"audio_device_name": "Missing Mic"}))

        self.assertEqual(capture.device_id, 0)
        self.assertIsNone(capture.preferred_device_id)
        with mock.patch.object(capture, "_get_pulse_default_source_device_id") as get_default:
            self.assertFalse(capture.refresh_default_input("test"))
        get_default.assert_not_called()

    def test_keepalive_moves_to_refreshed_default_device(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        config = FakeConfig({"keepalive_stream": True})

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=config)
            old_stream = capture._keepalive_stream
            self.assertEqual(old_stream.device, 0)
            self.assertTrue(capture.refresh_default_input("test"))

        self.assertTrue(old_stream.stopped)
        self.assertTrue(old_stream.closed)
        self.assertEqual(capture._keepalive_stream.device, 1)

    def test_keepalive_cycles_when_pulse_source_changes_on_same_portaudio_device(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "pulse",
            "max_input_channels": 32,
            "default_samplerate": 44100,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)
        config = FakeConfig({"keepalive_stream": True})

        with mock.patch("subprocess.run", side_effect=self._run_sources([
            "alsa_input.pci-0000_00_1f.3.analog-stereo",
            "bluez_input.68:F2:1F:03:3F:C9",
        ])):
            capture = module.AudioCapture(config_manager=config)
            old_stream = capture._keepalive_stream
            self.assertEqual(capture.device_id, 4)
            self.assertTrue(capture.refresh_default_input("test"))

        self.assertTrue(old_stream.stopped)
        self.assertTrue(old_stream.closed)
        self.assertEqual(capture.device_id, 4)
        self.assertEqual(capture._keepalive_stream.device, 4)

    def test_retriable_open_failure_reresolves_default_before_retry(self):
        fake_sd = FakeSoundDevice()
        fake_sd.fail_once_for_device = 0
        fake_sd.devices[1]["default_samplerate"] = 44100
        module = self._load_audio_capture(fake_sd)
        callback = FakeStreamingCallback()

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.old mic", "alsa_input.new mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording(streaming_callback=callback))
            capture.record_thread.join(timeout=1.0)

        self.assertEqual(fake_sd.started_devices[:2], [0, 1])
        self.assertEqual(capture.device_id, 1)
        self.assertEqual(callback.input_sample_rate, 44100)

    def test_stream_open_failure_records_error_state(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        def raise_host_error():
            raise RuntimeError("Unanticipated host error")

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"] * 8)):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = raise_host_error
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

            self.assertFalse(capture.stream_opened)
            self.assertIn("Unanticipated host error", capture.stream_open_error)

            # A later successful start must reset the recorded outcome
            capture.is_recording = False
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertTrue(capture.stream_opened)
        self.assertIsNone(capture.stream_open_error)

    def test_stream_open_success_sets_opened_flag(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertTrue(capture.stream_opened)
        self.assertIsNone(capture.stream_open_error)

    def test_unmatched_pulse_default_clears_stale_concrete_device(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic", "bluez_input.68:F2:1F:03:3F:C9"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertEqual(capture.device_id, 0)
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=1.0)

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)
        self.assertEqual(fake_sd.default.device[0], None)
        self.assertIn(None, fake_sd.started_devices)

    def test_output_monitor_default_is_rejected_at_startup(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "pulse",
            "max_input_channels": 32,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"

        # The mock stays active across start_recording: the entry guard re-polls
        # the default rather than trusting the cached rejection, so the monitor has
        # to still be the default for the refusal to stand.
        with mock.patch("subprocess.run", side_effect=self._run_sources([monitor] * 4)):
            capture = module.AudioCapture(config_manager=FakeConfig())

            self.assertIsNone(capture.device_id)
            self.assertIsNone(fake_sd.default.device[0])
            self.assertIn("output monitor", capture.get_input_selection_error())
            with self.assertRaisesRegex(RuntimeError, "output monitor"):
                capture.start_recording()

        self.assertEqual(capture.stream_open_error, capture.get_input_selection_error())

    def test_monitor_discovered_at_record_start_opens_no_stream(self):
        """The default can flip to a monitor after start_recording's entry guard.

        The record-start refresh then rejects it and clears the binding, which
        would leave device=None — and PortAudio resolves that straight back to
        the monitor. Nothing may be opened.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"

        with mock.patch("subprocess.run", side_effect=self._run_sources([
            "alsa_input.old mic",  # startup: a real microphone, so the guard passes
            monitor,               # record start: default moved to the monitor
        ])):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertEqual(capture.device_id, 0)
            self.assertIsNone(capture.get_input_selection_error())

            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertEqual(fake_sd.started_devices, [])
        self.assertFalse(capture.stream_opened)
        self.assertIn("output monitor", capture.stream_open_error)
        self.assertIsNone(capture.device_id)

    def test_input_selection_retry_does_not_run_while_recording(self):
        """The retry re-enters PortAudio, so it must sit behind the guards.

        `_initialize_sounddevice` rebinds the device and opens a keepalive stream;
        running it ahead of the `is_recording` early return would do that on top of
        a live recording stream, and ahead of the recovery wait would race
        `_reset_portaudio_state()` (#209).
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"] * 4)):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))

            capture.is_recording = True
            capture._input_selection_error = "stale rejection"
            with mock.patch.object(capture, "_retry_input_selection") as retry:
                self.assertTrue(capture.start_recording())

        retry.assert_not_called()

    def test_monitor_discovered_during_open_retry_opens_no_stream(self):
        """The retry loop's refresh can also surface the monitor for the first time.

        A transient stream failure re-resolves the default; if that now points at
        a monitor the binding is cleared, and retrying with device=None hands
        PortAudio the monitor.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"

        with mock.patch("subprocess.run", side_effect=self._run_sources(
            ["alsa_input.old mic"] * 2 + [monitor] * 4
        )):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertEqual(capture.device_id, 0)

            # First open fails transiently; the refresh that follows finds a monitor.
            fake_sd.fail_once_for_device = 0
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertEqual(fake_sd.started_devices, [0])  # no second, monitor-backed open
        self.assertFalse(capture.stream_opened)
        self.assertIn("output monitor", capture.stream_open_error)

    def test_monitor_rejection_clears_once_a_real_default_returns(self):
        """The rejection must not latch for the rest of the process.

        Nothing else re-tests it on the record path, so a user who fixes their
        default input in sound settings would otherwise be refused until they
        restarted the service.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"

        with mock.patch("subprocess.run", side_effect=self._run_sources(
            [monitor] + ["alsa_input.new mic"] * 8
        )):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertIsNotNone(capture.get_input_selection_error())

            # User picks a real microphone; the next record attempt must re-poll.
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertIsNone(capture.get_input_selection_error())
        self.assertTrue(capture.stream_opened)
        self.assertEqual(capture.device_id, 1)

    def test_configured_device_recovers_after_monitor_rejection(self):
        """A configured mic that was absent at startup must be retried.

        `_refresh_default_input_unlocked` returns early whenever a device is
        configured, so without an explicit retry the rejection recorded during
        startup fallback could never clear — and its advice ("set
        audio_device_name") is exactly what the user has already done.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"
        config = FakeConfig({
            "audio_device_name": "Preferred USB Mic",
            "stream_start_retry_delay": 0,
        })

        absent = [d for d in fake_sd.devices if d["name"] != "Preferred USB Mic"]
        present = list(fake_sd.devices)
        fake_sd.devices = absent

        with mock.patch("subprocess.run", side_effect=self._run_sources([monitor] * 8)):
            capture = module.AudioCapture(config_manager=config)
            self.assertIsNotNone(capture.get_input_selection_error())
            with self.assertRaisesRegex(RuntimeError, "output monitor"):
                capture.start_recording()

            # The mic is plugged back in.
            fake_sd.devices = present
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertIsNone(capture.get_input_selection_error())
        self.assertTrue(capture.stream_opened)

    def test_transient_pactl_failure_does_not_latch_the_rejection(self):
        """A failed poll leaves the last verdict standing, but not permanently.

        We cannot tell what the default is when `pactl` times out, so refusing is
        right — but the next attempt must re-poll rather than trust the cache.
        """
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"
        answers = iter([monitor, TimeoutError, "alsa_input.new mic", "alsa_input.new mic",
                        "alsa_input.new mic", "alsa_input.new mic"])

        def run(args, *rest, **kwargs):
            if 'list' in args:
                return FakeCompleted("")
            answer = next(answers)
            if answer is TimeoutError:
                raise subprocess.TimeoutExpired(cmd=args, timeout=2)
            return FakeCompleted(answer)

        with mock.patch("subprocess.run", side_effect=run):
            capture = module.AudioCapture(config_manager=FakeConfig({"stream_start_retry_delay": 0}))
            self.assertIsNotNone(capture.get_input_selection_error())

            # Poll fails: still refused, and the reason is unchanged.
            with self.assertRaisesRegex(RuntimeError, "output monitor"):
                capture.start_recording()

            # Next poll succeeds and finds a real source.
            fake_sd.on_stream_start = lambda: setattr(capture, "is_recording", False)
            self.assertTrue(capture.start_recording())
            capture.record_thread.join(timeout=2.0)

        self.assertIsNone(capture.get_input_selection_error())
        self.assertTrue(capture.stream_opened)

    def test_output_monitor_transition_blocks_until_real_source_returns(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)
        monitor = "bluez_output.88_0E_85_0F_49_80.1.monitor"

        with mock.patch("subprocess.run", side_effect=self._run_sources([
            "alsa_input.old mic",
            monitor,
            "alsa_input.new mic",
        ])):
            capture = module.AudioCapture(config_manager=FakeConfig())
            self.assertEqual(capture.device_id, 0)

            self.assertTrue(capture.refresh_default_input("monitor_selected"))
            self.assertIsNone(capture.device_id)
            self.assertIsNone(fake_sd.default.device[0])
            self.assertIn("output monitor", capture.get_input_selection_error())

            self.assertTrue(capture.refresh_default_input("microphone_selected"))

        self.assertEqual(capture.device_id, 1)
        self.assertIsNone(capture.get_input_selection_error())
        self.assertEqual(fake_sd.default.device[0], 1)

    def test_refresh_default_input_skips_during_recovery(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        capture.recovery_in_progress = True
        with mock.patch.object(capture, "_get_pulse_default_source_device_id") as get_default:
            self.assertFalse(capture.refresh_default_input("test"))

        get_default.assert_not_called()

    def test_sounddevice_fallback_rejects_output_only_default(self):
        fake_sd = FakeSoundDevice()
        fake_sd.default.device = [3, None]
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=1, stdout="")):
            capture = module.AudioCapture(config_manager=FakeConfig())

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)

    def test_pulse_default_rejects_output_only_match(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        capture.device_id = None
        capture.device_info = None
        fake_sd.default.device = [3, None]
        with (
            mock.patch.object(capture, "_get_pulse_default_source_device_id", return_value=3),
            mock.patch.object(capture, "_set_system_default_device"),
        ):
            self.assertFalse(capture.refresh_default_input("test"))

        self.assertIsNone(capture.device_id)
        self.assertIsNone(capture.device_info)


    def test_configured_pulse_source_name_resolves_to_portaudio_device(self):
        fake_sd = FakeSoundDevice()
        # PortAudio exposes the USB webcam under its ALSA-style display name,
        # which differs from the pactl source name the user configures.
        fake_sd.devices.append({
            "name": "C922 Pro Stream Webcam: USB Audio (hw:1,0)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)

        source = "alsa_input.usb-046d_C922_Pro_Stream_Webcam_D487FE4F-02.iec958-stereo"
        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(
                device_id=source,
                config_manager=FakeConfig({"audio_device_id": source}),
            )

        self.assertEqual(capture.device_id, 4)
        self.assertEqual(capture.preferred_device_id, source)
        self.assertEqual(fake_sd.default.device[0], 4)

    def test_configured_pulse_source_does_not_follow_default_change(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "C922 Pro Stream Webcam: USB Audio (hw:1,0)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)

        source = "alsa_input.usb-046d_C922_Pro_Stream_Webcam_D487FE4F-02.iec958-stereo"
        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(
                device_id=source,
                config_manager=FakeConfig({"audio_device_id": source, "stream_start_retry_delay": 0}),
            )
            self.assertEqual(capture.device_id, 4)
            # Explicit intent must not be overridden by a default-source refresh.
            with mock.patch.object(capture, "_get_pulse_default_source_device_id") as get_default:
                self.assertFalse(capture.refresh_default_input("test"))
            get_default.assert_not_called()

    def test_unmatched_pulse_source_falls_back_to_default(self):
        fake_sd = FakeSoundDevice()
        module = self._load_audio_capture(fake_sd)

        source = "alsa_input.usb-1234_Nonexistent_Mic-00.analog-stereo"
        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(
                device_id=source,
                config_manager=FakeConfig({"audio_device_id": source}),
            )

        # No PortAudio device matches the configured source, so init falls back
        # to the Pulse default rather than crashing.
        self.assertEqual(capture.device_id, 0)

    def test_fuzzy_match_requires_two_tokens(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "C922 Pro Stream Webcam: USB Audio (hw:1,0)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        # Only one distinctive token ("webcam") overlaps — below the >=2 bar.
        self.assertIsNone(
            capture._match_pulse_source_to_portaudio("alsa_input.usb-Acme_Webcam", fuzzy=True)
        )

    def test_fuzzy_match_refuses_ambiguous_tie(self):
        fake_sd = FakeSoundDevice()
        # Two identical webcams expose distinct PortAudio names (different hw
        # index) but share every model token, so the source can't disambiguate.
        for hw in (1, 2):
            fake_sd.devices.append({
                "name": f"C922 Pro Stream Webcam: USB Audio (hw:{hw},0)",
                "max_input_channels": 2,
                "default_samplerate": 48000,
                "hostapi": 0,
            })
        module = self._load_audio_capture(fake_sd)

        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        source = "alsa_input.usb-046d_C922_Pro_Stream_Webcam_D487FE4F-02.iec958-stereo"
        # Tie at the top score → refuse to guess rather than bind the wrong twin.
        self.assertIsNone(capture._match_pulse_source_to_portaudio(source, fuzzy=True))

    def test_default_path_does_not_fuzzy_match_concrete_device(self):
        fake_sd = FakeSoundDevice()
        # A "pulse" aggregate plus a concrete USB webcam. The default source is a
        # rich USB name that fuzzy matching WOULD bind to the concrete device,
        # but the default path must stay conservative and pick the aggregate.
        fake_sd.devices.append({
            "name": "pulse",
            "max_input_channels": 32,
            "default_samplerate": 44100,
            "hostapi": 0,
        })
        fake_sd.devices.append({
            "name": "C922 Pro Stream Webcam: USB Audio (hw:1,0)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)

        source = "alsa_input.usb-046d_C922_Pro_Stream_Webcam_D487FE4F-02.iec958-stereo"
        with mock.patch("subprocess.run", side_effect=self._run_sources([source])):
            capture = module.AudioCapture(config_manager=FakeConfig())

        # Index 4 is "pulse"; index 5 is the concrete C922.
        self.assertEqual(capture.device_id, 4)

    def test_configured_source_name_resolves_via_audio_device_name(self):
        fake_sd = FakeSoundDevice()
        fake_sd.devices.append({
            "name": "C922 Pro Stream Webcam: USB Audio (hw:1,0)",
            "max_input_channels": 2,
            "default_samplerate": 48000,
            "hostapi": 0,
        })
        module = self._load_audio_capture(fake_sd)

        source = "alsa_input.usb-046d_C922_Pro_Stream_Webcam_D487FE4F-02.iec958-stereo"
        with mock.patch("subprocess.run", side_effect=self._run_sources(["alsa_input.old mic"])):
            capture = module.AudioCapture(config_manager=FakeConfig({"audio_device_name": source}))

        self.assertEqual(capture.device_id, 4)


if __name__ == "__main__":
    unittest.main()
