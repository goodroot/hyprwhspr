import json
import sys
import time
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import qwen3_asr_backend  # noqa: E402
from backends.qwen3_asr_backend import Qwen3AsrBackend  # noqa: E402


class Config:
    def __init__(self):
        # Per-instance so a test that tweaks a setting cannot leak into others.
        self.values = {"qwen3_asr_model": "1.7b-q8_0", "qwen3_asr_timeout": 12,
                       "language": None, "threads": 3, "qwen3_asr_device": "cpu"}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)


class Manager:
    def __init__(self):
        self.config = Config()
        self.temp_dir = "/tmp"
        self.ready = True
        self.current_model = "1.7b-q8_0"
        self._last_use_time = 0


class Process:
    def __init__(self, running=True):
        self.running = running
        self.terminated = self.killed = self.waited = False

    def poll(self):
        return None if self.running else 1

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self, timeout=None):
        self.waited = True
        return 0


class SplitterTests(unittest.TestCase):
    SR = 16000
    CAP = 120

    def split(self, audio):
        return qwen3_asr_backend.split_for_transcription(audio, self.SR, self.CAP)

    def test_audio_within_the_cap_is_returned_untouched(self):
        for seconds in (0.5, 24, self.CAP):
            with self.subTest(seconds=seconds):
                audio = np.ones(int(seconds * self.SR), dtype=np.float32)
                chunks = self.split(audio)
                self.assertEqual(len(chunks), 1)
                self.assertIs(chunks[0], audio)

    def test_no_samples_are_lost_or_duplicated(self):
        for seconds in (120.1, 144, 300, 1800):
            with self.subTest(seconds=seconds):
                audio = np.arange(int(seconds * self.SR), dtype=np.float32)
                chunks = self.split(audio)
                np.testing.assert_array_equal(np.concatenate(chunks), audio)

    def test_no_chunk_exceeds_the_cap(self):
        for seconds in (120.1, 144, 300, 1800, 3600):
            with self.subTest(seconds=seconds):
                audio = np.ones(int(seconds * self.SR), dtype=np.float32)
                for chunk in self.split(audio):
                    self.assertGreater(len(chunk), 0)
                    self.assertLessEqual(len(chunk) / self.SR, self.CAP)

    def test_split_lands_inside_a_real_pause(self):
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(300 * self.SR) * 0.3).astype(np.float32)
        audio[int(105 * self.SR):int(107 * self.SR)] = 0.0
        cut = len(self.split(audio)[0]) / self.SR
        self.assertGreaterEqual(cut, 105)
        self.assertLessEqual(cut, 107)

    def test_every_cut_lands_in_a_pause_when_speech_is_regularly_spaced(self):
        # Measured against the real sidecar: a seam landing mid-utterance makes
        # the decoder complete or restart the sentence, inflating the transcript
        # by a whole repetition over ten minutes. A search window narrower than
        # the gap between sentences is what caused it.
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(600 * self.SR) * 0.3).astype(np.float32)
        pauses = []
        for k in range(1, 25):
            begin = int(k * 24 * self.SR)
            audio[begin:begin + self.SR // 2] = 0.0
            pauses.append(k * 24)
        cuts = np.cumsum([len(c) for c in self.split(audio)])[:-1] / self.SR
        self.assertGreater(len(cuts), 3)
        for cut in cuts:
            self.assertTrue(any(p <= cut <= p + 0.5 for p in pauses),
                            f"cut at {cut:.2f}s is not inside a pause")

    def test_pause_search_prefers_the_latest_pause_to_keep_chunks_long(self):
        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(300 * self.SR) * 0.3).astype(np.float32)
        # Two candidates in range; the later one should win.
        for at in (90, 118):
            audio[int(at * self.SR):int(at * self.SR) + self.SR // 2] = 0.0
        self.assertGreaterEqual(len(self.split(audio)[0]) / self.SR, 118)

    def test_uniformly_loud_audio_cuts_at_the_target_not_the_earliest_point(self):
        # argmin over equal buckets returns index 0, which would bias every cut
        # to the start of the search window and make extra chunks for nothing.
        audio = np.ones(300 * self.SR, dtype=np.float32)
        self.assertAlmostEqual(
            len(self.split(audio)[0]) / self.SR,
            qwen3_asr_backend._CHUNK_TARGET_SECONDS, places=1)

    def test_no_runt_final_chunk(self):
        # Seen live: a 600 s file split 5x~120 s plus a 0.2 s tail, costing a
        # whole sidecar round trip to transcribe nothing.
        for seconds in (240, 360, 599.998, 600, 720, 1200):
            with self.subTest(seconds=seconds):
                audio = np.ones(int(seconds * self.SR), dtype=np.float32)
                chunks = self.split(audio)
                self.assertGreaterEqual(
                    min(len(c) for c in chunks) / self.SR,
                    qwen3_asr_backend._CHUNK_MIN_TAIL_SECONDS)

    def test_degenerate_sample_rate_is_not_split(self):
        audio = np.ones(1000, dtype=np.float32)
        self.assertEqual(len(qwen3_asr_backend.split_for_transcription(audio, 0, self.CAP)), 1)


class QwenBackendTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.backend = Qwen3AsrBackend(Manager())
        self.backend._socket_path = Path(self._tmp.name) / "qwen3-asr.sock"
        self.backend._process = Process()
        self._log_path = Path(self._tmp.name) / "server.log"

    def _touch(self, name):
        path = Path(self._tmp.name) / name
        path.touch()
        return path

    def test_protocol_cleanup_preserves_unicode(self):
        self.assertEqual(self.backend._clean_text(
            " language Japanese<asr_text>こんにちは世界<|im_end|> "), "こんにちは世界")
        self.assertEqual(self.backend._clean_text("language is useful"), "language is useful")

    def test_split_preamble_reports_the_detected_language(self):
        self.assertEqual(
            self.backend._split_preamble(" language Japanese<asr_text>こんにちは<|im_end|> "),
            ("Japanese", "こんにちは"))
        # No preamble: nothing detected, text untouched.
        self.assertEqual(self.backend._split_preamble("language is useful"),
                         (None, "language is useful"))
        self.assertEqual(self.backend._split_preamble(""), (None, ""))

    def test_exactly_120_seconds_is_accepted(self):
        audio = np.ones(120 * 10, dtype=np.float32)
        response = json.dumps({"text": "fine"}).encode()
        with mock.patch.object(self.backend, "_request", return_value=(200, response)) as request:
            self.assertEqual(self.backend.transcribe(audio, sample_rate=10), "fine")
        request.assert_called_once()

    def test_over_120_seconds_is_chunked_and_rejoined(self):
        # llama.cpp returns nothing past ~2 min, so long audio is split at
        # pauses and rejoined rather than refused.
        audio = np.ones(300 * 16000, dtype=np.float32)
        with mock.patch.object(self.backend, "_request",
                               return_value=(200, b'{"text":"part"}')) as request:
            result = self.backend.transcribe(audio, sample_rate=16000)
        self.assertGreater(request.call_count, 1)
        self.assertEqual(result, "part part part")

    @staticmethod
    def _asr(text, language=None):
        """A llama-server response, optionally carrying the leaked preamble."""
        body = f"language {language}<asr_text>{text}" if language else text
        return (200, json.dumps({"text": body}).encode())

    def _language_fields(self, request):
        return [c.args[2].split(b'name="language"\r\n\r\n')[1].split(b"\r\n")[0].decode()
                for c in request.call_args_list
                if b'name="language"' in c.args[2]]

    def test_detected_language_is_pinned_across_later_chunks(self):
        # Auto-detect runs per request, so without pinning one ambiguous chunk
        # can switch languages midway through a transcript.
        audio = np.ones(300 * 16000, dtype=np.float32)
        responses = [self._asr("one", "Japanese"), self._asr("two", "English"),
                     self._asr("three", "English")]
        with mock.patch.object(self.backend, "_request", side_effect=responses) as request:
            self.assertEqual(
                self.backend.transcribe(audio, sample_rate=16000), "one two three")
        # Chunk 1 sends none (auto-detect); 2 and 3 carry chunk 1's detection.
        self.assertEqual(self._language_fields(request), ["Japanese", "Japanese"])

    def test_an_empty_leading_chunk_does_not_pin_the_session(self):
        # Silence or music first would otherwise mispin every later chunk.
        audio = np.ones(300 * 16000, dtype=np.float32)
        responses = [self._asr("", "English"), self._asr("real", "Japanese"),
                     self._asr("more", "English")]
        with mock.patch.object(self.backend, "_request", side_effect=responses) as request:
            self.backend.transcribe(audio, sample_rate=16000)
        self.assertEqual(self._language_fields(request), ["Japanese"])

    def test_an_explicit_language_is_never_overridden_by_detection(self):
        audio = np.ones(300 * 16000, dtype=np.float32)
        responses = [self._asr("a", "Japanese")] * 3
        with mock.patch.object(self.backend, "_request", side_effect=responses) as request:
            self.backend.transcribe(audio, sample_rate=16000, language_override="de")
        self.assertEqual(self._language_fields(request), ["German"] * 3)

    def test_no_preamble_means_no_pinning(self):
        # The graceful-degradation case: if llama.cpp#26749 is fixed upstream and
        # the preamble stops leaking, behaviour must match today's exactly.
        audio = np.ones(300 * 16000, dtype=np.float32)
        with mock.patch.object(self.backend, "_request",
                               side_effect=[self._asr("a"), self._asr("b"), self._asr("c")]) as request:
            self.assertEqual(self.backend.transcribe(audio, sample_rate=16000), "a b c")
        self.assertEqual(self._language_fields(request), [])

    def test_partial_failure_notifies_that_the_transcript_is_incomplete(self):
        audio = np.ones(300 * 16000, dtype=np.float32)
        responses = [self._asr("one"), (500, b""), self._asr("three")]
        with mock.patch.object(self.backend, "_notify_incomplete") as notify, \
                mock.patch.object(self.backend, "_request", side_effect=responses):
            self.assertEqual(
                self.backend.transcribe(audio, sample_rate=16000), "one three")
        notify.assert_called_once_with(1, 3)

    def test_full_success_and_total_failure_do_not_notify(self):
        audio = np.ones(300 * 16000, dtype=np.float32)
        for label, responses in [("all ok", [self._asr("x")] * 3),
                                 ("all fail", [(500, b"")] * 3)]:
            with self.subTest(label):
                with mock.patch.object(self.backend, "_notify_incomplete") as notify, \
                        mock.patch.object(self.backend, "_request", side_effect=responses):
                    self.backend.transcribe(audio, sample_rate=16000)
                notify.assert_not_called()

    def test_a_failed_chunk_does_not_discard_the_rest(self):
        audio = np.ones(300 * 16000, dtype=np.float32)
        responses = [(200, b'{"text":"one"}'), (500, b''), (200, b'{"text":"three"}')]
        with mock.patch.object(self.backend, "_request", side_effect=responses):
            self.assertEqual(self.backend.transcribe(audio, sample_rate=16000), "one three")

    def test_language_is_sent_as_a_name_and_no_prompt_is_sent(self):
        # llama.cpp appends this field verbatim to the ASR prompt, so Qwen must
        # receive "Japanese", not the ISO code hyprwhspr stores.
        audio = np.ones(10, dtype=np.float32)
        with mock.patch.object(self.backend, "_request",
                               return_value=(200, b'{"text":"ok"}')) as request:
            self.backend.transcribe(audio, sample_rate=10, language_override="ja")
        body = request.call_args.args[2]
        self.assertIn(b'name="language"\r\n\r\nJapanese', body)
        self.assertNotIn(b'name="prompt"', body)
        self.assertIn(b'RIFF', body)

    def test_unknown_language_code_passes_through(self):
        audio = np.ones(10, dtype=np.float32)
        with mock.patch.object(self.backend, "_request",
                               return_value=(200, b'{"text":"ok"}')) as request:
            self.backend.transcribe(audio, sample_rate=10, language_override="xx")
        self.assertIn(b'name="language"\r\n\r\nxx', request.call_args.args[2])

    def test_restart_cooldown_clears_on_reinitialize(self):
        self.backend._last_restart = time.monotonic()
        with mock.patch.object(self.backend, "_stop"), \
                mock.patch.object(self.backend, "_start", return_value=True):
            self.assertTrue(self.backend.reinitialize())
        self.assertIsNone(self.backend._last_restart)

    def test_crashes_outside_the_cooldown_each_get_a_restart(self):
        # The regression: a one-shot flag refused the second crash even hours
        # later, bricking the backend until the service was restarted.
        self.backend._process.running = False
        self.backend._last_restart = time.monotonic() - (
            qwen3_asr_backend._RESTART_COOLDOWN_SECONDS + 1)
        started = []

        def start():
            started.append(True)
            self.backend._process = Process()
            return True

        with mock.patch.object(self.backend, "_start", side_effect=start), \
                mock.patch.object(self.backend, "_stop"), \
                mock.patch.object(self.backend, "_request",
                                  return_value=(200, b'{"text":"ok"}')):
            self.assertEqual(
                self.backend.transcribe(np.ones(10, dtype=np.float32), 10), "ok")
        self.assertEqual(len(started), 1)

    def test_first_ever_crash_restarts_even_at_low_uptime(self):
        # time.monotonic() is seconds since boot on Linux, so a 0.0 sentinel sat
        # inside the cooldown window and refused the very first restart when the
        # service autostarted at boot.
        self.backend._process.running = False
        self.assertIsNone(self.backend._last_restart)
        with mock.patch.object(qwen3_asr_backend.time, "monotonic", return_value=12.0), \
                mock.patch.object(self.backend, "_stop"), \
                mock.patch.object(self.backend, "_start", return_value=True) as start:
            self.backend.transcribe(np.ones(10, dtype=np.float32), 10)
        start.assert_called_once()

    def test_chunks_are_joined_with_cjk_spacing_rules(self):
        # join_segments omits the space between CJK neighbours, which is the
        # reason this backend must not use a plain " ".join().
        audio = np.ones(300 * 16000, dtype=np.float32)
        cjk = [(200, '{"text":"你好"}'.encode())] * 3
        with mock.patch.object(self.backend, "_request", side_effect=cjk):
            self.assertEqual(
                self.backend.transcribe(audio, sample_rate=16000), "你好你好你好")
        latin = [(200, b'{"text":"hello"}')] * 3
        with mock.patch.object(self.backend, "_request", side_effect=latin):
            self.assertEqual(
                self.backend.transcribe(audio, sample_rate=16000), "hello hello hello")

    def _startup_window(self, timeout):
        """Virtual seconds _start() waits for health before giving up."""
        self.backend.config.values["qwen3_asr_timeout"] = timeout
        clock = {"t": 1000.0}

        def monotonic():
            clock["t"] += 10.0
            return clock["t"]

        start_t = clock["t"]
        with mock.patch.object(qwen3_asr_backend.time, "monotonic", monotonic), \
                mock.patch.object(qwen3_asr_backend.time, "sleep", lambda _: None), \
                mock.patch.object(qwen3_asr_backend.subprocess, "Popen",
                                  return_value=Process()), \
                mock.patch.object(qwen3_asr_backend, "server_path",
                                  return_value=self._touch("llama-server")), \
                mock.patch.object(qwen3_asr_backend, "model_paths",
                                  return_value=(self._touch("d.gguf"), self._touch("p.gguf"))), \
                mock.patch.object(qwen3_asr_backend, "QWEN3_ASR_LOG", self._log_path), \
                mock.patch.object(self.backend, "_request", side_effect=OSError("not up")), \
                mock.patch.object(self.backend, "_stop"):
            self.assertFalse(self.backend._start())
        return clock["t"] - start_t

    def test_startup_deadline_can_be_extended_by_config(self):
        # min() meant raising qwen3_asr_timeout could only ever shorten the
        # model-load window, never lengthen it. A cold 2.4 GB load needs longer.
        long_window = self._startup_window(400)
        short_window = self._startup_window(5)
        self.assertGreater(long_window, 300)
        # Below the floor the window must not shrink with the request timeout.
        self.assertLess(short_window, 120)

    def test_second_crash_inside_the_cooldown_is_refused(self):
        self.backend._process.running = False
        self.backend._last_restart = time.monotonic()
        with mock.patch.object(self.backend, "_start") as start:
            self.assertEqual(
                self.backend.transcribe(np.ones(10, dtype=np.float32), 10), "")
        start.assert_not_called()

    def test_unloaded_backend_does_not_silently_reload(self):
        # `model unload` stops the sidecar deliberately. WhisperManager also
        # refuses to dispatch while not ready, so this asserts the backend is
        # correct on its own terms rather than leaning on its caller.
        with tempfile.TemporaryDirectory() as directory:
            self.backend._socket_path = Path(directory) / "server.sock"
            self.backend.unload()
        with mock.patch.object(self.backend, "_start") as start:
            self.assertEqual(
                self.backend.transcribe(np.ones(10, dtype=np.float32), 10), "")
        start.assert_not_called()
        self.assertIsNone(self.backend._last_restart)

    def test_overlong_socket_path_is_reported_clearly(self):
        # Found during live verification: over the AF_UNIX 108-byte sun_path
        # limit, llama-server reports only "couldn't bind HTTP server socket …
        # port: 8080", which reads like a port conflict.
        self.backend._socket_path = Path("/tmp/" + "d" * 120 + "/qwen3-asr.sock")
        with mock.patch.object(qwen3_asr_backend.subprocess, "Popen") as popen:
            self.assertFalse(self.backend._start())
        popen.assert_not_called()

    def test_default_socket_path_is_within_the_kernel_limit(self):
        self.assertLessEqual(
            len(str(qwen3_asr_backend.QWEN3_ASR_SOCKET).encode()),
            qwen3_asr_backend._MAX_UNIX_SOCKET_BYTES)

    def test_stderr_is_never_an_unread_pipe(self):
        # An unread PIPE deadlocks llama-server once its 64 KiB buffer fills.
        import subprocess as sp
        captured = {}

        def fake_popen(args, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here")

        with mock.patch.object(qwen3_asr_backend.subprocess, "Popen", fake_popen), \
                mock.patch.object(qwen3_asr_backend, "server_path",
                                  return_value=self._touch("llama-server")), \
                mock.patch.object(qwen3_asr_backend, "model_paths",
                                  return_value=(self._touch("d.gguf"), self._touch("p.gguf"))), \
                mock.patch.object(qwen3_asr_backend, "QWEN3_ASR_LOG", self._log_path):
            self.assertFalse(self.backend._start())
        self.assertNotEqual(captured.get("stderr"), sp.PIPE)
        self.assertIsNotNone(captured.get("stderr"))

    def test_cleanup_terminates_and_reaps_child(self):
        process = self.backend._process
        with tempfile.TemporaryDirectory() as directory:
            self.backend._socket_path = Path(directory) / "server.sock"
            self.backend.cleanup()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertIsNone(self.backend._process)

    def test_a_still_dead_child_is_not_restarted_twice_in_one_request(self):
        self.backend._process.running = False
        new_process = Process()
        def start():
            self.backend._process = new_process
            return True
        with mock.patch.object(self.backend, "_start", side_effect=start) as start_mock, \
                mock.patch.object(self.backend, "_stop"), \
                mock.patch.object(self.backend, "_request", side_effect=OSError("died")):
            new_process.running = False
            self.assertEqual(self.backend.transcribe(np.ones(10, dtype=np.float32), 10), "")
        start_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
