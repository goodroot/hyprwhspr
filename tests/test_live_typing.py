"""Live typing (realtime_live_typing): word accounting, the delivery session,
the TextInjector stream session, and the recording lifecycle around them.

The injector stream-session tests are ported from Jacob Banghart's incremental
injection work (#261).
"""

import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))
sys.path.insert(0, str(ROOT / "lib"))
sys.modules.setdefault("websocket", types.SimpleNamespace(WebSocketApp=object))

from tests.test_suspend_resume_recovery import _import_main_isolated  # noqa: E402
from tests.text_injector_helpers import ConfigStub, make_injector  # noqa: E402

from backends import RealtimeWsBackend  # noqa: E402
from hallucination import could_be_hallucination  # noqa: E402
from live_typing import LiveTyper, LiveTypingSession  # noqa: E402
from realtime_client import RealtimeClient  # noqa: E402
from realtime_protocols import live_typing_conflict  # noqa: E402
from text_injector import InjectionOutcome  # noqa: E402


class LiveTyperTests(unittest.TestCase):
    def test_unfinished_last_word_waits(self):
        typer = LiveTyper()
        self.assertEqual(typer.update("", "hello wor"), (["hello"], False))
        self.assertEqual(typer.update("", "hello world "), (["world"], False))

    def test_committed_segment_releases_its_words_with_punctuation(self):
        typer = LiveTyper()
        typer.update("", "the quick brown fox")
        self.assertEqual(typer.update("The quick brown fox.", ""), (["fox."], True))
        self.assertEqual(typer.update("The quick brown fox.", "jumped "), (["jumped"], False))

    def test_opening_phantom_is_held_until_speech_diverges(self):
        typer = LiveTyper()
        self.assertEqual(typer.update("", "thank you "), ([], False))
        self.assertEqual(typer.update("Thank you.", ""), ([], True))
        self.assertEqual(typer.update("Thank you.", "for the review "),
                         (["Thank", "you.", "for", "the", "review"], False))

    def test_marker_words_after_real_text_are_typed(self):
        typer = LiveTyper()
        typer.update("Ship it.", "")
        self.assertEqual(typer.update("Ship it. Thank you.", ""), (["Thank", "you."], True))

    def test_finish_types_the_rest_aligned_by_word_count(self):
        typer = LiveTyper()
        typer.update("", "is it done ")
        self.assertEqual(typer.finish("Is it done? Yes."), ["Yes."])
        self.assertEqual(typer.finish("Is it"), [])

    def test_could_be_hallucination_matches_marker_prefixes(self):
        self.assertTrue(could_be_hallucination("Thank"))
        self.assertTrue(could_be_hallucination("thanks for"))
        self.assertFalse(could_be_hallucination("thanks for the"))
        self.assertFalse(could_be_hallucination(""))


class FakeInjector:
    def __init__(self, fail=False):
        self.chunks = []
        self.ended = []
        self.fail = fail

    def begin_stream(self):
        return types.SimpleNamespace(cancelled=threading.Event(), failed=False)

    def release_stream(self, handle):
        pass

    def inject_stream_chunk(self, handle, text, final=False):
        self.chunks.append((text, final))
        return InjectionOutcome.FAILED if self.fail else InjectionOutcome.INJECTED

    def end_stream(self, handle, submit=True):
        self.ended.append(submit)
        return InjectionOutcome.INJECTED if self.chunks else None


class LiveTypingSessionTests(unittest.TestCase):
    def test_words_type_in_order_and_finish_adds_the_rest(self):
        injector = FakeInjector()
        session = LiveTypingSession(injector)
        session.on_live_text("", "can you ")
        session.on_live_text("", "can you check the ")
        self.assertEqual(session.finish("Can you check the logs?"), InjectionOutcome.INJECTED)
        self.assertEqual(injector.chunks,
                         [("can you", False), ("check the", False), ("logs?", True)])
        self.assertEqual(injector.ended, [True])

    def test_nothing_typed_hands_the_transcript_back(self):
        injector = FakeInjector()
        session = LiveTypingSession(injector)
        session.on_live_text("", "thank you")
        self.assertIsNone(session.finish("Thank you."))
        self.assertEqual(injector.chunks, [])
        self.assertEqual(injector.ended, [False])

    def test_text_after_finish_is_ignored(self):
        injector = FakeInjector()
        session = LiveTypingSession(injector)
        session.on_live_text("", "hello there ")
        session.finish("hello there")
        session.on_live_text("hello there", "late words ")
        self.assertEqual(injector.chunks, [("hello there", False)])

    def test_a_failed_chunk_fails_the_dictation(self):
        session = LiveTypingSession(FakeInjector(fail=True))
        session.on_live_text("", "hello there ")
        self.assertEqual(session.finish("hello there"), InjectionOutcome.FAILED)

    def test_cancel_drops_pending_words_and_never_submits(self):
        injector = FakeInjector()
        release = threading.Event()
        original = injector.inject_stream_chunk

        def slow(handle, text, final=False):
            release.wait(2)
            return original(handle, text, final)

        injector.inject_stream_chunk = slow
        session = LiveTypingSession(injector)
        session.on_live_text("", "one ")
        session.on_live_text("", "one two ")
        threading.Timer(0.05, release.set).start()
        session.cancel()
        self.assertLessEqual(len(injector.chunks), 1)
        self.assertEqual(injector.ended, [False])
        self.assertEqual(session.finish("one two"), InjectionOutcome.FAILED)


class LiveTypingShutdownTests(unittest.TestCase):
    def _session(self, settings=None):
        injector = make_injector()
        injector.config_manager = ConfigStub({
            "append_trailing_space": False, **(settings or {}),
        })
        session = LiveTypingSession(injector)
        session.FINISH_TIMEOUT_SECS = 0.03
        self.addCleanup(self._stop, session)
        return injector, session

    def _stop(self, session):
        session.cancel()
        session._worker.join(1)
        self.assertFalse(session._worker.is_alive())

    def _stall(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        return entered, release

    def test_timeout_discards_queue_and_reserves_delivery_through_cleanup(self):
        injector, session = self._session()
        entered, release = self._stall()
        calls = []

        def paste(text, **kwargs):
            calls.append(text)
            entered.set()
            self.assertTrue(release.wait(2))
            return True

        with (
            mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste),
            mock.patch.object(injector, "_send_enter_if_auto_submit") as enter,
        ):
            session.on_live_text("", "hello ")
            self.assertTrue(entered.wait(1))
            session.on_live_text("", "hello queued ")
            self.assertEqual(session.finish("Hello queued remainder."), InjectionOutcome.FAILED)
            self.assertTrue(injector.stream_busy())
            self.assertEqual(injector.inject_text("overlap"), InjectionOutcome.FAILED)
            for action in ("copy_last", "paste_last"):
                self.assertFalse(injector.recover_last(action)[0])
            with self.assertRaises(RuntimeError):
                LiveTypingSession(injector)
            release.set()
            self.assertTrue(session._done.wait(1))
            self.assertEqual(calls, ["hello"])
            enter.assert_not_called()
            self.assertEqual(injector._last_text, "Hello queued remainder.")
            self.assertFalse(injector.stream_busy())
            self.assertEqual(session.finish("ignored"), InjectionOutcome.FAILED)
            next_session = LiveTypingSession(injector)
            next_session.cancel()

    def test_explicit_cancel_keeps_only_attempted_text(self):
        injector, session = self._session()
        entered, release = self._stall()

        def paste(text, **kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return True

        with (
            mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste) as delivery,
            mock.patch.object(injector, "_send_enter_if_auto_submit") as enter,
        ):
            session.on_live_text("", "hello ")
            self.assertTrue(entered.wait(1))
            session.on_live_text("", "hello queued ")
            session.cancel()
            self.assertTrue(injector.stream_busy())
            release.set()
            self.assertTrue(session._done.wait(1))
            self.assertEqual(delivery.call_count, 1)
            self.assertEqual(injector._last_text, "hello")
            enter.assert_not_called()

    def test_timeout_during_each_finalization_stage_suppresses_later_delivery(self):
        for stage in ("held", "space", "restore"):
            with self.subTest(stage=stage):
                injector, session = self._session({"append_trailing_space": True})
                entered, release = self._stall()
                calls = []

                def paste(text, stream=None, **kwargs):
                    calls.append(text)
                    stream.saved_clipboard = b"original"
                    stream.last_pasted = text
                    if (stage == "held" and text == " question") or (stage == "space" and text == " "):
                        entered.set()
                        self.assertTrue(release.wait(2))
                    return True

                def restore(*args, **kwargs):
                    if stage == "restore":
                        entered.set()
                        self.assertTrue(release.wait(2))

                with (
                    mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste),
                    mock.patch.object(injector, "_restore_clipboard", side_effect=restore) as cleanup,
                    mock.patch.object(injector, "_send_enter_if_auto_submit") as enter,
                ):
                    session.on_live_text("", "open question ")
                    result = []
                    finisher = threading.Thread(target=lambda: result.append(session.finish("open question")))
                    finisher.start()
                    self.assertTrue(entered.wait(1))
                    finisher.join(1)
                    self.assertFalse(finisher.is_alive())
                    self.assertEqual(result, [InjectionOutcome.FAILED])
                    self.assertTrue(injector.stream_busy())
                    release.set()
                    self.assertTrue(session._done.wait(1))
                    enter.assert_not_called()
                    cleanup.assert_called_once()
                    if stage == "held":
                        self.assertEqual(calls, ["open", " question"])
                    self.assertEqual(session.finish("ignored"), InjectionOutcome.FAILED)

    def test_timeout_during_submit_does_not_dispatch_a_fallback_enter(self):
        injector, session = self._session({"auto_submit": True})
        entered, release = self._stall()

        def native_enter(chord):
            self.assertEqual(chord, "enter")
            entered.set()
            self.assertTrue(release.wait(2))
            return False

        with (
            mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=True),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_send_shortcut_hyprland", side_effect=native_enter),
            mock.patch.object(injector, "_run_ydotool") as fallback,
        ):
            session.on_live_text("", "hello ")
            self.assertEqual(session.finish("hello"), InjectionOutcome.FAILED)
            self.assertTrue(entered.is_set())
            release.set()
            self.assertTrue(session._done.wait(1))
            fallback.assert_not_called()

    def test_stale_handles_cannot_modify_or_close_next_stream(self):
        injector = make_injector()
        old = injector.begin_stream()
        injector.end_stream(old, submit=False)
        current = injector.begin_stream()
        self.assertEqual(injector.inject_stream_chunk(old, "stale"), InjectionOutcome.FAILED)
        self.assertEqual(injector.end_stream(old), InjectionOutcome.FAILED)
        injector.release_stream(old)
        self.assertIs(injector._stream, current)
        self.assertEqual(current.raw_parts, [])
        injector.end_stream(current, submit=False)

    def test_delivery_exception_prevents_submit_and_cleanup_runs_once(self):
        injector, session = self._session()
        with (
            mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=RuntimeError("paste failed")),
            mock.patch.object(injector, "end_stream", wraps=injector.end_stream) as cleanup,
            mock.patch.object(injector, "_send_enter_if_auto_submit") as enter,
        ):
            session.on_live_text("", "hello ")
            self.assertEqual(session.finish("hello"), InjectionOutcome.FAILED)
            session.cancel()
            self.assertEqual(session.finish("hello"), InjectionOutcome.FAILED)
            cleanup.assert_called_once()
            enter.assert_not_called()
            self.assertFalse(injector.stream_busy())

    def test_no_live_text_fallback_waits_for_cleanup_and_fails_on_timeout(self):
        injector, session = self._session()
        entered, release = self._stall()

        def cleanup(*args):
            entered.set()
            self.assertTrue(release.wait(2))
            return None

        with mock.patch.object(injector, "_finish_stream", side_effect=cleanup):
            self.assertEqual(session.finish("Thank you."), InjectionOutcome.FAILED)
            self.assertTrue(entered.is_set())
            self.assertTrue(injector.stream_busy())
            release.set()
            self.assertTrue(session._done.wait(1))
            self.assertEqual(session.finish("Thank you."), InjectionOutcome.FAILED)

    def test_thread_start_failure_releases_reservation(self):
        injector = make_injector()
        with mock.patch("live_typing.threading.Thread.start", side_effect=RuntimeError("no thread")):
            with self.assertRaises(RuntimeError):
                LiveTypingSession(injector)
        self.assertFalse(injector.stream_busy())

    def test_cleanup_exception_releases_ownership_and_fails(self):
        injector, session = self._session()
        with mock.patch.object(injector, "_finish_stream", side_effect=RuntimeError("cleanup failed")):
            self.assertEqual(session.finish("thank you"), InjectionOutcome.FAILED)
        self.assertTrue(session._done.is_set())
        self.assertFalse(injector.stream_busy())

    def test_timeout_does_not_expose_blocked_app_transcript(self):
        injector, session = self._session()
        entered, release = self._stall()
        injector._last_text = "previous"

        def blocked(text, stream=None, **kwargs):
            stream.recovery_allowed = False
            stream.not_pasted = True
            entered.set()
            self.assertTrue(release.wait(2))
            return True

        with mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=blocked):
            session.on_live_text("", "secret ")
            self.assertTrue(entered.wait(1))
            self.assertEqual(session.finish("secret final words"), InjectionOutcome.FAILED)
            release.set()
            self.assertTrue(session._done.wait(1))
            self.assertEqual(injector._last_text, "previous")


class RealtimeClientToInjectorTests(unittest.TestCase):
    """The real OpenAI-protocol client feeding a session and the real injector."""

    def _run(self, events, settings=None):
        injector = make_injector()
        injector.config_manager = ConfigStub({"append_trailing_space": False, **(settings or {})})
        pasted = []

        def paste(text, auto_submit=True, retain=False, stream=None):
            pasted.append(text)
            return True

        client = RealtimeClient(mode="transcribe")
        session = LiveTypingSession(injector)
        client.set_live_text_listener(session.on_live_text)
        with (
            mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste),
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch.object(injector, "_restore_clipboard"),
        ):
            for kind, text in events:
                event_type = "conversation.item.input_audio_transcription." + kind
                key = "delta" if kind == "delta" else "transcript"
                client._handle_event({"type": event_type, key: text})
            session.finish(client._full_committed_text_locked())
        return "".join(pasted)

    def test_new_line_completing_after_the_previous_word_was_typed(self):
        events = [("delta", d) for d in ["I said hello", " new", " line world", " again"]]
        events.append(("completed", "I said hello new line world again."))
        self.assertEqual(self._run(events), "I said hello\nworld again.")

    def test_question_mark_split_across_deltas(self):
        events = [("delta", d) for d in ["is it", " done question", " mark"]]
        events.append(("completed", "is it done question mark"))
        self.assertEqual(self._run(events), "is it done?")

    def test_override_split_across_deltas_and_segments(self):
        events = [("delta", d) for d in ["I use", " hyper", " whisper daily"]]
        events.append(("completed", "I use hyper whisper daily."))
        events += [("delta", " It works"), ("completed", "It works.")]
        self.assertEqual(
            self._run(events, {"word_overrides": {"hyper whisper": "hyprwhspr"}}),
            "I use hyprwhspr daily. It works.",
        )


class TextInjectorStreamSessionTests(unittest.TestCase):
    """Layer 4: TextInjector.inject_stream_chunk / end_stream."""

    def _injector(self, settings=None):
        injector = make_injector()
        injector.config_manager = ConfigStub(settings or {})
        self.pasted = []

        def paste(text, auto_submit=True, retain=False, stream=None):
            self.assertFalse(auto_submit)
            self.assertFalse(retain)
            self.assertIsNotNone(stream)
            self.pasted.append(text)
            return True

        patcher = mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", side_effect=paste)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.enter = mock.patch.object(injector, "_send_enter_if_auto_submit").start()
        self.restore = mock.patch.object(injector, "_restore_clipboard").start()
        self.addCleanup(mock.patch.stopall)
        return injector, injector.begin_stream()

    def test_chunks_are_joined_with_a_leading_space(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "can you", final=False)
        injector.inject_stream_chunk(handle, "check the logs", final=False)
        self.assertEqual(self.pasted, ["can you", " check the logs"])

    def test_auto_submit_enter_is_sent_once_at_the_end(self):
        injector, handle = self._injector({"auto_submit": True})
        injector.inject_stream_chunk(handle, "can you", final=False)
        injector.inject_stream_chunk(handle, "check the logs", final=True)
        self.enter.assert_not_called()
        self.assertEqual(injector.end_stream(handle), InjectionOutcome.INJECTED)
        self.enter.assert_called_once()

    def test_segments_stay_separated_when_trailing_space_is_off(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "Hello.", final=True)
        injector.inject_stream_chunk(handle, "How are you?", final=True)
        injector.end_stream(handle)
        self.assertEqual("".join(self.pasted), "Hello. How are you?")

    def test_trailing_space_setting_applies_once_at_the_end(self):
        injector, handle = self._injector({"append_trailing_space": True})
        injector.inject_stream_chunk(handle, "Hello there.", final=True)
        injector.end_stream(handle)
        self.assertEqual(self.pasted, ["Hello there.", " "])

    def test_spoken_punctuation_attaches_to_the_previous_chunk(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "hello", final=False)
        injector.inject_stream_chunk(handle, "comma world", final=False)
        self.assertEqual("".join(self.pasted), "hello, world")

    def test_multi_word_command_split_across_chunks_is_held_together(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "is it done question", final=False)
        injector.inject_stream_chunk(handle, "mark", final=True)
        self.assertEqual("".join(self.pasted), "is it done?")

    def test_new_line_split_across_chunks_survives(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "first item new", final=False)
        injector.inject_stream_chunk(handle, "line second item", final=True)
        self.assertEqual("".join(self.pasted), "first item\nsecond item")

    def test_new_line_after_already_typed_words_survives(self):
        # "hello" is typed before "new" completes; the newline then starts a
        # chunk, where preprocessing's strip would otherwise drop it.
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "I said hello", final=False)
        injector.inject_stream_chunk(handle, "new", final=False)
        injector.inject_stream_chunk(handle, "line world", final=False)
        injector.inject_stream_chunk(handle, "again.", final=True)
        self.assertEqual("".join(self.pasted), "I said hello\nworld again.")

    def test_segment_starting_with_new_line_keeps_it(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "Dear John,", final=True)
        injector.inject_stream_chunk(handle, "new line thanks for the update.", final=True)
        self.assertEqual("".join(self.pasted), "Dear John,\nthanks for the update.")

    def test_trailing_new_line_is_dropped_like_inject_text_does(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "last item new line", final=True)
        injector.end_stream(handle)
        self.assertEqual("".join(self.pasted), "last item")

    def test_multi_word_filler_split_across_chunks_is_filtered(self):
        injector, handle = self._injector({
            "append_trailing_space": False,
            "filter_filler_words": True,
            "filler_words": ["you know"],
        })
        injector.inject_stream_chunk(handle, "it was you", final=False)
        injector.inject_stream_chunk(handle, "know great", final=True)
        self.assertNotIn("you know", "".join(self.pasted))
        self.assertNotIn("know", "".join(self.pasted))

    def test_new_line_at_a_segment_end_waits_for_the_next_word(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "first item new line", final=True)
        injector.inject_stream_chunk(handle, "second item", final=True)
        injector.end_stream(handle)
        self.assertEqual("".join(self.pasted), "first item\nsecond item")

    def test_consecutive_new_lines_at_a_chunk_boundary_survive(self):
        for first, second, final in (
            ("first item new line new line", "second item", True),
            ("first item new line new", "line second item", False),
        ):
            with self.subTest(first=first):
                injector, handle = self._injector({"append_trailing_space": False})
                injector.inject_stream_chunk(handle, first, final=final)
                injector.inject_stream_chunk(handle, second, final=True)
                injector.end_stream(handle)
                self.assertEqual("".join(self.pasted), "first item\n\nsecond item")

    def test_multi_word_override_split_across_chunks_matches(self):
        injector, handle = self._injector({
            "append_trailing_space": False,
            "word_overrides": {"hyper whisper": "hyprwhspr"},
        })
        injector.inject_stream_chunk(handle, "I use hyper", final=False)
        injector.inject_stream_chunk(handle, "whisper daily", final=False)
        self.assertEqual("".join(self.pasted), "I use hyprwhspr daily")

    def test_held_words_are_delivered_at_the_end(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "open the question", final=False)
        injector.end_stream(handle)
        self.assertEqual("".join(self.pasted), "open the question")

    def test_whole_dictation_is_retained_for_recovery(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector.inject_stream_chunk(handle, "hello", final=False)
        injector.inject_stream_chunk(handle, "there friend.", final=True)
        injector.end_stream(handle)
        self.assertEqual(injector._last_text, "hello there friend.")

    def test_cancel_skips_held_words_and_enter(self):
        injector, handle = self._injector({"auto_submit": True, "append_trailing_space": True})
        injector.inject_stream_chunk(handle, "open the question", final=False)
        injector.end_stream(handle, submit=False)
        self.assertEqual(self.pasted, ["open the"])
        self.enter.assert_not_called()

    def test_failed_chunk_makes_the_dictation_fail_without_enter(self):
        injector, handle = self._injector({"auto_submit": True})
        with mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=False):
            self.assertEqual(
                injector.inject_stream_chunk(handle, "hello", final=True), InjectionOutcome.FAILED
            )
        self.assertEqual(injector.end_stream(handle), InjectionOutcome.FAILED)
        self.enter.assert_not_called()

    def test_all_failed_chunks_retain_the_current_dictation_for_recovery(self):
        injector, handle = self._injector({"append_trailing_space": False})
        injector._last_text = "previous dictation"
        with mock.patch.object(injector, "_inject_via_clipboard_and_hotkey", return_value=False):
            injector.inject_stream_chunk(handle, "hello", final=False)
            injector.inject_stream_chunk(handle, "there friend", final=True)
        self.assertEqual(injector.end_stream(handle), InjectionOutcome.FAILED)
        self.assertEqual(injector._last_text, "hello there friend")
        self.enter.assert_not_called()

    def test_end_without_delivered_text_is_a_noop(self):
        injector, handle = self._injector()
        self.assertIsNone(injector.end_stream(handle))
        self.enter.assert_not_called()
        self.restore.assert_not_called()


class TextInjectorStreamClipboardTests(unittest.TestCase):
    """The clipboard is saved before the first chunk and restored once."""

    def test_no_enter_when_injection_is_disabled_for_the_focused_app(self):
        injector = make_injector()
        handle = injector.begin_stream()
        injector.config_manager = ConfigStub({"auto_submit": True})
        injector._last_text = "previous dictation"
        with (
            mock.patch.object(injector, "_active_window_lookup_needed", return_value=False),
            mock.patch.object(injector, "_is_gnome_wayland_session", return_value=False),
            mock.patch.object(injector, "_resolve_paste_chord", return_value=(False, "keepassxc")),
            mock.patch.object(injector, "_paste_via_clipboard") as paste,
            mock.patch.object(injector, "_send_enter_if_auto_submit") as enter,
            mock.patch.object(injector, "_restore_clipboard") as restore,
        ):
            injector.inject_stream_chunk(handle, "hunter two", final=True)
            injector.end_stream(handle)

        paste.assert_not_called()
        enter.assert_not_called()
        restore.assert_not_called()
        self.assertEqual(injector._last_text, "previous dictation")

    def test_clipboard_saved_once_and_restored_once_to_the_original(self):
        injector = make_injector()
        handle = injector.begin_stream()
        injector.config_manager = ConfigStub({"append_trailing_space": False})
        clipboard = {"value": b"https://original.example"}

        def copy(text):
            clipboard["value"] = text.encode("utf-8")
            return True

        with (
            mock.patch.object(injector, "_save_clipboard", side_effect=lambda: clipboard["value"]) as save,
            mock.patch.object(injector, "_copy_text_to_clipboard", side_effect=copy),
            mock.patch.object(injector, "_is_x11_session", return_value=False),
            mock.patch.object(injector, "_is_hyprland_session", return_value=True),
            mock.patch.object(injector, "_is_gnome_wayland_session", return_value=False),
            mock.patch.object(injector, "_active_window_lookup_needed", return_value=False),
            mock.patch.object(injector, "_resolve_paste_chord", return_value=("ctrl+v", None)),
            mock.patch.object(injector, "_send_shortcut_hyprland", return_value=True),
            mock.patch.object(injector, "_restore_clipboard") as restore,
            mock.patch.object(injector, "_send_enter_if_auto_submit"),
            mock.patch("text_injector.time.sleep"),
        ):
            injector.inject_stream_chunk(handle, "hello there", final=False)
            injector.inject_stream_chunk(handle, "friend", final=True)
            restore.assert_not_called()
            injector.end_stream(handle)

        save.assert_called_once()
        restore.assert_called_once()
        self.assertEqual(restore.call_args.args[0], b"https://original.example")
        self.assertEqual(restore.call_args.kwargs["injected"], b" friend")


class RecordingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def _app(self, transcript="", settings=None, supported=True):
        main = self.main
        app = main.hyprwhsprApp.__new__(main.hyprwhsprApp)
        values = {"realtime_live_typing": True, **(settings or {})}
        app.config = types.SimpleNamespace(
            get_setting=lambda key, default=None: values.get(key, default),
            get_hallucination_markers=lambda: None,
        )
        app.text_injector = mock.Mock()
        app.text_injector.stream_busy.return_value = False
        app._live_typing = None
        app._recording_lock = threading.Lock()
        app.is_processing = False
        app._current_language_override = None
        app.audio_capture = types.SimpleNamespace(sample_rate=16000, abort_recovery=mock.Mock())
        app.audio_manager = mock.Mock()
        app._recording_control_server = mock.Mock()
        app._recording_control_server.has_capture_subscriber.return_value = False
        app._recording_control_server.is_trace_capture.return_value = False
        app._notify_user = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._inject_text = mock.Mock(return_value=InjectionOutcome.INJECTED)
        app._mark_delivery_healthy = mock.Mock()
        app.whisper_manager = types.SimpleNamespace(
            transcribe_audio=mock.Mock(return_value=transcript),
            realtime_live_typing_supported=lambda: supported,
        )
        return app

    def test_eligibility(self):
        self.assertTrue(self._app()._live_typing_eligible())
        self.assertFalse(self._app(supported=False)._live_typing_eligible())
        self.assertFalse(self._app(settings={"realtime_live_typing": False})._live_typing_eligible())
        app = self._app()
        app._recording_control_server.has_capture_subscriber.return_value = True
        self.assertFalse(app._live_typing_eligible())

    def test_typed_live_finishes_without_a_second_paste(self):
        app = self._app(transcript="hello there friend")
        app._live_typing = session = mock.Mock()
        session.finish.return_value = InjectionOutcome.INJECTED
        app._process_audio([0.0])
        session.finish.assert_called_once_with("hello there friend")
        app._inject_text.assert_not_called()
        app._mark_delivery_healthy.assert_called_once_with()
        app._show_result_and_hide.assert_called_once_with(True)
        self.assertIsNone(app._live_typing)

    def test_failed_live_typing_reports_failure_once(self):
        app = self._app(transcript="")
        app._live_typing = session = mock.Mock()
        session.finish.return_value = InjectionOutcome.FAILED
        app._process_audio([0.0])
        app._notify_user.assert_called_once()
        app._mark_delivery_healthy.assert_not_called()
        app.audio_manager.play_error_sound.assert_not_called()
        app._show_result_and_hide.assert_called_once_with(False)

    def test_nothing_typed_falls_back_to_the_normal_path(self):
        app = self._app(transcript="Thank you.")
        app._live_typing = session = mock.Mock()
        session.finish.return_value = None
        app._process_audio([0.0])
        app._inject_text.assert_not_called()  # a phantom: dropped like any other
        app.audio_manager.play_error_sound.assert_called_once()

    def test_busy_delivery_records_without_live_typing(self):
        app = self._app()
        injector = make_injector()
        app.text_injector = injector
        held, release = threading.Event(), threading.Event()

        def paste():  # the previous dictation is still pasting
            with injector._delivery_lock:
                held.set()
                release.wait(5)

        worker = threading.Thread(target=paste)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(release.set)
        held.wait(5)
        app._open_live_typing()
        self.assertIsNone(app._live_typing)
        self.assertFalse(injector.stream_busy())

    def test_cleanup_cancels_live_typing(self):
        app = self._app()
        app._live_typing = session = mock.Mock()
        app._cancel_live_typing()
        session.cancel.assert_called_once_with()
        self.assertIsNone(app._live_typing)

    def test_listener_forwards_only_to_the_open_session(self):
        app = self._app()
        app._on_live_text("a", "b")  # no session: ignored
        app._live_typing = session = mock.Mock()
        app._on_live_text("a", "b")
        session.on_live_text.assert_called_once_with("a", "b")


class LiveTypingConflictTests(unittest.TestCase):
    NEMO = {"transcription_backend": "realtime-ws", "websocket_provider": "custom",
            "websocket_live_text": "append_only"}

    def conflict(self, **overrides):
        values = {**self.NEMO, **overrides}
        return live_typing_conflict(lambda key, default=None: values.get(key, default))

    def test_append_only_transcription_has_no_conflict(self):
        self.assertIsNone(self.conflict())
        self.assertIsNone(self.conflict(recording_mode="push_to_talk", post_transcription_hook="  "))

    def test_each_whole_dictation_consumer_is_named(self):
        self.assertEqual(self.conflict(recording_mode="continuous"), "continuous needs the whole dictation")
        self.assertEqual(self.conflict(recording_mode="long_form"), "long_form needs the whole dictation")
        self.assertIn("post_transcription_hook", self.conflict(post_transcription_hook="cat"))
        self.assertEqual(self.conflict(realtime_mode="converse"), "converse mode")
        self.assertEqual(self.conflict(websocket_live_text="revisable"), "this model may revise words")
        self.assertIn("realtime-ws", self.conflict(transcription_backend="onnx-asr"))


class BackendSupportTests(unittest.TestCase):
    def _backend(self, values):
        config = types.SimpleNamespace(get_setting=lambda key, default=None: values.get(key, default))
        return RealtimeWsBackend(types.SimpleNamespace(config=config, _realtime_partial_callback=None))

    def test_only_append_only_transcription_supports_live_typing(self):
        nemo = {"websocket_provider": "custom", "websocket_live_text": "append_only"}
        self.assertTrue(self._backend(nemo).live_typing_supported)
        self.assertFalse(self._backend({**nemo, "realtime_mode": "converse"}).live_typing_supported)
        self.assertFalse(self._backend({**nemo, "websocket_live_text": "revisable"}).live_typing_supported)
        self.assertFalse(self._backend({"websocket_provider": "openai",
                                        "websocket_model": "gpt-live-transcribe"}).live_typing_supported)


if __name__ == "__main__":
    unittest.main()
