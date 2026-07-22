import unittest

from mic_osd.transcript_preview import (
    PillTranscriptAnimator,
    PillTranscriptConfig,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PillTranscriptConfigTests(unittest.TestCase):
    def test_loads_and_clamps_public_settings(self):
        values = {
            "mic_osd_pill_transcript_enabled": True,
            "mic_osd_pill_transcript_word_limit": 99,
            "mic_osd_pill_transcript_idle_timeout_ms": 800,
        }
        config = PillTranscriptConfig.from_getter(
            lambda key, default=None: values.get(key, default)
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.word_limit, 12)
        self.assertEqual(config.idle_timeout_seconds, 0.8)

    def test_preview_is_opt_in_by_default(self):
        config = PillTranscriptConfig.from_getter(
            lambda _key, default=None: default
        )
        self.assertFalse(config.enabled)


class PillTranscriptAnimatorTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.config = PillTranscriptConfig(
            enabled=True,
            word_limit=4,
            idle_timeout_seconds=1.4,
        )
        self.animator = PillTranscriptAnimator(self.config, clock=self.clock)

    def test_keeps_only_latest_four_words(self):
        self.animator.set_text("zero one two three four", now=0.0)
        self.assertEqual(
            self.animator.current_words,
            ("one", "two", "three", "four"),
        )

    def test_character_growth_does_not_restart_animation(self):
        self.animator.set_text("trans", now=0.0)
        transition_started_at = self.animator.transition_started_at
        for index, text in enumerate(
            ("transc", "transcr", "transcri", "transcrip", "transcript"),
            start=1,
        ):
            self.animator.set_text(text, now=index * 0.03)
        self.assertEqual(
            self.animator.transition_started_at,
            transition_started_at,
        )
        self.assertEqual(self.animator.current_words, ("transcript",))

    def test_250_wpm_stream_never_queues_and_converges_on_latest_words(self):
        completed = []
        now = 0.0
        for word_index in range(12):
            target = f"word{word_index}"
            for char_index in range(1, len(target) + 1):
                provisional = completed + [target[:char_index]]
                self.animator.set_text(" ".join(provisional), now=now)
                now += 0.04
            completed.append(target)
            now = max(now, (word_index + 1) * 0.24)
        self.assertEqual(
            self.animator.current_words,
            tuple(completed[-4:]),
        )
        self.assertFalse(hasattr(self.animator, "pending_transitions"))

    def test_rolling_window_keeps_overlapping_words_stable(self):
        self.animator.set_text("one two three four", now=0.0)
        self.animator.set_text("two three four five", now=0.24)
        self.assertEqual(
            self.animator.matches,
            ((1, 0), (2, 1), (3, 2)),
        )
        frame = self.animator.frame(now=0.34)
        stable = [word for word in frame.words if word.matched_from is not None]
        incoming = [
            word
            for word in frame.words
            if word.source == "current" and word.matched_from is None
        ]
        outgoing = [word for word in frame.words if word.source == "previous"]
        self.assertEqual([word.text for word in stable], ["two", "three", "four"])
        self.assertEqual([word.text for word in incoming], ["five"])
        self.assertEqual([word.text for word in outgoing], ["one"])

    def test_idle_timeout_starts_upward_exit(self):
        self.animator.set_text("last four spoken words", now=0.0)
        frame = self.animator.frame(now=1.41)
        self.assertEqual(frame.current_words, ())
        self.assertTrue(frame.previous_words)
        later = self.animator.frame(now=1.51)
        outgoing = [word for word in later.words if word.source == "previous"]
        self.assertTrue(all(word.y_offset < 0 for word in outgoing))
        self.assertTrue(all(0 < word.alpha < 1 for word in outgoing))

    def test_disabled_preview_renders_nothing(self):
        animator = PillTranscriptAnimator(
            PillTranscriptConfig(),
            clock=self.clock,
        )
        animator.set_text("should stay hidden", now=0.0)
        self.assertEqual(animator.frame(now=0.1).words, ())


if __name__ == "__main__":
    unittest.main()
