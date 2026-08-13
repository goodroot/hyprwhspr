import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from hallucination import DEFAULT_HALLUCINATION_MARKERS, is_hallucination


class HallucinationTests(unittest.TestCase):
    def test_every_default_marker_is_caught(self):
        for marker in DEFAULT_HALLUCINATION_MARKERS:
            self.assertTrue(is_hallucination(marker), marker)

    def test_normalization_is_the_same_in_every_mode(self):
        # Continuous mode used to strip only '.!? ', so these pasted there while
        # being filtered in toggle and long-form.
        for text in ("[Silence]", "blank_audio", "Thank you.", "(music playing)", "  you  "):
            self.assertTrue(is_hallucination(text), text)

    def test_music_note_prefix_is_caught(self):
        self.assertTrue(is_hallucination("♪ la la la ♪"))

    def test_real_transcriptions_pass_through(self):
        for text in ("hello there", "thank you for the coffee", "you know what I mean"):
            self.assertFalse(is_hallucination(text), text)

    def test_empty_input_is_safe(self):
        self.assertFalse(is_hallucination(""))
        self.assertFalse(is_hallucination(None))

    def test_custom_markers_replace_the_defaults(self):
        markers = ["silence"]
        self.assertTrue(is_hallucination("silence", markers))
        # The point of the setting: 'you' is no longer swallowed.
        self.assertFalse(is_hallucination("you", markers))

    def test_hand_written_markers_are_normalized_too(self):
        # The docs promise case/bracket/underscore-insensitive matching; that has
        # to hold for user-supplied entries, not just the pre-normalized defaults.
        self.assertTrue(is_hallucination("Thank you.", ["Thank You"]))
        self.assertTrue(is_hallucination("silence", ["[Silence]"]))
        self.assertTrue(is_hallucination("[Background noise]", ["Background_Noise"]))

    def test_empty_marker_entries_are_ignored(self):
        self.assertFalse(is_hallucination("anything", ["", None]))

    def test_accepts_any_iterable_of_markers(self):
        for markers in ({"silence"}, ["silence"], ("silence",)):
            self.assertTrue(is_hallucination("Silence.", markers), type(markers))


if __name__ == "__main__":
    unittest.main()
