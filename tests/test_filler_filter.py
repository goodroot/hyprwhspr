"""Rule-level coverage for filler removal.

Each table maps a rule in filler_filter's module docstring to the outputs it is
supposed to produce. Integration through the config and the spoken-symbol pass
lives in test_text_injector_injection.py.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib" / "src"))

from filler_filter import filter_filler_words


DEFAULTS = ["uh", "um", "er", "ah", "eh", "hmm", "hm", "mm", "mhm"]


class FillerFilterTests(unittest.TestCase):
    def assertTable(self, cases, words=None):
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(filter_filler_words(raw, words or DEFAULTS), expected)

    def test_punctuated_fillers_do_not_strand_their_marks(self):
        # Issue #242: backends that punctuate their transcripts attach the mark
        # to the filler, and deleting only the letters left ". , what about".
        self.assertTable({
            "Fair enough. Um. Uh, what about the um, second option?":
                "Fair enough. What about the second option?",
            "Uh, what about it": "What about it",
            "Um. Fair enough.": "Fair enough.",
            "That's it um.": "That's it.",
        })

    def test_separator_on_a_filler_leaves_with_it(self):
        # The comma marked the pause the filler created; the comma that belongs
        # to the sentence sits in front of the filler and stays.
        self.assertTable({
            "what about the um, second option?": "what about the second option?",
            "I said, um, no.": "I said, no.",
            "So, um, anyway": "So, anyway",
            "It was, um; complicated.": "It was, complicated.",
            "well um okay": "well okay",
        })

    def test_sentence_break_outlives_the_filler(self):
        self.assertTable({
            "Well um. Okay.": "Well. Okay.",
            # The filler's break supersedes the mark that introduced it, rather
            # than being dropped as though the comma were the sentence end.
            "Well, um. Okay.": "Well. Okay.",
            "I mean, uh. That's it.": "I mean. That's it.",
            "Right; um. Next.": "Right. Next.",
            # A break already in front of the filler wins, kept as spelled.
            "Sure... Um. Fine.": "Sure... Fine.",
            "Really?! Um?! Yes.": "Really?! Yes.",
        })

    def test_repeated_fillers_are_judged_against_surviving_text(self):
        # A second "um." must not keep a period whose sentence the first "um."
        # already took away.
        self.assertTable({
            "Um. Um. Hello.": "Hello.",
            "Um. Uh. Er. Right.": "Right.",
            "Uh, uh, what about it": "What about it",
            "So um, um, anyway": "So anyway",
            "one, um; two: um. three": "one, two. three",
        })

    def test_wrapper_pair_leaves_with_the_filler(self):
        self.assertTable({
            '"Um," he said.': "He said.",
            "(um) fine": "fine",
            "He said 'um.' loudly": "He said loudly",
            "[um] ok": "ok",
        })

    def test_padding_before_the_closer_is_part_of_the_wrapper(self):
        self.assertTable({
            "(um ) fine": "fine",
            "(um  ) fine": "fine",
            "He said (um ) fine": "He said fine",
        })
        self.assertTable({"\u300c\u90a3\u4e2a \u300d\u597d": "\u597d"}, words=["\u90a3\u4e2a"])

    def test_a_quote_closing_an_earlier_word_does_not_open_the_filler(self):
        # A quote is both an opener and a closer, so the opener has to sit
        # directly against the filler or "yes" would lend the filler its quote
        # and the sentence would lose its period.
        self.assertTable({'Say "yes" um. Now.': 'Say "yes". Now.'})

    def test_unpaired_wrapper_is_not_the_fillers_and_stays(self):
        self.assertTable({
            "(so um) fine": "(so) fine",
            "[so um] ok": "[so] ok",
            "don't um, worry": "don't worry",
            'Say "yes" um. Now.': 'Say "yes". Now.',
        })

    def test_an_unpaired_opener_survives_where_it_was_written(self):
        # The mark on a filler that opens a bracket is inside the bracket with
        # it, so it leaves with the filler rather than closing the sentence in
        # front of the bracket -- and the bracket keeps its position.
        self.assertTable({
            "(Um. Next": "(Next",
            "He said (Um. Next": "He said (Next",
            "He said (um, next": "He said (next",
        })

    def test_mismatched_delimiters_are_not_a_wrapper_pair(self):
        self.assertTable({
            "(Um] Next": "(] Next",
            "[Um) Next": "[) Next",
            "\u201cUm\u2019 Next": "\u201c\u2019 Next",
        })

    def test_full_width_and_cjk_wrapper_pairs_are_recognized(self):
        self.assertTable({"\u201cum\u201d ok": "ok", "\uff08um\uff09 \u597d": "\u597d"})
        self.assertTable({"\u300c\u90a3\u4e2a\u300d\u597d": "\u597d"}, words=["\u90a3\u4e2a"])

    def test_only_a_sentence_initial_filler_hands_over_its_capital(self):
        self.assertTable({
            "umbrella. Um. rain": "umbrella. Rain",
            # A lowercase filler was mid-sentence, so nothing is promoted.
            "see e.g. um. foo": "see e.g. foo",
        })

    def test_abbreviations_and_measurements_are_untouched(self):
        for text in ("U.S.A. today", "5mm. wide", "at 10:30. done", "umbrella"):
            with self.subTest(text=text):
                self.assertEqual(filter_filler_words(text, DEFAULTS), text)

    def test_an_all_filler_utterance_filters_to_nothing(self):
        self.assertTable({"Um. Uh.": "", "um uh": "", "Hmm...": ""})

    def test_cjk_terms_carry_their_marks_without_word_boundaries(self):
        self.assertTable({"我那个，觉得": "我觉得", "我那个觉得": "我觉得"}, words=["那个"])
        # A shorter filler listed first must not strip half of a longer one.
        self.assertTable({"我那个觉得": "我觉得"}, words=["那", "那个"])

    def test_a_line_break_is_a_boundary_not_a_gap(self):
        # word_overrides runs before filtering and may map a word to a newline,
        # so filtered text can carry line breaks. A filler opening a line has
        # nothing in front of it to punctuate, whatever the previous line ends with.
        self.assertTable({
            "hello\nUm. World": "hello\nWorld",
            "hello\num, world": "hello\nworld",
            "hello.\nUm. World": "hello.\nWorld",
            "hello\n(Um. World": "hello\n(World",
            # A mark inside the match still stands: it is on this line.
            "hello\n. Um. World": "hello\n. World",
        })

    def test_absent_or_empty_configuration_is_a_no_op(self):
        self.assertEqual(filter_filler_words("well um okay", []), "well um okay")
        self.assertEqual(filter_filler_words("well um okay", None), "well um okay")
        self.assertEqual(filter_filler_words("", DEFAULTS), "")
        self.assertEqual(filter_filler_words("well um okay", ["", None]), "well um okay")


if __name__ == "__main__":
    unittest.main()
