import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from text_script import (
    boundaried_pattern,
    ends_with_no_space_script,
    is_no_space_char,
    join_segments,
    needs_word_boundaries,
)


class NoSpaceScriptTests(unittest.TestCase):
    def test_cjk_characters_are_no_space(self):
        for char in ("中", "日", "あ", "ア", "한", "。", "、", "，", "！", "「"):
            self.assertTrue(is_no_space_char(char), char)

    def test_spaced_scripts_are_not_no_space(self):
        for char in ("a", "Z", "0", "ß", "é", "ж", "α", "ก", ".", " ", "_"):
            self.assertFalse(is_no_space_char(char), char)

    def test_less_common_cjk_blocks_are_covered(self):
        for char in ("ｱ", "ﾊ", "ㄅ", "ㄱ", "ㇰ", "\U00020000"):
            self.assertTrue(is_no_space_char(char), repr(char))

    def test_ends_with_no_space_script(self):
        self.assertTrue(ends_with_no_space_script("你好世界"))
        self.assertTrue(ends_with_no_space_script("こんにちは。"))
        self.assertTrue(ends_with_no_space_script("안녕하세요"))
        self.assertTrue(ends_with_no_space_script("你好世界   "))
        self.assertFalse(ends_with_no_space_script("hello there"))
        self.assertFalse(ends_with_no_space_script("我用的是 Python"))
        self.assertFalse(ends_with_no_space_script(""))
        self.assertFalse(ends_with_no_space_script("   "))

    def test_documented_imperfect_cases(self):
        self.assertFalse(ends_with_no_space_script("你好."))
        self.assertTrue(ends_with_no_space_script("the character 中"))


class WordBoundaryTests(unittest.TestCase):
    def test_latin_multichar_term_is_guarded_on_both_edges(self):
        self.assertEqual(needs_word_boundaries("hello"), (True, True))
        self.assertEqual(needs_word_boundaries("Straße"), (True, True))

    def test_single_character_is_unguarded(self):
        self.assertEqual(needs_word_boundaries("ß"), (False, False))
        self.assertEqual(needs_word_boundaries("中"), (False, False))

    def test_cjk_term_is_unguarded(self):
        self.assertEqual(needs_word_boundaries("你好"), (False, False))

    def test_mixed_latin_and_cjk_terms_are_unguarded(self):
        # A Latin edge would be guarded by (?<!\w), which CJK text can never
        # satisfy — the term would silently never match mid-sentence.
        self.assertEqual(needs_word_boundaries("AI助手"), (False, False))
        self.assertEqual(needs_word_boundaries("助手AI"), (False, False))
        self.assertEqual(re.sub(boundaried_pattern("AI助手"), "X", "我用AI助手吧"), "我用X吧")
        self.assertEqual(re.sub(boundaried_pattern("助手AI"), "X", "我用助手AI吧"), "我用X吧")

    def test_punctuation_edges_are_unguarded(self):
        self.assertEqual(needs_word_boundaries("c++"), (True, False))
        self.assertEqual(needs_word_boundaries("..."), (False, False))

    def test_empty_term(self):
        self.assertEqual(needs_word_boundaries(""), (False, False))

    def test_cjk_override_now_matches_mid_sentence(self):
        self.assertEqual(
            re.sub(boundaried_pattern("你好"), "HI", "我说你好世界"), "我说HI世界"
        )

    def test_latin_term_still_respects_word_boundaries(self):
        pattern = boundaried_pattern("cat")
        self.assertEqual(re.sub(pattern, "dog", "the cat sat"), "the dog sat")
        self.assertEqual(re.sub(pattern, "dog", "concatenate"), "concatenate")

    def test_punctuation_term_matches_where_backslash_b_could_not(self):
        self.assertEqual(re.sub(boundaried_pattern("c++"), "rust", "c++ rules"), "rust rules")
        # Only the trailing edge is unguarded; the leading 'c' still needs a boundary.
        self.assertEqual(re.sub(boundaried_pattern("c++"), "rust", "abc++"), "abc++")


class JoinSegmentsTests(unittest.TestCase):
    def test_cjk_segments_join_without_spaces(self):
        self.assertEqual(join_segments(["你好", "世界"]), "你好世界")

    def test_latin_segments_keep_spaces(self):
        self.assertEqual(join_segments(["hello", "there"]), "hello there")

    def test_mixed_segments_follow_the_left_neighbour(self):
        self.assertEqual(join_segments(["你好", "world"]), "你好world")
        self.assertEqual(join_segments(["hello", "世界"]), "hello 世界")

    def test_empty_and_whitespace_parts_are_dropped(self):
        self.assertEqual(join_segments(["hello", "", "  ", "there"]), "hello there")
        self.assertEqual(join_segments([]), "")
        self.assertEqual(join_segments(None), "")

    def test_parts_are_stripped_before_joining(self):
        self.assertEqual(join_segments(["  hello  ", "there  "]), "hello there")

    def test_whisper_style_leading_spaces_do_not_survive_a_cjk_join(self):
        self.assertEqual(join_segments([" 你好", " 世界"]), "你好世界")
        self.assertEqual(join_segments([" Hello", " there"]), "Hello there")


if __name__ == "__main__":
    unittest.main()
