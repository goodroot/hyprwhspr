import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import keyboard_layout


def _keymap(overrides=None):
    """A keymap covering every ASCII key, with per-key placeholder keysyms."""
    overrides = overrides or {}
    keys = '\n'.join(
        f'\t\tkey <{name}> {{\t[ {overrides.get(name, f"k_{name}, K_{name}")} ] }};'
        for name in sorted(keyboard_layout._ASCII_KEY_NAMES)
    )
    return 'xkb_keymap {\n\txkb_symbols "(unnamed)" {\n' + keys + (
        '\n\t\tkey <CAPS> {\t[ Caps_Lock ] };\n\t};\n};\n'
    )


US_KEYMAP = _keymap({'SPCE': 'space'})

# Polish: same ASCII positions, its own characters on the AltGr levels.
PL_KEYMAP = _keymap({
    'SPCE': 'space, space, nobreakspace, nobreakspace',
    'AD01': 'k_AD01, K_AD01, Greek_pi, Greek_OMEGA',
})

# German: one key carries another's keysyms, as z/y are swapped.
DE_KEYMAP = _keymap({'SPCE': 'space', 'AB01': 'k_AD06, K_AD06'})


class KeyboardLayoutTests(unittest.TestCase):
    def setUp(self):
        keyboard_layout.reset_caches()
        self.addCleanup(keyboard_layout.reset_caches)

    def _with_keymaps(self, active_keymap):
        def fake_compile(layout, variant='', timeout=2.0):
            return US_KEYMAP if (layout, variant) == ('us', '') else active_keymap

        return mock.patch.object(keyboard_layout, "compile_keymap", fake_compile)

    def test_layout_adding_altgr_levels_matches_us(self):
        with self._with_keymaps(PL_KEYMAP):
            self.assertIs(keyboard_layout.ascii_positions_match_us("pl"), True)

    def test_layout_moving_a_key_does_not_match_us(self):
        with self._with_keymaps(DE_KEYMAP):
            self.assertIs(keyboard_layout.ascii_positions_match_us("de"), False)

    def test_single_level_key_shifts_to_itself(self):
        """US space has one level, pl's has four: not a difference that matters."""
        levels = keyboard_layout._parse_ascii_levels(US_KEYMAP)
        self.assertEqual(levels["SPCE"], ("space", "space"))

    def test_non_ascii_keys_are_ignored(self):
        self.assertNotIn("CAPS", keyboard_layout._parse_ascii_levels(US_KEYMAP))

    def test_numeric_index_brackets_are_not_read_as_keysyms(self):
        """libxkbcommon writes `symbols[1]=` for any key with an explicit type."""
        keymap = """
	xkb_symbols "(unnamed)" {
		key <AD01> {
			type= "FOUR_LEVEL",
			symbols[1]= [               q,               Q,              oe,              OE ]
		};
	};
"""
        self.assertEqual(
            keyboard_layout._parse_ascii_levels(keymap), {"AD01": ("q", "Q")}
        )

    def test_group_brackets_are_not_read_as_keysyms(self):
        keymap = """
	xkb_symbols "(unnamed)" {
		key <AD01> {
			type[group1]= "ALPHABETIC",
			symbols[Group1]= [               q,               Q ]
		};
	};
"""
        self.assertEqual(keyboard_layout._parse_ascii_levels(keymap), {"AD01": ("q", "Q")})

    def test_uncompilable_keymap_is_unanswerable(self):
        with mock.patch.object(keyboard_layout, "compile_keymap", return_value=None):
            self.assertIsNone(keyboard_layout.ascii_positions_match_us("zz"))

    def test_compile_failure_is_cached_not_retried(self):
        failed = types.SimpleNamespace(returncode=1, stdout="")

        with mock.patch("keyboard_layout.subprocess.run", return_value=failed) as run:
            self.assertIsNone(keyboard_layout.compile_keymap("zz"))
            self.assertIsNone(keyboard_layout.compile_keymap("zz"))

        run.assert_called_once()

    def test_transient_compile_failure_is_retried_after_a_pause(self):
        """A timeout must not disable layout awareness for the process lifetime,
        nor make every injection pay for it again."""
        completed = types.SimpleNamespace(returncode=0, stdout=US_KEYMAP)

        with (
            mock.patch(
                "keyboard_layout.subprocess.run",
                side_effect=[subprocess.TimeoutExpired("xkbcli", 2.0), completed],
            ) as run,
            mock.patch("keyboard_layout.time.monotonic", side_effect=[10.0, 10.5, 99.0]),
        ):
            self.assertIsNone(keyboard_layout.compile_keymap("us"))
            self.assertIsNone(keyboard_layout.compile_keymap("us"))  # inside the pause
            self.assertEqual(run.call_count, 1)
            self.assertEqual(keyboard_layout.compile_keymap("us"), US_KEYMAP)

    def test_unanswered_match_is_retried(self):
        with mock.patch.object(keyboard_layout, "compile_keymap", return_value=None):
            self.assertIsNone(keyboard_layout.ascii_positions_match_us("pl"))

        with self._with_keymaps(PL_KEYMAP):
            self.assertIs(keyboard_layout.ascii_positions_match_us("pl"), True)

    def test_missing_xkbcli_is_cached_not_retried(self):
        with mock.patch(
            "keyboard_layout.subprocess.run", side_effect=FileNotFoundError
        ) as run:
            self.assertIsNone(keyboard_layout.compile_keymap("us"))
            self.assertIsNone(keyboard_layout.compile_keymap("us"))

        run.assert_called_once()

    def test_incomplete_us_reference_gives_no_answer(self):
        """Half a reference could call a moved key type-safe; refuse instead."""
        trimmed = US_KEYMAP.replace("key <AB01>", "key <IGNORED>")

        with mock.patch.object(keyboard_layout, "compile_keymap", return_value=trimmed):
            self.assertIsNone(keyboard_layout.ascii_positions_match_us("de"))

    def test_compile_keymap_passes_variant(self):
        completed = types.SimpleNamespace(returncode=0, stdout=US_KEYMAP)

        with mock.patch("keyboard_layout.subprocess.run", return_value=completed) as run:
            keyboard_layout.compile_keymap("pl", "dvorak")

        self.assertEqual(
            run.call_args[0][0],
            ["xkbcli", "compile-keymap", "--layout", "pl", "--variant", "dvorak"],
        )


@unittest.skipUnless(shutil.which("xkbcli"), "xkbcli not installed")
class RealKeymapTests(unittest.TestCase):
    """The claim this all rests on, checked against xkeyboard-config itself."""

    def setUp(self):
        keyboard_layout.reset_caches()
        self.addCleanup(keyboard_layout.reset_caches)

    def test_real_us_reference_parses_completely(self):
        """The completeness guard is only useful if the real keymap satisfies it."""
        levels = keyboard_layout._parse_ascii_levels(keyboard_layout.compile_keymap("us"))
        self.assertEqual(len(levels), len(keyboard_layout._ASCII_KEY_NAMES))

    def test_us_ascii_compatible_layouts(self):
        # us(mac) writes its keysyms in long form; it is still plain US ASCII.
        for layout, variant in [
            ("us", ""), ("us", "mac"), ("us", "euro"), ("pl", ""), ("ro", ""), ("tr", "alt")
        ]:
            with self.subTest(layout=layout, variant=variant):
                self.assertIs(
                    keyboard_layout.ascii_positions_match_us(layout, variant), True
                )

    def test_layouts_that_move_ascii(self):
        for layout, variant in [
            ("de", ""),
            ("fr", ""),
            ("gb", ""),
            ("us", "intl"),
            ("us", "dvorak"),
            ("pl", "qwertz"),
            ("pl", "dvorak"),
            ("pl", "dvp"),
        ]:
            with self.subTest(layout=layout, variant=variant):
                self.assertIs(
                    keyboard_layout.ascii_positions_match_us(layout, variant), False
                )


if __name__ == "__main__":
    unittest.main()
