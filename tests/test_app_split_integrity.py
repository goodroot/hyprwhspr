"""hyprwhsprApp is main.py's core class plus mixins in lib/src/app/.

A method moved between those modules keeps working only if its new module
imports every global it reads, as the same object the others see. A missing
import is a NameError on the day that path first runs, often a rare one
(suspend, recovery), so resolve every global up front instead.
"""

import builtins
import re
import unittest
from pathlib import Path

from tests.test_suspend_resume_recovery import _import_main_isolated, app_functions, global_reads

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


class AppSplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _import_main_isolated().hyprwhsprApp
        cls.functions = list(app_functions(cls.app))

    def test_the_scan_covers_the_whole_app(self):
        # Floor: guards over an empty scan would pass vacuously.
        self.assertGreaterEqual(len(self.functions), 70)
        self.assertGreaterEqual(len({f.__module__ for _k, _n, f in self.functions}), 7)

    def test_every_global_a_method_reads_resolves_in_its_module(self):
        missing = [
            f"{klass.__name__}.{name}: {read}"
            for klass, name, function in self.functions
            for read in sorted(global_reads(function.__code__))
            if read not in function.__globals__ and not hasattr(builtins, read)
        ]
        self.assertEqual(missing, [], "import these where the method now lives")

    def test_modules_share_one_object_per_global(self):
        # `from src.paths import X` next to `from paths import X` would load
        # a second module copy with its own state; every binding must agree.
        seen = {}
        for _klass, _name, function in self.functions:
            for read in global_reads(function.__code__):
                if read in function.__globals__:
                    seen.setdefault(read, {})[id(function.__globals__[read])] = function.__module__
        split = {name: sorted(where.values()) for name, where in seen.items() if len(where) > 1}
        self.assertEqual(split, {})

    def test_annotations_resolve(self):
        # Python 3.14 evaluates annotations lazily; older interpreters do it
        # when the module loads, so an unimported annotation name must fail here.
        for klass, name, function in self.functions:
            with self.subTest(member=f"{klass.__name__}.{name}"):
                function.__annotations__

    def test_no_member_is_defined_twice(self):
        owners = {}
        for klass in self.app.__mro__[:-1]:
            for name in vars(klass):
                if name.startswith("__") and name.endswith("__") and name != "__init__":
                    continue
                owners.setdefault(name, []).append(klass.__name__)
        duplicated = {name: where for name, where in owners.items() if len(where) > 1}
        self.assertEqual(duplicated, {})

    def test_tests_never_patch_main_globals_by_name(self):
        # A name patched on main misses every mixin that reads its own copy;
        # use patch_app_global, which patches each module that reads it.
        patterns = (
            re.compile(r"patch\.object\(\s*(?:self\.|cls\.)?main\s*,\s*[\"'](\w+)[\"']"),
            re.compile(r"patch\(\s*[\"']main\.(\w+)[\"']"),
        )
        found = sorted(
            f"{path.name}: {name}"
            for path in TESTS.glob("test_*.py")
            for pattern in patterns
            for name in pattern.findall(path.read_text(encoding="utf-8"))
        )
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
