"""hyprwhsprApp is main.py's core class plus mixins in lib/src/app/.

A method moved between those modules keeps working only if its new module
imports every global it reads. A missing import is a NameError on the day
that path first runs, often a rare one (suspend, recovery), so resolve every
global up front instead.
"""

import ast
import builtins
import dis
import re
import types
import unittest
from pathlib import Path

from tests.test_suspend_resume_recovery import _import_main_isolated

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def _functions(member):
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    if isinstance(member, property):
        return [f for f in (member.fget, member.fset, member.fdel) if f]
    return [member] if isinstance(member, types.FunctionType) else []


def _global_reads(code):
    names = {
        ins.argval for ins in dis.get_instructions(code)
        if ins.opname == "LOAD_GLOBAL"
    }
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_reads(const)
    return names


class AppSplitIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _import_main_isolated().hyprwhsprApp
        cls.classes = [klass for klass in cls.app.__mro__ if klass is not object]

    def test_every_global_a_method_reads_resolves_in_its_module(self):
        missing = []
        for klass in self.classes:
            for name, member in vars(klass).items():
                for function in _functions(member):
                    for read in sorted(_global_reads(function.__code__)):
                        if read not in function.__globals__ and not hasattr(builtins, read):
                            missing.append(f"{klass.__name__}.{name}: {read}")
        self.assertEqual(missing, [], "import these where the method now lives")

    def test_no_member_is_defined_twice(self):
        owners = {}
        for klass in self.classes:
            for name in vars(klass):
                if name.startswith("__") and name.endswith("__") and name != "__init__":
                    continue
                owners.setdefault(name, []).append(klass.__name__)
        duplicated = {name: where for name, where in owners.items() if len(where) > 1}
        self.assertEqual(duplicated, {})

    def test_patches_on_main_target_names_main_reads(self):
        # A patch on a name main.py only imports reaches no moved method;
        # patch through the method with patch_app_global instead.
        tree = ast.parse((ROOT / "lib" / "main.py").read_text(encoding="utf-8"))
        read = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        pattern = re.compile(r"patch\.object\(\s*(?:self\.|cls\.)?main\s*,\s*[\"'](\w+)[\"']")
        stale = sorted(
            f"{path.name}: {name}"
            for path in TESTS.glob("test_*.py")
            for name in pattern.findall(path.read_text(encoding="utf-8"))
            if name not in read
        )
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
