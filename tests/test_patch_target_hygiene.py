"""Guard for test patch targets on the lib/src/cli/ package.

Tests in this suite monkeypatch attributes on CLI modules, e.g.
``mock.patch.object(uninstall, "VENV_DIR", ...)``. Such a patch only
affects code whose global namespace is that module. When a function moves
to another module, the patch still applies cleanly but reaches nothing:
the moved code reads its own module's global. Tests then pass while
testing nothing.

This guard makes that failure loud. For every name patched on a cli
package module, the target module's source must actually READ that name
somewhere — a name that is bound but never read cannot be reached by a
patch, so the moment code moves between cli modules without its test
patch targets moving along, this test fails.

Scan limitations (keep test code within them):
- CLI modules must be referenced unaliased in patch calls, either as
  ``patch.object(<module_name>, "NAME")`` or ``patch("<module_name>.NAME")``.
- Dotted targets like ``patch.object(install.shutil, "which")`` patch a
  global stdlib object, are move-safe, and are deliberately not matched.
"""

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / 'lib' / 'src'
TESTS_DIR = REPO_ROOT / 'tests'

# patch.object(uninstall, "name")  /  patch.object(keyboard, 'name')
_PATCH_OBJECT_RE = re.compile(
    r"patch\.object\(\s*(\w+)\s*,\s*[\"'](\w+)[\"']")
# patch("uninstall.name")  — string form, module path with final attr
_PATCH_STRING_RE = re.compile(
    r"patch(?:\.dict)?\(\s*[\"']([\w.]+)\.(\w+)[\"']")


def _cli_family_modules() -> dict:
    """Map unaliased module identifier -> source path for cli package modules."""
    return {
        path.stem: path
        for path in (SRC_DIR / 'cli').glob('*.py')
        if path.stem != '__init__'
    }


def _collect_patch_targets(family: dict) -> set:
    """Return {(module_identifier, patched_name)} used across the test suite."""
    targets = set()
    for test_file in TESTS_DIR.glob('test_*.py'):
        if test_file.name == Path(__file__).name:
            continue
        text = test_file.read_text()
        for module, name in _PATCH_OBJECT_RE.findall(text):
            if module in family:
                targets.add((module, name))
        for dotted, name in _PATCH_STRING_RE.findall(text):
            module = dotted.split('.')[-1]
            if module in family:
                targets.add((module, name))
    return targets


def _names_read_in_module(path: Path) -> set:
    """All identifiers the module reads (ast.Load), i.e. patches can reach them."""
    tree = ast.parse(path.read_text())
    return {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


class PatchTargetHygieneTest(unittest.TestCase):

    def test_patched_names_are_read_by_their_target_module(self):
        family = _cli_family_modules()
        targets = _collect_patch_targets(family)
        # Sanity: the scan itself works — the known patch sites must be seen.
        self.assertGreaterEqual(
            len(targets), 10,
            "Patch-target scan found suspiciously few patches on CLI modules; "
            "the regexes or test imports may have drifted.")

        stale = []
        reads_cache = {}
        for module, name in sorted(targets):
            path = family[module]
            if path not in reads_cache:
                reads_cache[path] = _names_read_in_module(path)
            if name not in reads_cache[path]:
                stale.append(f"{module}.{name}")

        self.assertEqual(
            stale, [],
            "These test patch targets are bound but never read in the target "
            "module (probably left behind by a code move — the patch reaches "
            "nothing). Re-point the tests at the module the code now lives "
            f"in: {stale}")


if __name__ == '__main__':
    unittest.main()
