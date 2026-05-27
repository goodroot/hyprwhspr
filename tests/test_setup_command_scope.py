import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))


def _stub_if_missing(name, **attrs):
    """Install a stub module only if the real one cannot be imported.

    cli_commands transitively imports global_shortcuts -> evdev (a hard import)
    and rich (prompt/console/table). On a dev machine these are installed and we
    must NOT shadow them; in a bare CI we fall back to a minimal stub so the
    import still works.
    """
    try:
        importlib.import_module(name)
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_if_missing("rich")
_stub_if_missing("rich.prompt", Prompt=object, Confirm=object)
_stub_if_missing("rich.console", Console=object)
_stub_if_missing("rich.table", Table=object)
_stub_if_missing(
    "evdev",
    InputDevice=object,
    categorize=lambda *a, **k: None,
    ecodes=types.SimpleNamespace(),
    UInput=object,
)

import cli_commands  # noqa: E402


class SetupCommandScopeTests(unittest.TestCase):
    def test_run_command_is_not_function_local(self):
        # Regression: a conditional local re-import of run_command made it a
        # function-local throughout setup_command, so non-cloud GNOME setups hit
        # UnboundLocalError at the gsettings (toolkit-accessibility) call. It must
        # resolve to the module-level helper instead.
        self.assertNotIn(
            "run_command", cli_commands.setup_command.__code__.co_varnames
        )


if __name__ == "__main__":
    unittest.main()
