import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))

import dependency_plan


def _finder(absent):
    """A find_spec stand-in where only the named modules are absent."""
    return lambda name: None if name in absent else object()


class MissingImportsTests(unittest.TestCase):
    def test_reports_core_import_absent_from_the_environment(self):
        with mock.patch.object(dependency_plan.importlib.util, 'find_spec', _finder({'soxr'})):
            self.assertEqual(dependency_plan.missing_imports('vulkan'), ('soxr',))

    def test_complete_environment_reports_nothing(self):
        with mock.patch.object(dependency_plan.importlib.util, 'find_spec', _finder(set())):
            self.assertEqual(dependency_plan.missing_imports('vulkan'), ())

    def test_checks_the_imports_of_the_selected_provider(self):
        absent = {'websocket', 'elevenlabs'}
        with mock.patch.object(dependency_plan.importlib.util, 'find_spec', _finder(absent)):
            self.assertEqual(dependency_plan.missing_imports('realtime-ws', 'openai'), ('websocket',))
            self.assertEqual(dependency_plan.missing_imports('realtime-ws', 'elevenlabs'), ('elevenlabs',))

    def test_unknown_backend_has_no_plan_to_check(self):
        with mock.patch.object(dependency_plan.importlib.util, 'find_spec', _finder({'soxr'})):
            self.assertEqual(dependency_plan.missing_imports('no-such-backend'), ())

    def test_loaded_module_without_spec_counts_as_present(self):
        def find_spec(name):
            if name == 'numpy':
                raise ValueError('numpy.__spec__ is None')
            return object()

        with mock.patch.object(dependency_plan.importlib.util, 'find_spec', find_spec), \
                mock.patch.dict(sys.modules, {'numpy': object()}):
            self.assertEqual(dependency_plan.missing_imports('vulkan'), ())


if __name__ == '__main__':
    unittest.main()
