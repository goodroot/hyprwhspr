"""Exercise CI orchestration without installing packages or touching the network."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'dependency_smoke', ROOT / '.github/scripts/smoke_python_dependencies.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class DependencySmokeTests(unittest.TestCase):
    def test_matrix_includes_every_runtime_plan_and_cli_at_ceiling(self):
        entries = smoke.matrix()['include']
        self.assertEqual({entry['plan'] for entry in entries}, {'cli', *smoke.PLAN_SPECS})
        self.assertEqual(len(entries), len(smoke.PLAN_SPECS) + 1)
        for entry in entries:
            expected = '3.14' if entry['plan'] in (
                'onnx-cpu', 'onnx-gpu', 'faster-cpu', 'faster-cuda') else '3.15'
            self.assertEqual(entry['python'], expected)

    def test_install_check_and_real_imports_use_isolated_environment(self):
        for key, extra in [('cli', 'rich'), ('onnx-cpu', 'onnxruntime'),
                           ('faster-cuda', 'ctranslate2'), ('pywhispercpp', '_pywhispercpp')]:
            with self.subTest(key=key), mock.patch.object(smoke.subprocess, 'run') as run:
                smoke.smoke(key)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 5)
            self.assertEqual(commands[0][1:4], ['-I', '-m', 'venv'])
            self.assertNotIn('--system-site-packages', commands[0])
            self.assertEqual(commands[1][-3:], ['install', '-r', str(ROOT / smoke.plans()[key][0])])
            self.assertEqual(commands[2][-1], 'check')
            self.assertIn(f'import {extra}\n', commands[3][-1])
            for name in smoke.plans()[key][1]:
                self.assertIn(f'import {name}\n', commands[3][-1])
            self.assertEqual(commands[4][-2:], ['list', '--format=json'])
            for call in run.call_args_list[1:]:
                self.assertEqual(call.args[0][0], str(Path(commands[0][-1]) / 'bin/python'))
                self.assertEqual(call.kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertFalse(Path(commands[0][-1]).parent.exists())

    def test_failure_is_propagated_and_package_list_retained(self):
        for failing_step in (1, 2, 3):
            error = subprocess.CalledProcessError(1, ['fixture'])
            calls = [None] * failing_step + [error, None]
            with self.subTest(step=failing_step), \
                    mock.patch.object(smoke.subprocess, 'run', side_effect=calls) as run:
                with self.assertRaises(subprocess.CalledProcessError) as result:
                    smoke.smoke('rest')
            self.assertIs(result.exception, error)
            self.assertEqual(run.call_count, failing_step + 2)
            self.assertEqual(run.call_args.args[0][-2:], ['list', '--format=json'])
            self.assertFalse(run.call_args.kwargs['check'])


if __name__ == '__main__':
    unittest.main()
