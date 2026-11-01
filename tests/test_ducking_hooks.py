import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))
from ducking_hooks import DuckingHooks, _DeviceWorker, _PersistentDeviceWorker


class DuckingHooksTests(unittest.TestCase):
    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail('background job did not reach expected state')

    def worker(self, command, grace=1):
        worker = _DeviceWorker('test', command, grace)
        def cleanup():
            worker.close()
            worker._thread.join(4)
        self.addCleanup(cleanup)
        return worker

    def test_invalid_hooks_are_skipped_without_starting_commands(self):
        with mock.patch('ducking_hooks._DeviceWorker') as worker:
            hooks = DuckingHooks([{}, {'command': 'shell string'},
                                 {'command': ['ok'], 'restore_timeout': float('nan')},
                                 {'command': ['ok'], 'enabled': False}])
            hooks.set_recording(True)
            hooks.close()
            worker.assert_not_called()

    def test_stop_is_nonblocking_and_restores_on_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            code = (
                'import sys,pathlib; p=pathlib.Path(sys.argv[1]); '
                'line=sys.stdin.readline(); '
                'p.write_text("duck") if line else None; '
                'sys.stdin.read(); p.write_text("restored")'
            )
            worker = self.worker([sys.executable, '-c', code, str(events)])
            worker.set_recording(True)
            self.wait_for(lambda: events.exists() and events.read_text() == 'duck')
            start = time.monotonic()
            worker.set_recording(False)
            self.assertLess(time.monotonic() - start, 0.1)
            self.wait_for(lambda: events.read_text() == 'restored')
            worker.close()
            worker._thread.join(3)
            self.assertFalse(worker._thread.is_alive())

    def test_rapid_restart_serializes_restore_before_next_duck(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            code = '''import sys,time
p=sys.argv[1]
def event(text):
    with open(p, 'a') as f: f.write(text+'\\n')
if sys.stdin.readline():
    event('duck')
    sys.stdin.read()
    time.sleep(.15)
    event('restore')
'''
            worker = self.worker([sys.executable, '-c', code, str(events)])
            worker.set_recording(True)
            self.wait_for(lambda: events.exists() and 'duck' in events.read_text())
            worker.set_recording(False)
            worker.set_recording(True)
            self.wait_for(lambda: events.read_text().count('duck') == 2)
            worker.close()
            worker._thread.join(3)
            self.assertEqual(events.read_text().splitlines(), ['duck', 'restore', 'duck', 'restore'])

    def test_launch_failure_is_not_retried_until_next_recording(self):
        with mock.patch('ducking_hooks.subprocess.Popen', side_effect=OSError('offline')) as popen:
            worker = self.worker(['missing'])
            worker.set_recording(True)
            self.wait_for(lambda: popen.call_count == 1)
            worker.set_recording(True)
            time.sleep(.05)
            self.assertEqual(popen.call_count, 1)
            worker.set_recording(False)
            worker.set_recording(True)
            self.wait_for(lambda: popen.call_count == 2)
            worker.close()
            worker._thread.join(3)

    def test_hung_command_is_terminated_after_restore_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / 'ready'
            code = 'import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)'
            worker = self.worker([sys.executable, '-c', code, str(ready)])
            worker.set_recording(True)
            self.wait_for(ready.exists)
            worker.close()
            worker._thread.join(4)
            self.assertFalse(worker._thread.is_alive())

    def test_devices_have_independent_workers(self):
        with mock.patch('ducking_hooks._DeviceWorker') as worker:
            hooks = DuckingHooks([{'name': 'a', 'command': ['one']},
                                 {'name': 'b', 'command': ['two']}])
            self.assertEqual(worker.call_count, 2)
            hooks.set_recording(True)
            hooks.close()
            self.assertEqual(worker.return_value.set_recording.call_count, 2)
            self.assertEqual(worker.return_value.close.call_count, 2)

    def test_capture_state_signals_hooks_even_if_status_file_fails(self):
        tree = ast.parse((ROOT / 'lib/main.py').read_text())
        publish = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == '_write_recording_status')
        status_path = mock.Mock()
        status_path.parent.mkdir.side_effect = OSError('unwritable runtime directory')
        namespace = {'RECORDING_STATUS_FILE': status_path}
        exec(compile(ast.Module(body=[publish], type_ignores=[]), '<test>', 'exec'), namespace)
        hooks = mock.Mock()
        app = types.SimpleNamespace(ducking_hooks=hooks)
        namespace['_write_recording_status'](app, True)
        namespace['_write_recording_status'](app, False)
        self.assertEqual(hooks.set_recording.call_args_list, [mock.call(True), mock.call(False)])

    def test_capture_state_and_shutdown_are_wired_to_hooks(self):
        tree = ast.parse((ROOT / 'lib/main.py').read_text())
        functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        publish = functions['_write_recording_status']
        self.assertTrue(any(isinstance(n, ast.Attribute) and n.attr == 'set_recording'
                            for n in ast.walk(publish)))
        cleanup = functions['_cleanup']
        self.assertTrue(any(isinstance(n, ast.Constant) and n.value == 'ducking_hooks'
                            for n in ast.walk(cleanup)))


class PersistentDuckingTests(unittest.TestCase):
    wait_for = DuckingHooksTests.wait_for

    def worker(self, command):
        worker = _PersistentDeviceWorker('persistent-test', command, 1)
        def cleanup():
            worker.close()
            worker._thread.join(4)
        self.addCleanup(cleanup)
        return worker

    def test_warms_while_idle_and_reuses_connection_across_takes(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            code = '''import sys
p=sys.argv[1]
def event(text):
    with open(p, 'a') as f: f.write(text+'\\n')
event('start')
print('ready', flush=True)
for line in sys.stdin:
    event(line.strip())
    print('ok', flush=True)
event('restored')
'''
            worker = self.worker([sys.executable, '-u', '-c', code, str(events)])
            self.wait_for(lambda: events.exists() and 'idle' in events.read_text())
            for take in range(2):
                worker.set_recording(True)
                self.wait_for(lambda: events.read_text().count('recording') >= take + 1)
                worker.set_recording(False)
                self.wait_for(lambda: events.read_text().count('idle') >= take + 2)
            worker.close()
            worker._thread.join(3)
            lines = events.read_text().splitlines()
            self.assertEqual(lines.count('start'), 1)
            self.assertEqual(lines[-1], 'restored')

    def test_slow_connection_uses_current_state_not_stale_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            code = '''import sys,time
p=sys.argv[1]
with open(p, 'w') as f: f.write('starting\\n')
time.sleep(.2)
print('ready', flush=True)
for line in sys.stdin:
    with open(p, 'a') as f: f.write(line)
    print('ok', flush=True)
'''
            worker = self.worker([sys.executable, '-u', '-c', code, str(events)])
            worker.set_recording(True)
            self.wait_for(events.exists)
            worker.set_recording(False)
            self.wait_for(lambda: 'idle' in events.read_text())
            worker.close()
            worker._thread.join(3)
            self.assertNotIn('recording', events.read_text())

    def test_missing_ack_triggers_cleanup_without_blocking_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            code = '''import sys,pathlib
p=pathlib.Path(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
p.write_text('waiting')
sys.stdin.read()
p.write_text('restored')
'''
            with mock.patch.object(_PersistentDeviceWorker, 'ACK_TIMEOUT', .15):
                worker = self.worker([sys.executable, '-u', '-c', code, str(events)])
                self.wait_for(lambda: events.exists() and events.read_text() == 'waiting')
                start = time.monotonic()
                worker.set_recording(False)
                self.assertLess(time.monotonic() - start, .1)
                self.wait_for(lambda: events.read_text() == 'restored')
                worker.close()
                worker._thread.join(3)

    def test_lost_connection_reconnects_with_latest_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / 'events'
            marker = Path(tmp) / 'connected-once'
            code = '''import pathlib,sys
p=pathlib.Path(sys.argv[1]); marker=pathlib.Path(sys.argv[2])
first=not marker.exists()
marker.touch()
print('ready', flush=True)
for line in sys.stdin:
    with p.open('a') as f: f.write(line)
    if first and line.strip() == 'recording': sys.exit(1)
    print('ok', flush=True)
'''
            worker = self.worker([sys.executable, '-u', '-c', code, str(events), str(marker)])
            worker.set_recording(True)
            self.wait_for(lambda: events.exists() and 'recording' in events.read_text())
            worker.set_recording(False)
            self.wait_for(lambda: 'idle' in events.read_text())
            worker.close()
            worker._thread.join(3)
            self.assertEqual(events.read_text().splitlines().count('recording'), 1)

    def test_offline_start_retries_without_a_hotkey(self):
        # A failed cold connection must recover even while idle.
        with mock.patch('ducking_hooks.subprocess.Popen', side_effect=OSError('offline')) as popen:
            worker = self.worker(['missing'])
            self.wait_for(lambda: popen.call_count >= 2)
            worker.close()
            worker._thread.join(3)


@unittest.skipUnless(all(shutil.which(cmd) for cmd in ('bash', 'flock', 'timeout')), 'requires shell utilities')
class TermuxHookTests(unittest.TestCase):
    def test_remote_script_restores_saved_volume_on_eof(self):
        spec = importlib.util.spec_from_file_location('duck_termux', ROOT / 'scripts/duck-termux.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / 'events'
            for name, body in {
                'termux-volume': 'import sys,os,json\nfrom pathlib import Path\nif len(sys.argv)==1: print(json.dumps([{"stream":"music","volume":14}]))\nelse:\n with open(os.environ["EVENTS"],"a") as f: f.write(" ".join(sys.argv[1:])+"\\n")\n',
                'jq': 'import sys\nsys.stdin.read()\nprint(14)\n',
                'su': 'import os,sys,shlex\nargs=shlex.split(sys.argv[-1])\nvalue=args[args.index("--set")+1] if "--set" in args else "14"\nif "--set" in args:\n with open(os.environ["EVENTS"],"a") as f: f.write("music "+value+"\\n")\nprint("[V] volume is "+value+" in range [0..15]")\n',
            }.items():
                path = root / name
                path.write_text(f'#!{sys.executable}\n' + body)
                path.chmod(0o700)
            env = dict(os.environ, TMPDIR=tmp, EVENTS=str(events), PATH=tmp + os.pathsep + os.environ['PATH'])
            result = subprocess.run(
                ['bash', '-c', module.REMOTE_SCRIPT, 'test', 'music', '2'],
                input='recording\n', text=True, capture_output=True, env=env, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['music 2', 'music 14'])
            self.assertFalse((root / 'hyprwhspr-duck/music.volume').exists())
            events.unlink()
            # Connection completing after stop must not duck at all.
            result = subprocess.run(
                ['bash', '-c', module.REMOTE_SCRIPT, 'test', 'music', '2'],
                input='', text=True, capture_output=True, env=env, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(events.exists())
            # Keep stdin open but stop heartbeats: the remote lease must restore.
            process = subprocess.Popen(
                ['bash', '-c', module.REMOTE_SCRIPT.replace('-t 6', '-t 0.2'),
                 'test', 'music', '2'],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env,
            )
            try:
                process.stdin.write(b'recording\n')
                process.stdin.flush()
                process.wait(timeout=5)
                self.assertEqual(process.returncode, 0)
                self.assertEqual(events.read_text().splitlines(), ['music 2', 'music 14'])
            finally:
                process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
            # A journal left by a killed session is recovered before ducking.
            events.unlink()
            (root / 'hyprwhspr-duck/music.volume').write_text('9\n')
            result = subprocess.run(
                ['bash', '-c', module.REMOTE_SCRIPT, 'test', 'music', '2'],
                input='recording\n', text=True, capture_output=True, env=env, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(events.read_text().splitlines(), ['music 9', 'music 2', 'music 14'])
            events.unlink()
            # Warm root-backed protocol: idle is harmless, heartbeat is
            # idempotent, and explicit idle restores without closing SSH.
            result = subprocess.run(
                ['bash', '-c', module.REMOTE_SCRIPT, 'test', 'music', '2',
                 'android-root', 'persistent'],
                input='idle\nrecording\nrecording\nidle\nidle\n', text=True,
                capture_output=True, env=env, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ['ready'] + ['ok'] * 5)
            self.assertEqual(events.read_text().splitlines(), ['music 2', 'music 14'])


if __name__ == '__main__':
    unittest.main()
