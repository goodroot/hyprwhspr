"""service_log.log is print(x, flush=True) as one atomic write, plus a guard on converted modules."""
import ast
import contextlib
import io
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from service_log import log  # noqa: E402

# Service modules whose line output goes through log(); print() is reserved
# for the few deliberate stderr writes (file=...).
CONVERTED = [
    'lib/src/whisper_manager.py',
    'lib/src/audio_capture.py',
    'lib/src/global_shortcuts.py',
    'lib/src/text_injector.py',
    'lib/src/backends/base.py',
    'lib/src/backends/cohere_backend.py',
    'lib/src/backends/faster_whisper_backend.py',
    'lib/src/backends/onnx_asr_backend.py',
    'lib/src/backends/pywhispercpp_backend.py',
    'lib/src/backends/qwen3_asr_backend.py',
    'lib/src/backends/realtime_ws_backend.py',
    'lib/src/backends/rest_api_backend.py',
    'lib/main.py',
    # hyprwhsprApp's mixins, split out of main.py
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / 'lib' / 'src' / 'app').glob('*.py')),
]


class ChunkRecorder(io.StringIO):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def write(self, text):
        self.chunks.append(text)
        return super().write(text)


class ServiceLogTests(unittest.TestCase):
    def test_output_matches_print_with_flush(self):
        for message in ('[TAG] hello', '', 42, None, 'multi\nline'):
            expected, actual = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(expected):
                print(message, flush=True)
            with contextlib.redirect_stdout(actual):
                log(message)
            self.assertEqual(actual.getvalue(), expected.getvalue(), repr(message))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            log()
        self.assertEqual(out.getvalue(), '\n')

    def test_each_line_is_a_single_write_even_across_threads(self):
        out = ChunkRecorder()
        with contextlib.redirect_stdout(out):
            threads = [threading.Thread(target=lambda n=n: [log(f'[T{n}] line') for _ in range(200)])
                       for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(out.chunks), 1600)
        self.assertTrue(all(c.endswith(' line\n') and c.count('\n') == 1 for c in out.chunks))

    def test_detached_stdout_is_ignored(self):
        with contextlib.redirect_stdout(None):
            log('nowhere')

    def test_converted_modules_do_not_reintroduce_plain_print(self):
        for relative in CONVERTED:
            tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
            plain = [node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'print' and not any(k.arg == 'file' for k in node.keywords)]
            self.assertEqual(plain, [], f'{relative}: use service_log.log for line output')


if __name__ == '__main__':
    unittest.main()
