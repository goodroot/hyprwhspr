import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'scripts/install.sh').read_text(encoding='utf-8')
GATE = re.search(r'^if \[\[ .*\]\]; then\n    release_bootstrap "\$@"\nfi$', SCRIPT, re.MULTILINE).group(0)


class BootstrapDefaultPathTests(unittest.TestCase):
    def path_taken(self, *args, legacy=None, managed=False):
        """Evaluate the real gate with the two paths stubbed out."""
        with tempfile.TemporaryDirectory() as home:
            if managed:
                state = Path(home) / '.local/share/hyprwhspr'
                state.mkdir(parents=True)
                (state / 'current.json').write_text('{}')
            env = {'HOME': home, 'PATH': os.environ['PATH']}
            if legacy is not None:
                env['HYPRWHSPR_LEGACY_BOOTSTRAP'] = legacy
            script = f'release_bootstrap() {{ echo release; exit 0; }}\n{GATE}\necho legacy\n'
            return subprocess.run(['bash', '-c', script, 'install.sh', *args], env=env,
                                  capture_output=True, text=True, timeout=30).stdout.strip()

    def test_fresh_install_takes_the_release_path(self):
        self.assertEqual(self.path_taken(), 'release')
        self.assertEqual(self.path_taken(legacy='0'), 'release')

    def test_escape_hatch_keeps_the_clone_path(self):
        self.assertEqual(self.path_taken(legacy='1'), 'legacy')

    def test_lifecycle_flags_and_release_installs_ignore_the_escape_hatch(self):
        self.assertEqual(self.path_taken('--repair', legacy='1'), 'release')
        self.assertEqual(self.path_taken(legacy='1', managed=True), 'release')

    def test_old_opt_in_variable_is_gone(self):
        for path in ('scripts/install.sh', 'docs/MANAGED_INSTALLATION.md'):
            self.assertNotIn('HYPRWHSPR_RELEASE_BOOTSTRAP', (ROOT / path).read_text(encoding='utf-8'), path)

    def test_release_path_is_defined_before_the_gate_and_never_falls_through(self):
        self.assertLess(SCRIPT.index('release_bootstrap() {'), SCRIPT.index(GATE))
        function = SCRIPT.split('release_bootstrap() {', 1)[1].split('\n}\n', 1)[0]
        self.assertRegex(function, r'\nexec "\$python_path" ')


if __name__ == '__main__':
    unittest.main()
