"""Waybar notification controls with no desktop, service, or user config access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / 'config/hyprland/hyprwhspr-tray.sh').read_text()
# Only function definitions: never execute the script's host probes or dispatcher.
FUNCTIONS = SCRIPT[SCRIPT.index('# Click results obey'):SCRIPT.index('# Act on a recovery result')]


class WaybarNotificationTests(unittest.TestCase):
    def run_shell(self, command, level='all'):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'hyprwhspr'
            config.mkdir()
            (config / 'config.json').write_text(json.dumps({'notification_level': level}))
            env = dict(os.environ, XDG_CONFIG_HOME=tmp, SYSTEM_PYTHON=sys.executable)
            result = subprocess.run(['bash', '-c', FUNCTIONS + '\n' + command],
                                    env=env, capture_output=True, text=True, timeout=5)
            return result

    def test_off_blocks_the_actual_notification_sender(self):
        for level, expected in [('all', 'sent'), ('action_required', 'sent'), ('off', '')]:
            with self.subTest(level=level):
                result = self.run_shell('''
notify-send() { printf sent; }
gdbus() { printf unexpected; }
show_notification hyprwhspr failure critical
''', level)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_command_success_and_failure_are_reported_honestly(self):
        for action, past in [('start', 'Started'), ('stop', 'Stopped'), ('restart', 'Restarted')]:
            for status in (0, 1):
                with self.subTest(action=action, status=status):
                    result = self.run_shell(f'''
systemctl() {{ return {status}; }}
show_notification() {{ printf '%s' "$2"; }}
service_action {action} {past}
''')
                    self.assertEqual(result.returncode, status)
                    if status:
                        self.assertIn(f'Service {action} failed', result.stdout)
                        self.assertNotIn(past, result.stdout)
                    else:
                        self.assertEqual(result.stdout, past)

    def test_every_explicit_service_action_uses_the_checked_helper(self):
        # The recovery restart intentionally has no click notification.
        dispatcher = SCRIPT[SCRIPT.index('# Function to toggle hyprwhspr'):]
        self.assertNotRegex(dispatcher, r'systemctl --user (?:start|stop|restart) hyprwhspr.service')
