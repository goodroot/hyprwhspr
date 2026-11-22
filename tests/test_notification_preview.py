"""The notification presenter carries live preview text in the status bubble."""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))

from mic_osd import notification_presenter as presenter_module  # noqa: E402
from mic_osd.notification_presenter import NotificationPresenter  # noqa: E402


class NotificationPreviewTests(unittest.TestCase):
    def setUp(self):
        self.bodies = []
        self.patch_send = mock.patch.object(
            presenter_module, 'send_notification_with_id',
            side_effect=self._send)
        self.patch_send.start()
        self.addCleanup(self.patch_send.stop)
        patch_which = mock.patch.object(presenter_module.shutil, 'which',
                                        return_value='/usr/bin/notify-send')
        patch_which.start()
        self.addCleanup(patch_which.stop)
        self.presenter = NotificationPresenter()
        self.addCleanup(self.presenter.stop)

    def _send(self, name, body, **kwargs):
        self.bodies.append(body)
        return len(self.bodies)

    def test_preview_text_is_appended_to_the_recording_state(self):
        self.presenter.show()
        self.presenter.set_preview_text('hello there')
        self.assertIn('Recording', self.bodies[-1])
        self.assertIn('hello there', self.bodies[-1])

    def test_clear_preview_restores_the_plain_state(self):
        self.presenter.show()
        self.presenter.set_preview_text('hello there')
        self.presenter.clear_preview_text()
        self.assertIn('Recording', self.bodies[-1])
        self.assertNotIn('hello there', self.bodies[-1])

    def test_identical_preview_does_not_rewrite_the_bubble(self):
        self.presenter.show()
        self.presenter.set_preview_text('same')
        count = len(self.bodies)
        self.presenter.set_preview_text('same')
        self.assertEqual(len(self.bodies), count)

    def test_preview_before_a_bubble_exists_is_ignored(self):
        self.presenter.set_preview_text('early')
        self.assertEqual(self.bodies, [])

    def test_empty_preview_is_ignored(self):
        self.presenter.show()
        count = len(self.bodies)
        self.presenter.set_preview_text('')
        self.assertEqual(len(self.bodies), count)

    def test_long_preview_keeps_the_newest_characters(self):
        self.presenter.show()
        self.presenter.set_preview_text('a' * 400 + 'TAIL')
        self.assertIn('TAIL', self.bodies[-1])
        preview_line = self.bodies[-1].split('\n', 1)[1]
        self.assertEqual(len(preview_line), presenter_module.NotificationPresenter._PREVIEW_MAX_CHARS)

    def test_hide_drops_the_preview(self):
        self.presenter.show()
        self.presenter.set_preview_text('gone soon')
        self.presenter.hide()
        self.presenter.show()
        self.assertNotIn('gone soon', self.bodies[-1])


if __name__ == '__main__':
    unittest.main()
