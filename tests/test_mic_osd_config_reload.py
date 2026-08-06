import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from mic_osd.osd_config import MicOSDConfig
import mic_osd.main as main_module


class MicOSDConfigReloadTests(unittest.TestCase):
    def _app(self, config):
        app = object.__new__(main_module.MicOSD)
        app._config_signature = None
        app.osd_config = config
        app.visualization = mock.Mock()
        app.visualization.apply_config = mock.Mock()
        app.window = mock.Mock()
        app.window.apply_osd_config = mock.Mock()
        return app

    def test_glib_signal_callbacks_dispatch_show_and_hide_directly(self):
        original = main_module._app
        app = mock.Mock()
        main_module._app = app
        try:
            self.assertTrue(main_module._glib_sigusr1_handler())
            self.assertTrue(main_module._glib_sigusr2_handler())
        finally:
            main_module._app = original

        app._show.assert_called_once_with()
        app._hide.assert_called_once_with()

    def test_valid_change_is_applied_without_restarting_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            config_file.write_text('{"mic_osd_spectrum_gain": 2.5}')
            old_config = MicOSDConfig()
            new_config = MicOSDConfig(spectrum_gain=2.5)
            app = self._app(old_config)

            with mock.patch.object(main_module, "CONFIG_FILE", config_file), \
                    mock.patch.object(
                        main_module.MicOSDConfig,
                        "load",
                        return_value=new_config,
                    ):
                self.assertTrue(app._poll_config_file())

            self.assertEqual(app.osd_config, new_config)
            app.visualization.apply_config.assert_called_once_with(new_config)
            app.window.apply_osd_config.assert_called_once_with(new_config)

    def test_partial_json_keeps_current_presentation_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = Path(tmp) / "config.json"
            config_file.write_text('{"mic_osd_spectrum_gain":')
            config = MicOSDConfig()
            app = self._app(config)

            with mock.patch.object(main_module, "CONFIG_FILE", config_file), \
                    mock.patch.object(main_module.MicOSDConfig, "load") as load:
                self.assertTrue(app._poll_config_file())

            self.assertEqual(app.osd_config, config)
            self.assertIsNone(app._config_signature)
            load.assert_not_called()
            app.visualization.apply_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
