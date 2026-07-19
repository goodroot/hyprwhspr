import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class X11ServiceContractTests(unittest.TestCase):
    def test_readiness_uses_selected_wayland_socket_or_display(self):
        service = (ROOT / "config/systemd/hyprwhspr.service").read_text(encoding="utf-8")
        prestart = next(line for line in service.splitlines() if line.startswith("ExecStartPre="))

        self.assertIn('${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}', prestart)
        self.assertIn('[ -S ', prestart)
        self.assertIn('[ -n "$DISPLAY" ]', prestart)
        self.assertIn('${XDG_RUNTIME_DIR}"/wayland-*', prestart)
        self.assertIn('/*)', prestart)
        self.assertNotIn('/tmp/.X11-unix/X*', prestart)

    def test_setup_imports_wayland_and_x11_session_variables(self):
        setup = (ROOT / "lib/src/cli/systemd.py").read_text(encoding="utf-8")
        for variable in (
            "WAYLAND_DISPLAY", "DISPLAY", "XAUTHORITY", "XDG_SESSION_TYPE",
            "XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION",
        ):
            self.assertIn(f"'{variable}'", setup)


if __name__ == "__main__":
    unittest.main()
