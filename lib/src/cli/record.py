"""
Recording control commands for hyprwhspr (external hotkey integration)
"""

import os
import sys
import socket

try:
    from ..paths import (RECORDING_CONTROL_FILE, RECORDING_STATUS_FILE,
                         SOCKET_FILE)
except ImportError:
    from paths import (RECORDING_CONTROL_FILE, RECORDING_STATUS_FILE,
                       SOCKET_FILE)

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error


def record_command(action: str, language: str = None):
    """
    Control recording via CLI - useful when keyboard grab is not possible.

    This writes to the recording control FIFO to trigger start/stop/cancel/toggle
    without requiring keyboard grab. Useful for users with:
    - External hotkey systems (KDE, GNOME, sxhkd, etc.)
    - Keyboard remappers that grab devices (Espanso, keyd, kmonad)
    - Multiple keyboard tools that conflict with grab_keys

    Args:
        action: The action to perform (start, stop, cancel, toggle, status)
        language: Optional language code for transcription (e.g., 'en', 'it', 'de')
    """
    import stat

    def is_recording() -> bool:
        """Check if currently recording (status file exists with 'true')"""
        if not RECORDING_STATUS_FILE.exists():
            return False
        try:
            content = RECORDING_STATUS_FILE.read_text().strip().lower()
            return content == 'true'
        except Exception:
            return False

    def send_control(command: str) -> bool:
        """Send a command to the recording control FIFO"""
        if not RECORDING_CONTROL_FILE.exists():
            log_error("Recording control file not found.")
            log_error("Is the hyprwhspr service running?")
            log_info("Start it with: systemctl --user start hyprwhspr")
            return False

        # Check if it's a FIFO (named pipe)
        try:
            file_stat = RECORDING_CONTROL_FILE.stat()
            is_fifo = stat.S_ISFIFO(file_stat.st_mode)
        except Exception:
            is_fifo = False

        try:
            if is_fifo:
                # Open FIFO in non-blocking mode with timeout
                import select
                fd = os.open(str(RECORDING_CONTROL_FILE), os.O_WRONLY | os.O_NONBLOCK)
                fd_closed = False
                try:
                    # Wait for FIFO to be ready for writing (service is listening)
                    _, ready, _ = select.select([], [fd], [], 2.0)
                    if not ready:
                        os.close(fd)
                        fd_closed = True
                        log_error("Service not responding (timeout waiting for FIFO)")
                        log_info("The service may be busy or not running properly")
                        return False
                    os.write(fd, (command + '\n').encode())
                finally:
                    if not fd_closed:
                        os.close(fd)
            else:
                # Fall back to regular file write
                RECORDING_CONTROL_FILE.write_text(command + '\n')
            return True
        except OSError as e:
            if e.errno == 6:  # ENXIO - no reader on FIFO
                log_error("Service not listening on control FIFO")
                log_info("Is the hyprwhspr service running?")
                log_info("Start it with: systemctl --user start hyprwhspr")
            else:
                log_error(f"Failed to send command: {e}")
            return False
        except Exception as e:
            log_error(f"Failed to send command: {e}")
            return False

    # Build start command with optional language
    start_cmd = f'start:{language}' if language else 'start'

    if action == 'start':
        if is_recording():
            log_warning("Already recording")
            return
        if send_control(start_cmd):
            msg = f"Recording started (language: {language})" if language else "Recording started"
            log_success(msg)

    elif action == 'stop':
        if not is_recording():
            log_warning("Not currently recording")
            return
        if send_control('stop'):
            log_success("Recording stopped")

    elif action == 'cancel':
        if not is_recording():
            log_warning("Not currently recording")
            return
        if send_control('cancel'):
            log_success("Recording cancelled (audio discarded)")

    elif action == 'toggle':
        if is_recording():
            if send_control('stop'):
                log_success("Recording stopped")
        else:
            if send_control(start_cmd):
                msg = f"Recording started (language: {language})" if language else "Recording started"
                log_success(msg)

    elif action == 'status':
        if is_recording():
            log_info("Status: Recording in progress")
        else:
            log_info("Status: Idle")

    else:
        log_error(f"Unknown action: {action}")
        log_info("Available actions: start, stop, cancel, toggle, status")


def record_capture_command(language: str = None):
    """
    Connect to the capture socket, trigger a recording, stream the transcription to stdout.

    Blocks until the daemon closes the connection. If daemon is idle, this self-triggers a recording via the socket.
    If daemon is already recording, this attaches to the in-flight transcription.

    Args:
      language: Language code (e.g., 'en', 'it', 'fr') or None for auto-detect
    """

    if not SOCKET_FILE.exists():
        log_error("Capture socket not found.")
        log_error("Is the hyprwhspr service running?")
        log_error("Start it with: systemctl --user start hyprwhspr")
        sys.exit(1)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(str(SOCKET_FILE))
            request = "capture"
            if language:
                request += f":{language}"
            request += "\n"
            s.sendall(request.encode())

            first = True
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                if first:
                    first = False
                    if chunk.startswith(b"ERROR:"):
                        msg = chunk.decode().strip().removeprefix("ERROR:")
                        log_error(f"Capture rejected: {msg}")
                        sys.exit(1)
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
    except KeyboardInterrupt:
        sys.exit(130)
    except ConnectionRefusedError:
        log_error("Capture socket refused connection. Daemon may be shutting down.")
        sys.exit(1)
    except OSError as e:
        log_error(f"Capture socket error: {e}")
        sys.exit(1)
