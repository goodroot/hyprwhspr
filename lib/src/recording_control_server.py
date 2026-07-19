"""Daemon-side transports for recording control and capture clients."""

import os
import select
import socket
import threading
from pathlib import Path


class RecordingControlServer:
    """Own the recording-control FIFO and single-subscriber capture socket."""

    _VALID_COMMANDS = {
        "start", "stop", "cancel", "submit", "model_unload", "model_reload",
    }

    def __init__(self, fifo_path, socket_path, on_command, is_recording):
        self.fifo_path = Path(fifo_path)
        self.socket_path = Path(socket_path)
        self._on_command = on_command
        self._is_recording = is_recording

        self._lifecycle_lock = threading.Lock()
        self._stop_event = None
        self._fifo_thread = None
        self._capture_thread = None
        self._listening_socket = None

        self._capture_subscriber = None
        self._capture_subscriber_lock = threading.Lock()
        self._capture_subscriber_done = threading.Event()

    @classmethod
    def parse_commands(cls, raw_data):
        """Return the last valid ``(action, language)`` command, if any."""
        parsed = []
        for line in (line.strip() for line in raw_data.splitlines()):
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("start:"):
                language = line.split(":", 1)[1].strip() or None
                parsed.append(("start", language))
            elif lower in cls._VALID_COMMANDS:
                parsed.append((lower, None))
        return parsed[-1] if parsed else None

    def prepare_fifo(self):
        """Ensure the control path is a FIFO, replacing legacy regular files."""
        try:
            self.fifo_path.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(2):
                if self.fifo_path.exists():
                    if self.fifo_path.is_fifo():
                        print("[INIT] Recording control FIFO already exists", flush=True)
                        return True
                    try:
                        self.fifo_path.unlink()
                        print("[INIT] Removed old recording_control file (replacing with FIFO)", flush=True)
                    except Exception as exc:
                        print(f"[WARN] Failed to remove old recording_control file: {exc}", flush=True)
                        return False
                try:
                    os.mkfifo(str(self.fifo_path))
                    print(f"[INIT] Created recording control FIFO: {self.fifo_path}", flush=True)
                    return True
                except FileExistsError:
                    continue
            print("[WARN] Failed to create recording control FIFO: file kept reappearing", flush=True)
            print("[WARN] Recording control will fall back to file polling (1 second delay)", flush=True)
        except OSError as exc:
            print(f"[WARN] Failed to create recording control FIFO: {exc}", flush=True)
            print("[WARN] Recording control will fall back to file polling (1 second delay)", flush=True)
        except Exception as exc:
            print(f"[WARN] Unexpected error creating recording control FIFO: {exc}", flush=True)
        return False

    def start(self):
        """Start both listeners, using a fresh stop generation each time."""
        with self._lifecycle_lock:
            workers = (self._fifo_thread, self._capture_thread)
            if any(worker is not None and worker.is_alive() for worker in workers):
                print("[WARN] Recording control server is already running", flush=True)
                return False

            stop_event = threading.Event()
            self._stop_event = stop_event
            self._fifo_thread = None
            self._capture_thread = None

            if self.fifo_path.exists() and self.fifo_path.is_fifo():
                fifo_thread = threading.Thread(
                    target=self._fifo_listener, args=(stop_event,), daemon=True,
                    name="RecordingControlListener")
                try:
                    fifo_thread.start()
                except Exception as exc:
                    print(f"[WARN] Failed to start recording control FIFO listener: {exc}", flush=True)
                else:
                    self._fifo_thread = fifo_thread
                    print("[INIT] Started recording control FIFO listener", flush=True)
            else:
                print("[WARN] Recording control FIFO not available, using fallback polling", flush=True)

            listening_socket = self._setup_capture_socket()
            if listening_socket is not None:
                self._listening_socket = listening_socket
                capture_thread = threading.Thread(
                    target=self._capture_listener, args=(stop_event, listening_socket),
                    daemon=True, name="CaptureSocketListener")
                try:
                    capture_thread.start()
                except Exception as exc:
                    print(f"[WARN] Failed to start capture socket listener: {exc}", flush=True)
                    self._listening_socket = None
                    listening_socket.close()
                    try:
                        self.socket_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    self._capture_thread = capture_thread
                    print("[INIT] Started capture socket listener", flush=True)

            return self._fifo_thread is not None or self._capture_thread is not None

    def stop(self, fifo_timeout=1.0, capture_timeout=2.0):
        """Stop listeners and release the active capture client."""
        with self._lifecycle_lock:
            stop_event = self._stop_event
            if stop_event is None:
                return True
            stop_event.set()
            listening_socket = self._listening_socket
            self._listening_socket = None

        if listening_socket is not None:
            try:
                listening_socket.close()
            except OSError:
                pass

        self._capture_subscriber_done.set()
        with self._capture_subscriber_lock:
            subscriber = self._capture_subscriber
        if subscriber is not None:
            try:
                subscriber.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        self._wake_fifo()
        all_stopped = True
        for worker, timeout, label in (
            (self._fifo_thread, fifo_timeout, "recording control FIFO listener"),
            (self._capture_thread, capture_timeout, "capture socket listener"),
        ):
            if worker is not None and worker.is_alive() and worker is not threading.current_thread():
                print(f"[SHUTDOWN] Stopping {label}...", flush=True)
                worker.join(timeout=timeout)
                if worker.is_alive():
                    all_stopped = False
                    print(f"[WARN] {label} did not stop cleanly", flush=True)

        with self._lifecycle_lock:
            if self._fifo_thread is not None and not self._fifo_thread.is_alive():
                self._fifo_thread = None
            if self._capture_thread is not None and not self._capture_thread.is_alive():
                self._capture_thread = None
            if self._fifo_thread is None and self._capture_thread is None:
                self._stop_event = None
        return all_stopped

    def has_capture_subscriber(self):
        with self._capture_subscriber_lock:
            return self._capture_subscriber is not None

    def notify_capture(self, text, final):
        """Stream transcription text and signal completion to the subscriber."""
        with self._capture_subscriber_lock:
            subscriber = self._capture_subscriber
            if subscriber is None:
                if final:
                    self._capture_subscriber_done.set()
                return
            if text:
                try:
                    subscriber.sendall(text.encode("utf-8"))
                except (BrokenPipeError, ConnectionError, OSError) as exc:
                    print(f"[CAPTURE] Subscriber write failed: {exc}", flush=True)
            if final:
                self._capture_subscriber_done.set()

    def _fifo_listener(self, stop_event):
        while not stop_event.is_set():
            fd = None
            try:
                if not self.fifo_path.exists() or not self.fifo_path.is_fifo():
                    if stop_event.is_set():
                        break
                    if self.fifo_path.exists():
                        self.fifo_path.unlink()
                    os.mkfifo(str(self.fifo_path))
                    print("[CONTROL] Recreated recording control FIFO", flush=True)
                fd = os.open(str(self.fifo_path), os.O_RDONLY | os.O_NONBLOCK)
                raw_data = None
                while not stop_event.is_set():
                    if not self.fifo_path.exists() or not self.fifo_path.is_fifo():
                        raise FileNotFoundError(self.fifo_path)
                    readable, _, _ = select.select([fd], [], [], 0.5)
                    if not readable:
                        continue
                    chunks = []
                    while True:
                        try:
                            chunk = os.read(fd, 4096)
                        except BlockingIOError:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                    if chunks:
                        raw_data = b"".join(chunks).decode("utf-8", errors="replace")
                        break
                    # EOF remains readable until this descriptor is replaced.
                    # Reopen it so an empty writer cannot leave the daemon spinning.
                    break
                else:
                    break
                if raw_data is None:
                    continue
                command = self.parse_commands(raw_data)
                if command is None:
                    lines = [line.strip() for line in raw_data.splitlines() if line.strip()]
                    if lines:
                        print(f"[CONTROL] No valid commands in: {lines}", flush=True)
                    continue
                if not stop_event.is_set() and stop_event is self._stop_event:
                    self._on_command(*command)
            except FileNotFoundError:
                if not stop_event.is_set():
                    print("[CONTROL] FIFO deleted, will recreate on next iteration", flush=True)
                    stop_event.wait(0.1)
            except OSError as exc:
                if not stop_event.is_set():
                    print(f"[CONTROL] FIFO error: {exc}, retrying...", flush=True)
                    stop_event.wait(0.1)
            except Exception as exc:
                if not stop_event.is_set():
                    print(f"[CONTROL] Error in FIFO listener: {exc}", flush=True)
                    stop_event.wait(0.1)
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def _wake_fifo(self):
        try:
            fd = os.open(str(self.fifo_path), os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError:
            pass

    def _write_fifo(self, command):
        try:
            fd = os.open(str(self.fifo_path), os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, command.encode())
            finally:
                os.close(fd)
            return True
        except OSError:
            return False

    def _setup_capture_socket(self):
        try:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self.socket_path.exists():
                try:
                    self.socket_path.unlink()
                    print("[INIT] Removed stale capture socket", flush=True)
                except OSError as exc:
                    print(f"[WARN] Failed to remove stale capture socket: {exc}", flush=True)
                    return None
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(str(self.socket_path))
                os.chmod(str(self.socket_path), 0o600)
                listener.listen(4)
                listener.settimeout(1.0)
            except Exception:
                listener.close()
                raise
            print(f"[INIT] Created capture socket: {self.socket_path}", flush=True)
            return listener
        except OSError as exc:
            print(f"[WARN] Failed to create capture socket: {exc}", flush=True)
            print("[WARN] `record capture` will be unavailable this session", flush=True)
            return None

    def _capture_listener(self, stop_event, listener):
        try:
            while not stop_event.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if stop_event.is_set():
                        break
                    print(f"[CAPTURE] Accept error: {exc}", flush=True)
                    stop_event.wait(0.1)
                    continue
                if stop_event.is_set() or stop_event is not self._stop_event:
                    conn.close()
                    break
                threading.Thread(target=self._handle_capture_connection, args=(conn, stop_event),
                                 daemon=True, name="CaptureConnectionHandler").start()
        finally:
            try:
                listener.close()
            except OSError:
                pass
            try:
                self.socket_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _handle_capture_connection(self, conn, stop_event=None):
        stop_event = stop_event or self._stop_event or threading.Event()
        try:
            conn.settimeout(1.0)
            line = conn.recv(256).split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
            verb, _, language = line.partition(":")
            language = language.strip() or None
            if verb != "capture":
                print(f"[CAPTURE] Unknown request: {line!r}", flush=True)
                return
            conn.settimeout(None)
            with self._capture_subscriber_lock:
                if self._capture_subscriber is not None:
                    print("[CAPTURE] Rejecting — slot occupied", flush=True)
                    try:
                        conn.sendall(b"ERROR:slot_occupied\n")
                    except OSError:
                        pass
                    return
                if stop_event.is_set() or stop_event is not self._stop_event:
                    return
                self._capture_subscriber = conn
                self._capture_subscriber_done.clear()

            if not self._is_recording():
                command = f"start:{language}\n" if language else "start\n"
                if not self._write_fifo(command):
                    print("[CAPTURE] Failed to self-trigger via FIFO", flush=True)

            client_disconnected = False
            while not stop_event.is_set():
                if self._capture_subscriber_done.is_set():
                    break
                try:
                    readable, _, _ = select.select([conn], [], [], 0.5)
                except (OSError, ValueError):
                    client_disconnected = True
                    break
                if readable:
                    try:
                        peek = conn.recv(1, socket.MSG_PEEK)
                    except OSError:
                        peek = b""
                    if not peek:
                        client_disconnected = True
                        break
            if client_disconnected and not stop_event.is_set() and self._is_recording():
                print("[CAPTURE] Client disconnected, cancelling recording", flush=True)
                self._write_fifo("cancel\n")
        except Exception as exc:
            print(f"[CAPTURE] Handler error: {exc}", flush=True)
        finally:
            with self._capture_subscriber_lock:
                if self._capture_subscriber is conn:
                    self._capture_subscriber = None
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
