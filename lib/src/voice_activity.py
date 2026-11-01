"""Optional connection-scoped voxd input leases; stdlib only, no mic access.

voxd owns output arbitration and lightd's speech-input intent. Each recording
thread owns its own session, so stale cleanup cannot release a newer take.
"""
import json
import os
import socket
import threading
import time


def default_socket_path():
    return os.environ.get('VOXD_SOCKET') or os.path.join(
        os.environ.get('XDG_RUNTIME_DIR') or '/tmp',
        'voxd.sock' if os.environ.get('XDG_RUNTIME_DIR') else f'voxd-{os.getuid()}.sock',
    )


class VoiceActivity:
    def __init__(self, path=None):
        self.path = path or default_socket_path()
        self._lock = threading.Lock()
        self._sessions = set()
        self._closed = False

    def begin(self, active=None):
        session = _Session(self.path, self._forget, active)
        with self._lock:
            closed = self._closed
            if not closed:
                self._sessions.add(session)
        if closed:
            session.close()
            return session
        # Bound the initial acknowledgement wait before starting capture. A
        # missing voxd must not break dictation; retry off the capture thread.
        session.start()
        return session

    def _forget(self, session):
        with self._lock:
            self._sessions.discard(session)

    def close(self):
        with self._lock:
            self._closed = True
            sessions = list(self._sessions)
        for session in sessions:
            session.close()


class _Session:
    TIMEOUT = 0.5
    HEARTBEAT = 1.0  # voxd expires the connection after five seconds without begin

    def __init__(self, path, forget, active=None):
        self._active = active
        self.path = path
        self._forget = forget
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._socket = None
        self._sequence = 0
        self._warned = False
        self._thread = None

    def start(self):
        self._renew()
        if not self._stop.is_set():
            self._thread = threading.Thread(target=self._run, name='voxd-input', daemon=True)
            self._thread.start()

    def _disconnect(self):
        with self._lock:
            sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()  # EOF releases this connection's lease, even on failure

    def close(self):
        self._stop.set()
        self._disconnect()
        self._forget(self)

    def _renew(self):
        try:
            if self._active is not None and not self._active():
                self.close()
            if self._stop.is_set():
                return
            deadline = time.monotonic() + self.TIMEOUT
            with self._lock:
                if self._stop.is_set():
                    return
                sock = self._socket
                connect = sock is None
                if connect:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self._socket = sock
            sock.settimeout(self.TIMEOUT)
            if connect:
                sock.connect(self.path)
            if self._stop.is_set():
                self._disconnect()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('input connection timed out')
            sock.settimeout(remaining)
            self._sequence += 1
            request_id = str(self._sequence)
            sock.sendall((json.dumps({'v': 1, 'request_id': request_id, 'op': 'input_begin'}) + '\n').encode())
            data = bytearray()
            while b'\n' not in data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('input acknowledgement timed out')
                sock.settimeout(remaining)
                chunk = sock.recv(4096 - len(data))
                if not chunk:
                    raise OSError('voxd closed input connection')
                data.extend(chunk)
                if len(data) >= 4096:
                    raise ValueError('oversized input acknowledgement')
            response = json.loads(data)
            if (response.get('v') != 1 or response.get('request_id') != request_id
                    or response.get('ok') is not True or response.get('type') != 'input'
                    or response.get('active') is not True):
                raise ValueError('voxd rejected input lease (daemon may need upgrading)')
            self._warned = False
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self._disconnect()
            if not self._stop.is_set() and not self._warned:
                print(f'[VOXD] Input lease unavailable: {exc}; dictation continues, retrying', flush=True)
                self._warned = True

    def _run(self):
        while not self._stop.wait(self.HEARTBEAT):
            self._renew()
