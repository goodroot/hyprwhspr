"""Sway focus over IPC, so paste never lands in a window the user left.

Stdlib only: importable and testable without a Sway session. The injector owns
transports and logging; this module only answers "which view has focus" and
"does it still".
"""

import json
import os
import re
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GET_TREE = 4
GET_SEATS = 101

# Why a lookup found no target. IPC_UNAVAILABLE covers a missing, dead or
# malformed reply; NO_VIEW means Sway answered but focus is on a workspace or
# container. Keyboard routing for a focused container cannot be verified, so
# both fail closed.
IPC_UNAVAILABLE = 'ipc'
NO_VIEW = 'no-view'

_MAGIC = b'i3-ipc'
_HEADER = struct.Struct('=II')  # i3-ipc uses host byte order
_BUDGET_SECS = 0.5
_PROBE_SECS = 0.1
_MAX_REPLY = 8 * 1024 * 1024


def is_sway_session(environ=os.environ) -> bool:
    """Prefer the current desktop identity over inherited IPC variables.

    Sway itself sets no desktop variable. Started from a TTY, the service may
    still carry NIRI_SOCKET or HYPRLAND_INSTANCE_SIGNATURE from an earlier
    session, so a SWAYSOCK that answers outranks them.
    """
    if environ.get('XDG_SESSION_TYPE', '').lower() == 'x11':
        return False
    for key in ('XDG_CURRENT_DESKTOP', 'XDG_SESSION_DESKTOP', 'DESKTOP_SESSION'):
        tokens = [token.strip() for token in re.split(r'[:;,]', environ.get(key, '').lower())]
        tokens = [token for token in tokens if token]
        if tokens:
            return any(token in ('sway', 'swayfx') for token in tokens)
    swaysock = environ.get('SWAYSOCK')
    if swaysock and SwayIPC._answers(swaysock):
        return True
    if environ.get('NIRI_SOCKET') or environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
        return False
    return bool(swaysock)


def delivery_seat(seats: Any) -> Optional[Dict[str, Any]]:
    """The seat injected keys land on, or None when that is ambiguous.

    The transports do not share a verified seat selection mechanism. Even an
    empty seat can receive a virtual keyboard, so require exactly one seat.
    """
    if not isinstance(seats, list) or len(seats) != 1 or not isinstance(seats[0], dict):
        return None
    return seats[0]


class SwayIPC:
    """One Sway instance, addressed by its socket; follows a restart only when
    exactly one live instance is left."""

    def __init__(self, socket_path: Optional[str], runtime_dir: str, uid: Optional[int] = None):
        self.socket_path = socket_path or None
        self.runtime_dir = runtime_dir
        self.uid = os.getuid() if uid is None else uid

    def request(self, kind: int) -> Any:
        """One exchange bounded to 500 ms and 8 MiB, reconnect included.

        Returns the decoded reply, or None on any failure.
        """
        deadline = time.monotonic() + _BUDGET_SECS
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ipc:
                def budget():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Sway IPC deadline')
                    ipc.settimeout(remaining)

                def receive(count):
                    data = bytearray()
                    while len(data) < count:
                        budget()
                        part = ipc.recv(min(count - len(data), 65536))
                        if not part:
                            raise ValueError('Sway IPC truncated')
                        data.extend(part)
                    return data

                path = self.socket_path or self.rediscover()
                if not path:
                    return None
                budget()
                try:
                    ipc.connect(path)
                except (FileNotFoundError, ConnectionRefusedError):
                    path = self.rediscover()
                    if not path:
                        return None
                    budget()
                    ipc.connect(path)
                self.socket_path = path
                budget()
                ipc.sendall(_MAGIC + _HEADER.pack(0, kind))
                header = receive(len(_MAGIC) + _HEADER.size)
                length, reply_kind = _HEADER.unpack(header[len(_MAGIC):])
                if header[:len(_MAGIC)] != _MAGIC or reply_kind != kind or length > _MAX_REPLY:
                    return None
                return json.loads(receive(length))
        except (OSError, ValueError, TypeError, RecursionError, struct.error):
            return None

    def rediscover(self) -> Optional[str]:
        """The single live Sway socket in the runtime directory, else None.

        Liveness is a connect probe: socket files outlive their compositor,
        and a stale file's PID may since belong to an unrelated process.
        """
        try:
            entries = [str(path) for path in Path(self.runtime_dir).glob(f'sway-ipc.{self.uid}.*.sock')]
        except OSError:
            return None
        live = [path for path in entries if self._answers(path)]
        return live[0] if len(live) == 1 else None

    @staticmethod
    def _answers(path: str) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(_PROBE_SECS)
                probe.connect(path)
            return True
        except OSError:
            return False

    def lookup(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """The focused view as window info, or (None, reason)."""
        pending = [self.request(GET_TREE)]
        if not isinstance(pending[0], dict):
            return None, IPC_UNAVAILABLE
        while pending:
            node = pending.pop()
            if not isinstance(node, dict):
                return None, IPC_UNAVAILABLE
            props = node.get('window_properties') or {}
            if not isinstance(props, dict):
                return None, IPC_UNAVAILABLE
            app_id = node.get('app_id')
            wm_class = props.get('class')
            # pid/shell belong to views; a focused workspace or split container
            # must not be mistaken for an unidentified application.
            is_view = bool(app_id or wm_class or node.get('pid') or node.get('shell') or node.get('window'))
            if node.get('focused') is True and is_view:
                identity = app_id or wm_class or ''
                if not isinstance(identity, str) or type(node.get('id')) is not int:
                    return None, IPC_UNAVAILABLE
                return {
                    'class': identity, 'app_id': app_id,
                    'title': node.get('name') or '', 'source': 'sway',
                    'sway_id': node['id'], 'sway_socket': self.socket_path,
                    'xwayland': node.get('shell') == 'xwayland' or node.get('window') is not None or bool(wm_class),
                    'x11_window_id': node.get('window'),
                }, None
            for key in ('nodes', 'floating_nodes'):
                children = node.get(key, [])
                if not isinstance(children, list):
                    return None, IPC_UNAVAILABLE
                pending.extend(children)
        return None, NO_VIEW

    def focus_matches(self, expected: Optional[Dict[str, Any]]) -> bool:
        """True while the delivery seat still focuses the expected view.

        GET_SEATS stays small however many windows are open. Container IDs are
        unique within one compositor instance, so the socket must match too.
        """
        if not expected or type(expected.get('sway_id')) is not int:
            return False
        seat = delivery_seat(self.request(GET_SEATS))
        return (
            seat is not None
            and expected.get('sway_socket') == self.socket_path
            and seat.get('focus') == expected['sway_id']
        )
