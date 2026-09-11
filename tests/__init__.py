"""Test-suite guards: keep tests off the real network and out of the real home.

The XDG guard exists because ownership receipts are now recorded by every
install flavour, so any test exercising a config/credential save writes a
receipt. A test that forgets to redirect XDG_STATE_HOME would write into the
developer's own ~/.local/state/hyprwhspr (this happened). Redirecting the XDG
roots for the whole session keeps a missing mock contained in a temp dir
instead of mutating real user state; HOME itself is left alone so tests that
deliberately assert on it still behave.

Network guard: fail loudly if a test slips past its mocks and touches
the real network (this happened: a mispatched mock once downloaded a real
model from HuggingFace mid-test-run). Loopback and unix sockets stay allowed.

Two layers: socket.connect catches any HTTP client on machines with direct
network access; the urllib wrappers inspect the requested URL so proxied
environments (where every connection is to a localhost proxy) are covered too.
"""
import atexit
import os
import shutil
import socket
import tempfile
import urllib.request
from urllib.parse import urlparse

_XDG_SANDBOX = tempfile.mkdtemp(prefix='hyprwhspr-tests-')
atexit.register(shutil.rmtree, _XDG_SANDBOX, True)
for _name in ('XDG_STATE_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME', 'XDG_RUNTIME_DIR'):
    os.environ[_name] = os.path.join(_XDG_SANDBOX, _name.lower())
    os.makedirs(os.environ[_name], exist_ok=True)

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _refuse(target):
    raise RuntimeError(
        f"test attempted real network access to {target!r} - a mock is missing or mispatched"
    )


_real_connect = socket.socket.connect


def _guarded_connect(self, address, *args, **kwargs):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            _refuse(address)
    return _real_connect(self, address, *args, **kwargs)


socket.socket.connect = _guarded_connect


def _check_url(url):
    if not isinstance(url, str):
        url = getattr(url, "full_url", str(url))  # urllib.request.Request
    host = urlparse(url).hostname
    if host and host not in _LOOPBACK:
        _refuse(url)


_real_urlopen = urllib.request.urlopen
_real_urlretrieve = urllib.request.urlretrieve


def _guarded_urlopen(url, *args, **kwargs):
    _check_url(url)
    return _real_urlopen(url, *args, **kwargs)


def _guarded_urlretrieve(url, *args, **kwargs):
    _check_url(url)
    return _real_urlretrieve(url, *args, **kwargs)


urllib.request.urlopen = _guarded_urlopen
urllib.request.urlretrieve = _guarded_urlretrieve
