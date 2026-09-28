"""Thread-safe line output for the long-running service.

print() writes a message and its newline as separate calls, so concurrent
threads interleave into garbled journald lines. log() writes the whole line
in one call under a lock and flushes, producing exactly what
print(message, flush=True) would have.
"""
import sys
import threading

# Reentrant so a signal handler that logs while the main thread holds it cannot deadlock.
_lock = threading.RLock()


def log(message: object = '') -> None:
    """Write `message` and a newline to the current sys.stdout as one line."""
    line = f'{message}\n'
    with _lock:
        # Resolved per call so redirect_stdout (tests, CLI capture) still applies.
        stream = sys.stdout
        if stream is None:  # pythonw / detached stdout
            return
        stream.write(line)
        stream.flush()
