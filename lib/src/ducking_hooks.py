"""Nonblocking recording-scoped external ducking sessions.

Commands receive ``recording\n`` on stdin every second. EOF means restore and
exit. Commands must also implement a remote lease timeout: killing a local SSH
process cannot guarantee delivery across a broken network.
"""

import math
import os
import signal
import select
import subprocess
import time
import threading


class DuckingHooks:
    """One serialized worker per device; no subprocess I/O on recording threads."""

    def __init__(self, hooks):
        self._workers = []
        if not isinstance(hooks, list):
            print('[DUCKING_HOOK] Expected a list of hooks', flush=True)
            return
        for index, hook in enumerate(hooks):
            try:
                if not isinstance(hook, dict):
                    raise ValueError('expected an object')
                if hook.get('enabled', True) is False:
                    continue
                command = hook.get('command')
                if (not isinstance(command, list) or not command
                        or any(not isinstance(arg, str) or not arg or '\0' in arg
                               for arg in command)):
                    raise ValueError('command must be a nonempty argv array')
                grace = float(hook.get('restore_timeout', 10))
                if not math.isfinite(grace) or not 1 <= grace <= 60:
                    raise ValueError('restore_timeout must be between 1 and 60 seconds')
                persistent = hook.get('persistent', False)
                if not isinstance(persistent, bool):
                    raise ValueError('persistent must be a boolean')
                worker_class = _PersistentDeviceWorker if persistent else _DeviceWorker
                worker = worker_class(str(hook.get('name', index)), command, grace)
                self._workers.append(worker)
            except (ValueError, TypeError) as exc:
                print(f'[DUCKING_HOOK] Skipping hook {index}: {exc}', flush=True)

    def set_recording(self, active):
        for worker in self._workers:
            worker.set_recording(active)

    def close(self):
        """Request restoration without joining workers on the shutdown path.

        If Python exits first, OS pipe closure still sends EOF to each child.
        Remote commands must honor EOF and their lease independently of us.
        """
        for worker in self._workers:
            worker.close()


class _DeviceWorker:
    HEARTBEAT_SECONDS = 1.0

    def __init__(self, name, command, restore_timeout):
        self.name = name
        self.command = list(command)
        self.restore_timeout = restore_timeout
        self._condition = threading.Condition()
        self._active = False
        self._closed = False
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run, name=f'ducking-{name}', daemon=True,
        )
        self._thread.start()

    def set_recording(self, active):
        with self._condition:
            if self._closed or self._active == bool(active):
                return
            self._active = bool(active)
            self._generation += 1
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._active = False
            self._condition.notify_all()

    def _log(self, message):
        print(f'[DUCKING_HOOK] {self.name}: {message}', flush=True)

    @staticmethod
    def _signal_group(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def _restore(self, process):
        # Unbuffered stdin: closing never flushes a blocked pipe.
        process.stdin.close()
        try:
            process.wait(timeout=self.restore_timeout)
        except subprocess.TimeoutExpired:
            self._log('restore deadline exceeded; terminating command group')
            self._signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
                process.wait()
        if process.returncode:
            self._log(f'command exited with status {process.returncode}')

    def _run(self):
        handled = -1
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or (self._active and self._generation != handled)
                )
                if self._closed:
                    return
                generation = handled = self._generation
            process = None
            try:
                process = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    bufsize=0, start_new_session=True,
                )
                os.set_blocking(process.stdin.fileno(), False)
                while process.poll() is None:
                    with self._condition:
                        if self._closed or not self._active or self._generation != generation:
                            break
                        try:
                            os.write(process.stdin.fileno(), b'recording\n')
                        except BlockingIOError:
                            # A hung command must not accumulate unlimited heartbeats.
                            self._log('command is not consuming heartbeats')
                            break
                        self._condition.wait(timeout=self.HEARTBEAT_SECONDS)
            except (OSError, ValueError) as exc:
                self._log(f'session failed: {exc}')
            finally:
                if process is not None:
                    try:
                        self._restore(process)
                    except OSError as exc:
                        self._log(f'cleanup failed: {exc}')
            # Never retry within a take: repeated remote side effects are unsafe.
            # A new recording may start only after this session's cleanup ends.


class _PersistentDeviceWorker(_DeviceWorker):
    """Warm, acknowledged sessions. Reconnect with latest state, never replay.

    stdout must contain ``ready`` on startup and ``ok`` after each applied
    ``idle``/``recording`` command. Only one request is in flight. A broken or
    silent peer is disconnected, allowing its remote lease to restore volume.
    """

    READY_TIMEOUT = 10.0
    ACK_TIMEOUT = 5.0
    MAX_RETRY_SECONDS = 15.0

    def _read_reply(self, process, expected, timeout):
        deadline = time.monotonic() + timeout
        data = bytearray()
        while time.monotonic() < deadline:
            with self._condition:
                if self._closed:
                    raise InterruptedError('closing')
            remaining = deadline - time.monotonic()
            readable, _, _ = select.select([process.stdout], [], [], max(0, min(.1, remaining)))
            if not readable:
                continue
            chunk = os.read(process.stdout.fileno(), 1)
            if not chunk:
                raise OSError('control connection closed')
            if chunk == b'\n':
                if data.decode('ascii', errors='replace') != expected:
                    raise OSError('invalid control acknowledgement')
                return
            data.extend(chunk)
            if len(data) > 64:
                raise OSError('oversized control acknowledgement')
        raise TimeoutError(f'timed out waiting for {expected}')

    def _run(self):
        retry = 1.0
        while True:
            with self._condition:
                if self._closed:
                    return
            process = None
            try:
                process = subprocess.Popen(
                    self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=0, start_new_session=True,
                )
                os.set_blocking(process.stdin.fileno(), False)
                self._read_reply(process, 'ready', self.READY_TIMEOUT)
                self._log('persistent connection ready')
                while True:
                    with self._condition:
                        if self._closed:
                            break
                        active, generation = self._active, self._generation
                    os.write(process.stdin.fileno(), b'recording\n' if active else b'idle\n')
                    self._read_reply(process, 'ok', self.ACK_TIMEOUT)
                    retry = 1.0
                    with self._condition:
                        # A state change during the acknowledgement wait is sent
                        # immediately, not delayed by the heartbeat interval.
                        self._condition.wait_for(
                            lambda: self._closed or self._generation != generation,
                            timeout=self.HEARTBEAT_SECONDS,
                        )
            except (OSError, ValueError) as exc:
                if not self._closed:
                    self._log(f'connection unavailable: {exc}; reconnecting in background')
            finally:
                if process is not None:
                    try:
                        self._restore(process)
                    except OSError as exc:
                        self._log(f'cleanup failed: {exc}')
                    finally:
                        process.stdout.close()
            with self._condition:
                if self._closed:
                    return
                # State changes do not bypass backoff or queue stale commands.
                self._condition.wait_for(lambda: self._closed, timeout=retry)
            retry = min(self.MAX_RETRY_SECONDS, retry * 2)
