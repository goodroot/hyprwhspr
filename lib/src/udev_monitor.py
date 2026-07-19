"""Shared serialized lifecycle for internal pyudev hotplug monitors."""

import queue
import threading
from typing import Optional


class SerializedUdevMonitor:
    """Own one observer and one FIFO callback worker at a time."""

    SHUTDOWN_TIMEOUT = 2.0

    def __init__(self, pyudev_module, subsystem, normalize_event, dispatch_event, log_prefix):
        self._pyudev = pyudev_module
        self._subsystem = subsystem
        self._normalize_event = normalize_event
        self._dispatch_event = dispatch_event
        self._log_prefix = log_prefix
        self.observer = None
        self.monitor = None
        self.context = None
        self.is_running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._accepting_events = False

    def _log(self, message):
        print(f"[{self._log_prefix}] {message}")

    def _dispatch_loop(self, event_queue):
        while True:
            item = event_queue.get()
            if item is None:
                break
            try:
                self._dispatch_event(*item)
            except Exception as e:
                self._log(f"Error handling event: {e}")

    def start(self):
        if self._pyudev is None:
            return False
        if self.is_running:
            return True
        if self._worker is not None:
            if self._worker.is_alive():
                self._log("Previous callback worker is still shutting down")
                return False
            self._worker = None

        event_queue = queue.Queue()
        observer = None
        worker = None
        try:
            context = self._pyudev.Context()
            monitor = self._pyudev.Monitor.from_netlink(context)
            monitor.filter_by(subsystem=self._subsystem)

            def handle_event(action, device):
                if not self._accepting_events or self._event_queue is not event_queue:
                    return
                try:
                    normalized = self._normalize_event(action, device)
                    if normalized is not None:
                        event_queue.put(normalized)
                except Exception as e:
                    self._log(f"Error handling event: {e}")

            observer = self._pyudev.MonitorObserver(monitor, handle_event)
            self._event_queue = event_queue
            self.context = context
            self.monitor = monitor
            self.observer = observer
            self._accepting_events = True
            observer.start()

            worker = threading.Thread(target=self._dispatch_loop, args=(event_queue,), daemon=True)
            self._worker = worker
            worker.start()
            self.is_running = True
            return True
        except Exception as e:
            self._log(f"Failed to start: {e}")
            self._accepting_events = False
            if observer is not None:
                try:
                    observer.stop()
                except Exception:
                    pass
            if worker is not None and worker.is_alive():
                event_queue.put(None)
                worker.join(timeout=self.SHUTDOWN_TIMEOUT)
                self._worker = worker if worker.is_alive() else None
            else:
                self._worker = None
            self.observer = None
            self.monitor = None
            self.context = None
            self.is_running = False
            self._event_queue = queue.Queue()
            return False

    def stop(self):
        if not self.is_running:
            return

        self._accepting_events = False
        observer = self.observer
        self.observer = None
        self.monitor = None
        self.context = None
        self.is_running = False
        if observer is not None:
            try:
                observer.stop()
            except Exception as e:
                self._log(f"Error stopping observer: {e}")

        worker = self._worker
        if worker is not None:
            self._event_queue.put(None)
            worker.join(timeout=self.SHUTDOWN_TIMEOUT)
            if not worker.is_alive():
                self._worker = None
