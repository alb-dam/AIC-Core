"""Thread Helpers: code thread-safe con drop-frame policy e worker threads."""

import time
import queue
import threading
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class DropFrameQueue:
    """Coda thread-safe con drop-frame policy per garantire minima latenza."""

    def __init__(self, maxsize: int = 2) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: Any) -> None:
        while True:
            try:
                self.q.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout: Optional[float] = None) -> Any:
        return self.q.get(timeout=timeout)

    def clear(self) -> None:
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break


class WorkerThread(threading.Thread):
    def __init__(self, name: str, target: Callable[[], None], stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self.target_func = target
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.target_func()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Errore non gestito nel thread {self.name}: {e}")
                time.sleep(1.0)
