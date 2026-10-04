"""Threading for the UI.

All engine work (tasks, re-checks, lock/delete) runs on ONE background thread,
in order, so the graph never sees two writers. Engine events are deep-copied
on that thread and delivered to the UI thread through a Qt signal.
"""
from __future__ import annotations

import copy
import itertools
import queue
import threading
from typing import Any, Callable

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from sciai.graph.events import Event, EventBus
from sciai.graph.model import Node


class EventBridge(QObject):
    event = pyqtSignal(object)

    def __init__(self, bus: EventBus) -> None:
        super().__init__()
        self._unsub = bus.subscribe(self._forward)

    def _forward(self, ev: Event) -> None:
        try:
            snap = copy.deepcopy(ev)
        except Exception:  # noqa: BLE001 - never break the engine thread over a copy
            snap = ev
        self.event.emit(snap)

    def close(self) -> None:
        self._unsub()


class EngineExecutor(QObject):
    """Single worker thread. ``submit`` returns a job id; ``done`` reports (id, result, error).

    The executor lives on the UI thread. The worker only posts ``_finished``; the
    pending count drops in ``_on_finished`` on the UI thread, right after ``done``
    is delivered, so ``is_busy`` never reads False while a result is still queued.
    """

    done = pyqtSignal(int, object, object)
    busy = pyqtSignal(bool)
    _finished = pyqtSignal(int, object, object)

    def __init__(self) -> None:
        super().__init__()
        self._q: queue.Queue = queue.Queue()
        self._ids = itertools.count(1)
        self._pending = 0
        self._lock = threading.Lock()
        self._finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._thread = threading.Thread(target=self._loop, name="sciai-engine", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any]) -> int:
        job = next(self._ids)
        with self._lock:
            self._pending += 1
            if self._pending == 1:
                self.busy.emit(True)
        self._q.put((job, fn))
        return job

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            job, fn = item
            try:
                result, error = fn(), None
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                result, error = None, exc
            self._finished.emit(job, result, error)

    def _on_finished(self, job: int, result: Any, error: Any) -> None:
        try:
            self.done.emit(job, result, error)
        finally:
            with self._lock:
                self._pending -= 1
                idle = self._pending == 0
            if idle:
                self.busy.emit(False)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._pending > 0

    def shutdown(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)


class RootConfirmer(QObject):
    """Lets the controller thread block until the user confirms the formalized problem."""

    ask = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._event = threading.Event()
        self._answer: bool | str = True

    def __call__(self, node: Node) -> bool | str:
        self._event.clear()
        self.ask.emit(copy.deepcopy(node))
        self._event.wait()
        return self._answer

    def answer(self, value: bool | str) -> None:
        self._answer = value
        self._event.set()
