"""Plain-Python observer events. The UI bridges these onto Qt signals."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str  # node_added | node_updated | edge_added | cascade | notice | message | loop_state
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Callable[[Event], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsubscribe

    def emit(self, event_kind: str, /, **payload: Any) -> None:
        with self._lock:
            subs = list(self._subs)
        ev = Event(event_kind, payload)
        for fn in subs:
            try:
                fn(ev)
            except Exception:  # a broken listener must never break the engine
                log.exception("event listener failed for %s", event_kind)
