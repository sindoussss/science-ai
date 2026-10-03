"""Tool execution.

``SandboxRunner`` runs every tool (and every parse of model-written text) in a
separate worker process: a hung SymPy call is killed at the timeout and the
worker is restarted, network sockets are disabled, and on POSIX the address
space is capped. ``InlineRunner`` has the same interface for fast unit tests.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_TIMEOUT = 20.0
MEMORY_LIMIT_BYTES = 3 * 1024**3


@dataclass
class ToolOutcome:
    ok: bool
    value: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: float = 0.0
    timed_out: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class Runner(Protocol):
    def run(self, tool: str, args: dict[str, Any], timeout: float | None = None) -> ToolOutcome: ...
    def canonicalize(self, tool: str, args: dict[str, Any]) -> ToolOutcome: ...
    def close(self) -> None: ...


def _execute(op: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    from sciai.tools.canonical import canonicalize
    from sciai.tools.registry import get

    if op == "canon":
        return {"canonical": canonicalize(tool, args)}
    spec = get(tool)
    spec.validate(args)
    out = spec.fn(args)
    if not isinstance(out, dict) or "result" not in out:
        raise TypeError(f"tool {tool} returned a malformed value")
    return out


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:600]


class InlineRunner:
    """Same process. Tests only: no timeout enforcement."""

    def run(self, tool: str, args: dict[str, Any], timeout: float | None = None) -> ToolOutcome:
        t0 = time.perf_counter()
        try:
            value = _execute("run", tool, args)
            return ToolOutcome(True, value, duration_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001 - tool errors become error results
            return ToolOutcome(False, error=_short_error(exc), duration_ms=(time.perf_counter() - t0) * 1000)

    def canonicalize(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        try:
            return ToolOutcome(True, _execute("canon", tool, args))
        except Exception as exc:  # noqa: BLE001
            return ToolOutcome(False, error=_short_error(exc))

    def close(self) -> None:
        pass


def _harden_worker() -> None:
    import socket

    class _NoNetwork(socket.socket):
        def __init__(self, *a: Any, **k: Any) -> None:
            raise PermissionError("network access is disabled inside the tool sandbox")

    socket.socket = _NoNetwork  # type: ignore[misc]
    socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[assignment]
        PermissionError("network access is disabled inside the tool sandbox"))
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass  # Windows or a platform without RLIMIT_AS: timeout still applies


def _worker_main(conn: Any) -> None:
    _harden_worker()
    from sciai.tools.registry import load_builtin_tools

    load_builtin_tools()
    conn.send({"ready": True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        t0 = time.perf_counter()
        try:
            value = _execute(msg["op"], msg["tool"], msg["args"])
            conn.send({"ok": True, "value": value, "ms": (time.perf_counter() - t0) * 1000})
        except MemoryError:
            conn.send({"ok": False, "error": "MemoryError: tool exceeded the memory limit",
                       "ms": (time.perf_counter() - t0) * 1000})
        except Exception as exc:  # noqa: BLE001
            conn.send({"ok": False, "error": _short_error(exc), "trace": traceback.format_exc(limit=3),
                       "ms": (time.perf_counter() - t0) * 1000})


class SandboxRunner:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT, start_timeout: float = 60.0) -> None:
        self.timeout = timeout
        self.start_timeout = start_timeout
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._proc: Any = None
        self._conn: Any = None

    def _ensure(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=_worker_main, args=(child,), daemon=True, name="sciai-tools")
        proc.start()
        child.close()
        if not parent.poll(self.start_timeout):
            proc.kill()
            raise RuntimeError("tool sandbox failed to start")
        parent.recv()
        self._proc, self._conn = proc, parent

    def _kill(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.join(timeout=5)
        self._proc = self._conn = None

    def _call(self, op: str, tool: str, args: dict[str, Any], timeout: float) -> ToolOutcome:
        with self._lock:
            self._ensure()
            t0 = time.perf_counter()
            try:
                self._conn.send({"op": op, "tool": tool, "args": args})
                if not self._conn.poll(timeout):
                    self._kill()
                    return ToolOutcome(False, error=f"timed out after {timeout:.0f}s", timed_out=True,
                                       duration_ms=(time.perf_counter() - t0) * 1000)
                reply = self._conn.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
                self._kill()
                return ToolOutcome(False, error="tool worker crashed", duration_ms=(time.perf_counter() - t0) * 1000)
        if reply.get("ok"):
            return ToolOutcome(True, reply["value"], duration_ms=reply["ms"])
        return ToolOutcome(False, error=reply["error"], duration_ms=reply["ms"])

    def run(self, tool: str, args: dict[str, Any], timeout: float | None = None) -> ToolOutcome:
        return self._call("run", tool, args, timeout or self.timeout)

    def canonicalize(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        return self._call("canon", tool, args, min(self.timeout, 10.0))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                except (BrokenPipeError, OSError):
                    pass
            self._kill()
