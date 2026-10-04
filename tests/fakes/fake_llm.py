"""Scripted stand-in for the local model, so tests run without Ollama."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from sciai.llm.client import LLMReply

Step = dict[str, Any] | str | Callable[[str], "dict[str, Any] | str"]


class ScriptedLLM:
    model_name = "scripted"

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, str]] = []

    def chat(self, system: str, user: str, schema: dict[str, Any]) -> LLMReply:
        self.calls.append({"system": system, "user": user})
        if not self.steps:
            raise AssertionError(f"model called more often than scripted; last prompt:\n{user}")
        step = self.steps.pop(0)
        if callable(step):
            step = step(user)
        return LLMReply(step if isinstance(step, str) else json.dumps(step))


def last_handle(prompt: str, tool: str | None = None, status: str | None = None) -> str:
    """Latest handle in the digest, optionally filtered by tool and status."""
    best = None
    for line in prompt.splitlines():
        m = re.match(r"^(n\d+) \[([^\]]+)\] (\S+?)\(", line) or re.match(r"^(n\d+) \[([^\]]+)\] (\S+)", line)
        if not m:
            continue
        h, st, t = m.groups()
        if tool and t != tool:
            continue
        if status and not st.startswith(status):
            continue
        best = h
    if best is None:
        raise AssertionError(f"no node for tool={tool} status={status} in prompt:\n{prompt}")
    return best
