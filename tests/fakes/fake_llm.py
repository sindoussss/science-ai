"""Scripted stand-in for the local model, so tests run without Ollama.

Every scripted reply is checked against the schema the controller passed to ``chat``, which is
the same schema a real Ollama call generates under. A mock may therefore only say things a real
model could say. Without that check the mocks drifted: the suite passed on replies carrying a
``statement`` the request never required, while every live reply was rejected for not having
one. A test that means to send malformed output asks for it explicitly (``loose=True``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import jsonschema

from sciai.llm.client import LLMReply

Step = dict[str, Any] | str | Callable[[str], "dict[str, Any] | str"]


class ScriptedLLM:
    model_name = "scripted"

    def __init__(self, steps: list[Step], loose: bool = False) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, str]] = []
        self.loose = loose

    def chat(self, system: str, user: str, schema: dict[str, Any]) -> LLMReply:
        self.calls.append({"system": system, "user": user})
        if not self.steps:
            raise AssertionError(f"model called more often than scripted; last prompt:\n{user}")
        step = self.steps.pop(0)
        if callable(step):
            step = step(user)
        text = step if isinstance(step, str) else json.dumps(step)
        if not self.loose:
            self._check(text, schema)
        return LLMReply(text)

    @staticmethod
    def _check(text: str, schema: dict[str, Any]) -> None:
        """A scripted reply must satisfy the schema the model was given, or it is not a reply a
        real model could have produced and the test proves nothing."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return  # deliberately malformed text, which the controller is meant to reject
        if not isinstance(data, dict):
            return
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            raise AssertionError(
                "scripted reply does not satisfy the schema the controller sent to the model, so "
                f"no real model could have produced it: {exc.message}\nreply: {text}") from None


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
