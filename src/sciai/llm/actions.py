"""Controller actions: the only things the model may emit.

The schema is flat (one object, ``action`` picks the variant) because flat
schemas constrain small models reliably under Ollama's structured output.
Per-action requirements are checked here with messages written for the retry.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

ACTIONS = ("formalize", "call_tool", "run_check", "finish", "ask_user")
_HANDLE = {"type": "string", "pattern": "^n[0-9]+$"}

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"enum": list(ACTIONS)},
        "thought": {"type": "string", "maxLength": 300},
        "tool": {"type": "string", "maxLength": 60},
        "args": {"type": "object"},
        "depends_on": {"type": "array", "items": _HANDLE, "maxItems": 12},
        "replaces": _HANDLE,
        "title": {"type": "string", "maxLength": 80},
        "node": _HANDLE,
        "check": {"type": "string", "maxLength": 60},
        "answer_template": {"type": "string", "maxLength": 800},
        "answer_nodes": {"type": "array", "items": _HANDLE, "maxItems": 12},
        "question": {"type": "string", "maxLength": 500},
        "statement": {"type": "string", "maxLength": 2000},
        "assumptions": {"type": "object", "additionalProperties": {"type": "string"}},
        "goal": {"type": "object"},
    },
    "required": ["action"],
}

_REQUIRED = {
    "formalize": ("statement",),
    "call_tool": ("tool", "args"),
    "run_check": ("node",),
    "finish": ("answer_template", "answer_nodes"),
    "ask_user": ("question",),
}


class ActionError(ValueError):
    """Invalid model output; the message is fed back to the model on the retry."""


@dataclass
class Action:
    kind: str
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ActionError("reply was not valid JSON; reply with exactly one JSON object")


def parse_action(text: str, allowed: tuple[str, ...] = ACTIONS) -> Action:
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ActionError("reply must be a JSON object")
    try:
        jsonschema.validate(data, ACTION_SCHEMA)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(p) for p in exc.absolute_path) or "action"
        raise ActionError(f"invalid field {path}: {exc.message}") from None
    kind = data["action"]
    if kind not in allowed:
        raise ActionError(f"action must be one of {', '.join(allowed)}")
    missing = [f for f in _REQUIRED[kind] if f not in data]
    if missing:
        raise ActionError(f"{kind} needs field(s): {', '.join(missing)}")
    return Action(kind, data)


PLACEHOLDER = re.compile(r"\{\{\s*(n[0-9]+)\s*\}\}")
_DIGIT = re.compile(r"\d")


def check_answer_template(template: str, answer_nodes: list[str]) -> list[str]:
    """Hard rule: no digits outside placeholders; placeholders must name answer nodes."""
    stripped = PLACEHOLDER.sub("", template)
    if _DIGIT.search(stripped):
        raise ActionError("answer_template must not contain digits; put every value in a {{node}} placeholder")
    refs = PLACEHOLDER.findall(template)
    if not refs:
        raise ActionError("answer_template must reference at least one {{node}}")
    unknown = [r for r in refs if r not in answer_nodes]
    if unknown:
        raise ActionError(f"placeholders {unknown} are not listed in answer_nodes")
    return refs
