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

from sciai.controller.recipes import ROUTES

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
        # formalize, Phase 2: the problem type picks the default assumption checklist (an
        # unknown type gets the general one); givens are quantities {"value", "unit", "kind"?}.
        "problem_type": {"type": "string", "maxLength": 40},
        "givens": {"type": "object", "maxProperties": 20,
                   "additionalProperties": {"type": "object"}},
        "modelling_assumptions": {"type": "array", "maxItems": 8,
                                  "items": {"type": "string", "maxLength": 120}},
        # formalize, Phase 4: the chemistry operation the question asks for. The controller
        # declines when no registered tool performs it, so an operation the model invents
        # (a synthesis route, a dose) is declined by having no tool, not by its wording.
        "operation": {"type": "string", "maxLength": 40},
        # formalize, the router: which recipe answers the question, from a closed list. The
        # enum is the whole of the model's planning freedom. "none" means out of scope, and the
        # two delegated routes hand the question to the dataset and chemistry paths.
        "recipe": {"enum": list(ROUTES)},
        # The recipe's slots, flat and all strings: a slot holds one number, one unit, one
        # expression or one name. Nothing nested, because a nested object is what the model
        # got wrong, and code coerces each string to the kind its slot declares.
        "slots": {"type": "object", "maxProperties": 24,
                  "additionalProperties": {"type": ["string", "number", "boolean"]}},
    },
    "required": ["action"],
}

_REQUIRED = {
    "formalize": ("statement", "recipe"),
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


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove reasoning a thinking model wrote into its reply, so it is never parsed as the action.

    Handles a complete <think>...</think> block, a block cut off by the token limit (no closing
    tag: everything after <think> is reasoning), and a reply whose opening tag was part of the
    prompt template (only </think>: everything before it is reasoning)."""
    text = _THINK.sub("", text)
    lower = text.lower()
    if "</think>" in lower:
        text = text[lower.rfind("</think>") + len("</think>"):]
        lower = text.lower()
    if "<think>" in lower:
        text = text[: lower.find("<think>")]
    return text


def _extract_json(text: str) -> Any:
    raw = text
    text = strip_think(text).strip()
    if not text and "<think>" in raw.lower():
        raise ActionError("reply was cut off while thinking; reply with exactly one JSON object and no reasoning")
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
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_answer_template(template: str, answer_nodes: list[str], question: str = "") -> list[str]:
    """Hard rule: a computed value only ever reaches the answer through a placeholder.

    The rule used to be "no digits at all outside a placeholder", which also threw out the
    question's own numbers. "y(2) = {{n5}}" names the point the question asked about and is the
    clearest way to write that answer; refusing it taught the model nothing, because there was
    no wording that would pass. So a bare number is allowed when the question contains it, and
    refused otherwise -- which is still the whole of the invariant: a number the question never
    mentioned can only have come from the model, and the model does not compute.
    """
    stripped = PLACEHOLDER.sub("", template)
    if _DIGIT.search(stripped):
        from sciai.controller.provenance import numbers_in

        asked = numbers_in(question)
        invented = sorted({n for n in _NUMBER.findall(stripped) if not numbers_in(n) <= asked})
        if invented:
            raise ActionError(
                f"answer_template may not contain the number(s) {', '.join(invented)}: they are "
                "not in the question, so put every computed value in a {{node}} placeholder")
    refs = PLACEHOLDER.findall(template)
    if not refs:
        raise ActionError("answer_template must reference at least one {{node}}")
    unknown = [r for r in refs if r not in answer_nodes]
    if unknown:
        raise ActionError(f"placeholders {unknown} are not listed in answer_nodes")
    return refs


# ask_user exists for one thing: a required value the question itself does not contain. A
# formula, a constant, a conversion factor and a method are all things the system holds, so
# asking the user for one is the model offloading its own job and leaves the user stuck. Each
# rule names what to do instead, because the message is what the retry sees.
_ASK_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bformula|\bequations?\b.{0,20}\b(use|need|appl)|\bexpression\b.{0,20}\buse",
                re.IGNORECASE),
     "the recipe holds the formula; ask only for a value the question leaves out"),
    (re.compile(r"\bconstants?\b|\bvalue of\s+(?:g|c|h|pi|e)\b|\bgravit\w*\b.{0,20}\bvalue|"
                r"\bplanck|\bspeed of light", re.IGNORECASE),
     "constants come from phys.constant, never from the user"),
    (re.compile(r"\bconversion\s+factor|"
                r"\bhow many\s+\w+\s+(?:\w+\s+){0,2}(?:in|per)\s+(?:a|an|one)\s", re.IGNORECASE),
     "unit conversion is units.convert's job, not the user's"),
    (re.compile(r"\bmethods?\b|\bapproach\b|\bwhich tool\b|\bhow should i\b|\bshould i use\b",
                re.IGNORECASE),
     "choosing the method is your job, not the user's"),
)


def check_question(question: str) -> None:
    """What ask_user may ask for. Raises ActionError with what to do instead."""
    text = question.strip()
    if not text:
        raise ActionError("ask_user needs a question")
    for pattern, instead in _ASK_FORBIDDEN:
        if pattern.search(text):
            raise ActionError(f"ask_user may not ask about that: {instead}")
