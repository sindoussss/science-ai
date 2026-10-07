"""Controller actions: the only things the model may emit.

One schema per call, built by ``action_schema``, serves as both the output format sent to
Ollama and the rules the reply is validated against. That is deliberate and it is the fix for
the bug that broke every live run of the recipe branch: the two used to be separate, the
request required only ``action``, and the validator required three fields the grammar never
forced the model to write.

An action is one flat object -- flat schemas constrain small models reliably -- with ``action``
naming which one it is, its own fields, and its own required list. When a call allows several
actions, the schema is an ``anyOf`` of those objects, so each keeps its own requirements.
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

# Every field any action may carry, defined once. ``FIELDS`` says which action may use which,
# and ``REQUIRED`` which it must. Nothing else defines a field: the schema sent to Ollama and
# the schema the reply is validated against are built from these by ``action_schema``, so the
# two cannot drift -- which is what went wrong before. The request declared only ``action``
# required, so Ollama's grammar let the model stop after {"action","recipe","slots"}, while the
# validator demanded "statement" as well. Every live reply was rejected twice and the run died
# at formalization; no mock could catch it, because a mock writes whatever the test wrote.
PROPERTIES: dict[str, Any] = {
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
    # formalize: a restatement of the problem, for the user to confirm or edit. Optional,
    # because code writes a precise one from the recipe and its slots when the model leaves it
    # out, and a field the model must produce is a field the model can fail to produce.
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
}

# What each action may carry, beyond "action" and "thought", which every one may.
FIELDS: dict[str, tuple[str, ...]] = {
    "formalize": ("recipe", "slots", "statement", "problem_type", "goal", "givens",
                  "assumptions", "modelling_assumptions", "operation"),
    "call_tool": ("tool", "args", "depends_on", "replaces", "title"),
    "run_check": ("node", "check"),
    "finish": ("answer_template", "answer_nodes"),
    "ask_user": ("question",),
}

# What each action must carry. These become the schema's own "required" list, and the only
# requirement check made on a reply is that list, so the request and the validator agree by
# construction. formalize asks for the two fields that decide everything -- which recipe, and
# its slots -- and nothing more.
REQUIRED: dict[str, tuple[str, ...]] = {
    "formalize": ("recipe", "slots"),
    "call_tool": ("tool", "args"),
    "run_check": ("node",),
    "finish": ("answer_template", "answer_nodes"),
    "ask_user": ("question",),
}


def action_variant(kind: str) -> dict[str, Any]:
    """The schema for one action: its own fields, and its own required list."""
    props = {"action": {"enum": [kind]}, "thought": PROPERTIES["thought"]}
    props.update({f: PROPERTIES[f] for f in FIELDS[kind]})
    return {"type": "object", "properties": props, "required": ["action", *REQUIRED[kind]]}


def action_schema(allowed: tuple[str, ...] = ACTIONS) -> dict[str, Any]:
    """The one schema for a call: sent to Ollama as the output grammar and used to validate
    the reply. One allowed action is a plain object, so its required fields are forced; several
    are an ``anyOf`` of those objects, one per action, each with its own required fields.

    The ``anyOf`` is what lets the loop's five actions each require their own fields, which one
    flat object cannot express. If a model ever generates badly against the union (the trace
    files from ``live_check.py --trace`` would show it: a reply that is valid JSON but not a
    valid action), the fallback is to return a flat object here, merging every allowed action's
    fields and requiring only ``action``; ``parse_action`` keeps working either way, because it
    checks whatever ``required`` list the schema it was handed carries.
    """
    variants = [action_variant(k) for k in allowed]
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


def variant_of(schema: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """The variant inside a schema that covers ``kind``, so validation uses the schema that was
    actually sent rather than a second copy of it."""
    for v in schema.get("anyOf", [schema]):
        if kind in (v.get("properties", {}).get("action", {}).get("enum") or ()):
            return v
    return None


# Kept for callers that validate a reply without a call in front of it (the UI's replay path
# and the tests): the whole contract, every action.
ACTION_SCHEMA: dict[str, Any] = action_schema()


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


def parse_action(text: str, allowed: tuple[str, ...] = ACTIONS,
                 schema: dict[str, Any] | None = None) -> Action:
    """Parse and validate one reply against the schema the model was given.

    ``schema`` is the object that was sent to Ollama as the output format. Passing it in is the
    whole point: the reply is judged by the rules the model was actually generating under, so a
    reply the grammar permitted can never be rejected here for a field the request did not ask
    for. Omitting it validates against the full contract, which is the same thing built the
    same way.
    """
    schema = action_schema(allowed) if schema is None else schema
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise ActionError("reply must be a JSON object")
    kind = data.get("action")
    if not isinstance(kind, str) or kind not in allowed:
        raise ActionError(f"action must be one of {', '.join(allowed)}")
    variant = variant_of(schema, kind)
    if variant is None:
        raise ActionError(f"action must be one of {', '.join(allowed)}")
    try:
        jsonschema.validate(data, variant)
    except jsonschema.ValidationError as exc:
        path = ".".join(str(p) for p in exc.absolute_path) or "action"
        if exc.validator == "required":
            # The schema's own required list, so this message can only ever name a field the
            # request asked the model for.
            missing = [f for f in variant["required"] if f not in data]
            raise ActionError(f"{kind} needs field(s): {', '.join(missing)}") from None
        raise ActionError(f"invalid field {path}: {exc.message}") from None
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
