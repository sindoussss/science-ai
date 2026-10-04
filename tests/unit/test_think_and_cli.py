"""The think setting, <think> stripping before action parsing, and the CLI model override."""
from __future__ import annotations

import json

import httpx
import pytest

from sciai.config import DEFAULT_FILE, Config, load
from sciai.llm.actions import ActionError, parse_action, strip_think
from sciai.llm.client import OllamaClient, full_tag

FINISH = {"action": "finish", "answer_template": "Answer: {{n2}}", "answer_nodes": ["n2"]}


# ------------------------------------------------------------------ stripping
@pytest.mark.parametrize("text", [
    "<think>maybe {x: 1} or {y}</think>" + json.dumps(FINISH),          # braces inside the reasoning
    "<THINK>\nlet me see\n</THINK>\n" + json.dumps(FINISH),               # any case, newlines
    "the template opened the tag\n</think>" + json.dumps(FINISH),        # only the closing tag
    "<think>a</think><think>b {}</think>```json\n" + json.dumps(FINISH) + "\n```",
])
def test_reasoning_is_removed_before_the_action_is_parsed(text):
    assert parse_action(text).kind == "finish"


def test_reply_cut_off_while_thinking_gets_a_clear_retry_message():
    with pytest.raises(ActionError, match="cut off while thinking"):
        parse_action("<think>first I will integrate {x^2")


def test_strip_think_leaves_plain_replies_alone():
    plain = json.dumps(FINISH)
    assert strip_think(plain) == plain
    assert strip_think(plain + "<think>trailing, unclosed") == plain


# ---------------------------------------------------------------------- client
def _client(cfg: Config, seen: list[dict]) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:latest", "model": "llama3.1:latest"}]})
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "<think>hm</think>" + json.dumps(FINISH)},
                                         "prompt_eval_count": 10, "eval_count": 5})

    c = OllamaClient(cfg.model)
    c._http = httpx.Client(base_url=cfg.model.ollama_url, transport=httpx.MockTransport(handler))
    return c


@pytest.mark.parametrize("think", [False, True])
def test_think_is_sent_explicitly(think):
    cfg = Config()
    cfg.model.think = think
    seen: list[dict] = []
    reply = _client(cfg, seen).chat("sys", "user", {"type": "object"})
    assert seen[0]["think"] is think
    assert parse_action(reply.text).kind == "finish"


def test_status_matches_an_untagged_model_name():
    cfg = Config()
    cfg.model.name = "llama3.1"
    assert _client(cfg, []).status()["loaded"] is True
    assert full_tag("llama3.1") == "llama3.1:latest"
    assert full_tag("qwen3:8b") == "qwen3:8b"
    assert full_tag("hf.co/org/model") == "hf.co/org/model:latest"
    assert full_tag("localhost:5000/model") == "localhost:5000/model:latest"


# ---------------------------------------------------------------------- config
def test_think_defaults_to_false_and_must_be_a_boolean(tmp_path):
    assert Config().model.think is False
    assert load([DEFAULT_FILE]).model.think is False
    p = tmp_path / "c.toml"
    p.write_text('[model]\nthink = "false"\n')  # a string would read as True
    with pytest.raises(ValueError, match="true or false"):
        load([p])
    p.write_text("[model]\nthink = true\n")
    assert load([p]).model.think is True


# ------------------------------------------------------------------------- CLI
def test_cli_overrides_model_and_think_and_passes_qt_args_through():
    from sciai.__main__ import apply_overrides, parse_args

    args, rest = parse_args(["--model", "qwen3:8b", "--think", "-platform", "offscreen"])
    cfg = Config()
    apply_overrides(cfg, args)
    assert (cfg.model.name, cfg.model.think) == ("qwen3:8b", True)
    assert rest == ["-platform", "offscreen"]
    args, _ = parse_args(["--no-think"])
    apply_overrides(cfg, args)
    assert cfg.model.think is False
    args, _ = parse_args([])
    keep = Config()
    apply_overrides(keep, args)
    assert (keep.model.name, keep.model.think) == (Config().model.name, False)


def test_config_file_saved_with_a_bom_still_loads(tmp_path):
    p = tmp_path / "c.toml"
    p.write_bytes("﻿[model]\nname = \"qwen3:8b\"\n".encode("utf-8"))  # e.g. saved by Windows Notepad
    assert load([p]).model.name == "qwen3:8b"
