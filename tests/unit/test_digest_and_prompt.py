"""Item 5: num_ctx is explicit and the digest is sized so a long graph still fits."""
from sciai.config import Config
from sciai.controller.loop import Controller
from sciai.graph.engine import GraphEngine
from sciai.graph.model import Layer, Node, NodeType, Status
from sciai.llm.client import estimate_tokens
from sciai.llm.roles import ROLES
from sciai.store.db import Database
from sciai.store.repository import Repository
from sciai.tools.sandbox import InlineRunner
from tests.fakes.fake_llm import ScriptedLLM


def test_long_graph_prompt_fits_context():
    db = Database(":memory:")
    engine = GraphEngine(Repository(db))
    engine.new_session("long")
    root = engine.add_node(Node(session_id="", layer=Layer.REASONING, type=NodeType.PROBLEM, title="Problem",
                                content="ROOT PROBLEM STATEMENT"))
    prev = root
    for i in range(800):
        n = engine.add_node(Node(session_id="", layer=Layer.REASONING, type=NodeType.TOOL_RESULT, title=f"s{i}",
                                 tool_name="sympy.expand", tool_inputs={"expr": f"(x+{i})**12"},
                                 result={"kind": "expr", "value": "x**12 + " * 30 + str(i)}), [prev.id])
        prev = n
    engine.set_status(n.id, Status.FAILED, "test")  # attention nodes are kept

    cfg = Config()
    ctl = Controller(engine, InlineRunner(), ScriptedLLM([]), cfg)
    prompt = ctl._prompt("controller", root, "LAST OBSERVATION")
    total = estimate_tokens(ROLES["controller"].system_prompt()) + estimate_tokens(prompt)
    assert total <= cfg.model.num_ctx - cfg.model.num_predict
    assert "ROOT PROBLEM STATEMENT" in prompt
    assert f"{engine.handle(n.id)} [failed]" in prompt
    assert "older nodes omitted" in prompt
    assert "LAST OBSERVATION" in prompt

    cfg.model.num_ctx = 4096  # a 12B model on 8GB gets a smaller window; still fits
    prompt = ctl._prompt("controller", root, "LAST OBSERVATION")
    assert estimate_tokens(ROLES["controller"].system_prompt()) + estimate_tokens(prompt) <= 4096 - 384
    db.close()


def test_ollama_request_sets_num_ctx(monkeypatch):
    import httpx

    from sciai.llm.client import OllamaClient

    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"action":"ask_user","question":"q"}'}})

    client = OllamaClient(Config().model)
    client._http = httpx.Client(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler))
    client.chat("sys", "user", {"type": "object"})
    assert seen["options"]["num_ctx"] == 8192 and seen["stream"] is False and seen["format"]


def test_ollama_must_be_local():
    import pytest

    from sciai.llm.client import OllamaClient

    cfg = Config().model
    cfg.ollama_url = "http://example.com:11434"
    with pytest.raises(ValueError):
        OllamaClient(cfg)
