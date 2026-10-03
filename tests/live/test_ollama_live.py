"""Run manually with a local Ollama: pytest -m live tests/live"""
from __future__ import annotations

import pytest

from sciai.config import load
from sciai.runtime import build_runtime

pytestmark = pytest.mark.live


def test_live_multistep(tmp_path):
    cfg = load()
    cfg.controller.auto_confirm_root = True
    rt = build_runtime(cfg, db_path=tmp_path / "live.db")
    try:
        rt.engine.new_session("live")
        res = rt.controller.run("Differentiate x^2*sin(x) and evaluate the derivative at x = pi")
        assert res.status in ("answered", "escalated"), res
        if res.status == "answered":
            assert "pi" in res.answer
    finally:
        rt.close()
