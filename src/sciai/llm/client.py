"""The one local model, served by Ollama. Localhost only; num_ctx is always explicit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from sciai.config import ModelConfig

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
CHARS_PER_TOKEN = 3.0  # conservative for English + math; real tokenizers average ~3.5-4


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLM(Protocol):
    model_name: str

    def chat(self, system: str, user: str, schema: dict[str, Any]) -> LLMReply: ...


class ContextOverflow(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: ModelConfig) -> None:
        host = urlparse(cfg.ollama_url).hostname
        if host not in LOCAL_HOSTS:
            raise ValueError(f"Ollama URL must be local (got {host!r}); no outside network at runtime")
        self.cfg = cfg
        self.model_name = cfg.name
        self._http = httpx.Client(base_url=cfg.ollama_url, timeout=cfg.request_timeout, trust_env=False)

    def chat(self, system: str, user: str, schema: dict[str, Any]) -> LLMReply:
        budget = self.cfg.num_ctx - self.cfg.num_predict
        if estimate_tokens(system) + estimate_tokens(user) > budget:
            raise ContextOverflow(f"prompt does not fit num_ctx={self.cfg.num_ctx}")
        body = {
            "model": self.cfg.name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
                "temperature": self.cfg.temperature,
            },
        }
        try:
            resp = self._http.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc
        data = resp.json()
        return LLMReply(
            text=data.get("message", {}).get("content", ""),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def status(self) -> dict[str, Any]:
        """Loaded model and VRAM use, for the Env tab. Never raises."""
        try:
            ps = self._http.get("/api/ps", timeout=3).json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return {"reachable": False, "model": self.cfg.name}
        for m in ps.get("models", []):
            if m.get("name") == self.cfg.name or m.get("model") == self.cfg.name:
                return {"reachable": True, "model": self.cfg.name, "loaded": True,
                        "vram_bytes": m.get("size_vram"), "size_bytes": m.get("size")}
        return {"reachable": True, "model": self.cfg.name, "loaded": False}
