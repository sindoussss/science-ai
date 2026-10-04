"""Wires the store, graph engine, tool runner, model and controller together."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sciai.config import Config
from sciai.controller.loop import ConfirmRoot, Controller
from sciai.graph.engine import GraphEngine
from sciai.graph.events import EventBus
from sciai.llm.client import LLM, OllamaClient
from sciai.store.db import Database
from sciai.store.repository import Repository
from sciai.tools.sandbox import Runner, SandboxRunner


@dataclass
class Runtime:
    cfg: Config
    db: Database
    repo: Repository
    engine: GraphEngine
    runner: Runner
    llm: LLM
    controller: Controller
    flagged_on_start: int

    def close(self) -> None:
        self.runner.close()
        self.db.close()


def build_runtime(cfg: Config, *, db_path: str | Path | None = None, llm: LLM | None = None,
                  runner: Runner | None = None, confirm_root: ConfirmRoot | None = None,
                  bus: EventBus | None = None) -> Runtime:
    db = Database(db_path if db_path is not None else cfg.db_path)
    repo = Repository(db)
    flagged = repo.flag_unverified()  # unverified results from earlier runs become hints
    engine = GraphEngine(repo, bus)
    runner = runner or SandboxRunner(timeout=cfg.tools.timeout)
    llm = llm or OllamaClient(cfg.model)
    controller = Controller(engine, runner, llm, cfg, confirm_root)
    return Runtime(cfg, db, repo, engine, runner, llm, controller, flagged)
