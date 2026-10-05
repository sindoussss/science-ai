"""Typed configuration loaded from config/default.toml plus an optional user file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_FILE = Path(__file__).resolve().parents[2] / "config" / "default.toml"
USER_FILE = Path("~/.sciai/config.toml").expanduser()


@dataclass
class ModelConfig:
    name: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_url: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    num_predict: int = 384
    temperature: float = 0.0
    keep_alive: str = "30m"
    request_timeout: float = 240
    think: bool = False  # reasoning ("thinking") mode for models that have one; off by default


@dataclass
class ControllerConfig:
    max_steps: int = 14
    max_invalid_streak: int = 2
    auto_confirm_root: bool = False


@dataclass
class RiskConfig:
    stakes_threshold: int = 2
    min_confidence: float = 0.9
    structural_int_max: int = 10


@dataclass
class ToolsConfig:
    timeout: float = 20


@dataclass
class PhysicsConfig:
    # Range rules by quantity kind (absolute temperature >= 0 K, 0 <= efficiency <= 1, speed <= c).
    plausibility: bool = True


@dataclass
class DataConfig:
    alpha: float = 0.05            # significance level when the question gives none
    max_rows: int = 5_000_000      # a larger file is refused at import instead of exhausting memory
    max_file_mb: int = 400         # the tool sandbox has 3 GB; pandas needs several times the file size
    data_dir: str = "~/.sciai/datasets"  # imported files, stored by their SHA-256

    @property
    def path(self) -> Path:
        return Path(self.data_dir).expanduser()


@dataclass
class ChemConfig:
    # Tanimoto (Morgan radius 2, 2048 bits) at or above which a structure counts as a close
    # analog of a restricted one and is refused. Lower refuses more; see domains/chem/restricted.
    analog_tanimoto: float = 0.70
    max_passage_chars: int = 1200   # a literature passage is cut to this before it is shown
    corpus_dir: str = "~/.sciai/corpus"  # imported documents and their index

    @property
    def corpus_path(self) -> Path:
        return Path(self.corpus_dir).expanduser()


@dataclass
class StoreConfig:
    db_path: str = "~/.sciai/knowledge.db"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chem: ChemConfig = field(default_factory=ChemConfig)
    store: StoreConfig = field(default_factory=StoreConfig)

    @property
    def db_path(self) -> Path:
        return Path(self.store.db_path).expanduser()


def _apply(section: Any, values: dict[str, Any]) -> None:
    known = {f.name for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown config key {type(section).__name__}.{key}")
        default = getattr(type(section)(), key)
        # a quoted "false" is a non-empty string and would read as True; only real booleans pass
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{type(section).__name__}.{key} must be true or false, got {value!r}")
        setattr(section, key, value)


def load(paths: list[Path] | None = None) -> Config:
    cfg = Config()
    for path in paths if paths is not None else [DEFAULT_FILE, USER_FILE]:
        if not path.exists():
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))  # Notepad may add a BOM
        for name, values in data.items():
            if not hasattr(cfg, name):
                raise ValueError(f"unknown config section [{name}] in {path}")
            _apply(getattr(cfg, name), values)
    return cfg
