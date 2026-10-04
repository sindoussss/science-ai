"""Importing a data file: hash it, keep a copy, read it twice, record it.

Runs in the main process (the reading happens in the tool sandbox). An import:

1. hashes the file (SHA-256) and copies it to ``data_dir/<sha>.<ext>`` (atomically: a
   temporary file in the same folder, checked, then renamed), so a later edit or deletion of
   the original can't change what an analysis read;
2. runs ``data.load`` (pandas) and ``data.check_load`` (the csv module or openpyxl): the
   import is refused unless both reads agree on rows, columns, missing counts and sums;
3. records the dataset under the file's name. Its id is the bytes plus the read options, so
   importing the same file again finds the same record (and the graph reuses its nodes),
   while a changed cell is a new dataset.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sciai.config import DataConfig
from sciai.domains.data import frames
from sciai.domains.data.frames import DataError
from sciai.graph.model import Status, TERMINAL_BAD
from sciai.store.repository import DatasetRecord, Repository
from sciai.tools.sandbox import Runner

if TYPE_CHECKING:
    from sciai.graph.engine import GraphEngine

STORED_SUFFIX = {"csv": ".csv", "tsv": ".tsv", "xlsx": ".xlsx"}


@dataclass
class ImportResult:
    record: DatasetRecord
    descriptor: dict[str, Any]
    check: dict[str, Any]
    reused: bool  # these bytes, read this way, were imported before


def _store_copy(src: Path, sha: str, fmt: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{sha}{STORED_SUFFIX[fmt]}"
    if target.exists() and frames.file_sha256(target) == sha:
        return target
    fd, tmp = tempfile.mkstemp(prefix=".import-", dir=data_dir)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        if frames.file_sha256(tmp) != sha:
            raise DataError(f"{src.name} changed while it was being copied; import it again")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def import_file(path: str | Path, repo: Repository, runner: Runner, cfg: DataConfig,
                name: str | None = None) -> ImportResult:
    src = Path(path).expanduser()
    if not src.is_file():
        raise DataError(f"{src} is not a file")
    fmt = frames.format_of(src)
    size_mb = src.stat().st_size / 1e6
    if size_mb > cfg.max_file_mb:
        raise DataError(f"{src.name} is {size_mb:.0f} MB; files above {cfg.max_file_mb} MB are refused before "
                        "loading (raise [data] max_file_mb if the machine has the memory)")
    sha = frames.file_sha256(src)
    stored = _store_copy(src, sha, fmt, cfg.path)
    name = (name or src.name).strip()
    load = runner.run("data.load", {"path": str(stored), "format": fmt, "sha256": sha, "name": name,
                                    "max_rows": int(cfg.max_rows)})
    if not load.ok:
        raise DataError(f"{src.name} could not be read: {load.error}")
    desc = load.value["result"]
    check = runner.run("data.check_load", {"loaded": desc})
    verdict = check.value["result"] if check.ok else {"outcome": "inconclusive", "error": check.error}
    if verdict.get("outcome") != "pass":
        problems = "; ".join(verdict.get("problems") or [verdict.get("error") or "the second read failed"])
        raise DataError(f"{src.name} reads differently with {verdict.get('method', 'a second reader')}: {problems}")
    options = desc["source"]["options"]
    dataset_id = frames.source_key(sha, fmt, options)
    reused = repo.dataset(dataset_id) is not None
    rec = repo.add_dataset(DatasetRecord(
        id=dataset_id, sha256=sha, name=name, format=fmt, options=options, stored_path=str(stored),
        original_path=str(src.resolve()), rows=int(desc["rows"]), columns=len(desc["columns"]),
        schema=desc["columns"]))
    return ImportResult(rec, desc, verdict, reused)


def load_args(rec: DatasetRecord, cfg: DataConfig) -> dict[str, Any]:
    """The ``data.load`` arguments that rebuild this dataset's descriptor (its options pinned,
    so the read is the one that was checked at import)."""
    return {"path": rec.stored_path, "format": rec.format, "sha256": rec.sha256, "name": rec.name,
            "options": rec.options, "max_rows": int(cfg.max_rows)}


def remove_dataset(rec: DatasetRecord, engine: "GraphEngine") -> list[str]:
    """The user removes an imported file: every node that read it is invalidated (in every
    session), which cascades to every result, plot and answer built on it. The record goes, and
    the stored copy too unless another record (the same bytes under another name or read
    another way) still uses it. Returns the invalidated dataset node ids."""
    hit = []
    for node in engine.repo.nodes_reading(rec.sha256):
        same_read = (node.tool_inputs or {}).get("options") == rec.options
        if same_read and node.status not in TERMINAL_BAD:
            engine.set_status(node.id, Status.INVALIDATED, f"dataset {rec.name} was removed")
            hit.append(node.id)
    engine.repo.delete_dataset(rec.id)
    if not engine.repo.datasets_with_sha(rec.sha256):
        Path(rec.stored_path).unlink(missing_ok=True)
    return hit
