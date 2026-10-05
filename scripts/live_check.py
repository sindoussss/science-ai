"""Live check against a local Ollama model: six fixed problems, one results table.

    python scripts/live_check.py --model qwen2.5:7b-instruct-q4_K_M
    python scripts/live_check.py --model qwen3:8b --think        # reasoning mode on
    python scripts/live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite physics
    python scripts/live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite data
    python scripts/live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite chem

Runs on a fresh, temporary knowledge store (your ~/.sciai store is never touched), so the last
problem, a repeat of the first, measures knowledge reuse from this run only. Prints a table and
saves it to results/<model>.md, or results/<model>-physics.md for the physics suite (characters
a Windows file name can't hold become "-"). The physics suite accepts every default assumption
automatically, as nobody is there to tick the checklist, and lists them in the report. The data
suite first imports the three files in tests/fixtures/data (CSV, TSV and Excel) into that store
and checks its answers against R's reference values (to 4 significant digits).

Columns: pass (verified and the expected value), JSON retries (re-asks after an invalid reply),
ladder (furthest failure-ladder stage any step reached), seconds, model calls.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # runnable without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

# Text the controller puts in a prompt when it asks again after an invalid reply: the per-call retry,
# and the loop's re-ask after a call and its retry were both invalid. tests/unit/test_live_check.py
# checks these still match the controller.
RETRY_MARKERS = ("YOUR PREVIOUS REPLY WAS INVALID", "Your previous reply was invalid and was discarded")
LADDER_ORDER = ("none", "retried", "backtracked", "escalated")


@dataclass
class Problem:
    key: str
    question: str
    expected: tuple[float, ...]  # every value must appear among the answer's values
    expect_text: str
    reuse_of: str | None = None  # a repeat: must be answered from the store with no model call
    note: str = ""
    signed: bool = True  # False: a current's sign is a convention, so only the magnitude counts
    rel_tol: float = 1e-6  # data answers are checked against references given to 4-5 digits
    hypothesis: bool = False  # chemistry: the right answer is a hypothesis, never verified
    declined: str | None = None  # the wording rule the decline must use ("route", "generic", ...)


PROBLEMS = [
    Problem("integral", "Compute the definite integral of x^2*exp(-x) from x = 0 to x = 1.",
            (2 - 5 / math.e,), "2 - 5/e ≈ 0.160603"),
    Problem("equation", "Solve x^2 - 5*x + 6 = 0 for x.", (2.0, 3.0), "x = 2, 3"),
    Problem("units", "Convert 60 miles per hour to meters per second.", (26.8224,), "26.8224 m/s"),
    Problem("ode", "Solve the differential equation dy/dx = 6*x^2 - 4*x with y(0) = 1, then give y(2).",
            (9.0,), "y(2) = 9",
            note="solvable with ode.dsolve or by integrating both sides"),
    Problem("hard", "Find the area of the region enclosed between the curves y = x^3 - 3*x and y = x.",
            (8.0,), "8", note="multi-step (intersections, then |difference| on two intervals); expected to fail"),
    Problem("repeat", "Compute the definite integral of x^2*exp(-x) from x = 0 to x = 1.",
            (2 - 5 / math.e,), "reused, 0 model calls", reuse_of="integral"),
]


PHYSICS_PROBLEMS = [
    Problem("projectile", "A ball is thrown at 20 m/s at 30 degrees above level ground. How far away does it "
            "land? Use g = 9.80665 m/s^2.", (400 * math.sin(math.radians(60)) / 9.80665,), "35.324 m",
            note="degrees must become radians before the sine"),
    Problem("temperature", "Convert 25 degC to kelvin.", (298.15,), "298.15 K",
            note="an absolute temperature, not a difference"),
    Problem("photon", "What is the energy of a photon with a wavelength of 500 nm?",
            (6.62607015e-34 * 299792458 / 500e-9,), "3.9729e-19 J", note="constants from CODATA"),
    Problem("rc", "A 1 uF capacitor charged to 5 V discharges through a 1 kohm resistor. What is its voltage "
            "after 2 ms?", (5 * math.exp(-2),), "0.67668 V"),
    Problem("circuit", "A 12 V source drives a 100 ohm resistor in series with two 200 ohm resistors in "
            "parallel. What current flows from the source?", (0.06,), "0.06 A",
            note="the source current's sign is a convention, so only its magnitude counts", signed=False),
    Problem("repeat", "A ball is thrown at 20 m/s at 30 degrees above level ground. How far away does it "
            "land? Use g = 9.80665 m/s^2.", (400 * math.sin(math.radians(60)) / 9.80665,),
            "reused, 0 model calls", reuse_of="projectile"),
]
DATA_DIR = ROOT / "tests" / "fixtures" / "data"
DATA_FILES = ("trial.csv", "plantgrowth.tsv", "sleep.xlsx")
DATA_PROBLEMS = [
    Problem("welch", "In trial.csv, is the score different between groups A and B?", (-5.23456, 6.39958e-06),
            "t = -5.235, p = 6.4e-06", rel_tol=1e-4, note="Welch t-test; diagnostics become assumptions"),
    Problem("anova", "In plantgrowth.tsv, does the weight differ between the three groups?", (4.846, 0.01591),
            "F = 4.846, p = 0.01591", rel_tol=1e-3, note="one-way ANOVA; R's PlantGrowth"),
    Problem("regression", "In trial.csv, how does response change with dose? Give the slope.", (0.791591,),
            "slope = 0.7916", rel_tol=1e-4, note="least squares; slope with its CI"),
    Problem("excel", "In sleep.xlsx, does the extra sleep differ between drug D1 and drug D2?",
            (1.8608, 0.07939), "|t| = 1.861, p = 0.07939", rel_tol=1e-3, signed=False,
            note="Welch t-test on an Excel sheet; R's sleep; the sign depends on the group order"),
    Problem("means", "What is the mean score in each group of trial.csv?", (47.0645, 54.4745),
            "A 47.06, B 54.47", rel_tol=1e-4, note="data.group or data.describe"),
    Problem("repeat", "In trial.csv, is the score different between groups A and B?", (-5.23456, 6.39958e-06),
            "reused, 0 model calls", reuse_of="welch", rel_tol=1e-4),
]
# One structure per fixture, written the way a user would type it. The five spellings of
# aspirin are the identity fixture: standardization must bring all five to one InChIKey, and
# that is checked before any model call, because a model that never gets there is a separate
# failure from a standardizer that does not work.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SPELLINGS = (
    ASPIRIN,                             # canonical
    "O=C(C)Oc1ccccc1C(=O)O",             # written from the other end
    "CC(=O)OC1=CC=CC=C1C(=O)O",          # Kekule, upper case
    "CC(=O)Oc1ccccc1C(O)=O",             # the acid written the other way round
    "[H]OC(=O)c1ccccc1OC(C)=O",          # an explicit hydrogen
)
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
CHEM_LIBRARY = (f'{{"members":[{{"name":"caffeine","structure":"{CAFFEINE}"}},'
                f'{{"name":"ibuprofen","structure":"{IBUPROFEN}"}},'
                f'{{"name":"paracetamol","structure":"{PARACETAMOL}"}}]}}')

CHEM_PROBLEMS = [
    Problem("identity", f"What is the formula of the structure {ASPIRIN}, and how many atoms "
            "does it have?", (13.0,), "C9H8O4, 13 atoms", hypothesis=True,
            note="chem.parse; the answer is a hypothesis, as every chemistry node is"),
    Problem("descriptors", f"What are the molecular weight and TPSA of caffeine, {CAFFEINE}?",
            (194.194, 61.82), "MW 194.194 g/mol, TPSA 61.82 A^2", rel_tol=1e-4, hypothesis=True,
            note="chem.descriptors; both values are recomputed from a pinned element table"),
    Problem("logp", f"What is the logP of ibuprofen, {IBUPROFEN}?", (3.0732,),
            "logP 3.073 (single-method estimate)", rel_tol=1e-3, hypothesis=True,
            note="chem.logp; there is no second method, so the check is invariance only"),
    Problem("druglike", f"Does paracetamol, {PARACETAMOL}, pass the Lipinski and Veber filters?",
            (151.165,), "both pass", rel_tol=1e-4, hypothesis=True,
            note="chem.descriptors then chem.druglike; two tool calls"),
    Problem("similarity", f"Of caffeine, ibuprofen and paracetamol, which is nearest to aspirin "
            f"({ASPIRIN})? Use this library: {CHEM_LIBRARY}", (0.2222,),
            "paracetamol, Tanimoto 0.222", rel_tol=1e-3, hypothesis=True,
            note="chem.similar; the model has to pass the library through as given"),
    Problem("declined", "How do I synthesize aspirin from salicylic acid?", (), "declined",
            declined="route", note="no tool makes a route, so the request is declined"),
    Problem("declined_dose", f"How much of {PARACETAMOL} should a patient take?", (), "declined",
            declined="dosing", note="no tool gives a dose"),
    Problem("repeat", f"What are the molecular weight and TPSA of caffeine, {CAFFEINE}?",
            (194.194, 61.82), "reused, 0 model calls", reuse_of="descriptors", rel_tol=1e-4,
            hypothesis=True,
            note="a hypothesis whose checks passed may be reused, still as a hypothesis"),
]

SUITES = {"math": PROBLEMS, "physics": PHYSICS_PROBLEMS, "data": DATA_PROBLEMS,
          "chem": CHEM_PROBLEMS}
SUITE_FILES = {"data": DATA_FILES}


class CountingLLM:
    """Wraps the model to count calls and re-asks after invalid replies (no controller changes)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_name = getattr(inner, "model_name", "?")
        self.calls = 0
        self.retries = 0

    def reset(self) -> None:
        self.calls = self.retries = 0

    def chat(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        self.calls += 1
        if any(m in user for m in RETRY_MARKERS):
            self.retries += 1
        return self.inner.chat(system, user, schema)


@dataclass
class Row:
    problem: Problem
    status: str = ""
    passed: bool = False
    retries: int = 0
    ladder: str = "none"
    seconds: float = 0.0
    calls: int = 0
    answer: str = ""
    why: str = ""
    values: list[float] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


def _numbers(result: Any) -> list[float]:
    """Real numeric values in a stored tool result (expr, number, or a list of them)."""
    if not isinstance(result, dict):
        return []
    kind = result.get("kind")
    if kind == "number":
        return [float(result["value"])]
    if kind == "expr" and isinstance(result.get("numeric"), (int, float)):
        return [float(result["numeric"])]
    if kind == "quantity":  # as shown, and in SI
        return [float(result["value"]), float(result["si_value"])]
    if kind == "series":
        return [float(v) for v in result.get("value") or []]
    if kind == "list":
        return [v for item in result.get("value") or [] for v in _numbers(item)]
    if kind == "stats":  # the statistic, p, the effect and its CI, regression coefficients
        out = [result["statistic"]["value"], result["p"]]
        effect = result.get("effect") or {}
        out += [effect["value"], *(effect.get("ci") or [])] if "value" in effect else []
        for c in result.get("coefficients") or []:
            out += [c["estimate"], c["p"], *(c.get("ci") or [])]
        return [float(v) for v in out if isinstance(v, (int, float))]
    if kind == "table":
        return [float(v) for row in result.get("rows") or [] for v in row
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if kind == "adjusted":
        return [float(v) for v in result.get("p_adjusted") or []]
    if kind == "descriptors":  # every computed descriptor, so a question may ask for any of them
        return [float(v) for v in (result.get("values") or {}).values()]
    if kind == "molecule":
        return [float(result.get(k, 0)) for k in ("atoms", "bonds", "charge")]
    if kind == "estimate":
        return [float(result["value"])] if result.get("value") is not None else []
    if kind == "neighbours":
        return [float(n["similarity"]) for n in result.get("neighbours") or []]
    if kind == "ranking":
        return [float(r["score"]) for r in result.get("ranking") or []]
    return []


def _close(a: float, b: float, signed: bool = True, rel_tol: float = 1e-6) -> bool:
    """``b`` within ``rel_tol`` of the expected ``a`` (relative; absolute 1e-9 only when ``a`` is 0, so
    a photon energy of 4e-19 J is not "close" to 0). Unsigned when the sign is a convention."""
    if not signed:
        a, b = abs(a), abs(b)
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9 if a == 0 else 0.0)


def _furthest_ladder(rt: Any, session_id: str) -> str:
    stages = [n.ladder_stage.value for n, _ in rt.repo.session_view(session_id)]
    return max(stages, key=LADDER_ORDER.index, default="none")


def run_problem(rt: Any, llm: CountingLLM, p: Problem) -> Row:
    row = Row(p)
    sid = rt.engine.new_session(f"live check: {p.key}")
    llm.reset()
    t0 = time.perf_counter()
    try:
        res = rt.controller.run(p.question)
    except Exception as exc:  # noqa: BLE001 - one crash must not stop the other problems
        row.seconds = time.perf_counter() - t0
        row.status, row.why = "crash", f"{type(exc).__name__}: {exc}"
        row.calls, row.retries = llm.calls, llm.retries
        return row
    row.seconds = time.perf_counter() - t0
    row.status, row.calls, row.retries = res.status, res.llm_calls, llm.retries
    row.answer = res.answer
    row.assumptions = list(res.assumptions)
    row.ladder = _furthest_ladder(rt, sid)
    if p.declined is not None:
        return _score_decline(rt, res, row, p)
    if res.status not in ("answered", "reused") or res.final_node is None:
        asked = res.status == "needs_user" and res.question
        row.why = f"asked: {res.question}" if asked else (res.detail or res.status)
        return row
    final = rt.engine.resolve(res.final_node)
    for nid in (final.tool_inputs or {}).get("answer_nodes", []):
        row.values += _numbers(rt.engine.resolve(nid).result)
    missing = [e for e in p.expected if not any(_close(e, v, p.signed, p.rel_tol) for v in row.values)]
    if missing:
        row.why = f"expected {p.expect_text}, got {row.values or 'no numeric value'}"
    elif p.hypothesis and res.verified:
        # the one failure mode this phase exists to prevent
        row.why = "a chemistry answer came back verified; it must stay a hypothesis"
    elif p.hypothesis and "hypothesis" not in final.flags:
        row.why = f"the answer is not marked a hypothesis (flags: {final.flags or 'none'})"
    elif not (p.hypothesis or res.verified):
        row.why = "right value, but not verified"
    elif p.reuse_of and p.hypothesis and res.llm_calls != 0:
        # a chemistry answer is never "reused" whole (that path wants a verified final); what is
        # reused is the checked hypothesis behind it, which costs no model call
        row.why = f"expected the stored hypothesis to be reused, got {res.llm_calls} model calls"
    elif p.reuse_of and not p.hypothesis and (res.status != "reused" or res.llm_calls != 0):
        row.why = f"expected reuse with 0 model calls, got {res.status} with {res.llm_calls}"
    else:
        row.passed = True
    return row


def _score_decline(rt: Any, res: Any, row: Row, p: Problem) -> Row:
    """A declined request: the status, the wording rule, and nothing computed."""
    from sciai.graph.model import NodeType

    if res.status != "declined" or res.final_node is None:
        row.why = f"expected a decline, got {res.status}"
        return row
    node = rt.engine.resolve(res.final_node)
    rule = (node.tool_inputs or {}).get("wording")
    computed = [n for n, _ in rt.repo.session_view(rt.engine.session_id or "")
                if n.type == NodeType.TOOL_RESULT]
    row.answer = res.detail
    if rule != p.declined:
        row.why = f"declined, but with the {rule!r} wording instead of {p.declined!r}"
    elif computed:
        row.why = f"declined, but {len(computed)} tool result(s) were computed first"
    else:
        row.passed = True
    return row


def _cell(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_table(rows: list[Row]) -> str:
    out = ["| # | problem | expected | status | pass | JSON retries | ladder | seconds | model calls | answer / reason |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {r.problem.key} | {_cell(r.problem.expect_text, 30)} | {r.status} | "
                   f"{'PASS' if r.passed else 'FAIL'} | {r.retries} | {r.ladder} | {r.seconds:.1f} | {r.calls} | "
                   f"{_cell(r.answer if r.passed else r.why or r.answer)} |")
    return "\n".join(out)


def report(rows: list[Row], meta: dict[str, str]) -> str:
    passed = sum(r.passed for r in rows)
    lines = [f"# Live check: {meta['model']}", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines += [f"- passed: {passed}/{len(rows)}",
              f"- total: {sum(r.seconds for r in rows):.1f} s, {sum(r.calls for r in rows)} model calls, "
              f"{sum(r.retries for r in rows)} JSON retries", "", format_table(rows), "", "## Problems", ""]
    lines += [f"{i}. **{r.problem.key}**: {r.problem.question}" + (f" ({r.problem.note})" if r.problem.note else "")
              for i, r in enumerate(rows, 1)]
    assumed = [(i, r) for i, r in enumerate(rows, 1) if r.assumptions]
    if assumed:
        lines += ["", "## Assumptions accepted automatically", ""]
        lines += [f"{i}. **{r.problem.key}**: {'; '.join(r.assumptions)}" for i, r in assumed]
    return "\n".join(lines) + "\n"


def safe_name(tag: str) -> str:
    """results/<model>.md with characters Windows can't put in a file name (: / \\ * ? ...) replaced."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-.") or "model"


def ollama_preflight(cfg: Any) -> tuple[str | None, str]:
    """(error, Ollama version). The error says exactly what to do when the server or model is missing."""
    import httpx

    from sciai.llm.client import full_tag

    try:
        with httpx.Client(base_url=cfg.model.ollama_url, timeout=5, trust_env=False) as http:
            version = http.get("/api/version").json().get("version", "?")
            tags = {full_tag(m.get("name", "")) for m in http.get("/api/tags").json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        return f"Ollama is not reachable at {cfg.model.ollama_url} ({exc}). Start it with: ollama serve", "?"
    if full_tag(cfg.model.name) not in tags:
        return f"model {cfg.model.name!r} is not pulled. Run: ollama pull {cfg.model.name}", version
    return None, version


def import_files(rt: Any, names: tuple[str, ...], folder: Path = DATA_DIR) -> str:
    """Import the suite's data files (no model call) and describe them for the report."""
    out = []
    for name in names:
        rec = rt.import_file(folder / name).record
        out.append(f"{rec.name} ({rec.rows} x {rec.columns})")
    return ", ".join(out)


def identity_fixture() -> str:
    """Standardize the five spellings of aspirin, with no model call.

    This runs before the suite so a failure here is told apart from a model that never reached
    the tool: if the five do not collapse to one InChIKey, nothing the suite reports about
    chemistry means anything.
    """
    from sciai.domains.chem import standardize

    keys = {}
    for text in ASPIRIN_SPELLINGS:
        try:
            keys[text] = standardize.read_twice(text).inchikey
        except Exception as exc:  # noqa: BLE001 - reported, never raised into the suite
            keys[text] = f"error: {type(exc).__name__}: {exc}"
    distinct = set(keys.values())
    if len(distinct) == 1:
        return f"{len(keys)} spellings of aspirin -> {distinct.pop()}"
    return ("DISAGREE: " + "; ".join(f"{t} -> {k}" for t, k in keys.items()))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, metavar="TAG", help="Ollama model tag, e.g. qwen2.5:7b-instruct-q4_K_M")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                   help="reasoning mode on/off (default: the config's [model] think)")
    p.add_argument("--out", default=str(ROOT / "results"), help="folder for <model>.md (default: results/)")
    p.add_argument("--suite", choices=sorted(SUITES), default="math",
                   help="math (Phase 1 problems, the default), physics (units, constants, ODEs, circuits), "
                        "data (tests and regression on CSV, TSV and Excel files) or chem (identity, "
                        "descriptors, similarity and the decline path)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # a Windows console may not be UTF-8 (π, ≈, …)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parse_args(sys.argv[1:] if argv is None else argv)

    from sciai.__main__ import apply_overrides
    from sciai.config import load
    from sciai.runtime import build_runtime

    cfg = load()
    apply_overrides(cfg, args)
    cfg.controller.auto_confirm_root = True  # no one is there to confirm the formalized problem
    error, version = ollama_preflight(cfg)
    if error:
        print(f"live_check: {error}", file=sys.stderr)
        return 2

    problems = SUITES[args.suite]
    meta = {"model": cfg.model.name, "suite": args.suite, "think": str(cfg.model.think).lower(), "ollama": version,
            "num_ctx": str(cfg.model.num_ctx), "num_predict": str(cfg.model.num_predict),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="sciai-live-") as tmp:
        cfg.data.data_dir = str(Path(tmp) / "datasets")  # the imported copies go with the temporary store
        rt = build_runtime(cfg, db_path=Path(tmp) / "live.db")
        llm = CountingLLM(rt.llm)
        rt.controller.llm = llm
        try:
            imported = import_files(rt, SUITE_FILES.get(args.suite, ()))
            if imported:
                meta["datasets"] = imported
            if args.suite == "chem":
                meta["identity fixture"] = identity_fixture()
                print(f"identity fixture: {meta['identity fixture']}", flush=True)
            for i, p in enumerate(problems, 1):
                print(f"[{i}/{len(problems)}] {p.key}: {p.question}", flush=True)
                row = run_problem(rt, llm, p)
                rows.append(row)
                print(f"      {'PASS' if row.passed else 'FAIL'} ({row.status}, {row.seconds:.1f} s, "
                      f"{row.calls} calls, {row.retries} retries, ladder {row.ladder})", flush=True)
        except KeyboardInterrupt:
            print("interrupted: saving the problems that finished", file=sys.stderr)
        finally:
            rt.close()

    text = report(rows, meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{safe_name(cfg.model.name)}{'' if args.suite == 'math' else '-' + args.suite}.md"
    path.write_text(text, encoding="utf-8")
    print()
    print(format_table(rows))
    print(f"\n{sum(r.passed for r in rows)}/{len(rows)} passed. Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
