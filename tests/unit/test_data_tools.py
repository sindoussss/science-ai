"""Dataset import, reading rules and the data tools with their independent checks."""
from __future__ import annotations

import copy
import math

import pytest

from sciai.config import DataConfig
from sciai.domains.data import frames
from sciai.domains.data.datasets import import_file, load_args
from sciai.domains.data.frames import DataError
from sciai.store.db import Database
from sciai.store.repository import Repository
from sciai.tools.registry import get
from sciai.tools.sandbox import InlineRunner

R = InlineRunner()

TRIAL = """id,group,score [points],mass [kg],height [m],note
1,A,12.5,70,1.75,
2,A,14.0,82,1.80,first visit
3,A,,65,1.62,
4,B,18.5,90,1.85,
5,B,17.0,NA,1.70,
6,B,19.5,77,1.68,repeat
"""


def run(tool, **args):
    out = R.run(tool, args)
    assert out.ok, out.error
    return out.value["result"]


def fails(tool, **args):
    out = R.run(tool, args)
    assert not out.ok
    return out.error


def write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "k.db")
    yield Repository(db), DataConfig(data_dir=str(tmp_path / "datasets"))
    db.close()


@pytest.fixture
def trial(tmp_path, store):
    repo, cfg = store
    return import_file(write(tmp_path, "trial.csv", TRIAL), repo, R, cfg).descriptor


# ------------------------------------------------------------------ import
def test_import_records_schema_and_reuses_the_same_bytes(tmp_path, store):
    repo, cfg = store
    path = write(tmp_path, "trial.csv", TRIAL)
    first = import_file(path, repo, R, cfg)
    assert not first.reused and first.check["outcome"] == "pass"
    rec = first.record
    assert (rec.name, rec.rows, rec.columns) == ("trial.csv", 6, 6)
    cols = {c["name"]: c for c in rec.schema}
    assert cols["score [points]"]["missing"] == 1 and cols["score [points]"]["type"] == "number"
    assert cols["mass [kg]"]["unit"] == "kg" and cols["group"]["levels"] == ["A", "B"]
    assert cols["note"]["missing"] == 4
    stored = tmp_path / "datasets" / f"{rec.sha256}.csv"
    assert rec.stored_path == str(stored) and stored.read_text() == TRIAL

    again = import_file(path, repo, R, cfg)
    assert again.reused and again.record.id == rec.id
    path.write_text(TRIAL.replace("12.5", "12.6"))
    changed = import_file(path, repo, R, cfg)
    assert not changed.reused and changed.record.id != rec.id
    assert repo.dataset_by_name("trial.csv").id == changed.record.id
    assert repo.dataset(rec.id) is not None  # the old version stays for the nodes that used it
    assert stored.read_text() == TRIAL       # and its stored copy is untouched


def test_model_never_sees_values(trial, store):
    repo, _ = store
    text = repo.dataset_by_name("trial.csv").schema_text()
    assert "score [points] (number; 1 missing)" in text  # "points" is not read as pint's typographic unit
    assert "mass [kg] (number; unit kg; 1 missing)" in text
    for value in ("12.5", "14.0", "first visit", "1.75"):
        assert value not in text


def test_semicolon_comma_decimal_and_windows_encoding(tmp_path, store):
    repo, cfg = store
    text = "Größe;Gewicht\n1,75;70,5\n1,80;82,25\n1,62;\n"
    res = import_file(write(tmp_path, "de.csv", text, "cp1252"), repo, R, cfg)
    opts = res.record.options
    assert (opts["delimiter"], opts["decimal"], opts["encoding"]) == (";", ",", "cp1252")
    assert [c["name"] for c in res.record.schema] == ["Größe", "Gewicht"]
    assert res.descriptor["checks"]["Gewicht"] == {"sum": 152.75, "count": 2}


def test_tsv_and_excel(tmp_path, store):
    from openpyxl import Workbook

    repo, cfg = store
    tsv = import_file(write(tmp_path, "t.tsv", TRIAL.replace(",", "\t")), repo, R, cfg)
    assert tsv.record.rows == 6 and tsv.record.options["delimiter"] == "\t"
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["dose [mg]", None, "ok"])  # an unnamed header cell becomes column2
    ws.append([1.5, 2, True])
    ws.append([2.5, None, False])
    ws.append([None, None, None])         # an empty row is dropped by both readers
    ws.append([4.0, 5, True])
    wb.save(tmp_path / "w.xlsx")
    x = import_file(tmp_path / "w.xlsx", repo, R, cfg)
    assert x.record.options["sheet"] == "Results"
    assert [c["name"] for c in x.record.schema] == ["dose [mg]", "column2", "ok"]
    assert x.record.rows == 3 and x.check["method"] == "openpyxl"
    assert x.descriptor["checks"]["ok"] == {"true": 2}


def test_import_refusals(tmp_path, store):
    repo, cfg = store
    with pytest.raises(DataError, match="duplicate column names: x"):
        import_file(write(tmp_path, "d.csv", "x,x\n1,2\n"), repo, R, cfg)
    with pytest.raises(DataError, match="unsupported file type"):
        import_file(write(tmp_path, "d.json", "{}"), repo, R, DataConfig(data_dir=cfg.data_dir))
    with pytest.raises(DataError, match="files above 0 MB are refused"):
        import_file(write(tmp_path, "s.csv", "x\n1\n"), repo, R, DataConfig(data_dir=cfg.data_dir, max_file_mb=0))
    with pytest.raises(DataError, match="more than 2 rows"):
        import_file(write(tmp_path, "big.csv", "x\n1\n2\n3\n"), repo, R,
                    DataConfig(data_dir=cfg.data_dir, max_rows=2))


def test_load_refuses_a_changed_stored_file(tmp_path, store):
    repo, cfg = store
    rec = import_file(write(tmp_path, "trial.csv", TRIAL), repo, R, cfg).record
    with open(rec.stored_path, "a") as fh:
        fh.write("7,B,1,1,1,\n")
    assert "changed since import" in fails("data.load", **load_args(rec, cfg))


def test_check_load_catches_a_wrong_read(trial):
    bad = copy.deepcopy(trial)
    bad["checks"]["score [points]"]["sum"] += 1.0
    out = run("data.check_load", loaded=bad)
    assert out["outcome"] == "fail" and "score [points]: sum" in out["problems"][0]
    bad = copy.deepcopy(trial)
    bad["rows"] = 5
    assert run("data.check_load", loaded=bad)["outcome"] == "fail"


# ------------------------------------------------------------------ columns
def test_column_resolution(trial):
    assert frames.resolve_column(trial, "score [points]") == "score [points]"
    assert frames.resolve_column(trial, "SCORE") == "score [points]"
    assert frames.resolve_column(trial, "mass") == "mass [kg]"
    with pytest.raises(DataError, match="column 'weight' is not in trial.csv; its columns are: id, group"):
        frames.resolve_column(trial, "weight")
    assert "is text, not numeric" in fails("data.describe", dataset=trial, columns=["group"])


# ------------------------------------------------------------------ describe
def test_describe_and_its_check(trial):
    t = run("data.describe", dataset=trial, columns=["score", "mass"])
    rows = {r[0]: r for r in t["rows"]}
    score = rows["score [points]"]
    assert score[1:3] == [5, 1]
    assert score[3] == pytest.approx(16.3)
    assert score[4] == pytest.approx(math.sqrt(8.825))
    assert score[5:] == [12.5, 14.0, 17.0, 18.5, 19.5]
    assert run("data.check_describe", dataset=trial, table=t)["outcome"] == "pass"
    alt = run("data.describe", dataset=trial, columns=["score", "mass"], method="numpy")
    assert alt["rows"] == [pytest.approx(r) for r in t["rows"]]
    wrong = copy.deepcopy(t)
    wrong["rows"][0][3] = 16.4  # an injected wrong mean
    out = run("data.check_describe", dataset=trial, table=wrong)
    assert out["outcome"] == "fail" and "mean" in out["problems"][0]


# ------------------------------------------------------------------ filter, derive, group
def test_filter_is_a_recipe_and_is_checked(trial):
    out = run("data.filter", dataset=trial, where=[{"column": "Group", "op": "==", "value": "B"},
                                                    {"column": "mass", "op": "not_missing"}])
    assert out["rows"] == 2 and out["parent_rows"] == 6
    assert out["recipe"] == [{"op": "filter", "where": [{"column": "group", "op": "==", "value": "B"},
                                                        {"column": "mass [kg]", "op": "not_missing"}]}]
    assert out["source"] == trial["source"] and out["key"] != trial["key"]
    assert run("data.check_frame", result=out)["outcome"] == "pass"
    bad = copy.deepcopy(out)
    bad["rows"] = 3
    assert run("data.check_frame", result=bad)["outcome"] == "fail"
    assert "keeps no rows" in fails("data.filter", dataset=trial,
                                    where=[{"column": "group", "op": "==", "value": "C"}])


def test_derive_and_its_check(trial):
    out = run("data.derive", dataset=trial, name="bmi", expr="m/h**2", vars={"m": "mass", "h": "height"})
    assert out["columns"][-1]["name"] == "bmi" and out["columns"][-1]["missing"] == 1
    assert run("data.check_frame", result=out)["outcome"] == "pass"
    df = frames.load(out)
    assert df["bmi"].iloc[0] == pytest.approx(70 / 1.75 ** 2)
    nested = run("data.describe", dataset=out, columns=["bmi"])
    assert nested["rows"][0][1] == 5
    assert "already exists" in fails("data.derive", dataset=trial, name="group", expr="m", vars={"m": "mass"})


def test_group_and_its_check(trial):
    t = run("data.group", dataset=trial, by="group", column="score", agg="mean")
    assert t["rows"] == [["A", pytest.approx(13.25), 2], ["B", pytest.approx(55 / 3), 3]]
    assert run("data.check_group", dataset=trial, table=t)["outcome"] == "pass"
    wrong = copy.deepcopy(t)
    wrong["rows"][1][1] = 18.0
    assert run("data.check_group", dataset=trial, table=wrong)["outcome"] == "fail"
    counts = run("data.group", dataset=trial, by="group", column="note", agg="count")
    assert [r[1] for r in counts["rows"]] == [1.0, 1.0]
    assert run("data.check_group", dataset=trial, table=counts)["outcome"] == "pass"


def test_fingerprints_use_the_dataset_key_and_real_column_names(trial):
    canon = get("data.describe").canonical
    a = canon({"dataset": trial, "columns": ["SCORE"]})
    b = canon({"dataset": trial, "columns": ["score [points]"]})
    assert a == b == {"dataset": trial["key"], "columns": ["score [points]"], "method": "pandas"}
    canon = get("stats.ttest").canonical
    assert canon({"dataset": trial, "column": "score", "by": "GROUP"}) == \
        canon({"dataset": trial, "column": "score [points]", "by": "group", "alternative": "two-sided",
               "alpha": 0.05})


# ------------------------------------------------------------------ plots (PlotSpec v2)
def test_plot_tools(trial):
    from sciai.graph import plotspec

    sc = run("plot.scatter", dataset=trial, x="mass", y="height", by="group")
    v = sc["value"]
    assert v["version"] == 2 and [s["label"] for s in v["series"]] == ["A", "B"]
    assert v["xlabel"] == "mass [kg]" and v["series"][0]["x"] == [70.0, 82.0, 65.0]
    assert v["x"] == v["series"][0]["x"]  # a v1 reader still sees the first series
    line = run("plot.line", dataset=trial, x="height", y="mass")["value"]["series"][0]
    assert line["x"] == sorted(line["x"]) and len(line["x"]) == 5  # the row without a mass is left out
    hist = run("plot.histogram", dataset=trial, column="score", bins=3)["value"]
    assert sum(hist["series"][0]["counts"]) == 5 and hist["notes"] == ["bins: 3 equal-width bins"]
    box = run("plot.box", dataset=trial, column="score", by="group")["value"]["series"][0]["boxes"]
    assert [b["label"] for b in box] == ["A", "B"] and box[1]["med"] == 18.5
    for spec in (v, hist, run("plot.box", dataset=trial, column="score")["value"]):
        assert plotspec.render_png(spec, 400, 300)[:4] == b"\x89PNG"
    old = {"x": [0, 1, 2], "y": [0, 1, None], "label": "x**2", "xlabel": "x"}  # a Phase 1 spec
    assert plotspec.normalize(old)["series"][0]["type"] == "line"
    assert plotspec.render_png(old, 300, 200)[:4] == b"\x89PNG"
    assert "is text, not numeric" in fails("plot.scatter", dataset=trial, x="group", y="mass")
