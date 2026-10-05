"""Phase 4 UI: the Molecule tab, the depiction, the graph thumbnail, and the tab row with the
most tabs it will ever have to fit.

The tab row is the measured part. Phase 3 left six tabs fitting a 440 px workspace only because
"Execution Log" shortens to "Log"; Phase 4 adds a seventh visible tab, so the gaps are measured
again at both widths rather than assumed to still work.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("rdkit")

from PyQt6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from sciai.graph.model import Layer, Node, NodeType, Status  # noqa: E402
from sciai.ui.canvas.layout import NODE_H, PLOT_H, has_thumbnail, node_height  # noqa: E402
from sciai.ui.canvas.node_item import NodeItem  # noqa: E402
from sciai.ui.mol_image import MolView, export, mol_image, structure_key  # noqa: E402
from sciai.ui.theme.theme import Theme, load_bundled_fonts  # noqa: E402
from sciai.graph.model import Evidence  # noqa: E402
from sciai.ui.workspace.mol_view import (  # noqa: E402
    HYPOTHESIS_TEXT,
    MoleculeTab,
    check_badge,
    descriptor_rows,
    molecule_of,
    verdict_text,
)
from sciai.ui.workspace.node_panel import SHORT_TAB_NAMES, TAB_NAMES  # noqa: E402

SHOTS = os.environ.get("SCIAI_SHOT_DIR")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def theme(qtbot):
    app = QApplication.instance()
    app.setStyle("Fusion")
    load_bundled_fonts()
    t = Theme.load("light")
    app.setStyleSheet(t.qss())
    return t


def molecule_result() -> dict:
    from sciai.tools import chem_tools

    return chem_tools.parse_fn({"structure": ASPIRIN})["result"]


def descriptors_result() -> dict:
    from sciai.tools import chem_tools

    return chem_tools.descriptors_fn({"structure": ASPIRIN})["result"]


def node(result: dict, title: str = "aspirin") -> Node:
    return Node(session_id="s", layer=Layer.DOMAIN, type=NodeType.TOOL_RESULT, title=title,
                result=result, status=Status.HYPOTHESIS)


# ----------------------------------------------------------------- the depiction

def test_a_structure_draws_into_an_image(theme) -> None:
    image = mol_image(theme, ASPIRIN, 400, 300, dpr=2)
    assert not image.isNull()
    assert (image.width(), image.height()) == (800, 600)
    assert image.devicePixelRatio() == 2.0
    colours = {image.pixel(x, y) for y in range(0, 600, 11) for x in range(0, 800, 11)}
    assert len(colours) > 8, "the depiction should not be a flat fill"


def test_the_drawing_is_cached_by_structure_and_size(theme) -> None:
    first = mol_image(theme, ASPIRIN, 300, 200)
    again = mol_image(theme, ASPIRIN, 300, 200)
    assert first is again
    assert mol_image(theme, ASPIRIN, 301, 200) is not first


def test_the_depiction_uses_the_theme_background(theme) -> None:
    markup = __import__("sciai.ui.mol_image", fromlist=["svg"]).svg(ASPIRIN, 200, 150, theme=theme)
    assert theme.hex("panel").lower().lstrip("#") in markup.lower().replace("#", "")


def test_a_structure_that_will_not_draw_is_handled(theme) -> None:
    view = MolView(theme)
    view.resize(300, 200)
    view.set_structure("not a molecule at all")
    assert view.image() is None       # no crash, nothing drawn


def test_the_view_redraws_only_once_the_size_settles(theme, qtbot) -> None:
    view = MolView(theme, ASPIRIN)
    qtbot.addWidget(view)
    view.resize(300, 220)
    view.show()
    assert view.image() is not None
    view.resize(460, 320)
    assert view._settle.isActive()    # not redrawn yet
    qtbot.waitUntil(lambda: view.image().width() == 460, timeout=2000)


def test_export_writes_both_formats(tmp_path) -> None:
    export(ASPIRIN, str(tmp_path / "a.svg"))
    export(ASPIRIN, str(tmp_path / "a.png"))
    assert (tmp_path / "a.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert (tmp_path / "a.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_key_is_the_structure(theme) -> None:
    assert structure_key(ASPIRIN) == structure_key(ASPIRIN)
    assert structure_key(ASPIRIN) != structure_key("CCO")


# ----------------------------------------------------------------- which nodes get the tab

def test_a_molecule_node_has_a_molecule() -> None:
    found = molecule_of(node(molecule_result()))
    assert found is not None and found["canonical_smiles"] == ASPIRIN


@pytest.mark.parametrize("kind", ["descriptors", "logp", "druglike", "substructure"])
def test_a_derived_result_carries_its_structure_back(kind: str) -> None:
    from sciai.tools import chem_tools

    desc = descriptors_result()
    result = {
        "descriptors": lambda: desc,
        "logp": lambda: chem_tools.logp_fn({"structure": ASPIRIN})["result"],
        "druglike": lambda: chem_tools.druglike_fn({"descriptors": desc, "logp": 1.3})["result"],
        "substructure": lambda: chem_tools.substructure_fn(
            {"structure": ASPIRIN, "smarts": "c1ccccc1"})["result"],
    }[kind]()
    found = molecule_of(node(result))
    assert found is not None and found["canonical_smiles"] == ASPIRIN


def test_a_similarity_result_shows_its_query() -> None:
    from sciai.tools import chem_tools

    library = {"members": [{"name": "salicylic acid", "structure": "OC(=O)c1ccccc1O"}]}
    result = chem_tools.similar_fn({"structure": ASPIRIN, "library": library})["result"]
    found = molecule_of(node(result))
    assert found is not None and found["canonical_smiles"] == ASPIRIN


@pytest.mark.parametrize("result", [{}, {"kind": "quantity", "value": 1},
                                    {"kind": "ranking", "ranking": []},
                                    {"kind": "literature", "hits": []}])
def test_other_nodes_get_no_molecule_tab(result: dict) -> None:
    assert molecule_of(node(result)) is None
    assert molecule_of(None) is None


# ----------------------------------------------------------------- the tab contents

def test_the_tab_shows_the_structure_and_its_identity(theme, qtbot) -> None:
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 620)
    tab.show()
    tab.show_molecule(molecule_result(), descriptors_result())
    assert tab.view.image() is not None
    assert "C9H8O4" in tab.identity.text()
    assert "BSYNRYMUTXBXSQ-UHFFFAOYSA-N" in tab.identity.text()
    assert tab.smiles.text() == ASPIRIN


def test_the_descriptor_table_names_the_tool_behind_each_row() -> None:
    rows = descriptor_rows(descriptors_result())
    assert rows, "the table should have rows"
    by_name = {r[0]: r for r in rows}
    assert by_name["mw"][2] == "g/mol"
    assert by_name["tpsa"][2] == "A^2"
    assert all(r[3] == "chem.descriptors" for r in rows)


def test_the_table_shows_logp_as_coming_from_its_own_tool() -> None:
    result = {**descriptors_result()}
    result["values"] = {**result["values"], "logp": 1.31}
    rows = {r[0]: r for r in descriptor_rows(result)}
    assert rows["logp"][3] == "chem.logp"


def test_counts_are_shown_as_integers_and_masses_are_not() -> None:
    rows = {r[0]: r[1] for r in descriptor_rows(descriptors_result())}
    assert rows["heavy atoms"] == "13"
    assert rows["rings"] == "1"
    assert rows["formal charge"] == "0"
    # four decimals on the masses: "180" would be a useless monoisotopic mass
    assert rows["mw"] == "180.1590"
    assert rows["exact mass"] == "180.0423"


def test_the_verdict_names_what_fails() -> None:
    text = verdict_text({"verdicts": {
        "lipinski": {"passed": False, "failing": ["mw", "logp"]},
        "veber": {"passed": True, "failing": []}}})
    assert "Lipinski: fails on mw, logp" in text
    assert "Veber: passes" in text


def test_one_lipinski_miss_reads_as_a_miss_not_a_failure() -> None:
    text = verdict_text({"verdicts": {"lipinski": {"passed": True, "failing": ["mw"]}}})
    assert "passes with one miss (mw)" in text


def test_the_hypothesis_label_is_pinned_outside_the_scroll_area(theme, qtbot) -> None:
    """It says the numbers describe an untested candidate, so it must not scroll away."""
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 300)
    tab.show()
    tab.show_molecule(molecule_result(), descriptors_result())
    assert "Hypothesis" in tab.hypothesis.text()
    assert "nothing here says the molecule works" in HYPOTHESIS_TEXT
    scroll = tab.findChild(QScrollArea)
    assert scroll is not None
    assert not scroll.isAncestorOf(tab.hypothesis)
    assert tab.hypothesis.isVisible()
    assert tab.hypothesis.y() < scroll.y()


def test_an_empty_tab_says_so_and_disables_export(theme, qtbot) -> None:
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.show_molecule(None)
    assert tab.identity.text() == "No structure"
    assert not tab.export_png.isEnabled() and not tab.export_svg.isEnabled()
    assert tab.descriptor_count() == 0


def test_standardization_steps_are_shown_when_there_were_any(theme, qtbot) -> None:
    from sciai.tools import chem_tools

    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.show()
    salt = chem_tools.parse_fn({"structure": "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"})["result"]
    tab.show_molecule(salt, {})
    assert tab.standardized.isVisible()
    assert "largest" in tab.standardized.text()
    tab.show_molecule(molecule_result(), {})
    assert not tab.standardized.isVisible()


# ------------------------------------------------------- the evidence badge

def _ev(method: str, outcome: str = "pass") -> Evidence:
    return Evidence(node_id="n", method=method, tool_name="chem.check", inputs={},
                    outcome=outcome, detail={})


def test_the_badge_says_what_the_checks_established_without_contradicting_the_label() -> None:
    assert check_badge([])[0] == "unchecked"
    badge, line = check_badge([_ev("reread"), _ev("recomputed")])
    assert badge == "checks passed"
    assert line == ("Identity re-read from the canonical form; masses and counts recomputed "
                    "from a pinned element table.")
    assert check_badge([_ev("recomputed", "fail"), _ev("reread")])[0] == "check failed"
    assert check_badge([_ev("reread", "inconclusive")])[0] == "inconclusive"
    # whatever the badge says, nothing in it claims the molecule is verified or works
    for evidence in ([], [_ev("reread")], [_ev("reread", "fail")]):
        words = " ".join(check_badge(evidence)).lower()
        assert "verified" not in words and "proven" not in words


def test_the_badge_and_the_hypothesis_label_are_both_pinned_above_the_scroll_area(theme, qtbot) -> None:
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 760)
    tab.show()
    tab.set_checks([_ev("reread"), _ev("recomputed")])
    tab.show_molecule(molecule_result(), descriptors_result())
    qtbot.waitUntil(lambda: tab.table is not None and tab.table.height() > 0, timeout=3000)
    scroll = tab.findChildren(QScrollArea)[0]
    for widget in (tab.hypothesis, tab.badge, tab.checks):
        assert widget.isVisible()
        assert not scroll.isAncestorOf(widget), "the banner must not be able to scroll away"
    assert tab.banner.height() + scroll.height() == tab.height()
    assert tab.badge.text() == "checks passed"


def test_the_descriptor_name_never_elides_and_the_unit_stays_whole(theme, qtbot) -> None:
    """At the default workspace width only "from", which repeats one string, gives width up."""
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 760)
    tab.show()
    tab.show_molecule(molecule_result(), descriptors_result())
    qtbot.waitUntil(lambda: tab.table is not None and tab.table.height() > 0, timeout=3000)
    table = tab.table
    assert table.elided() == []
    names, _, units, _ = (list(c) for c in zip(*table.rows))
    widths = table._widths()
    from PyQt6.QtGui import QFontMetricsF

    metrics = QFontMetricsF(theme.ui_font("size_ui_px"))
    pad = 2 * table.PAD
    assert widths[0] >= max(metrics.horizontalAdvance(n) for n in names) + pad
    assert widths[2] >= max(metrics.horizontalAdvance(u) for u in units if u) + pad


def test_the_tab_has_no_emoji_or_coloured_border(theme, qtbot) -> None:
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.show_molecule(molecule_result(), descriptors_result())
    tab.set_checks([_ev("reread")])
    text = " ".join([tab.hypothesis.text(), tab.identity.text(), tab.smiles.text(),
                     tab.verdict.text(), tab.badge.text(), tab.checks.text(), HYPOTHESIS_TEXT])
    assert all(ord(ch) < 0x2190 for ch in text), "plain style: no symbol glyphs"
    qss = theme.qss()
    block = qss[qss.index("QWidget#molBanner"):][:400]
    assert theme.hex("border") in block and theme.hex("accent") not in block
    assert theme.hex("accent") not in tab.badge.styleSheet()


# ----------------------------------------------------------------- the graph

def test_a_molecule_node_is_drawn_tall_with_a_thumbnail(theme, qtbot) -> None:
    n = node(molecule_result())
    assert has_thumbnail(n)
    assert node_height(n) == PLOT_H
    item = NodeItem(n, "n1", theme, _no_click)
    assert item.size()[1] == PLOT_H


def test_the_layout_height_and_the_item_size_agree(theme) -> None:
    """They read one predicate, so a new result kind cannot get one and not the other."""
    for result in (molecule_result(), {"kind": "plotspec", "value": {}}, {}, descriptors_result()):
        n = node(result)
        item = NodeItem(n, "n1", theme, _no_click)
        if n.type != NodeType.CHECK:
            assert item.size()[1] == node_height(n), result.get("kind")


def test_a_molecule_without_a_structure_stays_a_short_node(theme) -> None:
    n = node({"kind": "molecule", "canonical_smiles": ""})
    assert not has_thumbnail(n)
    assert node_height(n) == NODE_H


def test_the_graph_paints_a_molecule_node(theme, qtbot) -> None:
    from PyQt6.QtGui import QImage, QPainter

    item = NodeItem(node(molecule_result()), "n1", theme, _no_click)
    w, h = item.size()
    image = QImage(int(w), int(h), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    colours = {image.pixel(x, y) for y in range(0, int(h), 3) for x in range(0, int(w), 3)}
    assert len(colours) > 6, "the node body should carry a drawing"
    if SHOTS:
        image.save(os.path.join(SHOTS, "k-graph-molecule-node.png"))


# ----------------------------------------------------------------- the tab row

def test_the_tab_list_gained_molecule_with_a_short_label() -> None:
    assert "Molecule" in TAB_NAMES
    assert SHORT_TAB_NAMES["Molecule"] == "Mol"
    assert SHORT_TAB_NAMES["Execution Log"] == "Log"


@pytest.mark.parametrize("width", [640, 440])
def test_the_tab_row_fits_its_widest_case(theme, qtbot, width: int) -> None:
    """Every tab that can be visible at once, at the widest and narrowest workspace."""
    from sciai.ui.workspace.subtabs import SubTabs

    tabs = SubTabs(theme)
    qtbot.addWidget(tabs)
    for name in TAB_NAMES:
        tabs.addTab(_page(), name, SHORT_TAB_NAMES.get(name))
    # Data and Plot never show together, so the worst real case is one of them plus the rest.
    tabs.setTabVisible(tabs.indexOf("Plot"), False)
    tabs.resize(width, 40)
    tabs.show()
    qtbot.waitUntil(lambda: tabs.label_gaps() != [], timeout=2000)
    tabs.relayout()

    visible = tabs.visible_indexes()
    assert len(visible) == len(TAB_NAMES) - 1
    gaps = tabs.label_gaps()
    assert min(gaps) >= 12, f"{width}px: labels too close, gaps {gaps}"
    for i in visible:
        drawn = tabs.buttons[i].text()
        assert "…" not in drawn, f"{width}px: {drawn!r} was elided"
        full = tabs.labels[i]
        assert drawn in (full, SHORT_TAB_NAMES.get(full, full)), drawn


def _no_click(item, point) -> None:
    """NodeItem needs a click handler; these tests only measure and paint."""


def _page():
    from PyQt6.QtWidgets import QWidget

    return QWidget()


# ----------------------------------------------------------------- screenshots

def _settled(tab) -> bool:
    """Every widget sized and placed by its own layout, so a grab shows the finished tab.

    Three things arrive late and each one has been seen missing from a screenshot: the depiction
    (its view settles 120 ms after a resize), the descriptor table (added to a layout after the
    tab was shown) and the banner, which grows when the check badge is given its text.
    """
    if tab.view.image() is None or tab.table is None or tab.table.height() <= 0:
        return False
    banner = tab.banner.contentsRect()
    for widget in (tab.hypothesis, tab.badge, tab.checks):
        if not banner.contains(widget.geometry()):
            return False
    body = tab.table.parentWidget()
    return (tab.table.width() == tab.verdict.width()
            and tab.verdict.y() >= tab.table.y() + tab.table.height()
            and body.height() >= tab.verdict.y() + tab.verdict.height())


@pytest.mark.skipif(not SHOTS, reason="set SCIAI_SHOT_DIR to write screenshots")
def test_shots(theme, qtbot) -> None:
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 760)
    tab.show()
    tab.set_checks([_ev("reread"), _ev("recomputed")])
    tab.show_molecule(molecule_result(), {**descriptors_result(),
                                          "verdicts": {"lipinski": {"passed": True, "failing": []},
                                                       "veber": {"passed": True, "failing": []}}})
    qtbot.waitUntil(lambda: _settled(tab), timeout=5000)
    tab.grab().save(os.path.join(SHOTS, "j-node-molecule-420x760.png"))


def test_the_tab_lays_itself_out_with_nothing_overlapping(theme, qtbot) -> None:
    """The same settling the screenshot waits for, asserted rather than only looked at."""
    tab = MoleculeTab(theme)
    qtbot.addWidget(tab)
    tab.resize(420, 760)
    tab.show()
    tab.set_checks([_ev("reread"), _ev("recomputed")])
    tab.show_molecule(molecule_result(), {**descriptors_result(),
                                          "verdicts": {"lipinski": {"passed": True, "failing": []},
                                                       "veber": {"passed": True, "failing": []}}})
    qtbot.waitUntil(lambda: _settled(tab), timeout=5000)
    assert tab.verdict.y() >= tab.table.y() + tab.table.height()  # the verdict is below the table
    assert tab.table.y() >= tab.view.y() + tab.view.height()      # the table is below the depiction
