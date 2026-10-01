"""Help pages, per-event hover text, and event detection on already-loaded data."""
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

import help_text
import Pilatus_Integration_GUI as gui

insitu_seg = pytest.importorskip("insitu_seg")


@pytest.fixture
def win():
    app = QApplication.instance() or QApplication([])
    w = gui.PilatusIntegrationGUI()
    yield w, app
    w.close()


# ---------------------------------------------------------------- help
def test_events_help_explains_confidence_and_every_family(win):
    w, app = win
    dlg = w.show_help("Understanding Detected Events", help_text.EVENTS_HTML)
    text = dlg.browser.toPlainText()
    for word in ("[high]", "[low]", "sharpening", "broadening", "not detected", "baseline"):
        assert word in text, word
    for fam in ("area", "pointwise", "profile", "slope", "pearson"):
        assert fam in text, fam
    assert dlg.isVisible() and not dlg.isModal()
    dlg.close()


def test_live_help_opens(win):
    w, app = win
    dlg = w.show_help("Live Mode Guide", help_text.LIVE_HTML)
    assert ".pdi" in dlg.browser.toPlainText()
    dlg.close()


def test_event_hover_text_explains_that_event():
    ev = insitu_seg.Event(scan=96, span=(95, 97),
                          classes={"area+": 12.8, "area-": 6.8, "pearson": 9.2, "pointwise+": 39.7, "profile": 16.6},
                          T_C=905.0, profile_direction="sharpening")
    tip = help_text.event_explanation(ev)
    assert "Scan 96" in tip and "high confidence" in tip and "905" in tip
    assert "area (appearing and disappearing)" in tip and "one phase converting into another" in tip
    assert "pointwise (appearing)" in tip and "weak phase forming" in tip
    assert "sharpening" in tip and "crystallite growth" in tip
    only_gain = insitu_seg.Event(scan=96, span=(95, 97), classes={"area+": 12.8, "pearson": 9.2})
    t2 = help_text.event_explanation(only_gain)
    assert "area (appearing)" in t2 and "phase forming" in t2
    assert "consumed" not in t2, "an area+ event must not be explained as a loss"
    low = insitu_seg.Event(scan=30, span=(30, 30), classes={"area+": 8.5})
    assert "low confidence" in help_text.event_explanation(low)
    assert "false alarm" in help_text.event_explanation(low)


# ---------------------------------------------------------------- detection on loaded data
def synthetic(n=120, appear_at=60, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(5.0, 45.0, 0.005)
    host = rng.uniform(7, 43, 18)
    new = rng.uniform(7, 43, 6)
    out = []
    for i in range(n):
        xs = x + 0.0015 * i * (x / 25.0)                      # thermal expansion
        y = 40 + 15 * np.exp(-(x - 5) / 12)
        for c in host:
            y = y + 5 / (0.025 * 2.5) * np.exp(-0.5 * ((xs - c) / 0.025) ** 2)
        if i >= appear_at:
            g = min(1.0, (i - appear_at + 1) / 3)
            for c in new:
                y = y + g * 6 / (0.025 * 2.5) * np.exp(-0.5 * ((xs - c) / 0.025) ** 2)
        out.append((x, rng.poisson(y * 4) / 4.0))
    return out


def load(w, patterns, shuffle=True):
    order = list(range(len(patterns)))
    if shuffle:
        np.random.default_rng(0).shuffle(order)
    for k in order:
        x, y = patterns[k]
        name = f"synth_anneal_scan{k + 1}.xye"
        w.plot_data[name] = {"x": x, "y": y, "e": np.sqrt(y)}
        w.plot_list.addItem(name)


def test_detect_on_loaded_data_finds_new_phase_in_scan_order(win):
    w, app = win
    load(w, synthetic())                      # list is shuffled; detection must sort by scan number
    res = w.detect_events_in_loaded_data()
    assert res is not None
    assert [s for s, _ in w.batch_entries] == list(range(1, 121))
    hit = [e for e in res.events if e.span[0] - 4 <= 61 <= e.span[1] + 4 and "area" in e.families]
    assert hit and hit[0].confidence == "high", [e.as_dict() for e in res.events]
    assert w.event_list.count() == len(res.events)
    assert w.event_list.item(0).toolTip().startswith("<qt>")
    assert w.data_tabs.tabText(1) == f"Events ({len(res.events)})"


def test_detect_uses_selection_when_present(win):
    w, app = win
    load(w, synthetic(), shuffle=False)
    for i in range(29):                       # select scans 1-29: before the new phase appears
        w.plot_list.item(i).setSelected(True)
    res = w.detect_events_in_loaded_data()
    assert [s for s, _ in w.batch_entries] == list(range(1, 30))
    assert not [e for e in res.events if "area" in e.families and e.confidence == "high"]


def test_detect_refuses_too_few_patterns(win, monkeypatch):
    w, app = win
    warned = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    load(w, synthetic(n=10))
    assert w.detect_events_in_loaded_data() is None
    assert warned and "at least 25" in warned[0]
