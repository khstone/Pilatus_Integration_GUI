"""Live mode: scan-completion logic (no Qt), controller sequencing (fake worker), and an end
to end run through the GUI on the real May 2026 scan."""
import os
import shutil

import numpy as np
import pytest

import live_mode as lm
from conftest import CAL, DATA, IMAGES, REF, SPEC

HEADER = """#F test_anneal
#E 1779600000
#D Sun May 24 10:36:12 2026
#C twoc  User = b_stone
#O0 tth  th
"""
COLS = "#L tth  Epoch  Seconds  I0  TEMP  Monitor  Detector\n"


def scan_block(n, points, total=16, cmd=None):
    lines = [f"#S {n}  {cmd or f'ascan  tth 5 45  {total - 1} 3'}\n", "#P0 5 0\n", COLS]
    for k in range(points):
        lines.append(f"{5 + 40 * k / (total - 1):.4f} {60 * n + k} 3 184000 {25 + n} 176000 120\n")
    return "".join(lines)


def write_images(folder, spec_name, scan, n, pdi=True):
    for k in range(n):
        p = os.path.join(folder, lm.image_name("b_stone", spec_name, scan, k))
        with open(p, "wb") as fh:
            fh.write(b"\0" * lm.RAW_BYTES)
        if pdi:
            open(p + ".pdi", "w").close()


@pytest.fixture
def run(tmp_path):
    img = tmp_path / "images"
    img.mkdir()
    spec = tmp_path / "test_anneal"
    spec.write_text(HEADER)
    return spec, img


# ---------------------------------------------------------------- tracker (no Qt)
def test_expected_points():
    assert lm.expected_points("ascan  tth 5 45  15 3") == 16
    assert lm.expected_points("timescan 1 0") is None


def test_scan_ready_only_when_spec_and_images_complete(run):
    spec, img = run
    spec.write_text(HEADER + scan_block(1, 8))
    t = lm.ScanTracker(str(spec), str(img), "b_stone", start_scan=1)
    assert t.poll() == []                                    # 8 of 16 points
    spec.write_text(HEADER + scan_block(1, 16))
    assert t.poll() == []                                    # points done, no images
    write_images(str(img), "test_anneal", 1, 16, pdi=False)
    open(os.path.join(img, lm.image_name("b_stone", "test_anneal", 1, 0)) + ".pdi", "w").close()
    assert t.poll() == []                                    # detector writes .pdi: wait for all
    write_images(str(img), "test_anneal", 1, 16, pdi=True)
    got = t.poll()
    assert [s.number for s in got] == [1]
    assert got[0].mean_temp == pytest.approx(26.0)
    assert t.poll() == []                                    # never reported twice


def test_partial_image_is_not_ready(run):
    spec, img = run
    spec.write_text(HEADER + scan_block(1, 16))
    write_images(str(img), "test_anneal", 1, 16)
    last = os.path.join(img, lm.image_name("b_stone", "test_anneal", 1, 15))
    with open(last, "wb") as fh:
        fh.write(b"\0" * (lm.RAW_BYTES // 2))                # still being written
    assert lm.ScanTracker(str(spec), str(img), "b_stone", start_scan=1).poll() == []


def test_default_start_is_scan_in_progress_or_next(run):
    spec, img = run
    spec.write_text(HEADER + scan_block(1, 16) + scan_block(2, 5))
    assert lm.ScanTracker(str(spec), str(img), "b_stone").next_scan == 2
    spec.write_text(HEADER + scan_block(1, 16) + scan_block(2, 16))
    assert lm.ScanTracker(str(spec), str(img), "b_stone").next_scan == 3


def test_scan_without_images_skipped_after_grace(run):
    spec, img = run
    now = [0.0]
    spec.write_text(HEADER + scan_block(1, 16, cmd="ascan  vort_y 1 3  15 0.2") + scan_block(2, 16))
    write_images(str(img), "test_anneal", 2, 16)
    t = lm.ScanTracker(str(spec), str(img), "b_stone", start_scan=1, skip_after_s=30, clock=lambda: now[0])
    assert t.poll() == []                                    # scan 1 has no images yet
    now[0] = 31.0
    assert [s.number for s in t.poll()] == [2] and t.skipped == [1]


def test_unknown_scan_type_completes_when_next_starts(run):
    spec, img = run
    spec.write_text(HEADER + scan_block(1, 4, cmd="timescan 3 0"))
    write_images(str(img), "test_anneal", 1, 4)
    t = lm.ScanTracker(str(spec), str(img), "b_stone", start_scan=1)
    assert t.poll() == []
    spec.write_text(HEADER + scan_block(1, 4, cmd="timescan 3 0") + scan_block(2, 2))
    assert [s.number for s in t.poll()] == [1]


# ---------------------------------------------------------------- controller (Qt, fake worker)
from PyQt5.QtCore import QElapsedTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication


class FakeWorker(QThread):
    progress_updated = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)
    log = []

    def __init__(self, spec, scan, *a):
        super().__init__()
        self.scan = scan
        FakeWorker.log.append(scan)

    def run(self):
        x = np.arange(5, 45, 0.01)
        y = 100 + 50 * np.exp(-0.5 * ((x - 20) / 0.05) ** 2)
        self.result_ready.emit(f"test_anneal_scan{self.scan}.xye", x, y, np.sqrt(y))


def pump(app, cond, ms=5000):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < ms and not cond():
        app.processEvents()
    app.processEvents()


def test_controller_integrates_each_new_scan_once_in_order(run, tmp_path):
    app = QApplication.instance() or QApplication([])
    spec, img = run
    FakeWorker.log = []
    written, got = [], []
    c = lm.LiveController(lambda *a: FakeWorker(*a), lambda *a: written.append(a[1]), poll_ms=50)
    c.scan_integrated.connect(lambda name, scan, *r: got.append(scan))
    spec.write_text(HEADER + scan_block(1, 16))
    write_images(str(img), "test_anneal", 1, 16)
    c.start(str(spec), str(img), "b_stone", None, {"error_model": "poisson"}, str(tmp_path) + "/", start_scan=1,
            detect=True)
    pump(app, lambda: got == [1])
    spec.write_text(HEADER + scan_block(1, 16) + scan_block(2, 16) + scan_block(3, 16))
    write_images(str(img), "test_anneal", 2, 16)
    write_images(str(img), "test_anneal", 3, 16)
    pump(app, lambda: got == [1, 2, 3])
    c.stop()
    assert got == [1, 2, 3] and FakeWorker.log == [1, 2, 3]
    assert written == [f"test_anneal_scan{n}.xye" for n in (1, 2, 3)]
    assert len(c.monitor.scans) == 3                         # each pattern reached insitu-seg


# ---------------------------------------------------------------- end to end (real data, GUI)
def test_gui_live_mode_end_to_end(tmp_path):
    import Pilatus_Integration_GUI as gui
    app = QApplication.instance() or QApplication([])
    img = tmp_path / "images"
    shutil.copytree(IMAGES, img)
    spec = tmp_path / os.path.basename(SPEC)
    shutil.copy(SPEC, spec)
    out = tmp_path / "out"
    out.mkdir()
    w = gui.PilatusIntegrationGUI()
    w.spec_path, w.image_path, w.output_path, w.user = str(spec), str(img), str(out) + "/", "b_stone"
    w.read_calibration_parameters(CAL)
    w.live_start_input.setText("1")
    w.live_toggle.setChecked(True)
    assert not w.integrate_button.isEnabled(), "manual integration is disabled in live mode"
    pump(app, lambda: len(w.live_scans) == 1, 30000)
    w.live_toggle.setChecked(False)
    assert w.integrate_button.isEnabled()
    f = out / "KHS10_27D_CaCO3_CuO_anneal_scan1.xye"
    assert f.exists() and np.allclose(np.loadtxt(f), np.loadtxt(REF), rtol=1e-6)
    assert w.plot_list.count() == 1
    w.close()
