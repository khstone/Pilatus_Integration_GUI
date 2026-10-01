"""GUI control-flow tests (offscreen). Integration itself is replaced by a fake worker so
these test only what the GUI does with clicks, results, and thread completion."""
import numpy as np
import pytest
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

import Integration_worker
import Pilatus_Integration_GUI as gui
from conftest import CAL, IMAGES, SPEC

started = []


class FakeWorker(QThread):
    progress_updated = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    result_ready = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, spec_path, scan_num, image_path, user, xyz_map, settings, use_variance=False):
        super().__init__()
        self.scan_num = scan_num
        started.append(scan_num)

    def run(self):
        x = np.linspace(5, 45, 10)
        self.result_ready.emit(f"fake_scan{self.scan_num}.xye", x, x, x)


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(Integration_worker, "IntegrationWorker", FakeWorker)
    monkeypatch.setattr(gui.engine, "write_data", lambda *a, **k: None)
    started.clear()
    w = gui.PilatusIntegrationGUI()
    w.spec_path, w.image_path, w.output_path, w.user = SPEC, IMAGES, str(tmp_path) + "/", "b_stone"
    w.read_calibration_parameters(CAL)
    yield w, app
    w.close()


def pump(app, w, timeout_ms=5000):
    from PyQt5.QtCore import QElapsedTimer
    t = QElapsedTimer(); t.start()
    while t.elapsed() < timeout_ms:
        app.processEvents()
        if (w.worker is None or not w.worker.isRunning()) and not w.processing_scan_range:
            app.processEvents()
            return


def test_single_click_starts_one_integration(window):
    w, app = window
    w.scan_number_input.setText("7")
    w.plot_integrated_data()
    pump(app, w)
    assert started == [7], f"one click should integrate once, got {started}"


def test_scan_range_integrates_every_scan_in_order(window):
    w, app = window
    w.scan_toggle.setChecked(True)
    w.scan_start_input.setText("3")
    w.scan_end_input.setText("6")
    w.plot_integrated_data()
    pump(app, w)
    assert started == [3, 4, 5, 6], f"range should integrate 3..6 once each, got {started}"
    assert w.plot_list.count() == 4


def test_real_worker_end_to_end_writes_reference_xye(tmp_path):
    """Through the GUI with the real IntegrationWorker thread: integrate scan 1 and compare
    the written file with the reference produced at the beamline."""
    import os
    from conftest import REF
    app = QApplication.instance() or QApplication([])
    w = gui.PilatusIntegrationGUI()
    w.spec_path, w.image_path, w.output_path, w.user = SPEC, IMAGES, str(tmp_path) + "/", "b_stone"
    w.read_calibration_parameters(CAL)
    w.scan_number_input.setText("1")
    w.plot_integrated_data()
    pump(app, w, 30000)
    out = tmp_path / "KHS10_27D_CaCO3_CuO_anneal_scan1.xye"
    assert out.exists()
    got, ref = np.loadtxt(out), np.loadtxt(REF)
    assert got.shape == ref.shape and np.allclose(got, ref, rtol=1e-6)
    w.close()


def test_calibration_menu_does_not_crash(window, monkeypatch):
    w, app = window
    shown = []
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *a, **k: shown.append(a))
    before = dict(w.plot_settings)
    w.open_run_calib_settings()
    assert w.plot_settings == before, "calibration action must not touch plot settings"
    assert shown, "should tell the user to use the CLI calibration"
