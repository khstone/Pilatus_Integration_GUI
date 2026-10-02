"""Clear Data keeps the entered fields; Help > Open Manual looks next to the program, then online."""
import os

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

import Pilatus_Integration_GUI as gui
from conftest import CAL, IMAGES, SPEC


@pytest.fixture
def win(monkeypatch):
    app = QApplication.instance() or QApplication([])
    w = gui.PilatusIntegrationGUI()
    yield w, app, monkeypatch
    w.close()


def test_clear_data_clears_data_but_keeps_fields_and_settings(win):
    w, app, mp = win
    w.calib_path_input.setText(CAL); w.read_calibration_parameters(CAL)
    w.spec_path_input.setText(SPEC); w.spec_path = SPEC
    w.image_path_input.setText(IMAGES); w.image_path = IMAGES
    w.output_path_input.setText("out"); w.output_path = "out/"
    w.user_input.setText("b_stone"); w.user = "b_stone"
    w.integration_settings["stepsize"] = "0.01"; w.stepsize_input.setText("0.01")
    w.scan_toggle.setChecked(True); w.scan_start_input.setText("3"); w.scan_end_input.setText("9")
    x = np.linspace(5, 45, 50)
    for n in range(3):
        w.plot_data[f"a_scan{n}.xye"] = {"x": x, "y": x, "e": x}
        w.plot_list.addItem(f"a_scan{n}.xye")
    w.event_list.addItem("scan 1 [low] area")
    fields = {f: getattr(w, f).text() for f in ("calib_path_input", "spec_path_input", "image_path_input",
                                                "output_path_input", "user_input", "stepsize_input",
                                                "scan_start_input", "scan_end_input")}
    paths = (w.spec_path, w.image_path, w.output_path, w.user, w.xyz_map is not None)
    mp.setattr(gui.QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w.clear_data()
    assert w.plot_list.count() == 0 and w.plot_data == {} and w.event_list.count() == 0
    assert {f: getattr(w, f).text() for f in fields} == fields, "entered fields must be unchanged"
    assert (w.spec_path, w.image_path, w.output_path, w.user, w.xyz_map is not None) == paths
    assert w.scan_toggle.isChecked() and w.scan_stack.currentIndex() == 1
    assert w.integration_settings["stepsize"] == "0.01"


def test_clear_data_cancel_changes_nothing(win):
    w, app, mp = win
    w.plot_data["a_scan1.xye"] = {"x": [1], "y": [1], "e": [1]}
    w.plot_list.addItem("a_scan1.xye")
    mp.setattr(gui.QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    w.clear_data()
    assert w.plot_list.count() == 1


def test_program_dir_is_the_script_folder_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert gui.program_dir() == os.path.dirname(os.path.abspath(gui.__file__))
    assert os.path.isfile(gui.resource_path("icon.png")), "icons must be found from any working directory"


def test_program_dir_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(gui.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gui.sys, "executable", str(tmp_path / "Pilatus_Integration_GUI.exe"))
    assert gui.program_dir() == str(tmp_path)


def test_open_manual_uses_pdf_next_to_program(win, tmp_path):
    w, app, mp = win
    (tmp_path / "manual.pdf").write_bytes(b"%PDF-1.4")
    mp.setattr(gui, "program_dir", lambda: str(tmp_path))
    opened = []
    mp.setattr(gui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    assert w.open_manual() == str(tmp_path / "manual.pdf")
    assert opened[0].isLocalFile() and opened[0].toLocalFile().endswith("manual.pdf")


def test_open_manual_falls_back_to_github(win, tmp_path):
    w, app, mp = win
    mp.setattr(gui, "program_dir", lambda: str(tmp_path))          # no manual.pdf here
    opened = []
    mp.setattr(gui.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    assert w.open_manual() == gui.MANUAL_URL
    assert opened == [gui.MANUAL_URL]
