"""Capture GUI screenshots for the manual (offscreen, Windows fonts, Fusion style).
Writes PNGs plus callouts.json (widget rectangles in window pixel coordinates) to build/.

Run in the PilatusGUI env. Needs the May 2026 in-situ data (CCTO, TiO2_CuO runs with raw images
for CCTO); set MANUAL_DATA_DIR if it is not at the default location."""
import glob, json, os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_FONTDIR"] = r"C:\Windows\Fonts"
HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.dirname(os.path.dirname(HERE))          # repo root
sys.path.insert(0, GUI)
os.chdir(GUI)
from PyQt5.QtCore import QPoint, QElapsedTimer, Qt
from PyQt5.QtWidgets import QApplication, QStyleFactory
app = QApplication([])
app.setStyle(QStyleFactory.create("Fusion"))
import Pilatus_Integration_GUI as g
import help_text

OUT = os.path.join(HERE, "build")
os.makedirs(OUT, exist_ok=True)
B = os.environ.get("MANUAL_DATA_DIR",
                   r"C:\Users\khstone\Dropbox\Projects\Sikhumbuzo HEO\Rare Earth Perovskites\May2026")
TEST = os.path.join(GUI, "tests", "data", "CaCO3_CuO_scan1")
W, H = 1200, 780
callouts = {}


def pump(ms=300, cond=None, limit=60000):
    t = QElapsedTimer(); t.start()
    while t.elapsed() < (limit if cond else ms):
        app.processEvents()
        if cond and cond():
            break
    for _ in range(5):
        app.processEvents()


def rect(w, win):
    p = w.mapTo(win, QPoint(0, 0))
    return [p.x(), p.y(), w.width(), w.height()]


def union(*rs):
    x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
    x1 = max(r[0] + r[2] for r in rs); y1 = max(r[1] + r[3] for r in rs)
    return [x0, y0, x1 - x0, y1 - y0]


def new_window():
    w = g.PilatusIntegrationGUI()
    w.resize(W, H)
    w.show()
    pump()
    return w


def fill_paths(w, spec, images, out, calib, user="b_stone", run="CCTO", spec_name="KHS10_27A_CCTO_anneal"):
    """Real paths for the program; tidy example paths in the visible fields."""
    w.read_calibration_parameters(calib)
    w.spec_path, w.image_path, w.output_path, w.user = spec, images, out, user
    base = "D:\\BL2-1\\May2026\\" + run + "\\"
    w.calib_path_input.setText("D:\\BL2-1\\calibration\\directbeam_20260519_scan1_calib.cal")
    w.spec_path_input.setText(base + spec_name)
    w.image_path_input.setText(base + "images")
    w.output_path_input.setText(base + "xye")
    for f in (w.calib_path_input, w.spec_path_input, w.image_path_input, w.output_path_input):
        f.setCursorPosition(0)          # show the start of each path
    w.user_input.setText(user)


def load_xye(w, folder, scans=None):
    files = sorted(glob.glob(os.path.join(folder, "*_anneal_scan*.xye")),
                   key=lambda f: int(f.rsplit("scan", 1)[1][:-4]))
    for f in files:
        n = int(f.rsplit("scan", 1)[1][:-4])
        if scans and n not in scans:
            continue
        x, y, e = w.read_integrated_data(f)
        name = os.path.basename(f)
        w.plot_data[name] = {"x": x, "y": y, "e": e}
        w.plot_list.addItem(name)


# 1. Main window after integrating one scan (layout figure) --------------------------
tmp = os.path.join(OUT, "tmp_out"); os.makedirs(tmp, exist_ok=True)
w = new_window()
fill_paths(w, os.path.join(TEST, "KHS10_27D_CaCO3_CuO_anneal"), os.path.join(TEST, "images"), tmp + "/",
           os.path.join(TEST, "directbeam_20260519_scan1_calib.cal"),
           run="CaCO3_CuO", spec_name="KHS10_27D_CaCO3_CuO_anneal")
w.scan_number_input.setText("1")
w.plot_integrated_data()
pump(cond=lambda: w.plot_list.count() >= 1 and (w.worker is None or not w.worker.isRunning()))
pump(500)
w.grab().save(os.path.join(OUT, "01_layout.png"))
callouts["01_layout"] = {
    "1": rect(w.layout().itemAt(0).widget(), w),                                   # menu bar
    "2": union(rect(w.calib_path_input, w), rect(w.calib_path_button, w)),
    "3": union(rect(w.spec_path_input, w), rect(w.spec_path_button, w)),
    "4": union(rect(w.image_path_input, w), rect(w.image_path_button, w)),
    "5": union(rect(w.output_path_input, w), rect(w.output_path_button, w)),
    "6": union(rect(w.user_input, w), rect(w.stepsize_input, w)),
    "7": union(rect(w.scan_stack, w), rect(w.scan_toggle, w)),
    "8": rect(w.integrate_button, w),
    "9": union(rect(w.overlay_toggle, w), rect(w.contour_plot_toggle, w)),
    "10": rect(w.live_group, w),
    "11": rect(w.data_tabs, w),
    "12": rect(w.canvas, w),
    "13": rect(w.toolbar, w),
    "14": rect(w.status_bar, w),
}
callouts["01_layout_size"] = [w.width(), w.height()]

# 2. Menus ------------------------------------------------------------------------
mb = w.layout().itemAt(0).widget()
for i, act in enumerate(mb.actions()):
    m = act.menu()
    m.popup(w.mapToGlobal(QPoint(0, 0)))
    pump(200)
    m.grab().save(os.path.join(OUT, f"02_menu_{act.text().replace('&', '')}.png"))
    m.hide()

# 3. Settings dialogs ------------------------------------------------------------
d = g.IntegSettingsDialog(w.integration_settings, w); d.show(); pump(200)
d.grab().save(os.path.join(OUT, "03_integration_settings.png")); d.close()
d = g.PlotSettingsDialog(w.plot_settings, w); d.show(); pump(200)
d.grab().save(os.path.join(OUT, "04_plot_settings.png")); d.close()
w.close()

# 4. Overlay plot of several CCTO scans --------------------------------------------
w = new_window()
load_xye(w, os.path.join(B, "CCTO", "xye"), scans={20, 90, 100, 110, 140})
w.overlay_toggle.setChecked(True)
w.plot_settings.update({"min_x": 10.0, "max_x": 30.0})
for i in range(w.plot_list.count()):
    w.plot_list.item(i).setSelected(True)
w.replot_selected(); pump(300)
w.grab().save(os.path.join(OUT, "05_overlay.png"))
w.close()

# 5. Live mode at the end of a run (CCTO replay) + events tab ----------------------
import re
run = os.path.join(B, "CCTO"); src = os.path.join(run, "KHS10_27A_CCTO_anneal")
live = os.path.join(OUT, "live"); os.makedirs(live, exist_ok=True)
spec = os.path.join(live, "KHS10_27A_CCTO_anneal")
text = open(src, encoding="utf-8", errors="replace").read()
starts = [m.start() for m in re.finditer(r"^#S ", text, re.M)]
blocks = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
open(spec, "w").write(text[:starts[0]])
w = new_window()
fill_paths(w, spec, os.path.join(run, "images"), live + "/",
           os.path.join(B, "directbeam_20260519_scan1_calib.cal"))
w.plot_settings["sqrt_scale"] = True
w.live.timer.setInterval(20)
w.live_start_input.setText("1")
w.live_toggle.setChecked(True)
for i in range(288):
    with open(spec, "a") as fh:
        fh.write(blocks[i])
    pump(cond=lambda: len(w.live_scans) >= i + 1, limit=10000)
pump(800)
w.data_tabs.setCurrentIndex(1)
pump(200)
w.grab().save(os.path.join(OUT, "06_live.png"))
callouts["06_live"] = {"1": rect(w.live_toggle, w), "2": rect(w.live_start_input, w),
                       "3": rect(w.live_detect_toggle, w), "4": rect(w.live_follow_toggle, w),
                       "5": rect(w.live_status_label, w), "6": rect(w.data_tabs, w), "7": rect(w.canvas, w)}
callouts["06_live_size"] = [w.width(), w.height()]
callouts["06_live_events"] = [w.event_list.item(k).text() for k in range(w.event_list.count())]
w.live_toggle.setChecked(False); pump(300)
w.close()

# 6. Detect events on loaded data (TiO2 + CuO) ---------------------------------------
w = new_window()
load_xye(w, os.path.join(B, "TiO2_CuO", "xye"))
w.spec_path = os.path.join(B, "TiO2_CuO", "KHS10_27C_TiO2_CuO_anneal")
w.detect_events_in_loaded_data(); pump(300)
w.grab().save(os.path.join(OUT, "07_detect_loaded.png"))
callouts["07_events"] = [w.event_list.item(k).text() for k in range(w.event_list.count())]

# 7. Help dialog ----------------------------------------------------------------------
dl = w.show_help("Understanding Detected Events", help_text.EVENTS_HTML); dl.resize(820, 600); pump(300)
dl.grab().save(os.path.join(OUT, "08_help_events.png")); dl.close()
w.close()

json.dump(callouts, open(os.path.join(OUT, "callouts.json"), "w"), indent=1)
print("done:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
