import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DATA = os.path.join(os.path.dirname(__file__), "data", "CaCO3_CuO_scan1")
SPEC = os.path.join(DATA, "KHS10_27D_CaCO3_CuO_anneal")
IMAGES = os.path.join(DATA, "images")
CAL = os.path.join(DATA, "directbeam_20260519_scan1_calib.cal")
REF = os.path.join(DATA, "reference_scan1.xye")
USER = "b_stone"
SETTINGS = {"min_tth": 0.5, "max_tth": 180.0, "stepsize": "0.005", "img_clip_low": 20, "img_clip_high": 467}
