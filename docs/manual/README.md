# Regenerating the user manual

The manual (`manual.docx` / `manual.pdf` in the repo root, and `dist/manual.pdf` beside the
executable) is generated from the running GUI, so the screenshots match the program.

| Step | Script | Environment |
|---|---|---|
| 1. Screenshots + widget positions | `shots.py` | `PilatusGUI` (PyQt5) |
| 2. Red numbered callouts | `annotate.py` | `PilatusGUI` (matplotlib) |
| 3. Build `manual.docx` | `build_manual.py` | any env with `python-docx` |
| 4. Word: update contents, export PDF, render pages; `--install` copies to repo root and `dist/` | `to_pdf.py` | env with `pywin32` + `pymupdf`, Microsoft Word installed |

```
conda activate PilatusGUI
python docs/manual/shots.py          # ~2 min: includes a 288-scan live-mode replay
python docs/manual/annotate.py
conda activate harness-mcp           # or any env with python-docx, pywin32, pymupdf
python docs/manual/build_manual.py
python docs/manual/to_pdf.py 90 --install
```

Outputs go to `docs/manual/build/` (not tracked). Check `build/pages/p*.png` before installing.

**Data.** `shots.py` uses the repo test scan (`tests/data/CaCO3_CuO_scan1`) and the May 2026
in-situ runs (CCTO with raw images and SPEC file, TiO2_CuO `.xye` files). Set `MANUAL_DATA_DIR`
to the folder containing `CCTO/`, `TiO2_CuO/`, and `directbeam_20260519_scan1_calib.cal` if it is
not at the default Dropbox path.

**Text.** All manual text is in `build_manual.py`. Callout numbers in Figures 1 and 5 are set in
`annotate.py` and must match the numbered tables in `build_manual.py`.
