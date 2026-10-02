"""Open build/manual.docx in Word, update the table of contents, save, export PDF, and render
page images (build/pages/) for checking the layout.

    python to_pdf.py [dpi] [--install]

--install copies the result to the repo root (manual.docx, manual.pdf: what Help > Open Manual
PDF opens) and to dist/manual.pdf (shipped beside the executable). Needs Microsoft Word (pywin32)
and PyMuPDF."""
import os
import shutil
import sys

import pymupdf
import win32com.client as win32

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "build")
REPO = os.path.dirname(os.path.dirname(HERE))
docx_path = os.path.join(D, "manual.docx")
pdf_path = os.path.join(D, "manual.pdf")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
dpi = int(args[0]) if args else 60

word = win32.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(docx_path)
    for i in range(1, doc.TablesOfContents.Count + 1):
        doc.TablesOfContents(i).Update()
    doc.Fields.Update()
    doc.Save()
    doc.ExportAsFixedFormat(pdf_path, 17)          # 17 = wdExportFormatPDF
    pages = doc.ComputeStatistics(2)               # 2 = wdStatisticPages
    doc.Close(False)
finally:
    word.Quit()

pdf = pymupdf.open(pdf_path)
pdir = os.path.join(D, "pages")
os.makedirs(pdir, exist_ok=True)
for f in os.listdir(pdir):
    if f.endswith(".png"):
        os.remove(os.path.join(pdir, f))
for i, pg in enumerate(pdf):
    pg.get_pixmap(dpi=dpi).save(os.path.join(pdir, f"p{i + 1:02d}.png"))
print("pages:", pdf.page_count, "(Word says", pages, ")")
pdf.close()

if "--install" in sys.argv:
    os.makedirs(os.path.join(REPO, "dist"), exist_ok=True)
    for src, dst in ((pdf_path, "manual.pdf"), (os.path.join("dist", "manual.pdf"), None),
                     (docx_path, "manual.docx")):
        src, dst = (pdf_path, src) if dst is None else (src, dst)
        try:
            shutil.copy2(src, os.path.join(REPO, dst))
            print("installed:", dst)
        except PermissionError:
            print(f"SKIPPED {dst}: file is open in another program (close it and rerun with --install)")
