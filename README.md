This is a graphical user interface for the integration of powder diffraction data measured at SSRL beamline 2-1 using the Pilatus 100K small area detector.  This also provides visualization of data as it is integrated alongside previously integrated data.  Includes an executable of a stable version.

## Live mode
Integrates each scan automatically as soon as it is complete and shows a live waterfall
(intensity vs 2θ and scan number). No clicks are needed during the experiment.

1. Load the calibration file, SPEC file, image path, and output path as usual.
2. Tick **Live integration** in the Live Mode box. Leave **Start at scan** blank to begin
   with the scan in progress (or the next one); enter a number to also integrate earlier
   scans already collected.
3. Each finished scan is integrated, written as `.xye` to the output path, added to the
   data list, and drawn on the waterfall. Untick to stop. Manual **Integrate** is disabled
   while live mode runs.

A scan is integrated when its SPEC block has all its points (for `ascan`: intervals + 1;
other scan types: when the next scan starts) and every image is present at full size with
its `.pdi` file. Scans that never get images (e.g. alignment scans) are skipped after 30 s.

**Detect transformations** (optional, needs the `insitu-seg` package): flags reactions,
phase changes, reversible transitions, and peak sharpening/broadening, about 6 scans after
they happen, without knowing which phases are present. Events are listed in the Live Mode
box and drawn on the waterfall: solid lines = high confidence (two or more detector families
agree), dotted = low. Events are suggestions; you decide what they mean.

Rehearsed on a full in-situ run (288 scans): ~0.4 s per scan including the redraw.

## Event detection on loaded data
**Analysis > Detect Events in Selected Data** runs the same detection on patterns already in
the data list (imported or integrated): the selected patterns, or all if none are selected.
Patterns are ordered by the scan number in their names (`..._scanN.xye`); with the matching
SPEC file loaded, timestamps and furnace temperatures (when logged) are attached. This uses
the whole series as its baseline and usually gives fewer false alarms than live mode. Needs at
least 25 patterns.

## Help
- **Help > Live Mode Guide**: using live mode and when a scan counts as complete.
- **Help > Understanding Detected Events**: how to read an event such as
  `scan 96 [high] area+pearson+pointwise+profile+slope, sharpening`, what high/low confidence
  mean, what each detector family measures and may indicate physically, and what is not
  detected.
- **Hover over any event** in the Events tab for an explanation of that event.

## Tests
```
conda activate PilatusGUI
pytest tests            # offscreen; no display needed
```
- `test_engine.py`: one real scan (CaCO3 + CuO, May 2026, 0519 calibration) must reproduce
  the beamline `.xye`; azimuthal error model checked against an independent calculation.
- `test_gui.py`: one click integrates once; scan ranges integrate every scan in order; real
  worker end to end; calibration menu does not crash.
- `test_live.py`: scan-completion rules, live sequencing, and live mode end to end.

## Error models
- **poisson**: σ = √I.
- **azimuthal**: σ = spread (standard deviation) of the pixel values in each 2θ bin, so
  spotty rings get large errors and smooth rings errors near counting statistics.
