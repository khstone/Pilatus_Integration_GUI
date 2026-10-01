"""Live mode: integrate each scan as soon as it is complete, without any user action.

ScanTracker (pure Python, no Qt) reads the SPEC file and the image folder and reports scans
that are complete, in order, each exactly once. A scan is complete when
  - its #S block has all expected points (ascan/dscan/a2scan/d2scan: intervals + 1; other
    scan types: when the next #S has started), and
  - every image <user>_<spec>_scan<N>_<kkkk>.raw exists at full size, and its .raw.pdi
    sidecar exists if the detector writes them for this scan (the .pdi is written after the
    image, so it marks the image as finished).
A scan whose images never appear (e.g. a motor alignment scan) is skipped once a later scan
has been seen for `skip_after_s` seconds, so it cannot stall the queue.

LiveController (Qt) polls the tracker on a timer, integrates ready scans one at a time with
the existing IntegrationWorker, writes each .xye, and (optionally) feeds each pattern to an
insitu_seg.Monitor, which reports transformations a few scans behind.
"""
from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

RAW_BYTES = 195 * 487 * 4
_FIXED_POINT_SCANS = {"ascan", "dscan", "a2scan", "d2scan", "a3scan", "d3scan", "lup", "dscan_tth"}


@dataclass
class SpecScan:
    number: int
    command: str
    n_points: int = 0                  # data lines found so far
    expected: int | None = None        # from the scan command, if known
    closed: bool = False               # a later #S has started
    epoch: float | None = None         # first-point Epoch (absolute, uses #E)
    temps: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        if self.expected is not None:
            return self.n_points >= self.expected
        return self.closed and self.n_points > 0

    @property
    def mean_temp(self) -> float | None:
        t = [x for x in self.temps if abs(x) > 1e-9]
        return float(np.mean(t)) if t else None


def expected_points(command: str) -> int | None:
    """'ascan tth 5 45 15 3' -> 16. Unknown scan types -> None."""
    tok = command.split()
    if not tok or tok[0] not in _FIXED_POINT_SCANS:
        return None
    try:
        return int(float(tok[-2])) + 1
    except (ValueError, IndexError):
        return None


def parse_spec(path: str) -> dict[int, SpecScan]:
    scans: dict[int, SpecScan] = {}
    cur: SpecScan | None = None
    cols: list[str] | None = None
    e0 = 0.0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#E "):
                try:
                    e0 = float(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("#S "):
                if cur is not None:
                    cur.closed = True
                parts = line.split(None, 2)
                try:
                    n = int(parts[1])
                except (IndexError, ValueError):
                    continue
                cmd = parts[2].strip() if len(parts) > 2 else ""
                cur = SpecScan(n, cmd, expected=expected_points(cmd))
                scans[n] = cur
                cols = None
            elif line.startswith("#L ") and cur is not None:
                cols = line[3:].split()
            elif cur is not None and cols and line.strip() and not line.startswith("#"):
                v = line.split()
                if len(v) != len(cols):
                    continue
                cur.n_points += 1
                if cur.epoch is None and "Epoch" in cols:
                    cur.epoch = e0 + float(v[cols.index("Epoch")])
                if "TEMP" in cols:
                    cur.temps.append(float(v[cols.index("TEMP")]))
    return scans


def image_name(user: str, spec_name: str, scan: int, k: int) -> str:
    return f"{user}_{spec_name}_scan{scan}_{k:04d}.raw"


def images_ready(image_dir: str, user: str, spec_name: str, scan: int, n: int) -> bool:
    paths = [os.path.join(image_dir, image_name(user, spec_name, scan, k)) for k in range(n)]
    if not all(os.path.isfile(p) and os.path.getsize(p) == RAW_BYTES for p in paths):
        return False
    pdis = [os.path.isfile(p + ".pdi") for p in paths]
    return all(pdis) or not any(pdis)          # require .pdi only if the detector writes them


class ScanTracker:
    def __init__(self, spec_path: str, image_dir: str, user: str, start_scan: int | None = None,
                 skip_after_s: float = 30.0, clock=time.monotonic):
        self.spec_path = spec_path
        self.spec_name = os.path.basename(spec_path)
        self.image_dir = image_dir
        self.user = user
        self.skip_after_s = skip_after_s
        self.clock = clock
        self._sig = None
        self.scans: dict[int, SpecScan] = {}
        self.skipped: list[int] = []
        self._later_seen: dict[int, float] = {}
        # start_scan None: begin with the scan in progress, or the next one if the last is done
        self.next_scan = start_scan
        if self.next_scan is None:
            existing = parse_spec(spec_path) if os.path.isfile(spec_path) else {}
            if not existing:
                self.next_scan = 1
            else:
                last = existing[max(existing)]
                self.next_scan = last.number if not last.complete else last.number + 1

    def _refresh(self) -> None:
        st = os.stat(self.spec_path)
        sig = (st.st_size, st.st_mtime_ns)
        if sig != self._sig:
            self.scans = parse_spec(self.spec_path)
            self._sig = sig

    def poll(self) -> list[SpecScan]:
        """Scans that became ready since the last call, in order; each returned once."""
        if not os.path.isfile(self.spec_path):
            return []
        self._refresh()
        ready = []
        while True:
            s = self.scans.get(self.next_scan)
            if s is None:
                # gap in numbering: jump to the next existing scan if there is one
                later = [n for n in self.scans if n > self.next_scan]
                if later:
                    self.next_scan = min(later)
                    continue
                break
            if s.complete and images_ready(self.image_dir, self.user, self.spec_name, s.number, s.n_points):
                ready.append(s)
                self.next_scan += 1
                continue
            if any(n > s.number for n in self.scans):
                first = self._later_seen.setdefault(s.number, self.clock())
                if self.clock() - first >= self.skip_after_s:
                    self.skipped.append(s.number)
                    self.next_scan += 1
                    continue
            break
        return ready


# ---------------------------------------------------------------------- Qt controller
try:
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal
except ImportError:                                    # tracker stays usable without Qt
    QObject = object

if QObject is not object:
    class LiveController(QObject):
        scan_integrated = pyqtSignal(str, int, object, object, object)   # name, scan, x, y, e
        events_found = pyqtSignal(object)                                # list[insitu_seg.Event]
        status = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, worker_factory, write_data, poll_ms: int = 2000, parent=None):
            super().__init__(parent)
            self.worker_factory = worker_factory       # (spec, scan, image_path, user, xyz, settings, use_var)
            self.write_data = write_data               # (output_path, name, x, y, e)
            self.timer = QTimer(self)
            self.timer.setInterval(poll_ms)
            self.timer.timeout.connect(self.poll)
            self.queue: deque[SpecScan] = deque()
            self.worker = None
            self.tracker = None
            self.monitor = None
            self.n_done = 0
            self.active = False

        def start(self, spec_path, image_path, user, xyz_map, settings, output_path,
                  start_scan=None, detect=False, monitor_params=None):
            self.spec_path, self.image_path, self.user = spec_path, image_path, user
            self.xyz_map, self.settings, self.output_path = xyz_map, dict(settings), output_path
            self.tracker = ScanTracker(spec_path, image_path, user, start_scan)
            self.monitor = None
            if detect:
                from insitu_seg import Monitor, Params
                self.monitor = Monitor(monitor_params or Params())
            self.queue.clear()
            self.n_done = 0
            self.active = True
            self.status.emit(f"Live: watching {os.path.basename(spec_path)} from scan {self.tracker.next_scan}")
            self.timer.start()
            self.poll()

        def stop(self):
            self.active = False
            self.timer.stop()
            if self.monitor is not None:
                ev = self.monitor.flush()
                if ev:
                    self.events_found.emit(ev)
            self.status.emit(f"Live stopped after {self.n_done} scans")

        def poll(self):
            if not self.active or self.tracker is None:
                return
            try:
                for s in self.tracker.poll():
                    self.queue.append(s)
            except OSError as exc:
                self.status.emit(f"Live: cannot read SPEC file ({exc}); retrying")
                return
            self._start_next()

        def _start_next(self):
            if self.worker is not None or not self.queue:
                return
            s = self.queue.popleft()
            self._current = s
            w = self.worker_factory(self.spec_path, s.number, self.image_path, self.user, self.xyz_map,
                                    self.settings, self.settings.get("error_model") == "azimuthal")
            w.result_ready.connect(self._on_result)
            w.error_occurred.connect(self._on_error)
            w.finished.connect(self._on_finished)
            self.worker = w
            self.status.emit(f"Live: integrating scan {s.number} ({len(self.queue)} queued)")
            w.start()

        def _on_result(self, name, x, y, e):
            s = self._current
            try:
                self.write_data(self.output_path, name, x, y, e)
            except Exception as exc:                     # keep running; report
                self.error.emit(f"Live: could not write {name}: {exc}")
            self.n_done += 1
            self.scan_integrated.emit(name, s.number, x, y, e)
            if self.monitor is not None:
                ev = self.monitor.add(s.number, np.asarray(x), np.asarray(y), time=s.epoch, T=s.mean_temp)
                if ev:
                    self.events_found.emit(ev)

        def _on_error(self, msg):
            self.error.emit(f"Live: {msg}")

        def _on_finished(self):
            w = self.worker
            if w is not None:
                w.wait()
                w.deleteLater()
            self.worker = None
            if self.active:
                self._start_next()
            if self.active and not self.queue:
                self.status.emit(f"Live: {self.n_done} scans integrated; waiting for scan {self.tracker.next_scan}")
