from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import Integration_engine as engine


class IntegrationWorker(QThread):
    progress_updated = pyqtSignal(str)
    progress_percent = pyqtSignal(int)
    result_ready = pyqtSignal(
        str, np.ndarray, np.ndarray, np.ndarray
    )
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        spec_path,
        scan_num,
        image_path,
        user,
        xyz_map,
        settings,
        use_variance=False
    ):
        super().__init__()

        self.spec_path = spec_path
        self.scan_num = scan_num
        self.image_path = image_path
        self.user = user
        self.xyz_map = xyz_map
        self.settings = settings
        self.use_variance = use_variance

    def report_progress(self, fraction):
        """
        Receive progress from IntegrationEngine as a value from 0.0 to 1.0.
        """
        percent = max(0, min(100, int(round(fraction * 100))))
        self.progress_percent.emit(percent)
        self.progress_updated.emit(
            f"Integrating Scan {self.scan_num}: {percent}%"
        )

    def run(self):
        """Run the integration in the background thread."""
        try:
            self.progress_updated.emit(
                f"Starting integration for Scan {self.scan_num}..."
            )
            self.progress_percent.emit(0)

            # integrate() and integrate_var() are methods of this class.
            integration_engine = engine.IntegrationEngine()
            integration_engine.set_progress_callback(self.report_progress)

            if self.use_variance:
                scan_name, x, y, e = integration_engine.integrate_var(
                    self.spec_path,
                    self.scan_num,
                    self.image_path,
                    self.user,
                    self.xyz_map,
                    self.settings
                )
            else:
                scan_name, x, y, e = integration_engine.integrate(
                    self.spec_path,
                    self.scan_num,
                    self.image_path,
                    self.user,
                    self.xyz_map,
                    self.settings
                )

            self.progress_percent.emit(100)
            self.result_ready.emit(scan_name, x, y, e)
            self.progress_updated.emit(
                f"Scan {self.scan_num} completed!"
            )

        except Exception as exc:
            self.error_occurred.emit(
                f"Error in Scan {self.scan_num}: "
                f"{type(exc).__name__}: {exc}"
            )