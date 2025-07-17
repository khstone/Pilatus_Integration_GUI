from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import Integration_engine as engine

class IntegrationWorker(QThread):
    # Signals to communicate with the GUI thread
    #progress_updated = pyqtSignal(str)  # Status text
    #progress_value = pyqtSignal(int)    # Progress percentage (0-100)
    #progress_updated = pyqtSignal(str)          # Status messages (e.g., "Processing Scan 1...")
    #result_ready = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)  # scan_name, x, y, e
    #error_occurred = pyqtSignal(str)            # Error messages
    # Add progress signal
    progress_percent = pyqtSignal(int)  # New signal for percentage

    def __init__(self, spec_path, scan_num, image_path, user, xyz_map, settings, use_variance=False):
        super().__init__()
        self.spec_path = spec_path
        self.scan_num = scan_num
        self.image_path = image_path
        self.user = user
        self.xyz_map = xyz_map
        self.settings = settings
        self.use_variance = use_variance

    def run(self):
        """Runs in the background thread."""
        try:
            self.progress_updated.emit(f"Starting integration for Scan {self.scan_num}...")
            
            if self.use_variance:
                scan_name, x, y, e = integrate_var(
                    self.spec_path, self.scan_num, self.image_path, 
                    self.user, self.xyz_map, self.settings
                )
            else:
                scan_name, x, y, e = integrate(
                    self.spec_path, self.scan_num, self.image_path, 
                    self.user, self.xyz_map, self.settings
                )
            
            self.result_ready.emit(scan_name, x, y, e)
            self.progress_updated.emit(f"Scan {self.scan_num} completed!")
        
        except Exception as e:
            self.error_occurred.emit(f"Error in Scan {self.scan_num}: {str(e)}")