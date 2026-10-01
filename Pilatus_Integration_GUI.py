import sys
import os
import time
import traceback
import numpy as np
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm  # Import colormap module
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog, QMessageBox, QSizePolicy, QListWidget,
                             QCheckBox, QStatusBar, QMenuBar, QAction, QDialog, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QColorDialog, QComboBox, QGroupBox, QRadioButton, QAbstractItemView,
                             QListWidgetItem, QSlider, QStyleFactory, QProgressBar, QGridLayout,
                             QScrollArea, QSplitter, QStackedWidget, QFrame, QTabWidget)
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QThread, pyqtSignal
import Integration_engine as engine
import Integration_worker
import live_mode

try:                                    # optional: transformation detection in live mode
    import insitu_seg
    HAVE_INSITU_SEG = True
except ImportError:
    HAVE_INSITU_SEG = False

# This is only needed when using pyinstaller to create an executable
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("About Pilatus Integration GUI")
        self.setWindowIcon(QIcon(resource_path("icon.png")))  # Set the dialog's icon

        layout = QVBoxLayout()

        # Program Icon
        self.icon_label = QLabel(self)
        self.icon_label.setPixmap(QPixmap(resource_path("icon_100x100.png")).scaled(100, 100, Qt.KeepAspectRatio))  # Adjust size as needed
        self.icon_label.setAlignment(Qt.AlignCenter)

        # Description Text
        description = ("<h2>Pilatus Integration GUI</h2>"
                       "<p>Version 0.12</p>"
                       "<p>This program is designed to handle integration tasks related to Pilatus data collected at SSRL BL2-1.</p>"
                       "<p>It allows users to input calibration and spec files, integrated data, load previously integrated data, and visualize data plots.</p>"
                       "<p>Developed by: Kevin Stone</p>")
        self.text_label = QLabel(description)
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignCenter)

        # Add labels to layout
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

        self.setLayout(layout)
        self.setFixedSize(400, 300)  # Fix the size of the dialog

class IntegSettingsDialog(QDialog):
    def __init__(self,settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Integration Settings")
        self.settings = settings
        self.init_ui()
    
    def init_ui(self):
        layout = QFormLayout(self)
        
        # X-Axis Range Setting
        self.min_tth_spinbox = QDoubleSpinBox()
        self.min_tth_spinbox.setRange(0.0, 180.0)
        self.min_tth_spinbox.setSingleStep(0.1)
        self.min_tth_spinbox.setValue(self.settings["min_tth"])  # Default min X
        layout.addRow("Min 2-theta:", self.min_tth_spinbox)

        self.max_tth_spinbox = QDoubleSpinBox()
        self.max_tth_spinbox.setRange(0.0, 180.0)
        self.max_tth_spinbox.setSingleStep(0.1)
        self.max_tth_spinbox.setValue(self.settings["max_tth"])  # Default max X
        layout.addRow("Max 2-theta:", self.max_tth_spinbox)
        
        # Reset X-Axis Range to Auto Button
        self.reset_tth_button = QPushButton("Full 2-theta Range")
        self.reset_tth_button.clicked.connect(self.reset_tth_range)
        layout.addRow(self.reset_tth_button)
        
        # Binning step size
        self.stepsize_label = QLabel("Step Size:")
        self.stepsize_input = QLineEdit(self)
        self.stepsize_input.setText(self.settings['stepsize'])
        layout.addRow("Step Size:", self.stepsize_input)
        
        # Error model selection
        self.error_model_combobox = QComboBox()
        self.error_model_combobox.addItems(['poisson', 'azimuthal'])
        self.error_model_combobox.setCurrentText(self.settings["error_model"])
        layout.addRow("Error Model:", self.error_model_combobox)
        
        # Image clip range
        self.img_clip_low_spinbox = QSpinBox()
        self.img_clip_low_spinbox.setRange(0, 487)  # Set the range of allowable values
        self.img_clip_low_spinbox.setSingleStep(1)  # Set the step size
        self.img_clip_low_spinbox.setValue(self.settings["img_clip_low"])
        layout.addRow("Lower clipping range for images:", self.img_clip_low_spinbox)
        
        self.img_clip_high_spinbox = QSpinBox()
        self.img_clip_high_spinbox.setRange(0, 487)  # Set the range of allowable values
        self.img_clip_high_spinbox.setSingleStep(1)  # Set the step size
        self.img_clip_high_spinbox.setValue(self.settings["img_clip_high"])
        layout.addRow("Upper clipping range for images:", self.img_clip_high_spinbox)
        
        # Accept and Cancel Buttons
        buttons = QHBoxLayout()
        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(accept_button)
        buttons.addWidget(cancel_button)
        layout.addRow(buttons)
        
    def reset_tth_range(self):
        self.full_tth = True
        self.min_tth_spinbox.setValue(0.5)  # Default min X
        self.max_tth_spinbox.setValue(180.0)  # Default max X
        
    def accept(self):
        self.full_tth = False
        self.min_tth_spinbox.setEnabled(True)
        self.max_tth_spinbox.setEnabled(True)
        if self.img_clip_low_spinbox.value() >= self.img_clip_high_spinbox.value():
            QMessageBox.warning(self, 'Image Clipping Error',
                                    "Lower clipping range cannot be greater than upper clipping range, resetting to defaults.")
            self.img_clip_low_spinbox.setValue(20)
            self.img_clip_high_spinbox.setValue(467)
        super().accept()

    def get_settings(self):
        return {
            'min_tth': self.min_tth_spinbox.value() if not self.full_tth else None,
            'max_tth': self.max_tth_spinbox.value() if not self.full_tth else None,
            'full_tth': self.full_tth,
            'stepsize': self.stepsize_input.text(),
            'error_model': self.error_model_combobox.currentText(),
            'img_clip_low': self.img_clip_low_spinbox.value(),
            'img_clip_high': self.img_clip_high_spinbox.value()
            
        }

class PlotSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self.settings = settings
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # Line Width Setting
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setRange(0.5, 5.0)
        self.line_width_spinbox.setSingleStep(0.5)
        self.line_width_spinbox.setValue(self.settings["line_width"])  # Default line width
        layout.addRow("Line Width:", self.line_width_spinbox)

        # Line Style Setting
        self.line_style_combobox = QComboBox()
        self.line_style_combobox.addItems(['solid', 'dashed', 'dotted', 'dashdot'])
        self.line_style_combobox.setCurrentText(self.settings["line_style"])
        layout.addRow("Line Style:", self.line_style_combobox)

        # Line Color Setting
        self.line_color_button = QPushButton("Select Color")
        self.line_color_button.clicked.connect(self.open_color_dialog)
        self.line_color = QColor(Qt.blue)  # Default line color
        layout.addRow("Line Color:", self.line_color_button)
        
        # Colormap Setting
        self.colormap_combobox = QComboBox()
        self.colormap_combobox.addItems(plt.colormaps())  # Add all matplotlib colormaps
        self.colormap_combobox.setCurrentText(self.settings['colormap'])
        layout.addRow("Colormap:", self.colormap_combobox)

        # Marker Setting
        self.marker_combobox = QComboBox()
        self.marker_combobox.addItems(['None', 'o', 's', '^', 'v', '+', 'x', 'd'])
        self.marker_combobox.setCurrentText(self.settings["marker"])
        layout.addRow("Marker Style:", self.marker_combobox)
        
        # X-Axis Range Setting
        self.min_x_spinbox = QDoubleSpinBox()
        self.min_x_spinbox.setRange(0.0, 180.0)
        self.min_x_spinbox.setSingleStep(0.1)
        self.min_x_spinbox.setValue(self.settings["min_x"])  # Default min X
        layout.addRow("Min X:", self.min_x_spinbox)

        self.max_x_spinbox = QDoubleSpinBox()
        self.max_x_spinbox.setRange(0.0, 180.0)
        self.max_x_spinbox.setSingleStep(0.1)
        self.max_x_spinbox.setValue(self.settings["max_x"])  # Default max X
        layout.addRow("Max X:", self.max_x_spinbox)
        
        # Y-Axis Scale Options
        self.y_scale_group = QGroupBox("Y-Axis Scale")
        y_scale_layout = QVBoxLayout()

        self.linear_scale_button = QRadioButton("Linear Scale")
        self.sqrt_scale_button = QRadioButton("Square Root Scale")
        self.log_scale_button = QRadioButton("Log Scale")

        # Set initial checked state based on settings
        self.linear_scale_button.setChecked(not (self.settings['log_scale'] or self.settings['sqrt_scale']))
        self.sqrt_scale_button.setChecked(self.settings['sqrt_scale'])
        self.log_scale_button.setChecked(self.settings['log_scale'])

        y_scale_layout.addWidget(self.linear_scale_button)
        y_scale_layout.addWidget(self.sqrt_scale_button)
        y_scale_layout.addWidget(self.log_scale_button)

        self.y_scale_group.setLayout(y_scale_layout)
        layout.addRow(self.y_scale_group)

        # Reset X-Axis Range to Auto Button
        self.reset_x_button = QPushButton("Reset X-Axis")
        self.reset_x_button.clicked.connect(self.reset_x_range)
        layout.addRow(self.reset_x_button)

        # Accept and Cancel Buttons
        buttons = QHBoxLayout()
        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(accept_button)
        buttons.addWidget(cancel_button)
        layout.addRow(buttons)

    def open_color_dialog(self):
        color = QColorDialog.getColor(self.line_color, self, "Select Line Color")
        if color.isValid():
            self.line_color = color
            
    def reset_x_range(self):
        self.automatic_x = True
        self.min_x_spinbox.setValue(0.0)  # Default min X
        self.max_x_spinbox.setValue(120.0)  # Default max X
        
    def accept(self):
        self.automatic_x = False
        self.min_x_spinbox.setEnabled(True)
        self.max_x_spinbox.setEnabled(True)
        super().accept()

    def get_settings(self):
        return {
            'line_width': self.line_width_spinbox.value(),
            'line_style': self.line_style_combobox.currentText(),
            'line_color': self.line_color,
            'colormap': self.colormap_combobox.currentText(),
            'marker': self.marker_combobox.currentText() if self.marker_combobox.currentText() != 'None' else None,
            'min_x': self.min_x_spinbox.value() if not self.automatic_x else None,
            'max_x': self.max_x_spinbox.value() if not self.automatic_x else None,
            'automatic_x': self.automatic_x,
            'log_scale': self.log_scale_button.isChecked(),
            'sqrt_scale': self.sqrt_scale_button.isChecked()
        }

# ===================== Dialog to run calibration script =====================
class RunCalibDialog(QDialog):
    """
    Dialog that asks for:
      - calibration file (file path)
      - image directory
      - output directory
    and then runs an external Python script.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Run Calibration Script")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # CSV file
        csv_layout = QHBoxLayout()
        csv_label = QLabel("CSV file:", self)
        self.csv_edit = QLineEdit(self)
        csv_browse = QPushButton("Browse...", self)
        csv_browse.clicked.connect(self.browse_csv)
        csv_layout.addWidget(csv_label)
        csv_layout.addWidget(self.csv_edit)
        csv_layout.addWidget(csv_browse)
        layout.addLayout(csv_layout)

        # Image path (directory)
        img_layout = QHBoxLayout()
        img_label = QLabel("Image directory:", self)
        self.img_edit = QLineEdit(self)
        img_browse = QPushButton("Browse...", self)
        img_browse.clicked.connect(self.browse_img_dir)
        img_layout.addWidget(img_label)
        img_layout.addWidget(self.img_edit)
        img_layout.addWidget(img_browse)
        layout.addLayout(img_layout)

        # Output directory
        out_layout = QHBoxLayout()
        out_label = QLabel("Output directory:", self)
        self.out_edit = QLineEdit(self)
        out_browse = QPushButton("Browse...", self)
        out_browse.clicked.connect(self.browse_out_dir)
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(out_browse)
        layout.addLayout(out_layout)

        # Beam center [ x , y ] – x and y have separate boxes
        beam_layout = QHBoxLayout()
        beam_label = QLabel("Beam center:", self)
        left_bracket = QLabel("[", self)
        self.beam_x_edit = QLineEdit(self)
        self.beam_x_edit.setPlaceholderText("X")
        comma_label = QLabel(",", self)
        self.beam_y_edit = QLineEdit(self)
        self.beam_y_edit.setPlaceholderText("Y")
        right_bracket = QLabel("]", self)

        beam_layout.addWidget(beam_label)
        beam_layout.addWidget(left_bracket)
        beam_layout.addWidget(self.beam_x_edit)
        beam_layout.addWidget(comma_label)
        beam_layout.addWidget(self.beam_y_edit)
        beam_layout.addWidget(right_bracket)
        layout.addLayout(beam_layout)

        # Pixel size line: Pixel size:  [ value ]  microns
        px_layout = QHBoxLayout()
        px_label = QLabel("Pixel size:", self)
        self.pixel_size_edit = QLineEdit(self)
        self.pixel_size_edit.setText("172.0")  # default in microns
        px_units = QLabel("microns", self)

        px_layout.addWidget(px_label)
        px_layout.addWidget(self.pixel_size_edit)
        px_layout.addWidget(px_units)
        layout.addLayout(px_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        run_btn = QPushButton("Run", self)
        run_btn.clicked.connect(self.run_script)
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch(1)
        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def browse_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "All Files (*);;CSV Files (*.csv)")
        if file_name:
            self.csv_edit.setText(file_name)

    def browse_img_dir(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select Image Directory", "")
        if dir_name:
            self.img_edit.setText(dir_name)

    def browse_out_dir(self):
        dir_name = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "")
        if dir_name:
            self.out_edit.setText(dir_name)

    def _parse_beam_center(self):
        """
        Parse beam center from two line edits (x and y).
        Returns (x, y) as floats, or raises ValueError.
        """
        x_text = self.beam_x_edit.text().strip()
        y_text = self.beam_y_edit.text().strip()
        if not x_text or not y_text:
            raise ValueError("Both x and y must be provided.")
        x = float(x_text)
        y = float(y_text)
        return x, y

    def _parse_pixel_size(self):
        """
        Parse pixel size from the line edit.
        Returns pixel size as float (microns), or raises ValueError.
        """
        text = self.pixel_size_edit.text().strip()
        if not text:
            raise ValueError("Pixel size must be provided.")
        return float(text)

    def run_script(self):
        calib = self.csv_edit.text().strip()
        img_dir = self.img_edit.text().strip()
        out_dir = self.out_edit.text().strip()

        if not csv or not os.path.isfile(csv):
            QMessageBox.warning(self, "Input Error", "Please select a valid CSV file.")
            return
        if not img_dir or not os.path.isdir(img_dir):
            QMessageBox.warning(self, "Input Error", "Please select a valid image directory.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Input Error", "Please select a valid output directory.")
            return

        # Adjust this to point to your standalone script
        script_path = os.path.join(os.path.dirname(__file__), "standalone_script.py")

        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Error", f"Script not found:\n{script_path}")
            return

        try:
            # Example: pass arguments as command line args
            # e.g. python standalone_script.py calib_file img_dir out_dir
            cmd = [
            sys.executable, "standalone_script.py",
            "--calib", calib_path,
            "--images", img_dir,
            "--output", out_dir,
            "--beam-x", str(beam_x),
            "--beam-y", str(beam_y),
            "--pixel-size", str(pixel_size),
            ]
            subprocess.run(cmd, check=True)
            QMessageBox.information(self, "Started", "Calibration script has been started.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run script:\n{e}")

class PilatusIntegrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.calib_path = None
        self.spec_path = None
        self.output_path = None
        self.image_path = None  # To store the selected image path
        self.user = None
        self.db_pixel = None
        self.det_R = None
        self.xyz_map = None
        self.plot_data = {}  # Dictionary to store already integrated data
        self.overlay_plots = False  # Flag to control plot overlaying, default to single plots only
        self.contour_plot = False   # Flag to control contour plot
        self.plot_settings = {  # Default plot settings
            'line_width': 1.0,
            'line_style': 'solid',
            'line_color': QColor(Qt.blue),
            'colormap': 'viridis',  # Default colormap
            'marker': None,
            'min_x': 0.0,
            'max_x': 120.0,
            'automatic_x': True,
            'log_scale': False,
            'sqrt_scale': False
        }
        self.integration_settings = {
            'min_tth': 0.5,
            'max_tth': 180.0,
            'full_tth': True,
            'stepsize': '0.005',
            'error_model': 'poisson',
            'img_clip_low': 20,
            'img_clip_high': 467
        }
        self.init_ui()
        self.worker = None  # Track the active worker thread
        self.processing_scan_range = False
        
        # Add a progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # 0-100%
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setTextVisible(True)  # Show percentage text
        
        # Add it to your layout (e.g., at the bottom)
        self.layout().addWidget(self.progress_bar)  # Adjust based on your layout

        # Live mode controller. Factories resolve at call time so tests can substitute them.
        self.live = live_mode.LiveController(
            worker_factory=lambda *a: Integration_worker.IntegrationWorker(*a),
            write_data=lambda *a: engine.write_data(*a),
            parent=self)
        self.live.scan_integrated.connect(self.handle_live_result)
        self.live.events_found.connect(self.handle_live_events)
        self.live.status.connect(self.update_live_status)
        self.live.error.connect(self.update_live_status)
        self.live_scans = []      # (scan number, plot name) in arrival order
        self.live_events = []     # insitu_seg.Event objects

    def init_ui(self):
        # Layout Setup
        main_layout = QVBoxLayout()  # Changed to QVBoxLayout for toolbar placement
        right_layout = QVBoxLayout()

        # Menu Bar
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")
        settings_menu = menu_bar.addMenu("Settings")
        calibration_menu = menu_bar.addMenu("Calibration")
        help_menu = menu_bar.addMenu("Help")
        
        # Import Integrated Data Action
        import_data_action = QAction("Import Integrated Data", self)
        import_data_action.triggered.connect(self.import_integrated_data)
        file_menu.addAction(import_data_action)
        
        # Clear Data Action
        clear_data_action = QAction("Clear Data", self)
        clear_data_action.triggered.connect(self.clear_data)
        file_menu.addAction(clear_data_action)
        
        # Exit Action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)  # Connect to the close method to exit the application
        file_menu.addAction(exit_action)
        
        # Plot Settings Action
        plot_settings_action = QAction("Plot Settings", self)
        plot_settings_action.triggered.connect(self.open_plot_settings)  # Connect to open_plot_settings
        settings_menu.addAction(plot_settings_action)
        
        # Integration Settings Action
        integration_settings_action = QAction("Integration Settings", self)
        integration_settings_action.triggered.connect(self.open_integration_settings)  # Connect to open_integration_settings
        settings_menu.addAction(integration_settings_action)

        # Run Calibration Action
        run_calib_action = QAction("Run Calibration", self)
        run_calib_action.triggered.connect(self.open_run_calib_settings)  # Connect to open_run_calib_settings
        calibration_menu.addAction(run_calib_action)

        # Edit Calibration Action
        #edit_calib_action = QAction("Edit Calibration", self)
        #edit_calib_action.triggered.connect(self.open_edit_calib_settings)  # Connect to open_edit_calib_settings
        #calibration_menu.addAction(edit_calib_action)
        
        # Open Manual Action
        manual_action = QAction("Open Manual PDF", self)
        manual_action.triggered.connect(self.open_manual)
        help_menu.addAction(manual_action)
        
        # About Settings Action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # ---------------- Input fields: one row each (label | field | Browse) ----------------
        self.calib_path_label = QLabel("Calibration:")
        self.calib_path_input = QLineEdit(self)
        self.calib_path_button = QPushButton("Browse", self)
        self.calib_path_button.clicked.connect(self.browse_calib_file)

        self.spec_path_label = QLabel("SPEC file:")
        self.spec_path_input = QLineEdit(self)
        self.spec_path_button = QPushButton("Browse", self)
        self.spec_path_button.clicked.connect(self.browse_spec_file)

        self.image_path_label = QLabel("Images:")
        self.image_path_input = QLineEdit(self)
        self.image_path_button = QPushButton("Browse", self)
        self.image_path_button.clicked.connect(self.browse_image_directory)

        self.output_path_label = QLabel("Output:")
        self.output_path_input = QLineEdit(self)
        self.output_path_button = QPushButton("Browse", self)
        self.output_path_button.clicked.connect(self.browse_output_directory)

        # User and step size are read-only (set from the SPEC file and Integration Settings):
        # shown together on one compact line.
        self.user_label = QLabel("User:")
        self.user_input = QLineEdit(self)
        self.user_input.setReadOnly(True)
        self.user_input.setFrame(False)
        self.user_input.setToolTip("Read from the SPEC file")
        self.stepsize_label = QLabel("Step:")
        self.stepsize_input = QLineEdit(self)
        self.stepsize_input.setText(self.integration_settings["stepsize"])
        self.stepsize_input.setReadOnly(True)
        self.stepsize_input.setFrame(False)
        self.stepsize_input.setMaximumWidth(70)
        self.stepsize_input.setToolTip("Change in Settings > Integration Settings")
        info_row = QHBoxLayout()
        info_row.addWidget(self.user_input, 1)
        info_row.addWidget(self.stepsize_label)
        info_row.addWidget(self.stepsize_input)

        # Scan selection: one row; single/range pages in a stack so switching modes never
        # changes the panel height (it used to push the data list out of view).
        self.scan_toggle = QCheckBox("Range", self)
        self.scan_toggle.setToolTip("Integrate a range of scans")
        self.scan_toggle.stateChanged.connect(self.toggle_scan_input)
        self.scan_number_label = QLabel("")
        self.scan_number_input = QLineEdit(self)
        self.scan_number_input.setText("1")
        self.scan_range_label = QLabel("")
        self.scan_start_input = QLineEdit(self)
        self.scan_end_input = QLineEdit(self)
        single_page = QWidget()
        single_layout = QHBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(self.scan_number_input)
        self.scan_range_container = QWidget()
        scan_range_layout = QHBoxLayout(self.scan_range_container)
        scan_range_layout.setContentsMargins(0, 0, 0, 0)
        scan_range_layout.addWidget(self.scan_start_input)
        dash_label = QLabel("to")
        dash_label.setAlignment(Qt.AlignCenter)
        scan_range_layout.addWidget(dash_label)
        scan_range_layout.addWidget(self.scan_end_input)
        self.scan_stack = QStackedWidget()
        self.scan_stack.addWidget(single_page)
        self.scan_stack.addWidget(self.scan_range_container)
        scan_row = QHBoxLayout()
        scan_row.addWidget(self.scan_stack, 1)
        scan_row.addWidget(self.scan_toggle)

        form = QGridLayout()
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        for r, (lab, field, btn) in enumerate([
                (self.calib_path_label, self.calib_path_input, self.calib_path_button),
                (self.spec_path_label, self.spec_path_input, self.spec_path_button),
                (self.image_path_label, self.image_path_input, self.image_path_button),
                (self.output_path_label, self.output_path_input, self.output_path_button)]):
            form.addWidget(lab, r, 0)
            form.addWidget(field, r, 1)
            form.addWidget(btn, r, 2)
        form.addWidget(self.user_label, 4, 0)
        form.addLayout(info_row, 4, 1, 1, 2)
        form.addWidget(QLabel("Scan:"), 5, 0)
        form.addLayout(scan_row, 5, 1, 1, 2)
        form.setColumnStretch(1, 1)

        # Integrate button and plot options on one row
        integrate_button = QPushButton("Integrate", self)
        integrate_button.clicked.connect(self.plot_integrated_data)
        self.integrate_button = integrate_button
        self.overlay_toggle = QCheckBox("Overlay", self)
        self.overlay_toggle.stateChanged.connect(self.toggle_overlay)
        self.contour_plot_toggle = QCheckBox("Contour", self)
        self.contour_plot_toggle.stateChanged.connect(self.toggle_contour_plot)
        self.contour_plot_toggle.setEnabled(False)
        action_row = QHBoxLayout()
        action_row.addWidget(integrate_button, 1)
        action_row.addWidget(self.overlay_toggle)
        action_row.addWidget(self.contour_plot_toggle)

        # ---------------- Live Mode (compact) ----------------
        self.live_group = QGroupBox("Live Mode")
        live_layout = QVBoxLayout()
        live_layout.setSpacing(3)
        self.live_toggle = QCheckBox("Live integration", self)
        self.live_toggle.setToolTip("Integrate each new scan automatically as soon as it is complete")
        self.live_toggle.stateChanged.connect(self.toggle_live)
        self.live_start_input = QLineEdit(self)
        self.live_start_input.setPlaceholderText("current")
        self.live_start_input.setMaximumWidth(80)
        self.live_start_input.setToolTip("Leave blank to start with the scan in progress (or the next scan). "
                                         "Enter a number to also integrate earlier scans already collected.")
        live_row1 = QHBoxLayout()
        live_row1.addWidget(self.live_toggle)
        live_row1.addStretch(1)
        live_row1.addWidget(QLabel("Start at:"))
        live_row1.addWidget(self.live_start_input)
        self.live_detect_toggle = QCheckBox("Detect events", self)
        self.live_detect_toggle.setEnabled(HAVE_INSITU_SEG)
        self.live_detect_toggle.setChecked(HAVE_INSITU_SEG)
        self.live_detect_toggle.setToolTip(
            "insitu-seg: flags reactions, phase changes, and peak sharpening/broadening a few scans behind."
            if HAVE_INSITU_SEG else "Install the insitu-seg package to enable.")
        self.live_follow_toggle = QCheckBox("Live waterfall", self)
        self.live_follow_toggle.setChecked(True)
        self.live_follow_toggle.stateChanged.connect(lambda _: self.plot_live_waterfall())
        live_row2 = QHBoxLayout()
        live_row2.addWidget(self.live_detect_toggle)
        live_row2.addWidget(self.live_follow_toggle)
        live_row2.addStretch(1)
        self.live_status_label = QLabel("Live mode off", self)
        self.live_status_label.setWordWrap(True)
        # Detected events live in a tab beside the data list (more room, no cost to controls)
        self.event_list = QListWidget(self)
        self.event_list.setToolTip("Detected transformations: scan, confidence, detector families")
        live_layout.addLayout(live_row1)
        live_layout.addLayout(live_row2)
        live_layout.addWidget(self.live_status_label)
        self.live_group.setLayout(live_layout)

        # ---------------- Integrated data list ----------------
        self.plot_list = QListWidget(self)
        self.plot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.plot_list.itemClicked.connect(self.toggle_highlight)  # Connect itemClicked signal to toggle_highlight
        self.plot_list_label = QLabel("Integrated Data:")

        # Controls scroll if the window is short; a draggable divider separates them from the
        # data list, which always keeps usable space.
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 4, 0)
        controls_layout.addLayout(form)
        controls_layout.addLayout(action_row)
        controls_layout.addWidget(self.live_group)
        controls_layout.addStretch(1)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(controls)

        self.data_tabs = QTabWidget()
        self.data_tabs.addTab(self.plot_list, "Integrated Data")
        self.data_tabs.addTab(self.event_list, "Events")
        list_panel = QWidget()
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(self.data_tabs)
        self.plot_list.setMinimumHeight(80)

        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.addWidget(self.controls_scroll)
        self.left_splitter.addWidget(list_panel)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.setStretchFactor(0, 0)
        self.left_splitter.setStretchFactor(1, 1)
        self.left_splitter.setSizes([controls.sizeHint().height(), 300])
        
        # Status Bar
        self.status_bar = QStatusBar()
        
        # Matplotlib Plot
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Set Size Policy for expanding
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add menu bar, central layout, and status bar to the main layout
        main_layout.addWidget(menu_bar)

        # Add Navigation Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(QLabel("Integration Plot:"))
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.toolbar) # Add the toolbar to the right layout
        right_panel = QWidget()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_panel.setLayout(right_layout)

        # Left panel | plot, with a draggable divider so the left panel can be widened
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([340, 660])
        # Expanding + stretch 1: the splitter takes all extra space when the window grows.
        # (Without this, a QSplitter is 'Preferred' vertically and maximizing left the
        # contents at their old height.)
        self.main_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_splitter.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(self.main_splitter, 1)

        # Add central layout and status bar to the main layout
        main_layout.addWidget(self.status_bar)
        

        self.setLayout(main_layout)
        self.setWindowTitle("Pilatus Integration GUI")
        self.setWindowIcon(QIcon(resource_path("icon.png"))) # Sets the window icon
        # Default 1200x800, capped to the available screen
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        w0 = min(1200, int(avail.width() * 0.95)) if avail else 1200
        h0 = min(800, int(avail.height() * 0.9)) if avail else 800
        self.setGeometry(100, 60, w0, h0)
        
        self.status_bar.showMessage("Ready", 3000)  # Initial message

    def browse_calib_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", "",
                                                  "Text Files (*.cal);;All Files (*)", options=options)
        if file_path:
            self.calib_path_input.setText(file_path)
            self.calib_path = file_path
            self.read_calibration_parameters(file_path)

    def browse_spec_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Spec File", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            self.spec_path_input.setText(file_path)
            self.spec_path = file_path
            self.read_user_from_spec(file_path)
            file_path_only, file_name_only = os.path.split(file_path)
            if self.output_path == None:
                self.output_path = file_path_only + "/"
                self.output_path_input.setText(file_path_only)  # Return empty string if user not found

    def browse_image_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory", "", options=options)
        if dir_path:
            self.image_path_input.setText(dir_path)
            self.image_path = dir_path
            
    def browse_output_directory(self):
        options = QFileDialog.Options()
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", "", options=options)
        if dir_path:
            self.output_path_input.setText(dir_path)
            self.output_path = dir_path + "/"

    def read_user_from_spec(self, spec_file):
        """Read user from the provided spec file."""
        try:
            with open(spec_file, 'r') as f:
                for line in f:
                    if "User =" in line:  # assume user information is on a line starting with #USER
                        user = line.split()[-1]  # Capture the user after the #USER tag
                        self.user_input.setText(user)
                        self.user = user
                        return
            self.user_input.setText("")  # Return empty string if user not found
            self.user = None
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error reading user from spec file: {e}")
            self.user_input.setText("")
            self.user = None
            
    def import_integrated_data(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Import Integrated Data", "",
                                                  "XYE Files (*.xye);;Text Files (*.txt);;All Files (*)", options=options)
        if file_paths:
            for file_path in file_paths:
                try:
                    x, y, e = self.read_integrated_data(file_path)
                    file_path_only, plot_name = os.path.split(file_path)
                    self.plot_data[plot_name] = {'x': x, 'y': y, 'e': e}
                    self.plot_list.addItem(plot_name)  # Add item to list
                    self.status_bar.showMessage(f"Imported data from {file_path}", 5000)
                except Exception as e:
                    QMessageBox.warning(self, "Import Error", f"Error importing {file_path}: {e}")
                    self.status_bar.showMessage(f"Error importing {file_path}: {e}", 5000)

    def read_integrated_data(self, file_path):
        """Read x and y data from the given file."""
        x = []
        y = []
        e = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            xi, yi, ei = map(float, line.split())  # Unpack only x, y and e, ignore other columns if present
                            x.append(xi)
                            y.append(yi)
                            e.append(ei)
                        except ValueError:
                            continue  # Skip lines that don't have two values
            return np.array(x), np.array(y), np.array(e)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading  {e}")
            
    def read_calibration_parameters(self, calib_file):
        """Read parameters from the provided calibration file."""
        try:
            f = open(calib_file)
            line = f.readline()
            db_x = int(line.split()[-1])
            line = f.readline()
            db_y = int(line.split()[-1])
            line = f.readline()
            self.det_R = float(line.split()[-1])
            f.close()
            self.db_pixel = [db_x, db_y]
            self.xyz_map = engine.make_map(self.db_pixel, self.det_R)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error reading parameters from calibration file: {e}")
    
    def _inputs_ok(self):
        """Check the inputs needed for integration (manual or live); warn and return False if not."""
        checks = [
            (self.spec_path and os.path.isfile(self.spec_path), "Please select a valid SPEC file."),
            (self.image_path and os.path.isdir(self.image_path), "Please select a valid image directory."),
            (self.output_path and os.path.isdir(self.output_path), "Please select a valid output directory."),
            (self.xyz_map is not None, "Please select a valid calibration file."),
            (bool(self.user), "No user name was found in the SPEC file."),
        ]
        for ok, msg in checks:
            if not ok:
                QMessageBox.warning(self, "Input Error", msg)
                return False
        return True

    def plot_integrated_data(self):
        """Called when the Integrate button is clicked."""
        if self.live.active:
            QMessageBox.warning(self, "Live Mode", "Turn off live mode to integrate by hand.")
            return
        if not self._inputs_ok():
            return
        # Warn only in response to another button click.
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Integration Running",
                "An integration is already in progress."
            )
            return
        try:
            if self.scan_toggle.isChecked():
                start = int(self.scan_start_input.text())
                end = int(self.scan_end_input.text())

                if start > end:
                    raise ValueError(
                        "The starting scan must not exceed the ending scan."
                    )

                self.process_scans_sequentially(start, end)
            else:
                scan_num = int(self.scan_number_input.text())
                self.start_integration_thread(scan_num)

        except ValueError as exc:
            QMessageBox.warning(
                self, "Input Error", f"Invalid scan number: {exc}"
            )
        # (2026-10: a duplicate copy of the block above was removed; it started every
        # integration twice and replaced self.worker while the first thread was running.)

    def process_scans_sequentially(self, start, end):
        """Process scans one-by-one in the background."""
        self.processing_scan_range = True
        self.current_scan = start
        self.end_scan = end
        self.start_integration_thread(self.current_scan)
        
    def start_integration_thread(self, scan_num):
        """Start a worker thread for integration."""
        
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = Integration_worker.IntegrationWorker(
            spec_path=self.spec_path,
            scan_num=scan_num,
            image_path=self.image_path,
            user=self.user,
            xyz_map=self.xyz_map,
            settings=self.integration_settings,
            use_variance=(
                self.integration_settings["error_model"] == "azimuthal"
            )
        )

        self.worker.progress_updated.connect(self.update_status_bar)
        self.worker.progress_percent.connect(self.update_progress_bar)
        self.worker.result_ready.connect(self.handle_integration_result)
        self.worker.error_occurred.connect(self.show_error)
        # Advance a scan range when this thread ends (2026-10: was never connected, so a
        # range stopped after its first scan).
        self.worker.finished.connect(self.integration_thread_finished)

        self.worker.start()
        
    def update_status_bar(self, message):
        """Update the GUI status bar (thread-safe)."""
        self.status_bar.showMessage(message)

    def update_progress_bar(self, value):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(value)

        if value >= 100:
           self.progress_bar.setVisible(False)
        
    def handle_integration_result(self, scan_name, x, y, e):
        """Process results when integration finishes."""
        # Save data to file
        engine.write_data(self.output_path, scan_name, x, y, e)
        
        # Update plot data
        self.plot_data[scan_name] = {'x': x, 'y': y, 'e': e}
        
        # Add to plot list
        item = QListWidgetItem(scan_name)
        self.plot_list.addItem(item)
        item.setSelected(True)
        
        # Plot the data
        self.replot_selected()

    def integration_thread_finished(self):
        """
        Called after the current worker thread has completely stopped.
        Starts the next scan if a scan range is being processed.
        """
        # Let the thread fully exit before releasing it: destroying a QThread that is still
        # running aborts the whole process.
        finished = self.worker
        if finished is not None:
            finished.wait()
            finished.deleteLater()
        self.worker = None

        if (
            getattr(self, "processing_scan_range", False)
            and self.current_scan < self.end_scan
        ):
            self.current_scan += 1
            self.start_integration_thread(self.current_scan)
        else:
            self.processing_scan_range = False
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Integration complete", 5000)   

    def show_error(self, error_msg):
        """Show error messages in a dialog."""
        self.processing_scan_range = False
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)

    def showEvent(self, event):
        """On first show, guarantee the data list a share of the left panel: controls get at
        most ~60% of the height (they scroll if they need more). Later divider drags are kept."""
        super().showEvent(event)
        if not getattr(self, "_left_split_done", False):
            self._left_split_done = True
            self._apply_left_split()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Until the user drags the divider, keep the guaranteed split as the window resizes.
        if getattr(self, "_left_split_done", False) and not getattr(self, "_user_moved_split", False):
            self._apply_left_split()

    def _apply_left_split(self):
        total = sum(self.left_splitter.sizes()) or self.left_splitter.height()
        want = self.controls_scroll.widget().sizeHint().height()
        controls_h = int(min(want, 0.6 * total))
        self.left_splitter.blockSignals(True)
        self.left_splitter.setSizes([controls_h, max(total - controls_h, 1)])
        self.left_splitter.blockSignals(False)
        if not getattr(self, "_split_signal_connected", False):
            self.left_splitter.splitterMoved.connect(lambda *_: setattr(self, "_user_moved_split", True))
            self._split_signal_connected = True

    def closeEvent(self, event):
        if self.live.active:
            self.live.active = False
            self.live.timer.stop()
        if self.live.worker is not None:
            self.live.worker.wait()           # let a live integration finish (~0.2 s)
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()

    # ------------------------------------------------------------------ live mode
    def toggle_live(self, state):
        if state == Qt.Checked:
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.warning(self, "Live Mode", "Wait for the current integration to finish.")
                self.live_toggle.setChecked(False)
                return
            if not self._inputs_ok():
                self.live_toggle.setChecked(False)
                return
            text = self.live_start_input.text().strip()
            try:
                start = int(text) if text else None
            except ValueError:
                QMessageBox.warning(self, "Live Mode", f"Start scan must be a number: {text}")
                self.live_toggle.setChecked(False)
                return
            self.live_scans, self.live_events = [], []
            self.event_list.clear()
            self.data_tabs.setTabText(1, "Events")
            self.integrate_button.setEnabled(False)
            self.live_start_input.setEnabled(False)
            self.live_detect_toggle.setEnabled(False)
            self.live.start(self.spec_path, self.image_path, self.user, self.xyz_map,
                            self.integration_settings, self.output_path, start_scan=start,
                            detect=self.live_detect_toggle.isChecked())
        else:
            if self.live.active:
                self.live.stop()
            self.integrate_button.setEnabled(True)
            self.live_start_input.setEnabled(True)
            self.live_detect_toggle.setEnabled(HAVE_INSITU_SEG)

    def update_live_status(self, message):
        self.live_status_label.setText(message)
        self.status_bar.showMessage(message, 5000)

    def handle_live_result(self, scan_name, scan, x, y, e):
        """A live scan was integrated and written: store, list, and redraw the waterfall."""
        self.plot_data[scan_name] = {'x': np.asarray(x), 'y': np.asarray(y), 'e': np.asarray(e)}
        if not self.plot_list.findItems(scan_name, Qt.MatchExactly):
            self.plot_list.addItem(QListWidgetItem(scan_name))
        self.live_scans.append((scan, scan_name))
        self.plot_live_waterfall()

    def handle_live_events(self, events):
        for ev in events:
            self.live_events.append(ev)
            t = f", {ev.T_C:.0f} °C" if ev.T_C is not None else ""
            d = f", {ev.profile_direction}" if ev.profile_direction else ""
            self.event_list.addItem(f"scan {ev.scan} [{ev.confidence}] {'+'.join(ev.families)}{d}{t}")
        self.data_tabs.setTabText(1, f"Events ({len(self.live_events)})")
        self.plot_live_waterfall()

    def plot_live_waterfall(self):
        """Intensity vs 2theta and scan for all live scans, with detected events marked."""
        if not self.live_follow_toggle.isChecked() or not self.live_scans:
            return
        if hasattr(self, 'colorbar') and self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
        self.ax.clear()
        scans = [s for s, _ in self.live_scans]
        names = [n for _, n in self.live_scans]
        x0 = self.plot_data[names[0]]['x']
        if len(names) == 1:
            self.ax.plot(x0, self.plot_data[names[0]]['y'], linewidth=self.plot_settings['line_width'])
            self.ax.set_xlabel("2-theta")
            self.ax.set_ylabel("Integrated Intensity")
            self.ax.set_title(f"Live: scan {scans[0]}", loc="left")
            self.canvas.draw()
            return
        img = np.array([np.interp(x0, self.plot_data[n]['x'], self.plot_data[n]['y']) for n in names])
        if self.plot_settings['log_scale']:
            img, label = np.log(np.clip(img, 1e-12, None)), "log(Intensity)"
        elif self.plot_settings['sqrt_scale']:
            img, label = np.sqrt(np.clip(img, 0, None)), "SQRT(Intensity)"
        else:
            label = "Intensity"
        # 'antialiased': ~9000 2theta points shown in a few hundred pixels; 'nearest' aliases
        # narrow peaks in and out of view (seen as streaks in the first live rehearsal).
        im = self.ax.imshow(img, aspect="auto", origin="lower", interpolation="antialiased",
                            cmap=self.plot_settings['colormap'],
                            extent=[x0[0], x0[-1], scans[0] - 0.5, scans[-1] + 0.5],
                            vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.7))
        self.colorbar = self.fig.colorbar(im, ax=self.ax, label=label)
        for ev in self.live_events:
            self.ax.axhline(ev.scan, color="white", lw=1.0 if ev.confidence == "high" else 0.6,
                            ls="-" if ev.confidence == "high" else ":")
        if not self.plot_settings.get('automatic_x', True):
            self.ax.set_xlim(self.plot_settings['min_x'], self.plot_settings['max_x'])
        self.ax.set_xlabel("2-theta")
        self.ax.set_ylabel("Scan Number")
        self.ax.set_title(f"Live: scans {scans[0]}-{scans[-1]}"
                          + (f", {len(self.live_events)} events" if self.live_events else ""), loc="left")
        self.canvas.draw()

    def replot_selected(self):
        """Replots selected items, handling both single, overlay, and contour plot modes."""
        selected_items = self.plot_list.selectedItems()
        num_selected = len(selected_items)
    
        # Check if a colorbar exists and remove it
        if hasattr(self, 'colorbar') and self.colorbar:
            self.colorbar.remove()
            self.colorbar = None
    
        self.ax.clear()  # Clear the plot before replotting
    
        if self.contour_plot and num_selected > 4:
            # Contour plot mode:
            self.status_bar.showMessage("Generating Contour Plot...", 3000)
            tth_values = []
            intensity_values = []
    
            # Collect x and y data from all selected plots
            for item in selected_items:
                plot_name = item.text()
                if plot_name in self.plot_data:
                    data = self.plot_data[plot_name]
                    x, y, e = data['x'], data['y'], data['e']
                    tth_values.append(x)
                    # Apply y-axis scaling
                    if self.plot_settings['sqrt_scale']:
                        intensity_values.append(np.sqrt(y))
                    elif self.plot_settings['log_scale']:
                        intensity_values.append(np.log(y))
                    else:
                        intensity_values.append(y)
    
            # Create a grid of 2theta and scan number values
            tth = np.unique(np.concatenate(tth_values))
            scans = np.arange(1, num_selected + 1) # use scan number as a proxy for scan name
            tth_grid, scan_grid = np.meshgrid(tth, scans)
    
            # Interpolate the intensity values onto the grid
            intensity_grid = np.zeros_like(tth_grid)
            for i, (tth_data, intensity_data) in enumerate(zip(tth_values, intensity_values)):
                interp_func = interpolate.interp1d(tth_data, intensity_data, kind='linear', fill_value="extrapolate")
                intensity_grid[i, :] = interp_func(tth)
    
            # Create the contour plot
            contour = self.ax.contourf(tth_grid, scan_grid, intensity_grid, cmap=self.plot_settings['colormap'], levels=20) # change back to viridis when fixed
            self.colorbar = self.fig.colorbar(contour, ax=self.ax, label="Intensity") # save colorbar object
            self.ax.set_xlim(self.plot_settings['min_x'], self.plot_settings['max_x'])
            #self.fig.colorbar(contour, ax=self.ax, label="Intensity")
    
            self.ax.set_xlabel("2-theta")
            self.ax.set_ylabel("Scan Number")
            #self.ax.set_title("Contour Plot of Integrated Intensity")
    
        elif self.overlay_plots:
            # Overlay mode: plot all selected items
            self.status_bar.showMessage("Generating Overlay Plot...", 3000)
            if selected_items:
                for item in selected_items:
                    plot_name = item.text()
                    if plot_name in self.plot_data:
                        data = self.plot_data[plot_name]
                        x, y, e = data['x'], data['y'], data['e']
    
                        # Apply y-axis scaling
                        if self.plot_settings['sqrt_scale']:
                            y = np.sqrt(y)
                        elif self.plot_settings['log_scale']:
                            y = np.log(y)
    
                        self.ax.plot(x, y, linewidth=self.plot_settings['line_width'],
                                    linestyle=self.plot_settings['line_style'],
                                    marker=self.plot_settings['marker'],
                                    label=plot_name)  # Add label for each plot
    
                self.ax.set_xlim(self.plot_settings['min_x'], self.plot_settings['max_x'])
                self.ax.set_xlabel("2-theta")
    
                if self.plot_settings['sqrt_scale']:
                    self.ax.set_ylabel("SQRT(Integrated Intensity)")
                elif self.plot_settings['log_scale']:
                    self.ax.set_ylabel("log(Integrated Intensity)")
                else:
                    self.ax.set_ylabel("Integrated Intensity")
    
                self.ax.legend()  # Show legend to distinguish plots
    
        else:
            # Single plot mode: plot only the first selected item
            self.status_bar.showMessage("Generating Single Plot...", 3000)
            if selected_items:
                item = selected_items[0]  # Get the first selected item
                plot_name = item.text()
                if plot_name in self.plot_data:
                    data = self.plot_data[plot_name]
                    x, y, e = data['x'], data['y'], data['e']
    
                    # Apply y-axis scaling
                    if self.plot_settings['sqrt_scale']:
                        y = np.sqrt(y)
                    elif self.plot_settings['log_scale']:
                        y = np.log(y)
    
                    self.ax.plot(x, y, linewidth=self.plot_settings['line_width'],
                                linestyle=self.plot_settings['line_style'],
                                marker=self.plot_settings['marker'])
    
                    self.ax.set_xlim(self.plot_settings['min_x'], self.plot_settings['max_x'])
                    self.ax.set_xlabel("2-theta")
    
                    if self.plot_settings['sqrt_scale']:
                        self.ax.set_ylabel("SQRT(Integrated Intensity)")
                    elif self.plot_settings['log_scale']:
                        self.ax.set_ylabel("log(Integrated Intensity)")
                    else:
                        self.ax.set_ylabel("Integrated Intensity")
    
        self.canvas.draw()
        
    def toggle_highlight(self, item):
        """Toggle highlight state of the clicked item."""
        if self.overlay_plots:
            item.setSelected(item.isSelected())  # Toggle selection
            self.replot_selected()  # Replot to show changes
        else:
            for index in range(self.plot_list.count()):
                it = self.plot_list.item(index)
                if it != item:
                    it.setSelected(False)  # unselect it
                else:
                    it.setSelected(True)  # select this item
            self.replot_selected()
                
    def toggle_overlay(self, state):
        """Toggle the overlay plots flag."""
        self.overlay_plots = (state == Qt.Checked)
        self.contour_plot_toggle.setEnabled(self.overlay_plots)
        self.contour_plot_toggle.setCheckState(False) # Uncheck contour plot when overlay plot is unchecked
        
    def toggle_contour_plot(self, state):
        """Toggle the contour plots flag."""
        self.contour_plot = (state == Qt.Checked)
        self.status_bar.showMessage(f"Contour plot {'enabled' if self.contour_plot else 'disabled'}", 5000)

    def toggle_scan_input(self, state):
        """Switch between single-scan and range inputs. Both live in one stacked widget, so
        the panel height does not change (it used to push the data list out of view)."""
        use_scan_range = (state == Qt.Checked)
        self.scan_stack.setCurrentIndex(1 if use_scan_range else 0)
        self.status_bar.showMessage("Switched scan input mode", 3000)
        
    def open_plot_settings(self):
        """Open the Plot Settings dialog."""
        dialog = PlotSettingsDialog(self.plot_settings)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.plot_settings = dialog.get_settings()
            self.status_bar.showMessage("Plot settings applied", 3000)
            self.replot_selected()  # Replot with new settings
            
    def open_integration_settings(self):
        """Open the Integration Settings dialog."""
        dialog = IntegSettingsDialog(self.integration_settings)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.integration_settings = dialog.get_settings()
            self.stepsize_input.setText(self.integration_settings['stepsize'])
            self.status_bar.showMessage("Integration settings applied", 3000)

    def open_run_calib_settings(self):
        """Calibration from the GUI is not available yet (RunCalibDialog is work in progress:
        it references a script and variables that do not exist, and on success would have
        overwritten the plot settings). Until it is built in, point the user to the CLI."""
        QMessageBox.information(
            self, "Run Calibration",
            "Calibration from the GUI is not available yet.\n\n"
            "Run the command-line calibration script, then load the .cal file it writes "
            "with the Calibration File 'Browse' button.")
            
    def show_about_dialog(self):
        """Show the about dialog with program description and icon."""
        dialog = AboutDialog(self)
        dialog.exec_()
        
    def open_manual(self):
        """Open the PDF manual using the default PDF viewer."""
        pdf_path = "manual.pdf"  # Path to your PDF file
        # Ensure that the path is correct; adjust the path if necessary.
        if QDesktopServices.openUrl(QUrl.fromLocalFile(pdf_path)):
            self.status_bar.showMessage(f"Opened manual: {pdf_path}", 5000)
        else:
            QMessageBox.warning(self, "Error", f"Could not open manual: {pdf_path}")
            
    def clear_data(self):
        """Clears all data and resets the GUI, with a confirmation dialog."""
        reply = QMessageBox.question(self, 'Clear Data',
                                    "Are you sure you want to erase all of the data and start a new session?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        if reply == QMessageBox.Yes:
            # Stop live mode first, then reset its lists
            if self.live.active:
                self.live_toggle.setChecked(False)
            self.live_scans, self.live_events = [], []
            self.event_list.clear()
            self.data_tabs.setTabText(1, "Events")
            self.live_status_label.setText("Live mode off")

            # Clear the plot
            self.ax.clear()
            self.canvas.draw()

            # Reset input fields
            self.calib_path_input.clear()
            self.spec_path_input.clear()
            self.user_input.clear()
            self.stepsize_input.setText("0.005")
            self.image_path_input.clear()
            self.scan_number_input.setText("1")
            self.scan_start_input.clear()
            self.scan_end_input.clear()
            self.scan_toggle.setChecked(False)
    
            # Back to single-scan input (setChecked(False) above also does this via the signal)
            self.scan_stack.setCurrentIndex(0)
    
            # Clear plot data
            self.plot_list.clear() # clear items from plot list
            self.plot_data = {}    # clear stored plot data
    
            # Status bar message
            self.status_bar.showMessage("Data cleared, ready for a fresh start!", 5000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion')) # Set Fusion style
    ex = PilatusIntegrationGUI()
    ex.show()
    sys.exit(app.exec_())