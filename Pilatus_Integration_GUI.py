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
                             QListWidgetItem, QSlider, QStyleFactory, QProgressBar)
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QThread, pyqtSignal
import Integration_engine as engine
import Integration_worker

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
        
        # Add a progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # 0-100%
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setTextVisible(True)  # Show percentage text
        
        # Add it to your layout (e.g., at the bottom)
        self.layout().addWidget(self.progress_bar)  # Adjust based on your layout

    def init_ui(self):
        # Layout Setup
        main_layout = QVBoxLayout()  # Changed to QVBoxLayout for toolbar placement
        input_layout = QHBoxLayout() # Contains left and right layouts
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # Menu Bar
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("File")
        settings_menu = menu_bar.addMenu("Settings")
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
        
        # Open Manual Action
        manual_action = QAction("Open Manual PDF", self)
        manual_action.triggered.connect(self.open_manual)
        help_menu.addAction(manual_action)
        
        # About Settings Action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        # Input Fields
        self.calib_path_label = QLabel("Calibration File:")
        self.calib_path_input = QLineEdit(self)
        self.calib_path_button = QPushButton("Browse", self)
        self.calib_path_button.clicked.connect(self.browse_calib_file)
        calib_layout = QHBoxLayout()
        calib_layout.addWidget(self.calib_path_input)
        calib_layout.addWidget(self.calib_path_button)

        self.spec_path_label = QLabel("Spec File:")
        self.spec_path_input = QLineEdit(self)
        self.spec_path_button = QPushButton("Browse", self)
        self.spec_path_button.clicked.connect(self.browse_spec_file)
        spec_layout = QHBoxLayout()
        spec_layout.addWidget(self.spec_path_input)
        spec_layout.addWidget(self.spec_path_button)

        self.user_label = QLabel("User:")
        self.user_input = QLineEdit(self)
        self.user_input.setReadOnly(True)

        self.stepsize_label = QLabel("Step Size:")
        self.stepsize_input = QLineEdit(self)
        self.stepsize_input.setText(self.integration_settings["stepsize"])
        self.stepsize_input.setReadOnly(True)

        # Image Path Section
        self.image_path_label = QLabel("Image Path:")
        self.image_path_input = QLineEdit(self)
        self.image_path_button = QPushButton("Browse", self)
        self.image_path_button.clicked.connect(self.browse_image_directory)
        image_path_layout = QHBoxLayout()
        image_path_layout.addWidget(self.image_path_input)
        image_path_layout.addWidget(self.image_path_button)
        
        # Output Path Section
        self.output_path_label = QLabel("Output Path:")
        self.output_path_input = QLineEdit(self)
        self.output_path_button = QPushButton("Browse", self)
        self.output_path_button.clicked.connect(self.browse_output_directory)
        output_path_layout = QHBoxLayout()
        output_path_layout.addWidget(self.output_path_input)
        output_path_layout.addWidget(self.output_path_button)

        # Scan Number Input
        self.scan_toggle = QCheckBox("Use Scan Range", self)
        self.scan_toggle.stateChanged.connect(self.toggle_scan_input)

        self.scan_number_label = QLabel("Scan Number:")
        self.scan_number_input = QLineEdit(self)
        self.scan_number_input.setText("1")

        self.scan_range_label = QLabel("Scan Range:")
        self.scan_start_input = QLineEdit(self)
        self.scan_end_input = QLineEdit(self)

        # Scan Range Layout
        scan_range_layout = QHBoxLayout()
        scan_range_layout.addWidget(self.scan_start_input)
        dash_label = QLabel(" - ")
        dash_label.setAlignment(Qt.AlignCenter)
        scan_range_layout.addWidget(dash_label)
        scan_range_layout.addWidget(self.scan_end_input)

        # Create a container widget for the scan range layout
        self.scan_range_container = QWidget()
        self.scan_range_container.setLayout(scan_range_layout)
        
        # Plot Options
        self.overlay_toggle = QCheckBox("Overlay Plots", self)
        self.overlay_toggle.stateChanged.connect(self.toggle_overlay)

        self.contour_plot_toggle = QCheckBox("Contour Plot", self)
        self.contour_plot_toggle.stateChanged.connect(self.toggle_contour_plot)
        self.contour_plot_toggle.setEnabled(False)
	
        # Initial Visibility
        self.scan_number_label.setVisible(True)
        self.scan_number_input.setVisible(True)
        self.scan_range_label.setVisible(False)
        self.scan_range_container.setVisible(False)  # Hide the container instead

        # Integrate Button
        integrate_button = QPushButton("Integrate", self)
        integrate_button.clicked.connect(self.plot_integrated_data)
        
        # Plot List Widget
        self.plot_list = QListWidget(self)
        self.plot_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.plot_list.itemClicked.connect(self.toggle_highlight)  # Connect itemClicked signal to toggle_highlight
        self.plot_list_label = QLabel("Integrated Data:")

        # Add input fields to the left layout
        left_layout.addWidget(self.calib_path_label)
        left_layout.addLayout(calib_layout)
        left_layout.addWidget(self.spec_path_label)
        left_layout.addLayout(spec_layout)
        left_layout.addWidget(self.user_label)
        left_layout.addWidget(self.user_input)
        left_layout.addWidget(self.stepsize_label)
        left_layout.addWidget(self.stepsize_input)
        left_layout.addWidget(self.image_path_label)
        left_layout.addLayout(image_path_layout)  # Adding image path layout
        left_layout.addWidget(self.output_path_label)
        left_layout.addLayout(output_path_layout)  # Adding image path layout
        left_layout.addWidget(self.scan_toggle)
        left_layout.addWidget(self.scan_number_label)
        left_layout.addWidget(self.scan_number_input)
        left_layout.addWidget(self.scan_range_label)
        left_layout.addWidget(self.scan_range_container)  # Add the container instead
        left_layout.addWidget(integrate_button)
        left_layout.addWidget(self.overlay_toggle)  # Add the toggle to the layout
        left_layout.addWidget(self.contour_plot_toggle)
        left_layout.addWidget(self.plot_list_label)
        left_layout.addWidget(self.plot_list)
        
        # Add left and right to input layout
        input_layout.addLayout(left_layout, 1)
        input_layout.addLayout(right_layout, 3)
        
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

        # Add widgets to layout
        main_layout.addLayout(input_layout)
        right_layout.addWidget(self.toolbar) # Add the toolbar to the right layout
        
        # Add central layout and status bar to the main layout
        main_layout.addWidget(self.status_bar)
        

        self.setLayout(main_layout)
        self.setWindowTitle("Pilatus Integration GUI")
        self.setWindowIcon(QIcon(resource_path("icon.png"))) # Sets the window icon
        self.setGeometry(100, 100, 1000, 600)
        
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
    
    def plot_integrated_data(self):
        """Called when the 'Integrate' button is clicked."""
        try:
            if self.scan_toggle.isChecked():
                # Multi-scan mode (process one after another)
                start = int(self.scan_start_input.text())
                end = int(self.scan_end_input.text())
                self.process_scans_sequentially(start, end)
            else:
                # Single-scan mode
                scan_num = int(self.scan_number_input.text())
                self.start_integration_thread(scan_num)
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid scan number: {e}")
            
    def process_scans_sequentially(self, start, end):
        """Process scans one-by-one in the background."""
        self.current_scan = start
        self.end_scan = end
        self.start_integration_thread(self.current_scan)
        
    def start_integration_thread(self, scan_num):
        """Start a worker thread for integration."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()  # Stop any existing thread

        # Create a new worker
        self.worker = Integration_worker.IntegrationWorker(
            spec_path=self.spec_path,
            scan_num=scan_num,
            image_path=self.image_path,
            user=self.user,
            xyz_map=self.xyz_map,
            settings=self.integration_settings,
            use_variance=(self.integration_settings["error_model"] == "azimuthal")
        )

        # Connect signals
        self.worker.progress_updated.connect(self.update_status_bar)
        self.worker.result_ready.connect(self.handle_integration_result)
        self.worker.error_occurred.connect(self.show_error)

        # Start the thread
        self.worker.start()
        
    def update_status_bar(self, message):
        """Update the GUI status bar (thread-safe)."""
        self.status_bar.showMessage(message)
        
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
        
        # Process next scan in multi-scan mode
        if hasattr(self, 'current_scan') and self.current_scan < self.end_scan:
            self.current_scan += 1
            self.start_integration_thread(self.current_scan)

    def show_error(self, error_msg):
        """Show error messages in a dialog (thread-safe)."""
        QMessageBox.critical(self, "Error", error_msg)
        
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()

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
        """Toggle visibility of scan input fields based on checkbox state."""
        use_scan_range = (state == Qt.Checked)
        self.scan_number_label.setVisible(not use_scan_range)
        self.scan_number_input.setVisible(not use_scan_range)
        self.scan_range_label.setVisible(use_scan_range)
        self.scan_range_container.setVisible(use_scan_range)  # Show/hide the container
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
    
            # Reset visibility of scan input fields
            self.scan_number_label.setVisible(True)
            self.scan_number_input.setVisible(True)
            self.scan_range_label.setVisible(False)
            self.scan_range_container.setVisible(False)
    
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