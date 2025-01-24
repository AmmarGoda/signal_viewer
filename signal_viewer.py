import sys
import pandas as pd
import pyedflib
import numpy as np
import os
import wfdb
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMessageBox, QHBoxLayout, QComboBox, QScrollBar,
    QPushButton, QFileDialog, QLineEdit, QLabel, QMainWindow, QSlider, QColorDialog, QFrame,  QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import pyqtSignal
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyQt5.QtGui import QPixmap, QPainter
import psutil
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QVBoxLayout, QPushButton
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPen, QBrush
import numpy as np
import requests
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

class SignalGraph(QWidget):
    # signal_loaded = pyqtSignal(int)  # Signal to emit when a signal is loaded

    def __init__(self, graph_identifier, parent=None):
        super(SignalGraph, self).__init__(parent)
        self.graph_identifier = graph_identifier
        self.parent = parent

        # Initialize layout
        self.main_layout = QVBoxLayout(self)
        self.initialize_controls()

        # Initialize selected_color with a default color
        self.selected_color = '#1f77b4'  # Default color (blue)

        # Initialize is_playing
        self.is_playing = False

        # Initialize playback speed to match the slider's default value
        self.playback_speed = 50  # Default to 50 ms per frame

        # Initialize signal-related variables
        self.raw_signal_data = {}
        self.filtered_signal_data = {}
        self.time_axis = None
        self.sampling_frequency = None
        self.current_channel = None
        self.playback_position = 0
        self.display_window_duration = 0.1  # seconds
        self.draw_indicator_duration = 0.1  # seconds

        # Flag to indicate if cine mode is active
        self.cine_mode_active = False

        # Initialize plot after setting display_window_duration
        self.initialize_plot()

        # Initialize scrollbar for signal navigation
        self.navigation_scrollbar = QScrollBar(Qt.Horizontal)
        self.navigation_scrollbar.setMinimum(0)
        self.navigation_scrollbar.valueChanged.connect(self.on_scrollbar_moved)
        self.main_layout.addWidget(self.navigation_scrollbar)

        # Initialize playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback)

    def update_playback_speed(self, value):
        self.playback_speed = value
        self.playback_timer.setInterval(value)
        print(f"Playback speed updated to: {self.playback_speed} ms per frame")

    def initialize_controls(self):
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        # Create a frame to group controls
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("QFrame { background-color: #1E1E1E; border: none; }")
        controls_frame.setLayout(controls_layout)

        # Load Signal Button
        self.load_signal_button = QPushButton('Load Signal', self)
        self.load_signal_button.setToolTip("Load a signal file (CSV, EDF, WFDB)")
        self.load_signal_button.clicked.connect(self.load_signal_file)
        self.load_signal_button.setStyleSheet(self.button_style())
        # self.load_signal_button.setIcon(QIcon('icons/load.png'))  # Replace with your icon path
        controls_layout.addWidget(self.load_signal_button)

        # Channel Selection Dropdown
        channel_layout = QHBoxLayout()
        channel_label = QLabel('Channel:', self)
        channel_label.setStyleSheet("QLabel { color: #FFFFFF; }")
        self.channel_selection_dropdown = QComboBox(self)
        self.channel_selection_dropdown.setToolTip("Select Signal Channel")
        self.channel_selection_dropdown.currentIndexChanged.connect(self.on_channel_changed)
        self.channel_selection_dropdown.setStyleSheet(self.combo_box_style())
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_selection_dropdown)
        controls_layout.addLayout(channel_layout)

        # Color Change Button
        self.change_color_button = QPushButton('Change Color', self)
        self.change_color_button.setToolTip("Change Graph Line Color")
        self.change_color_button.clicked.connect(self.change_graph_color)
        self.change_color_button.setStyleSheet(self.button_style())
        # self.change_color_button.setIcon(QIcon('icons/color.png'))  # Replace with your icon path
        controls_layout.addWidget(self.change_color_button)

        # Playback Speed Slider
        speed_layout = QHBoxLayout()
        speed_label = QLabel('Playback Speed (ms):', self)
        speed_label.setStyleSheet("QLabel { color: #FFFFFF; }")
        self.playback_speed_slider = QSlider(Qt.Horizontal)
        self.playback_speed_slider.setRange(1, 200)  # 1 ms to 200 ms per frame
        self.playback_speed_slider.setValue(50)      # Default playback speed
        self.playback_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.playback_speed_slider.setTickInterval(20)
        self.playback_speed_slider.setToolTip("Adjust Playback Speed (ms per frame)")
        self.playback_speed_slider.valueChanged.connect(self.update_playback_speed)
        self.playback_speed_slider.setStyleSheet(self.slider_style())
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.playback_speed_slider)
        controls_layout.addLayout(speed_layout)

        # Play/Pause Button
        self.play_pause_button = QPushButton('▶ Play', self)
        self.play_pause_button.setToolTip("Start/Pause Playback")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setStyleSheet(self.button_style())
        # self.play_pause_button.setIcon(QIcon('icons/play.png'))  # Replace with your icon path
        controls_layout.addWidget(self.play_pause_button)

        # Rewind Button
        self.rewind_button = QPushButton('⏮ Rewind', self)
        self.rewind_button.setToolTip("Rewind to Start")
        self.rewind_button.clicked.connect(self.rewind_playback)
        self.rewind_button.setStyleSheet(self.button_style())
        # self.rewind_button.setIcon(QIcon('icons/rewind.png'))  # Replace with your icon path
        controls_layout.addWidget(self.rewind_button)

        # Add the controls frame to the main layout
        self.main_layout.addWidget(controls_frame)

    def button_style(self):
        return """
            QPushButton {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3E3E3E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
        """

    def combo_box_style(self):
        return """
            QComboBox {
                background-color: #1E1E1E;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
            }
            QComboBox:hover {
                border: 1px solid #BBBBBB;
            }
            QComboBox QAbstractItemView {
                background-color: #2E2E2E;
                selection-background-color: #3E3E3E;
                selection-color: #FFFFFF;
            }
        """

    def slider_style(self):
        return """
            QSlider::groove:horizontal {
                border: 1px solid #4A4A4A;
                height: 8px;
                background: #2E2E2E;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #BB86FC;
                border: 1px solid #4A4A4A;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
        """
    
    # def toggle_signal(self):
    #     # Assuming you have a reference to the other graph
    #     other_graph = self.parent.other_graph

    #     # Swap the signal data
    #     self.raw_signal_data, other_graph.raw_signal_data = other_graph.raw_signal_data, self.raw_signal_data
    #     self.filtered_signal_data, other_graph.filtered_signal_data = other_graph.filtered_signal_data, self.filtered_signal_data
    #     self.time_axis, other_graph.time_axis = other_graph.time_axis, self.time_axis
    #     self.sampling_frequency, other_graph.sampling_frequency = other_graph.sampling_frequency, self.sampling_frequency
    #     self.current_channel, other_graph.current_channel = other_graph.current_channel, self.current_channel
    #     self.playback_position, other_graph.playback_position = other_graph.playback_position, self.playback_position
    #     self.playback_speed, other_graph.playback_speed = other_graph.playback_speed, self.playback_speed

    #     # Swap the playback timer intervals
    #     self.playback_timer.setInterval(self.playback_speed)
    #     other_graph.playback_timer.setInterval(other_graph.playback_speed)

    #     # Swap the playback speed slider values
    #     self.playback_speed_slider.setValue(self.playback_speed)
    #     other_graph.playback_speed_slider.setValue(other_graph.playback_speed)

    #     # Replot the signals on both graphs
    #     self.plot_current_window()
    #     other_graph.plot_current_window()

    def initialize_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        
        self.plot = self.plot_widget.plot([], [], pen=pg.mkPen(color=self.selected_color, width=2), name='Signal')
        self.indicator_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.indicator_line)
        
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)

        # Restrict panning
        self.plot_widget.setLimits(xMin=0, xMax=None, yMin=None, yMax=None)
    
        # Initially, do not show any signal
        self.plot.setData([], [])
        self.plot_widget.setXRange(0, self.display_window_duration)

    def plot_initial_setup(self):
        if self.time_axis is not None and self.current_channel is not None:
            signal_values = self.filtered_signal_data[self.current_channel]
            min_amplitude = np.min(signal_values)
            max_amplitude = np.max(signal_values)
            amplitude_range = max_amplitude - min_amplitude
            padding = 0.1 * amplitude_range if amplitude_range != 0 else 1
            self.plot_widget.setYRange(min_amplitude - padding, max_amplitude + padding)
            self.indicator_line.setPos(self.display_window_duration - self.draw_indicator_duration)

            # Set x-axis limits
            self.plot_widget.setLimits(xMin=0, xMax=self.time_axis[-1])
        
        if self.time_axis is not None and self.sampling_frequency is not None:
            total_samples = len(self.time_axis)
            display_samples = int(self.display_window_duration * self.sampling_frequency)
            max_scroll = max(total_samples - display_samples, 0)
            self.navigation_scrollbar.setMaximum(max_scroll)
            self.navigation_scrollbar.setPageStep(display_samples)
            self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))
            self.navigation_scrollbar.setValue(self.playback_position)
    
    
    def start_playback(self):
        self.playback_timer.start(self.playback_speed_slider.value())
    
    def plot_polar_graph(self):
            # Read the radar signal data from the CSV file
            radar_signal = pd.read_csv('/Signals/radar_signal.csv')

            # Extract the angle and amplitude data
            angles = np.deg2rad(radar_signal.iloc[:, 0].values)  # Convert degrees to radians
            amplitudes = radar_signal.iloc[:, 1].values

            # Create a polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(angles, amplitudes, color='b', marker='o', linestyle='-')

            # Set the title of the plot
            ax.set_title('Radar Signal Polar Plot')

            # Enable zooming
            def on_scroll(event):
                base_scale = 1.1
                cur_ylim = ax.get_ylim()
                cur_radius = cur_ylim[1] - cur_ylim[0]
                scale_factor = base_scale if event.button == 'up' else 1 / base_scale
                new_radius = cur_radius * scale_factor
                ax.set_ylim([cur_ylim[0], cur_ylim[0] + new_radius])
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('scroll_event', on_scroll)

            # Display the plot
            plt.show()
        
    def change_graph_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color.name()
            self.plot.setPen(pg.mkPen(color=self.selected_color, width=2))
            self.plot_widget.repaint()
            
    def load_signal_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal File",
            "",
            "All Files (*);;CSV Files (*.csv);;EDF Files (*.edf);;WFDB Files (*.hea)",
            options=options
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith('.csv'):
                self.load_csv_file(file_path)
            elif file_path.lower().endswith('.edf'):
                self.load_edf_file(file_path)
            elif file_path.lower().endswith('.hea'):
                self.load_wfdb_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # # Emit signal with the graph identifier
            # self.signal_loaded.emit(self.graph_identifier)

            # Reset plot and playback position
            self.plot.setData([], [])
            self.playback_position = 0
            self.plot_widget.setXRange(0, self.display_window_duration)
            
            # Plot the current window
            self.plot_current_window()
            
            # Start playback automatically
            self.start_playback()
        except Exception as error:
            self.display_error_message("File Load Error", f"Error loading signal file '{file_path}': {str(error)}")
    ...
    def load_csv_file(self, file_path):
        try:
            file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            data_frame = pd.read_csv(file_path)
            self.sampling_frequency = 360
            if data_frame.shape[1] < 3:
                raise ValueError("CSV file must contain at least three columns: time, and two signal channels.")
            self.time_axis = data_frame.iloc[:, 1].values / 1000
            self.raw_signal_data = {f"{file_name_without_extension} Channel {i+1}": data_frame.iloc[:, i+2].values for i in range(len(data_frame.columns) - 2)}
            self.apply_initial_filters()
            self.update_channel_selection()
            self.plot_initial_setup()
        except Exception as error:
            self.display_error_message("CSV Load Error", f"Failed to load CSV file: {str(error)}")

    def load_edf_file(self, file_path):
        try:
            file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            with pyedflib.EdfReader(file_path) as edf_reader:
                self.sampling_frequency = edf_reader.getSampleFrequency(0)
                self.time_axis = np.arange(edf_reader.getNSamples()[0]) / self.sampling_frequency
                self.raw_signal_data = {
                    f"{file_name_without_extension} {edf_reader.getLabel(i)}": edf_reader.readSignal(i) for i in range(edf_reader.signals_in_file)
                }
            self.apply_initial_filters()
            self.update_channel_selection()
            self.plot_initial_setup()

            if self.channel_selection_dropdown.count() > 0:
                self.current_channel = self.channel_selection_dropdown.currentText()
                self.plot_current_window()
            else:
                raise ValueError("No valid channels found in EDF file.")
        except Exception as error:
            self.display_error_message("EDF Load Error", f"Failed to load EDF file: {str(error)}")

    def load_edf_file(self, file_path):
        try:
            # Extract file name without extension
            file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            
            with pyedflib.EdfReader(file_path) as edf_reader:
                self.sampling_frequency = edf_reader.getSampleFrequency(0)
                self.time_axis = np.arange(edf_reader.getNSamples()[0]) / self.sampling_frequency
                self.raw_signal_data = {
                    f"{file_name_without_extension} {edf_reader.getLabel(i)}": edf_reader.readSignal(i) for i in range(edf_reader.signals_in_file)
                }
            self.apply_initial_filters()
            self.update_channel_selection()
            self.plot_initial_setup()

            if self.channel_selection_dropdown.count() > 0:
                self.current_channel = self.channel_selection_dropdown.currentText()
                self.plot_current_window()
            else:
                raise ValueError("No valid channels found in EDF file.")
        except Exception as error:
            self.display_error_message("EDF Load Error", f"Failed to load EDF file: {str(error)}")

    def load_wfdb_file(self, file_path):
        try:
            # Extract file name without extension
            file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
            
            record = wfdb.rdrecord(file_path.replace('.hea', ''))
            self.sampling_frequency = record.fs
            self.time_axis = np.arange(record.sig_len) / self.sampling_frequency
            self.raw_signal_data = {f"{file_name_without_extension} Channel {i+1}": record.p_signal[:, i] for i in range(record.n_sig)}
            self.apply_initial_filters()
            self.update_channel_selection()
            self.plot_initial_setup()
        except Exception as error:
            self.display_error_message("WFDB Load Error", f"Failed to load WFDB file: {str(error)}")

    def display_error_message(self, title, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()

    def update_channel_selection(self):
        self.channel_selection_dropdown.clear()
        for channel in self.raw_signal_data.keys():
            self.channel_selection_dropdown.addItem(channel)
        self.current_channel = self.channel_selection_dropdown.currentText()
        self.plot_current_window()

    def on_channel_changed(self):
        self.current_channel = self.channel_selection_dropdown.currentText()
        if self.current_channel:
            self.apply_current_filter()
            self.plot_current_window()

    def apply_initial_filters(self):
        try:
            # Example filter application (replace with actual filter logic)
            for channel, signal in self.raw_signal_data.items():
                # Apply a dummy filter (replace with actual filter logic)
                self.filtered_signal_data[channel] = self.dummy_filter(signal)
        except Exception as error:
            self.display_error_message("Filter Application Error", f"Failed to apply filters: {str(error)}")

    def dummy_filter(self, signal):
        # Example dummy filter (replace with actual filter logic)
        return signal  # No actual filtering done
    

    def apply_current_filter(self):
            filter_type = 'Low Pass'  # Example filter type
            cutoff_frequency = 50  # Example cutoff frequency in Hz
            
            if self.current_channel in self.raw_signal_data:
                self.filtered_signal_data[self.current_channel] = self.apply_filter(
                    self.raw_signal_data[self.current_channel], filter_type, cutoff_frequency
                )
            else:
                print(f"Channel {self.current_channel} not found in raw signal data.")

    def apply_filter(self, data, filter_type, cutoff_frequency, order=5):
        nyquist_freq = 0.5 * self.sampling_frequency
        normalized_cutoff = cutoff_frequency / nyquist_freq

        if filter_type == "Low Pass":
            b, a = butter(order, normalized_cutoff, btype='low')
        elif filter_type == "High Pass":
            b, a = butter(order, normalized_cutoff, btype='high')
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return filtfilt(b, a, data)

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()


    def start_playback(self):
        self.play_pause_button.setText("⏸ Pause")
        self.is_playing = True
        self.playback_timer.start(self.playback_speed)

    def pause_playback(self):
        self.play_pause_button.setText("▶ Play")
        self.is_playing = False
        self.playback_timer.stop()

    def rewind_playback(self):
        self.playback_timer.stop()
        self.playback_position = 0
        self.plot.setData([], [])
        self.plot_widget.setXRange(0, self.display_window_duration)
        self.navigation_scrollbar.setValue(0)
    
    

    def update_playback(self):
        if self.time_axis is None or self.current_channel is None:
            return

        signal_values = self.filtered_signal_data[self.current_channel]
        total_samples = len(self.time_axis)
        display_samples = int(self.display_window_duration * self.sampling_frequency)
        
        # Calculate the new end position for the x-axis
        new_end_position = min(self.playback_position + display_samples, total_samples)
        
        # Update the plot with the new data range
        self.plot.setData(self.time_axis[:new_end_position], signal_values[:new_end_position])
        
        # Extend the x-axis range dynamically to follow the playback position
        start_position = max(0, self.playback_position - display_samples)
        end_position = self.playback_position + display_samples
        self.plot_widget.setXRange(self.time_axis[start_position], self.time_axis[end_position - 1])

        # Move the indicator line
        self.indicator_line.setPos(self.time_axis[self.playback_position])

        # Update playback position
        self.playback_position += int(self.playback_speed / 1000 * self.sampling_frequency / 10)  # Adjust the divisor to slow down the movement

        # Stop playback if the end of the signal is reached
        if self.playback_position >= total_samples:
            self.playback_timer.stop()
            self.playback_position = 0
            self.play_pause_button.setText("Play")
            self.is_playing = False

    # def on_scroll(self, event):
    #     if event.button == 'up':
    #         # Zoom in
    #         zoom_factor = 1.1
    #     elif event.button == 'down':
    #         # Zoom out
    #         zoom_factor = 0.9
    #     else:
    #         zoom_factor = 1.0

    #     if zoom_factor != 1.0:
    #         # Adjust the display window duration based on zoom
    #         new_duration = self.display_window_duration * zoom_factor
    #         new_duration = max(1, min(new_duration, 60))  # Clamp between 1 and 60 seconds
    #         self.display_window_duration = new_duration
    #         self.axes.set_xlim(0, self.display_window_duration)

    #         # Update the indicator line position
    #         draw_indicator_time = self.display_window_duration - self.draw_indicator_duration
    #         self.indicator_line.set_xdata([draw_indicator_time, draw_indicator_time])

    #         # Update scrollbar range based on new display window duration
    #         if self.time_axis is not None and self.sampling_frequency is not None:
    #             total_samples = len(self.time_axis)
    #             display_samples = int(self.display_window_duration * self.sampling_frequency)
    #             max_scroll = max(total_samples - display_samples, 0)
    #             self.navigation_scrollbar.setMaximum(max_scroll)
    #             self.navigation_scrollbar.setPageStep(display_samples)
    #             self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))  # Scroll by 10% of the window

    #         self.plot_current_window()

    def on_scrollbar_moved(self, value):
        self.playback_position = value
        self.plot_current_window()

    def plot_current_window(self):
        if self.current_channel and self.time_axis is not None:
            start_index = self.playback_position
            display_samples = int(self.display_window_duration * self.sampling_frequency)
            end_index = start_index + display_samples
            x_data = self.time_axis[start_index:end_index]
            y_data = self.filtered_signal_data[self.current_channel][start_index:end_index]
            self.plot.setData(x_data, y_data)

            # Adjust the view range to the current data, temporarily blocking signals
            self.plot_widget.blockSignals(True)
            self.plot_widget.setXRange(x_data[0], x_data[-1], padding=0)
            self.plot_widget.blockSignals(False)

    def disable_scroll(self):
        # Disable the horizontal scrollbar and visually indicate it's disabled
        self.navigation_scrollbar.setEnabled(False)
        self.navigation_scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #f0f0f0;
            }
            QScrollBar::handle:horizontal {
                background: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
            }
        """)

    def enable_scroll(self):
        # Enable the horizontal scrollbar and reset its style
        self.navigation_scrollbar.setEnabled(True)
        self.navigation_scrollbar.setStyleSheet("")

    # Setter methods for synchronization with other graphs
    def set_playback_speed(self, value):
        """Set the playback speed from an external source."""
        self.playback_speed_slider.blockSignals(True)
        self.playback_speed_slider.setValue(value)
        self.playback_speed_slider.blockSignals(False)
        self.update_playback_speed(value)

    def set_filter_type(self, filter_type_index):
        self.filter_type_dropdown.setCurrentIndex(filter_type_index)

    def set_cutoff_frequency(self, cutoff_value):
        self.cutoff_frequency_slider.setValue(cutoff_value)

    def set_navigation_position(self, value):
        """Set the navigation scrollbar position from an external source."""
        self.navigation_scrollbar.blockSignals(True)
        self.navigation_scrollbar.setValue(value)
        self.navigation_scrollbar.blockSignals(False)
        self.on_scrollbar_moved(value)
    
    def link_plot_widget(self, other_plot_widget):
        """Link this plot widget's view to another plot widget for synchronized zooming and panning."""
        self.plot_widget.setXLink(other_plot_widget)
        self.plot_widget.setYLink(other_plot_widget)

    def unlink_plot_widget(self):
        """Unlink this plot widget's view from any other plot widgets."""
        self.plot_widget.setXLink(None)
        self.plot_widget.setYLink(None)

    

    



class PolarGraphPage(QWidget):
    def __init__(self, parent=None):
        super(PolarGraphPage, self).__init__(parent)

        self.main_layout = QVBoxLayout(self)

        # Initialize an empty polar plot
        self.figure, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

        # Buttons for controls
        controls_layout = QHBoxLayout()

        self.load_button = QPushButton("Load Signal")
        self.load_button.clicked.connect(self.load_signal)
        controls_layout.addWidget(self.load_button)
        self.load_button.setStyleSheet(self.link_button_style())

        # self.play_button = QPushButton("Play")
        # self.play_button.clicked.connect(self.start_playback)
        # controls_layout.addWidget(self.play_button)
        # self.play_button.setStyleSheet(self.link_button_style())

        # self.pause_button = QPushButton("Pause")
        # self.pause_button.clicked.connect(self.pause_playback)
        # controls_layout.addWidget(self.pause_button)
        # self.pause_button.setStyleSheet(self.link_button_style())

        self.play_pause_button = QPushButton("▶ Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_pause_button)
        self.play_pause_button.setStyleSheet(self.link_button_style())
        self.is_playing = False

        self.rewind_button = QPushButton("Rewind")
        self.rewind_button.clicked.connect(self.rewind_playback)
        controls_layout.addWidget(self.rewind_button)
        self.rewind_button.setStyleSheet(self.link_button_style())

        self.color_button = QPushButton("Change Color")
        self.color_button.clicked.connect(self.change_color)
        controls_layout.addWidget(self.color_button)
        self.color_button.setStyleSheet(self.link_button_style())

        self.main_layout.addLayout(controls_layout)

        # Variables for playback and signal storage
        self.angles = None
        self.amplitudes = None
        self.playback_index = 0
        self.playback_active = False
        self.current_color = 'b'

        # Setup a playback timer
        self.animation = FuncAnimation(self.figure, self.update_plot, interval=100, blit=False)

    def load_signal(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Radar Signal", "", "CSV Files (*.csv)", options=options)

        if file_path:
            radar_signal = pd.read_csv(file_path)
            self.angles = np.deg2rad(radar_signal.iloc[:, 0].values)  # Convert degrees to radians
            self.amplitudes = radar_signal.iloc[:, 1].values
            self.playback_index = 0  # Reset playback index
            self.update_plot()  # Display the signal

    def toggle_play_pause(self):
        if self.playback_active:
            self.pause_playback()
        else:
            self.start_playback()

    def start_playback(self):
        self.play_pause_button.setText("⏸ Pause")
        self.playback_active = True
        #self.playback_timer.start()

    def pause_playback(self):
        self.play_pause_button.setText("▶ Play")
        self.playback_active = False
        #self.playback_timer.stop()

    def rewind_playback(self):
        self.playback_active = False
        self.playback_index = 0
        self.update_plot()

    def change_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color.name()
            self.update_plot()

    def link_button_style(self):
        return """
            QPushButton {
                background-color: #008CBA;
                color: white;
                border-radius: 10px;
                padding: 5px 10px;
            }
            QPushButton:checked {
                background-color: #005f6a;
            }
            QPushButton:hover {
                background-color: #007BA7;
            }
        """

    def update_plot(self, frame=None):
        if self.angles is None or self.amplitudes is None:
            return  # Exit if no signal is loaded

        self.ax.clear()
        self.ax.plot(self.angles[:self.playback_index], self.amplitudes[:self.playback_index], 
                     color=self.current_color, marker='o')

        self.ax.set_title("Radar Signal Polar Plot")
        self.canvas.draw()

        if self.playback_active and self.playback_index < len(self.angles):
            self.playback_index += 1

class RealTimeWindow(QWidget):
    def __init__(self, parent=None):
        super(RealTimeWindow, self).__init__(parent)
        self.setWindowTitle("Real-Time ISS Position Plot")
        self.setGeometry(100, 100, 900, 700)

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Initialize Matplotlib Figure and Axes
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)  # Add navigation toolbar

        # Add toolbar and canvas to the layout
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        # Data containers
        self.latitude_data = []
        self.longitude_data = []

        # Timer for periodic updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(5000)  # Update every 5 seconds
            # Fetch initial data to initialize containers

        initial_lon, initial_lat = None, None
        while initial_lat is None and initial_lon is None: 
            initial_lat, initial_lon = self.fetch_satellite_position()

        if initial_lat is not None and initial_lon is not None:
            self.latitude_data = [initial_lat]
            self.longitude_data = [initial_lon]

    def fetch_satellite_position(self):
        """Fetch latitude and longitude of the ISS from the Open Notify API."""
        url = "http://api.open-notify.org/iss-now.json"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                position = data['iss_position']
                latitude = float(position['latitude'])
                longitude = float(position['longitude'])
                return latitude, longitude
            else:
                print(f"API Error: {response.status_code}")
                return None, None
        except Exception as e:
            print(f"Network Error: {e}")
            return None, None
    


    def update_plot(self):
        """Fetch data and update the plot with actual latitude and longitude on a shared y-axis."""
        latitude, longitude = self.fetch_satellite_position()
        if latitude is not None and longitude is not None:
            self.latitude_data.append(latitude)
            self.longitude_data.append(longitude)

            # Keep the last 60 points
            if len(self.latitude_data) > 60:
                self.latitude_data.pop(0)
            if len(self.longitude_data) > 60:
                self.longitude_data.pop(0)

            # Clear the plot
            self.ax.clear()

            # Add grid and improved labels
            self.ax.set_title("Real-Time ISS Position", fontsize=14, fontweight="bold")
            self.ax.set_xlabel("Time (updates)", fontsize=12)
            self.ax.set_ylabel("Coordinates (Latitude/Longitude)", fontsize=12)
            self.ax.grid(True, linestyle="--", alpha=0.7)

            # Plot both latitude and longitude on the same y-axis
            self.ax.plot(self.latitude_data, label="Latitude", color="blue", marker="o", markersize=8, alpha=0.8)
            self.ax.plot(self.longitude_data, label="Longitude", color="green", marker="x", markersize=8, alpha=0.8)

            # Highlight the latest point
            if self.latitude_data and self.longitude_data:
                self.ax.scatter(len(self.latitude_data) - 1, self.latitude_data[-1], color="red",
                                label="Latest Latitude", zorder=5, s=100)
                self.ax.scatter(len(self.longitude_data) - 1, self.longitude_data[-1], color="orange",
                                label="Latest Longitude", zorder=5, s=100)

            # Set Y-axis limits to show full range for both latitude and longitude
            self.ax.set_ylim(-180, 180)

            # Add legend
            self.ax.legend(fontsize=10, loc="upper right")

            # Add numerical display of current latitude and longitude
            self.ax.text(0.02, 0.95, f"Current Latitude: {latitude:.2f}", transform=self.ax.transAxes,
                        fontsize=12, color="blue", bbox=dict(facecolor="white", alpha=0.7))
            self.ax.text(0.02, 0.90, f"Current Longitude: {longitude:.2f}", transform=self.ax.transAxes,
                        fontsize=12, color="green", bbox=dict(facecolor="white", alpha=0.7))

            # Draw canvas
            self.canvas.draw()

class SelectionWindow(QWidget):
    def __init__(self, signal, sampling_frequency=None, parent=None):
        super(SelectionWindow, self).__init__(parent)
        self.signal = signal
        self.sampling_frequency = sampling_frequency
        self.start_index = None
        self.end_index = None

        # Initialize layout
        self.main_layout = QVBoxLayout(self)

        # Add plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
        self.plot_widget.setXRange(0, len(self.signal) / self.sampling_frequency if self.sampling_frequency else len(self.signal))
        self.plot_widget.setYRange(-3, 3)
        self.plot_widget.plot(np.arange(len(self.signal)) / self.sampling_frequency  if self.sampling_frequency else np.arange(len(self.signal)) ,
                      self.signal, pen=pg.mkPen(color='b', width=2), name="Signal")

        self.main_layout.addWidget(self.plot_widget)

        # Selection line indicators
        self.start_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color='r', style=QtCore.Qt.DashLine))
        self.end_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color='g', style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.start_line)
        self.plot_widget.addItem(self.end_line)

        # Set initial positions of the selection lines
        self.start_line.setPos(0)
        self.end_line.setPos(len(self.signal) / self.sampling_frequency if self.sampling_frequency else len(self.signal))

        # Buttons to confirm/cancel selection
        self.controls_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirm_selection)
        self.controls_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        self.controls_layout.addWidget(self.cancel_button)

        self.main_layout.addLayout(self.controls_layout)

    def confirm_selection(self):
        self.start_index = int(self.start_line.value() * self.sampling_frequency) if self.sampling_frequency else int(self.start_line.value())
        self.end_index = int(self.end_line.value() * self.sampling_frequency) if self.sampling_frequency else int(self.end_line.value())
        self.close()

    def get_selected_indices(self):
        return self.start_index, self.end_index
            

class GlueGraphs(QWidget):
    def __init__(self, graph_identifier, graph1, graph2, parent=None):
        super(GlueGraphs, self).__init__(parent)
        self.graph_identifier = graph_identifier
        self.parent = parent


        self.graph1 = graph1
        self.graph2 = graph2

        # Initialize layout
        self.main_layout = QVBoxLayout(self)
        self.initialize_controls()

        # Initialize signal-related variables
        self.raw_signal_data = {}
        self.filtered_signal_data = {}
        self.time_axis = None
        self.sampling_frequency = None
        self.current_channel = None
        self.playback_position = 0
        self.display_window_duration = 0.1  # seconds
        self.draw_indicator_duration = 0.1  # seconds

        # Flag to indicate if cine mode is active
        self.cine_mode_active = False

        # Initialize plot after setting display_window_duration
        self.initialize_plot()

        # Initialize scrollbar for signal navigation
        self.navigation_scrollbar = QScrollBar(Qt.Horizontal)
        self.navigation_scrollbar.setMinimum(0)
        self.navigation_scrollbar.valueChanged.connect(self.on_scrollbar_moved)
        self.main_layout.addWidget(self.navigation_scrollbar)

        # Initialize playback timer
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.update_playback)

        self.playback_speed_slider = QSlider(Qt.Horizontal)
        self.playback_speed_slider.setRange(1, 100)
        self.playback_speed_slider.setValue(50)


    def update_playback_speed(self, value):
        self.playback_speed = value
        self.playback_timer.setInterval(value)
        print(f"Playback speed updated to: {self.playback_speed} ms per frame")


    def update_dropdowns(self, graph_identifier):
        if graph_identifier == self.signal_graph1.graph_identifier:
            self.signal1_dropdown.addItem(graph_identifier)
        elif graph_identifier == self.signal_graph2.graph_identifier:
            self.signal2_dropdown.addItem(graph_identifier)

    def initialize_controls(self):
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        # Create a frame to group controls
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("QFrame { background-color: #1E1E1E; border: none; }")
        controls_frame.setLayout(controls_layout)

        # Glue Signal Button
        self.glue_signal_button = QPushButton('Glue Signals', self)
        self.glue_signal_button.setToolTip("Glue both signals")
        self.glue_signal_button.clicked.connect(self.glue_signals)
        self.glue_signal_button.setStyleSheet(self.button_style())
        # self.glue_signal_button.setIcon(QIcon('icons/glue.png'))  # Replace with your icon path
        controls_layout.addWidget(self.glue_signal_button)

        # SnapShot Button
        self.snapShot_button = QPushButton('SnapShot', self)
        self.snapShot_button.setToolTip("Take a snapshot of the signals")
        self.snapShot_button.clicked.connect(self.take_snapshot_and_generate_pdf)
        self.snapShot_button.setStyleSheet(self.button_style())
        # self.snapShot_button.setIcon(QIcon('icons/snapshot.png'))  # Replace with your icon path
        controls_layout.addWidget(self.snapShot_button)

        # Open Selection Buttons
        self.open_selection_button1 = QPushButton("Select Part of Signal 1")
        self.open_selection_button1.setToolTip("Select a part of Signal 1")
        self.open_selection_button1.clicked.connect(lambda: self.open_selection_window(1))
        self.open_selection_button1.setStyleSheet(self.button_style())
        # self.open_selection_button1.setIcon(QIcon('icons/select1.png'))  # Replace with your icon path
        controls_layout.addWidget(self.open_selection_button1)

        self.open_selection_button2 = QPushButton("Select Part of Signal 2")
        self.open_selection_button2.setToolTip("Select a part of Signal 2")
        self.open_selection_button2.clicked.connect(lambda: self.open_selection_window(2))
        self.open_selection_button2.setStyleSheet(self.button_style())
        # self.open_selection_button2.setIcon(QIcon('icons/select2.png'))  # Replace with your icon path
        controls_layout.addWidget(self.open_selection_button2)

        # Gap/Overlap Input
        gap_layout = QHBoxLayout()
        gap_label = QLabel("Gap (+) / Overlap (-) Samples:", self)
        gap_label.setStyleSheet("QLabel { color: #FFFFFF; }")
        self.gap_input = QSpinBox(self)
        self.gap_input.setRange(-10000, 10000)  # Adjust the range as needed
        self.gap_input.setValue(0)
        self.gap_input.setSingleStep(1)
        self.gap_input.setToolTip("Enter positive value for gap, negative value for overlap (in samples)")
        self.gap_input.setStyleSheet(self.spin_box_style())
        gap_layout.addWidget(gap_label)
        gap_layout.addWidget(self.gap_input)
        controls_layout.addLayout(gap_layout)

        # Interpolation Order Input
        interp_layout = QHBoxLayout()
        interp_label = QLabel("Interpolation Order:", self)
        interp_label.setStyleSheet("QLabel { color: #FFFFFF; }")
        self.interpolation_input = QComboBox(self)
        self.interpolation_input.addItems(['1 (Linear)', '2 (Quadratic)', '3 (Cubic)'])  # Supported orders
        self.interpolation_input.setToolTip("Select the interpolation order")
        self.interpolation_input.setStyleSheet(self.combo_box_style())
        interp_layout.addWidget(interp_label)
        interp_layout.addWidget(self.interpolation_input)
        controls_layout.addLayout(interp_layout)

        # Add the controls frame to the main layout
        self.main_layout.addWidget(controls_frame)

    def button_style(self):
        return """
            QPushButton {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3E3E3E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
        """

    def combo_box_style(self):
        return """
            QComboBox {
                background-color: #1E1E1E;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
            }
            QComboBox:hover {
                border: 1px solid #BBBBBB;
            }
            QComboBox QAbstractItemView {
                background-color: #2E2E2E;
                selection-background-color: #3E3E3E;
                selection-color: #FFFFFF;
            }
        """

    def spin_box_style(self):
        return """
            QSpinBox {
                background-color: #1E1E1E;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #4A4A4A;
                border-radius: 4px;
            }
            QSpinBox:hover {
                border: 1px solid #BBBBBB;
            }
        """
    
    def on_signal_loaded(self, graph_identifier):
        print(f"Signal loaded in graph: {graph_identifier}")
        # Implement the logic to handle the signal loaded event
        if graph_identifier == self.graph1.graph_identifier:
            self.handle_graph1_update()
        elif graph_identifier == self.graph2.graph_identifier:
            self.handle_graph2_update()
    
    def handle_graph1_update(self):
        # Logic to handle updates in graph1
        print("Handling update for graph1")
        # Example: Sync some state or data between graphs
        self.sync_graphs()

    def handle_graph2_update(self):
        # Logic to handle updates in graph2
        print("Handling update for graph2")
        # Example: Sync some state or data between graphs
        self.sync_graphs()

    def sync_graphs(self):
        # Example logic to sync graphs
        print("Syncing graphs")
        # Implement the actual sync logic here

    def initialize_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        
        self.plot = self.plot_widget.plot([], [], pen=pg.mkPen(color='#1f77b4', width=2), name='Signal')
        self.indicator_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='k', style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.indicator_line)
        
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)

        # Restrict panning
        self.plot_widget.setLimits(xMin=0, xMax=None, yMin=None, yMax=None)
    
        # Initially, do not show any signal
        self.plot.setData([], [])
        self.plot_widget.setXRange(0, self.display_window_duration)

                # Connect the view range changed signal
        self.plot_widget.getViewBox().sigXRangeChanged.connect(self.on_view_range_changed)
    
    def on_view_range_changed(self):
        # Get current x-range
        x_range = self.plot_widget.viewRange()[0]
        visible_duration = x_range[1] - x_range[0]
        self.display_window_duration = visible_duration

        # Update scrollbar parameters
        if self.time_axis is not None and self.sampling_frequency is not None:
            total_samples = len(self.time_axis)
            display_samples = int(self.display_window_duration * self.sampling_frequency)
            max_scroll = max(total_samples - display_samples, 0)
            self.navigation_scrollbar.blockSignals(True)
            self.navigation_scrollbar.setMaximum(max_scroll)
            self.navigation_scrollbar.setPageStep(display_samples)
            self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))
            self.navigation_scrollbar.blockSignals(False)
    
    def update_scrollbar_parameters(self):
        if self.time_axis is not None and self.sampling_frequency is not None:
            total_samples = len(self.time_axis)
            display_samples = int(self.display_window_duration * self.sampling_frequency)
            max_scroll = max(total_samples - display_samples, 0)
            self.navigation_scrollbar.blockSignals(True)
            self.navigation_scrollbar.setMaximum(max_scroll)
            self.navigation_scrollbar.setPageStep(display_samples)
            self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))
            self.navigation_scrollbar.setValue(self.playback_position)
            self.navigation_scrollbar.blockSignals(False)

    def plot_initial_setup(self):
        if self.time_axis is not None and self.current_channel is not None:
            signal_values = self.filtered_signal_data[self.current_channel]
            min_amplitude = np.min(signal_values)
            max_amplitude = np.max(signal_values)
            amplitude_range = max_amplitude - min_amplitude
            padding = 0.1 * amplitude_range if amplitude_range != 0 else 1
            self.plot_widget.setYRange(min_amplitude - padding, max_amplitude + padding)
            self.indicator_line.setPos(self.display_window_duration - self.draw_indicator_duration)

            # Set x-axis limits
            self.plot_widget.setLimits(xMin=0, xMax=self.time_axis[-1])
        
        if self.time_axis is not None and self.sampling_frequency is not None:
            total_samples = len(self.time_axis)
            display_samples = int(self.display_window_duration * self.sampling_frequency)
            max_scroll = max(total_samples - display_samples, 0)
            self.navigation_scrollbar.setMaximum(max_scroll)
            self.navigation_scrollbar.setPageStep(display_samples)
            self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))
            self.navigation_scrollbar.setValue(self.playback_position)
    
    def start_playback(self):
        self.playback_timer.start(self.playback_speed_slider.value())

    def change_graph_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.plot.setPen(pg.mkPen(color=color.name(), width=2))
            self.plot_widget.repaint()

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_playback()
        else:
            self.start_playback()


    def start_playback(self):
        self.play_pause_button.setText("⏸ Pause")
        self.is_playing = True
        self.playback_timer.start(self.playback_speed)

    def pause_playback(self):
        self.play_pause_button.setText("▶ Play")
        self.is_playing = False
        self.playback_timer.stop()

    def rewind_playback(self):
        self.playback_timer.stop()
        self.playback_position = 0
        self.plot.setData([], [])
        self.plot_widget.setXRange(0, self.display_window_duration)
        self.navigation_scrollbar.setValue(0)

    def update_playback(self):
        if self.time_axis is None or self.current_channel is None:
            return

        signal_values = self.filtered_signal_data[self.current_channel]
        total_samples = len(self.time_axis)
        display_samples = int(self.display_window_duration * self.sampling_frequency)
        
        # Calculate the new end position for the x-axis
        new_end_position = min(self.playback_position + display_samples, total_samples)
        
        # Update the plot with the new data range
        self.plot.setData(self.time_axis[:new_end_position], signal_values[:new_end_position])
        
        # Extend the x-axis range
        self.plot_widget.setXRange(0, self.time_axis[new_end_position - 1])
        
        # Move the indicator line
        self.indicator_line.setPos(self.time_axis[new_end_position - 1])
        
        # Update playback position
        self.playback_position += int(self.playback_speed / 1000 * self.sampling_frequency)
        
        # Stop playback if the end of the signal is reached
        if self.playback_position >= total_samples:
            self.playback_timer.stop()
            self.playback_position = 0

    def on_scroll(self, event):
        if event.button == 'up':
            # Zoom in
            zoom_factor = 1.1
        elif event.button == 'down':
            # Zoom out
            zoom_factor = 0.9
        else:
            zoom_factor = 1.0

        if zoom_factor != 1.0:
            # Adjust the display window duration based on zoom
            new_duration = self.display_window_duration * zoom_factor
            new_duration = max(1, min(new_duration, 60))  # Clamp between 1 and 60 seconds
            self.display_window_duration = new_duration
            self.axes.set_xlim(0, self.display_window_duration)

            # Update the indicator line position
            draw_indicator_time = self.display_window_duration - self.draw_indicator_duration
            self.indicator_line.set_xdata([draw_indicator_time, draw_indicator_time])

            # Update scrollbar range based on new display window duration
            if self.time_axis is not None and self.sampling_frequency is not None:
                total_samples = len(self.time_axis)
                display_samples = int(self.display_window_duration * self.sampling_frequency)
                max_scroll = max(total_samples - display_samples, 0)
                self.navigation_scrollbar.setMaximum(max_scroll)
                self.navigation_scrollbar.setPageStep(display_samples)
                self.navigation_scrollbar.setSingleStep(int(0.1 * display_samples))  # Scroll by 10% of the window

            self.plot_current_window()

    def on_scrollbar_moved(self, value):
        self.playback_position = value
        self.plot_current_window()

    def plot_current_window(self):
        if self.current_channel and self.time_axis is not None:
            start_index = self.playback_position
            end_index = start_index + int(self.display_window_duration * self.sampling_frequency)
            self.plot.setData(self.time_axis[start_index:end_index], self.filtered_signal_data[self.current_channel][start_index:end_index])
            

    def disable_scroll(self):
        # Disable the horizontal scrollbar and visually indicate it's disabled
        self.navigation_scrollbar.setEnabled(False)
        self.navigation_scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background: #f0f0f0;
            }
            QScrollBar::handle:horizontal {
                background: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
            }
        """)

    def enable_scroll(self):
        # Enable the horizontal scrollbar and reset its style
        self.navigation_scrollbar.setEnabled(True)
        self.navigation_scrollbar.setStyleSheet("")

    # Setter methods for synchronization with other graphs
    def set_playback_speed(self, speed_value):
        self.playback_speed_slider.setValue(speed_value)

    

    def set_navigation_position(self, position_value):
        self.navigation_scrollbar.setValue(position_value)


    def open_selection_window(self, signal_number):
        if signal_number == 1:
            selected_signal_index = self.graph1.channel_selection_dropdown.currentIndex()
            if selected_signal_index >= 0:
                signal_key = list(self.graph1.raw_signal_data.keys())[selected_signal_index]
                signal_data = self.graph1.raw_signal_data[signal_key]
                sampling_frequency = self.graph1.sampling_frequency
                self.selection_window1 = SelectionWindow(signal_data, sampling_frequency)
                self.selection_window1.show()
            else:
                print("No valid signal selected for Signal 1.")
        elif signal_number == 2:
            selected_signal_index = self.graph2.channel_selection_dropdown.currentIndex()
            if selected_signal_index >= 0:
                signal_key = list(self.graph2.raw_signal_data.keys())[selected_signal_index]
                signal_data = self.graph2.raw_signal_data[signal_key]
                sampling_frequency = self.graph2.sampling_frequency
                self.selection_window2 = SelectionWindow(signal_data, sampling_frequency)
                self.selection_window2.show()
            else:
                print("No valid signal selected for Signal 2.")

    
    def glue_signals(self):
        # Check if selection windows are active
        if not hasattr(self, 'selection_window1') or not hasattr(self, 'selection_window2'):
            print("Please open selection windows for both signals.")
            return

        # Get the selected indices from both windows
        start1, end1 = self.selection_window1.get_selected_indices()
        start2, end2 = self.selection_window2.get_selected_indices()

        if start1 is None or end1 is None or start2 is None or end2 is None:
            print("Please select parts of both signals.")
            return

        # Extract parts of the signals
        part1 = self.selection_window1.signal[start1:end1]
        part2 = self.selection_window2.signal[start2:end2]

        # Get user-defined gap/overlap and interpolation
        try:
            gap_overlap = self.gap_input.value()  # Positive for gap, negative for overlap (in samples)
            interpolation_order_str = self.interpolation_input.currentText()
            interpolation_order = int(interpolation_order_str.split(' ')[0])
        except ValueError:
            print("Invalid input for gap or interpolation. Please enter integers.")
            return

        sampling_frequency = getattr(self.selection_window1, 'sampling_frequency', 1)

        # Initialize time arrays
        time_part1 = np.arange(len(part1)) / sampling_frequency
        time_part2 = None

        if gap_overlap >= 0:
            # Gap case
            gap_length = gap_overlap
            gap_signal = np.full(gap_length, np.nan)  # Use NaNs to represent the gap in the signal
            gap_time = np.arange(gap_length) / sampling_frequency + time_part1[-1] + (1 / sampling_frequency)

            # Time for Part 2
            time_part2 = np.arange(len(part2)) / sampling_frequency + gap_time[-1] + (1 / sampling_frequency)

            # Concatenate signal and time arrays
            glued_signal = np.concatenate([part1, gap_signal, part2])
            glued_time = np.concatenate([time_part1, gap_time, time_part2])
        else:
            # Overlap case
            overlap_size = abs(gap_overlap)
            overlap_size = min(overlap_size, len(part1), len(part2))

            # Overlapping regions
            overlap1 = part1[-overlap_size:]
            overlap2 = part2[:overlap_size]

            # Generate blending weights
            x = np.linspace(0, 1, overlap_size)

            # Blend using different blending functions
            def blend_weights(x, order):
                if order == 1:
                    return x  # Linear
                elif order == 2:
                    return x ** 2  # Quadratic
                elif order == 3:
                    return x ** 3  # Cubic
                else:
                    return x  # Default to linear

            weights = blend_weights(x, interpolation_order)
            interpolated_overlap = overlap1 * (1 - weights) + overlap2 * weights

            # Construct the glued signal and time arrays
            glued_signal = np.concatenate([part1[:-overlap_size], interpolated_overlap, part2[overlap_size:]])
            time_part1 = np.arange(len(part1[:-overlap_size])) / sampling_frequency
            time_overlap = np.arange(len(interpolated_overlap)) / sampling_frequency + time_part1[-1] + (1 / sampling_frequency)
            time_part2 = np.arange(len(part2[overlap_size:])) / sampling_frequency + time_overlap[-1] + (1 / sampling_frequency)
            glued_time = np.concatenate([time_part1, time_overlap, time_part2])

        # Plotting
        self.plot_widget.clear()

        # Plot Part 1
        self.plot_widget.plot(
            time_part1,
            part1[:len(time_part1)],
            pen=pg.mkPen(color=self.graph1.selected_color, width=2),
            name="Part 1"
        )

        if gap_overlap >= 0:
            # Plot the gap as a vertical line
            gap_indicator = pg.InfiniteLine(
                pos=gap_time[0],
                angle=90,
                pen=pg.mkPen(color='gray', style=QtCore.Qt.DashLine)
            )
            self.plot_widget.addItem(gap_indicator)
        else:
            # Plot the overlapping/interpolated region in a distinct color
            self.plot_widget.plot(
                time_overlap,
                interpolated_overlap,
                pen=pg.mkPen(color='green', width=2, style=QtCore.Qt.DashLine),
                name="Overlap"
            )

        # Plot Part 2
        self.plot_widget.plot(
            time_part2,
            part2[overlap_size:] if gap_overlap < 0 else part2,
            pen=pg.mkPen(color=self.graph2.selected_color, width=2),
            name="Part 2"
        )

        # Set labels and legend
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.addLegend()

        # Save the glued signal
        self.glued_signal = glued_signal
        self.glued_time = glued_time  # Save time data if needed elsewhere

        print("Signals glued successfully.")




    def load_data(self, file_name, signal_number):
        radar_signal = pd.read_csv(file_name)
        angles = np.deg2rad(radar_signal.iloc[:, 0].values)  # Convert degrees to radians
        amplitudes = radar_signal.iloc[:, 1].values

        if signal_number == 1:
            self.graph1.signal = amplitudes
            print("Signal 1 loaded:", self.graph1.signal)
        elif signal_number == 2:
            self.graph2.signal = amplitudes
            print("Signal 2 loaded:", self.graph2.signal)

        self.playback_index = 0  # Reset playback index
        self.plot()  # Display the signal

    def open_file_dialog(self, signal_number):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*);;CSV Files (*.csv)", options=options)
        if file_name:
            self.load_data(file_name, signal_number)

    def take_snapshot_and_generate_pdf(self):
        # Step 1: Capture the plot as an image
        pixmap = QPixmap(self.plot_widget.size())
        painter = QPainter(pixmap)
        self.plot_widget.render(painter)
        painter.end()
        image_path = "glued_signal_snapshot.png"
        pixmap.save(image_path)

        # Step 2: Calculate statistics
        glued_signal = self.glued_signal
        mean_value = np.mean(glued_signal)
        std_dev = np.std(glued_signal)
        min_value = np.min(glued_signal)
        max_value = np.max(glued_signal)
        duration = len(glued_signal) #/ self.sampling_frequency

        # Step 3: Generate PDF
        pdf_path = "glued_signal_report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # Add image to PDF
        c.drawImage(image_path, 50, height - 300, width=500, height=250)

        # Add statistics to PDF
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 320, f"Mean: {mean_value:.2f}")
        c.drawString(50, height - 340, f"Standard Deviation: {std_dev:.2f}")
        c.drawString(50, height - 360, f"Min Value: {min_value:.2f}")
        c.drawString(50, height - 380, f"Max Value: {max_value:.2f}")
        c.drawString(50, height - 400, f"Duration: {duration:.2f} seconds")

        c.save()

        print(f"Snapshot saved as {image_path} and PDF report generated as {pdf_path}")


    def display_error_message(self, title, message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Signal Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # # Set a custom icon (ensure the path is correct)
        # self.setWindowIcon(QIcon('path/to/your/icon.png'))  # Replace with your icon path

        # Set up the status bar
        self.statusBar().showMessage('Ready')

        # Create the main widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Main layout
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Top control layout
        top_control_layout = QHBoxLayout()
        top_control_layout.setContentsMargins(0, 0, 0, 0)
        top_control_layout.setSpacing(10)

        # Create a frame to hold the top controls
        top_controls_frame = QFrame()
        top_controls_frame.setFrameShape(QFrame.StyledPanel)
        top_controls_frame.setStyleSheet("QFrame { background-color: #1E1E1E; border: none; }")
        top_controls_frame.setLayout(top_control_layout)

        # Link graphs button
        self.link_graphs_button = QPushButton("Link Both Graphs", self)
        self.link_graphs_button.setCheckable(True)
        self.link_graphs_button.setToolTip("Synchronize controls between both graphs")
        self.link_graphs_button.setStyleSheet(self.button_style())
        self.link_graphs_button.clicked.connect(self.toggle_graph_linking)
        top_control_layout.addWidget(self.link_graphs_button)

        # Swap Signals Button
        self.swap_signals_button = QPushButton("Swap Signals", self)
        self.swap_signals_button.setToolTip("Swap signals between both graphs")
        self.swap_signals_button.setStyleSheet(self.button_style())
        self.swap_signals_button.clicked.connect(self.swap_signals)
        top_control_layout.addWidget(self.swap_signals_button)

        # Spacer to align buttons to the left
        top_control_layout.addStretch()

        # Add top controls frame to the main layout
        main_layout.addWidget(top_controls_frame)

        # Add two SignalGraph widgets
        self.graph_widget_1 = SignalGraph(graph_identifier=1, parent=self)
        self.graph_widget_2 = SignalGraph(graph_identifier=2, parent=self)
        self.graph_widget_3 = GlueGraphs(
            graph_identifier=3,
            graph1=self.graph_widget_1,
            graph2=self.graph_widget_2,
            parent=self
        )
        self.graph_widget_1.parent.other_graph = self.graph_widget_2
        self.graph_widget_2.parent.other_graph = self.graph_widget_1

        main_layout.addWidget(self.graph_widget_1)
        main_layout.addWidget(self.graph_widget_2)
        main_layout.addWidget(self.graph_widget_3)

        # Bottom control layout
        bottom_control_layout = QHBoxLayout()
        bottom_control_layout.setContentsMargins(0, 0, 0, 0)
        bottom_control_layout.setSpacing(10)

        # Create a frame to hold the bottom controls
        bottom_controls_frame = QFrame()
        bottom_controls_frame.setFrameShape(QFrame.StyledPanel)
        bottom_controls_frame.setStyleSheet("QFrame { background-color: #1E1E1E; border: none; }")
        bottom_controls_frame.setLayout(bottom_control_layout)

        # Polar graph button
        self.polar_button = QPushButton('Open Polar Graph', self)
        self.polar_button.setToolTip("Open Polar Graph")
        self.polar_button.setStyleSheet(self.button_style())
        self.polar_button.clicked.connect(self.open_polar_graph)
        bottom_control_layout.addWidget(self.polar_button)

        # Real-time button
        self.realtime_button = QPushButton('Open Realtime Graph', self)
        self.realtime_button.setToolTip("Open Real-time Graph")
        self.realtime_button.setStyleSheet(self.button_style())
        self.realtime_button.clicked.connect(self.open_realtime_graph)
        bottom_control_layout.addWidget(self.realtime_button)

        # Spacer to align buttons to the left
        bottom_control_layout.addStretch()

        # Add bottom controls frame to the main layout
        main_layout.addWidget(bottom_controls_frame)

        # Track graph linking state
        self.graphs_linked = False

    def button_style(self):
        return """
            QPushButton {
                background-color: #2E2E2E;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3E3E3E;
            }
            QPushButton:pressed {
                background-color: #1E1E1E;
            }
            QPushButton:checked {
                background-color: #5E5E5E;
            }
        """

    def open_realtime_graph(self):
        self.realtime_window = RealTimeWindow()
        self.realtime_window.setWindowFlags(Qt.Window)
        self.realtime_window.show()
        self.statusBar().showMessage('Real-time Graph Opened')

    def open_polar_graph(self):
        self.statusBar().showMessage('Polar Graph Opened')
        # Create and display the polar graph page
        self.polar_graph_page = PolarGraphPage(self)
        self.setCentralWidget(self.polar_graph_page)

        # Add back button to return to the main page
        self.back_button = QPushButton("Back to Main", self.polar_graph_page)
        self.back_button.setStyleSheet(self.button_style())
        self.back_button.clicked.connect(self.back_to_main_page)
        self.polar_graph_page.main_layout.addWidget(self.back_button)
    
    def back_to_main_page(self):
        # Reset the central widget to the main page
        self.setCentralWidget(self.main_widget)
        self.statusBar().showMessage('Returned to Main Page')

    def toggle_graph_linking(self, checked):
        self.graphs_linked = checked
        if self.graphs_linked:
            self.statusBar().showMessage('Graphs Linked')
            self.link_graphs_button.setChecked(True)
            self.link_graphs()
        else:
            self.statusBar().showMessage('Graphs Unlinked')
            self.link_graphs_button.setChecked(False)
            self.unlink_graphs()

    def swap_signals(self):
        # Stop playback timers to prevent conflicts during swapping
        self.graph_widget_1.playback_timer.stop()
        self.graph_widget_2.playback_timer.stop()

        # Swap all relevant attributes between graph_widget_1 and graph_widget_2
        attrs_to_swap = [
            'raw_signal_data',
            'filtered_signal_data',
            'time_axis',
            'sampling_frequency',
            'current_channel',
            'playback_position',
            'display_window_duration',
            'draw_indicator_duration',
            'is_playing',
            'selected_color',
            'playback_speed',
        ]

        for attr in attrs_to_swap:
            temp = getattr(self.graph_widget_1, attr)
            setattr(self.graph_widget_1, attr, getattr(self.graph_widget_2, attr))
            setattr(self.graph_widget_2, attr, temp)

        # Swap play/pause button states and text
        temp_button_text = self.graph_widget_1.play_pause_button.text()
        self.graph_widget_1.play_pause_button.setText(self.graph_widget_2.play_pause_button.text())
        self.graph_widget_2.play_pause_button.setText(temp_button_text)

        # Swap the playback speed sliders
        temp_slider_value = self.graph_widget_1.playback_speed_slider.value()
        self.graph_widget_1.playback_speed_slider.setValue(self.graph_widget_2.playback_speed_slider.value())
        self.graph_widget_2.playback_speed_slider.setValue(temp_slider_value)

        # Swap the navigation scrollbar positions
        temp_scrollbar_value = self.graph_widget_1.navigation_scrollbar.value()
        self.graph_widget_1.navigation_scrollbar.setValue(self.graph_widget_2.navigation_scrollbar.value())
        self.graph_widget_2.navigation_scrollbar.setValue(temp_scrollbar_value)

        # Swap the plots
        # Extract data from the plots
        plot_data_1 = self.graph_widget_1.plot.getData()
        plot_data_2 = self.graph_widget_2.plot.getData()
        # Swap the data
        self.graph_widget_1.plot.setData(*plot_data_2)
        self.graph_widget_2.plot.setData(*plot_data_1)
        # Swap the pen colors
        temp_pen = self.graph_widget_1.plot.opts['pen']
        self.graph_widget_1.plot.setPen(self.graph_widget_2.plot.opts['pen'])
        self.graph_widget_2.plot.setPen(temp_pen)

        # Swap the channel selection dropdown selections
        temp_index = self.graph_widget_1.channel_selection_dropdown.currentIndex()
        self.graph_widget_1.channel_selection_dropdown.setCurrentIndex(
            self.graph_widget_2.channel_selection_dropdown.currentIndex())
        self.graph_widget_2.channel_selection_dropdown.setCurrentIndex(temp_index)

        # Swap any additional UI elements or states as needed
        # ...

        # Update playback timer intervals
        self.graph_widget_1.playback_timer.setInterval(self.graph_widget_1.playback_speed)
        self.graph_widget_2.playback_timer.setInterval(self.graph_widget_2.playback_speed)

        # Resume playback timers if necessary
        if self.graph_widget_1.is_playing:
            self.graph_widget_1.playback_timer.start(self.graph_widget_1.playback_speed)
        if self.graph_widget_2.is_playing:
            self.graph_widget_2.playback_timer.start(self.graph_widget_2.playback_speed)



    def link_graphs(self):
        # Connect playback speed sliders
        self.graph_widget_1.playback_speed_slider.valueChanged.connect(self.graph_widget_2.set_playback_speed)
        self.graph_widget_2.playback_speed_slider.valueChanged.connect(self.graph_widget_1.set_playback_speed)

        # Connect navigation scrollbars
        self.graph_widget_1.navigation_scrollbar.valueChanged.connect(self.graph_widget_2.set_navigation_position)
        self.graph_widget_2.navigation_scrollbar.valueChanged.connect(self.graph_widget_1.set_navigation_position)

        # Link plot widgets for synchronized zooming and panning
        self.graph_widget_1.link_plot_widget(self.graph_widget_2.plot_widget)
        self.graph_widget_2.link_plot_widget(self.graph_widget_1.plot_widget)

                # Link plot widgets for synchronized zooming and panning
        self.graph_widget_1.plot_widget.setXLink(self.graph_widget_2.plot_widget)
        self.graph_widget_1.plot_widget.setYLink(self.graph_widget_2.plot_widget)

    def unlink_graphs(self):
        # Safely disconnect playback speed sliders
        try:
            self.graph_widget_1.playback_speed_slider.valueChanged.disconnect(self.graph_widget_2.set_playback_speed)
            self.graph_widget_2.playback_speed_slider.valueChanged.disconnect(self.graph_widget_1.set_playback_speed)
        except TypeError:
            pass  # Handles case where signals are not connected

        # Safely disconnect navigation scrollbars
        try:
            self.graph_widget_1.navigation_scrollbar.valueChanged.disconnect(self.graph_widget_2.set_navigation_position)
            self.graph_widget_2.navigation_scrollbar.valueChanged.disconnect(self.graph_widget_1.set_navigation_position)
        except TypeError:
            pass

        # Unlink plot widgets
        self.graph_widget_1.unlink_plot_widget()
        self.graph_widget_2.unlink_plot_widget()

                # Unlink plot widgets
        self.graph_widget_1.plot_widget.setXLink(None)
        self.graph_widget_1.plot_widget.setYLink(None)


import qdarkstyle

if __name__ == "__main__":

    
    app = QApplication(sys.argv)

        # Apply the dark theme
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())


    from PyQt5.QtGui import QFont
    app.setFont(QFont('Segoe UI', 9))
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
