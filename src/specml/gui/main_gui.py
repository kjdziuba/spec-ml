import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider, QHBoxLayout, QLineEdit, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from specml.data.spectroscopic_data import SpectroscopicData
from specml.gui.gui_plotting import plot_image_over_wavenumber_gui, plot_dim_reduction_result
from matplotlib.backend_bases import MouseButton
from specml.feature_engineering.dim_reduction import apply_pca
import numpy as np

class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = None  # Placeholder for loaded data
        self.reduced_data = None  # Placeholder for reduced (PCA/UMAP) data
        self.labels = None  # Placeholder for clustering labels (if any)
        self.wavenumbers = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Spectroscopic Data Analysis")
        self.setGeometry(100, 100, 1024, 768)  # Set window size

        # Set up main layout
        main_layout = QVBoxLayout()

        # Load data button with a pleasant style
        self.load_button = QPushButton("Load Data")
        self.load_button.setStyleSheet("background-color: lightblue; font-size: 14px; padding: 10px;")
        self.load_button.clicked.connect(self.load_data)
        main_layout.addWidget(self.load_button)

        # Slider for wavenumbers
        slider_layout = QHBoxLayout()
        self.slider_label = QLabel('Wavenumber: 1650')
        slider_layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(950)
        self.slider.setMaximum(3550)
        self.slider.setValue(1650)  # Default wavenumber
        self.slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.slider)
        main_layout.addLayout(slider_layout)

        # Plot canvas for image
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        main_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()


        # Preprocessing Options
        preprocess_layout = QVBoxLayout()
        self.area_range_label = QLabel('Area Range (e.g., 1600, 1700):')
        preprocess_layout.addWidget(self.area_range_label)

        self.area_range_input = QLineEdit('1600, 1700')
        preprocess_layout.addWidget(self.area_range_input)

        self.area_threshold_label = QLabel('Area Threshold (e.g., 1.5):')
        preprocess_layout.addWidget(self.area_threshold_label)

        self.area_threshold_input = QLineEdit('1.5')
        preprocess_layout.addWidget(self.area_threshold_input)

        # Preprocess Button
        self.preprocess_button = QPushButton("Run Preprocessing")
        self.preprocess_button.setStyleSheet("background-color: lightgreen; font-size: 14px; padding: 10px;")
        self.preprocess_button.clicked.connect(self.run_preprocessing)
        preprocess_layout.addWidget(self.preprocess_button)

        main_layout.addLayout(preprocess_layout)

        # Add a separate canvas for the spectrum plot
        self.spectrum_canvas = FigureCanvas(Figure(figsize=(5, 1)))
        main_layout.addWidget(self.spectrum_canvas)
        self.spectrum_ax = self.spectrum_canvas.figure.subplots()

        # Connect click event to the plot
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Add a separate canvas for the dimensionality reduction plot
        self.dr_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        main_layout.addWidget(self.dr_canvas)
        self.dr_ax = self.dr_canvas.figure.subplots()

        # Add a button for applying dimensionality reduction (PCA/UMAP)
        self.dimensionality_button = QPushButton("Run Dimensionality Reduction")
        self.dimensionality_button.setStyleSheet("background-color: lightcoral; font-size: 14px; padding: 10px;")
        self.dimensionality_button.clicked.connect(self.apply_dimensionality_reduction)
        main_layout.addWidget(self.dimensionality_button)


        # Connect click event to the plot
        self.dr_canvas.mpl_connect('button_press_event', self.on_dr_click)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Set main window layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_data(self):
        # Open a file dialog to load the data
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.dmt *.csv)")

        if file_path:
            # Show loading dialog
            self.loading_dialog = QProgressDialog("Loading data...", None, 0, 0, self)
            self.loading_dialog.setWindowModality(Qt.WindowModal)
            self.loading_dialog.show()

            # Load data with a timer to keep the UI responsive
            QTimer.singleShot(100, lambda: self._load_data(file_path))

    def _load_data(self, file_path):
        # Simulate loading data
        self.data = SpectroscopicData(file_path)
        self.wavenumbers = self.data.wavenumbers  # Assume wavenumbers are part of the data
        self.plot_image(1650)  # Plot image over default wavenumber

        # Close loading dialog
        self.loading_dialog.close()
        print(f"Data loaded: {file_path}")

    def plot_image(self, wavenumber):
        if self.data is not None:
            # Clear the previous plot
            self.ax.clear()

            # Plot the image over the given wavenumber using the GUI plotting function
            plot_image_over_wavenumber_gui(self.data, target_wavenumber=wavenumber, ax=self.ax)

            # Redraw the canvas with the updated plot
            self.canvas.draw()

    def update_plot(self, value):
        # Update the label with the current wavenumber
        self.slider_label.setText(f'Wavenumber: {value}')

        # Update the plot dynamically based on the slider value
        self.plot_image(value)

    def run_preprocessing(self):
        if self.data is None:
            return

        # Retrieve values from input fields
        area_range = [int(x.strip()) for x in self.area_range_input.text().split(',')]
        area_threshold = float(self.area_threshold_input.text())

        # Run the preprocessing function (example with user-specified area range and threshold)
        self.data.easy_preprocess(area_range=area_range, area_threshold=area_threshold)
        print(f"Preprocessing complete with area_range={area_range} and area_threshold={area_threshold}")
        self.plot_image(1650)  # Replot after preprocessing


    def on_click(self, event):
        """
        Handle click events on the plot. Extract the pixel coordinates and plot
        the corresponding spectrum (wavenumbers vs absorbance) for the clicked pixel.
        """
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            print(f"Clicked on pixel: ({x}, {y})")

            # Extract the spectrum for the clicked pixel
            if self.data is not None and self.data.data.ndim == 3:
                spectrum = self.data.data[y, x, :]  # Extract the spectrum at (y, x)
                self.plot_wavenumbers(spectrum)

    def plot_wavenumbers(self, spectrum):
        """
        Plot the wavenumbers vs absorbance for a given spectrum (clicked pixel).
        This plot is displayed in the spectrum canvas in the same window.
        """
        # Clear the previous spectrum plot
        self.spectrum_ax.clear()

        # Plot the spectrum on the second canvas
        self.spectrum_ax.plot(self.wavenumbers, spectrum)
        self.spectrum_ax.set_title("Spectrum for Selected Pixel")
        self.spectrum_ax.set_xlabel("Wavenumbers")
        self.spectrum_ax.set_ylabel("Absorbance")

        # Redraw the spectrum canvas
        self.spectrum_canvas.draw()


    def on_dr_click(self, event):
        """
        Handle click events on the dimensionality reduction plot. 
        Highlights the corresponding pixel in the image.
        """
        if event.inaxes == self.dr_ax and self.reduced_data is not None:
            # Get the index of the closest point clicked
            distances = np.linalg.norm(self.reduced_data - [event.xdata, event.ydata], axis=1)
            clicked_index = np.argmin(distances)
            print(f"Clicked on reduced point: {clicked_index}")

            # Find corresponding pixel in the original image
            ypixels, xpixels = self.data.ypixels, self.data.xpixels
            flat_index = clicked_index  # Assuming flat index maps to original
            y, x = divmod(flat_index, xpixels)  # Reconvert to (y, x)

            # Highlight the pixel in the original image
            self.ax.clear()
            plot_image_over_wavenumber_gui(self.data, target_wavenumber=self.slider.value(), ax=self.ax)
            self.ax.plot(x, y, 'ro', markersize=10)  # Mark the clicked pixel
            self.canvas.draw()

    def plot_dim_reduction(self, dim_result, labels=None):
        """
        Plot the dimensionality reduction result (e.g., PCA or UMAP).
        """
        self.dr_ax.clear()

        # Plot the dimensionality reduction result with optional labels
        plot_dim_reduction_result(dim_result, labels=labels, ax=self.dr_ax)

        # Redraw the dimensionality reduction plot
        self.dr_canvas.draw()

    def apply_dimensionality_reduction(self):
        if self.data is None:
            return

        # Apply PCA (you can also apply UMAP here)
        pca_result = apply_pca(self.data, n_components=2)

        # Store the reduced data for later reference
        self.reduced_data = pca_result.reduced_data

        # Plot the dimensionality reduction result
        self.plot_dim_reduction(pca_result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainGUI()
    main_window.show()
    sys.exit(app.exec_())
