from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider

class SliderWidget(QWidget):
    def __init__(self, label, min_val, max_val, init_val):
        super().__init__()
        self.initUI(label, min_val, max_val, init_val)

    def initUI(self, label, min_val, max_val, init_val):
        layout = QVBoxLayout()

        # Label for the slider
        self.label = QLabel(f"{label}: {init_val}")
        layout.addWidget(self.label)

        # Slider for adjusting values
        self.slider = QSlider()
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(init_val)
        self.slider.valueChanged.connect(self.update_label)

        layout.addWidget(self.slider)
        self.setLayout(layout)

    def update_label(self, value):
        self.label.setText(f"{self.label.text().split(':')[0]}: {value}")
