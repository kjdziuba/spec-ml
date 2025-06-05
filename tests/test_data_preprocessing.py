# tests/test_spectroscopic_data.py

import unittest
import numpy as np
from specml.data.spectroscopic_data import SpectroscopicData
from unittest.mock import patch, MagicMock

class TestSpectroscopicData(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.mock_wavenumbers = np.linspace(1000, 3500, 400)  # Wavenumbers from 1000 to 2000 cm^-1
        self.mock_data_2d = np.random.rand(10, 1000)  # 10 spectra, each with 1000 data points
        self.mock_data_3d = np.random.rand(5, 5, 1000)  # 5x5 image with spectra at each pixel
        self.mock_mask = np.random.choice(a=[False, True], size=(5, 5))
        self.mock_labels = np.random.randint(0, 3, size=(25,))  # 3 classes

        self.metadata = {'xpixels': 5, 'ypixels': 5}