# tests/test_spectroscopic_data.py

import unittest
import numpy as np
from specml.data.spectroscopic_data import SpectroscopicData
from unittest.mock import patch, MagicMock
import logging

class TestSpectroscopicData(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.mock_wavenumbers = np.linspace(1000, 3500, 400)  # Wavenumbers from 1000 to 2000 cm^-1
        self.mock_data_2d = np.random.rand(10, 1000)  # 10 spectra, each with 1000 data points
        self.mock_data_3d = np.random.rand(5, 5, 1000)  # 5x5 image with spectra at each pixel
        self.mock_mask = np.random.choice(a=[False, True], size=(5, 5))
        self.mock_labels = np.random.randint(0, 3, size=(25,))  # 3 classes

        self.metadata = {'xpixels': 5, 'ypixels': 5}


    def test_init_with_2d_data(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers,
            metadata=self.metadata
        )
        self.assertEqual(spec_data.data.shape, self.mock_data_2d.shape)
        self.assertEqual(spec_data.wavenumbers.shape, self.mock_wavenumbers.shape)
        self.assertEqual(spec_data.xpixels, 5)
        self.assertEqual(spec_data.ypixels, 5)

    def test_init_with_3d_data(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_3d,
            wavenumbers=self.mock_wavenumbers,
            mask=self.mock_mask,
            metadata=self.metadata
        )
        self.assertNotEqual(spec_data.data.shape, self.mock_data_3d.shape)
        self.assertEqual(spec_data.wavenumbers.shape, self.mock_wavenumbers.shape)
        self.assertEqual(spec_data.xpixels, 5)
        self.assertEqual(spec_data.ypixels, 5)
        self.assertTrue(hasattr(spec_data, 'original_shape'))
        self.assertEqual(spec_data.original_shape, (5, 5))

    def test_copy(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        spec_data_copy = spec_data.copy()
        self.assertNotEqual(id(spec_data), id(spec_data_copy))
        np.testing.assert_array_equal(spec_data.data, spec_data_copy.data)
        np.testing.assert_array_equal(spec_data.wavenumbers, spec_data_copy.wavenumbers)

    def test_flatten(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_3d,
            wavenumbers=self.mock_wavenumbers,
            mask=self.mock_mask
        )
        flattened_spec_data = spec_data.flatten()
        expected_data_shape = (np.sum(self.mock_mask), 1000)
        self.assertEqual(flattened_spec_data.data.shape, expected_data_shape)
        self.assertEqual(flattened_spec_data.data.ndim, 2)

    def test_to_image(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_3d,
            wavenumbers=self.mock_wavenumbers,
            mask=self.mock_mask
        )
        flattened_spec_data = spec_data.flatten()
        image_spec_data = flattened_spec_data.to_image()
        self.assertEqual(image_spec_data.data.shape, self.mock_data_3d.shape)

    def test_adjust_min_to_zero(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d - 5,  # Ensure data has negative values
            wavenumbers=self.mock_wavenumbers
        )
        adjusted_spec_data = spec_data.adjust_min_to_zero()
        self.assertAlmostEqual(adjusted_spec_data.data.min(), 0.0)

    def test_normalize_vector(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        normalized_spec_data = spec_data.normalize(method='vector')
        norms = np.linalg.norm(normalized_spec_data.data, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms))

    def test_normalize_z_score(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        normalized_spec_data = spec_data.normalize(method='z_score')
        mean = np.mean(normalized_spec_data.data, axis=0)
        std = np.std(normalized_spec_data.data, axis=0)
        np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-7)
        np.testing.assert_allclose(std, np.ones_like(std), atol=1e-7)

    def test_keep_range(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        lower, upper = 1200, 1800
        new_spec_data = spec_data.keep_range(lower, upper)
        self.assertTrue(np.all(new_spec_data.wavenumbers >= lower))
        self.assertTrue(np.all(new_spec_data.wavenumbers <= upper))
        expected_num_points = np.sum((self.mock_wavenumbers >= lower) & (self.mock_wavenumbers <= upper))
        self.assertEqual(new_spec_data.data.shape[1], expected_num_points)

    def test_remove_range(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        lower, upper = 1200, 1800
        new_spec_data = spec_data.remove_range(lower, upper)
        self.assertTrue(np.all((new_spec_data.wavenumbers < lower) | (new_spec_data.wavenumbers > upper)))
        expected_num_points = np.sum((self.mock_wavenumbers < lower) | (self.mock_wavenumbers > upper))
        self.assertEqual(new_spec_data.data.shape[1], expected_num_points)

    def test_calculate_area(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        lower, upper = 1200, 1800
        areas = spec_data.calculate_area(lower, upper, method='trapz')
        self.assertEqual(areas.shape[0], spec_data.data.shape[0])
        self.assertTrue(np.all(areas >= 0))

    def test_apply_mask(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers,
            labels=self.mock_labels
        )
        mask = self.mock_labels == 1
        masked_spec_data = spec_data.apply_mask(mask)
        self.assertEqual(masked_spec_data.data.shape[0], np.sum(mask))
        np.testing.assert_array_equal(masked_spec_data.labels, spec_data.labels[mask])

    def test_mean_center(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_2d,
            wavenumbers=self.mock_wavenumbers
        )
        mean_centered_spec_data = spec_data.mean_center()
        mean_spectrum = np.mean(mean_centered_spec_data.data, axis=0)
        np.testing.assert_allclose(mean_spectrum, np.zeros_like(mean_spectrum), atol=1e-7)


    @patch('specml.data.loaders.DataLoader.load_data')
    def test_load_data_method(self, mock_load_data):
        # Mock the DataLoader.load_data method
        mock_load_data.return_value = {
            'data': self.mock_data_2d,
            'wavenumbers': self.mock_wavenumbers,
            'metadata': self.metadata,
            'mask': self.mock_mask.flatten(),
            'labels': self.mock_labels,
            'label_mapping': {0: 'Class0', 1: 'Class1', 2: 'Class2'},
            'annotations': {'info': 'Test annotations'}
        }
        spec_data = SpectroscopicData()
        spec_data.load_data('dummy_path')
        self.assertEqual(spec_data.data.shape, self.mock_data_2d.shape)
        self.assertEqual(spec_data.wavenumbers.shape, self.mock_wavenumbers.shape)
        self.assertEqual(spec_data.labels.shape, self.mock_labels.shape)
        self.assertEqual(spec_data.label_mapping, {0: 'Class0', 1: 'Class1', 2: 'Class2'})
        self.assertEqual(spec_data.annotations, {'info': 'Test annotations'})

    def test_get_spectrum_at(self):
        spec_data = SpectroscopicData(
            data=self.mock_data_3d,
            wavenumbers=self.mock_wavenumbers,
            metadata=self.metadata
        )
        x, y = 2, 3
        spectrum = spec_data.get_spectrum_at(x, y)
        expected_spectrum = self.mock_data_3d[y, x, :]
        np.testing.assert_array_equal(spectrum, expected_spectrum)

if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)
    logging.getLogger('specml').setLevel(logging.CRITICAL)
    unittest.main()
