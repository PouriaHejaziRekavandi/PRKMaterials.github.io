import unittest
import numpy as np
from eeg_utils import to_array, median_mad

class DummyData:
    def __init__(self, data):
        self.data = data

class TestEegUtils(unittest.TestCase):

    def test_to_array_with_data_attr(self):
        obj = DummyData([1, 2, 3])
        arr = to_array(obj)
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape, (3, 1))

    def test_to_array_1d_input(self):
        arr = to_array([1, 2, 3])
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape, (3, 1))

    def test_to_array_2d_input(self):
        arr = to_array([[1, 2], [3, 4]])
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertEqual(arr.ndim, 2)
        self.assertEqual(arr.shape, (2, 2))

    def test_median_mad_normal(self):
        data = [1, 2, 3, 4, 5]
        med, mad = median_mad(data)
        self.assertEqual(med, 3.0)
        self.assertEqual(mad, 1.0)

    def test_median_mad_with_nan(self):
        data = [1, 2, np.nan, 4, 5]
        med, mad = median_mad(data)
        self.assertEqual(med, 3.0)
        self.assertEqual(mad, 1.5)

    def test_median_mad_empty(self):
        data = []
        med, mad = median_mad(data)
        self.assertTrue(np.isnan(med))
        self.assertTrue(np.isnan(mad))

    def test_median_mad_all_nan(self):
        data = [np.nan, np.nan]
        med, mad = median_mad(data)
        self.assertTrue(np.isnan(med))
        self.assertTrue(np.isnan(mad))

if __name__ == '__main__':
    unittest.main()
