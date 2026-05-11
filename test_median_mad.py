import unittest
import math

class MockNumpyArray:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data),) if isinstance(data, list) else ()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, mask):
        # Very simple boolean masking simulation for median_mad logic: np.isfinite
        if isinstance(mask, MockNumpyArray) and len(mask.data) > 0 and isinstance(mask.data[0], bool):
            return MockNumpyArray([d for d, m in zip(self.data, mask.data) if m])
        if isinstance(mask, MockNumpyArray) and len(mask.data) == 0:
            return MockNumpyArray([])
        return MockNumpyArray(self.data[mask])

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return MockNumpyArray([x - other for x in self.data])
        raise NotImplementedError()

class MockNumpy:
    def array(self, data):
        return MockNumpyArray(data)

    def isfinite(self, arr):
        return MockNumpyArray([not math.isinf(x) and not math.isnan(x) for x in arr.data])

    def abs(self, arr):
        return MockNumpyArray([abs(x) for x in arr.data])

    def median(self, arr):
        sorted_data = sorted(arr.data)
        n = len(sorted_data)
        if n == 0:
            return float('nan')
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        return sorted_data[mid]

    @property
    def nan(self):
        return float('nan')

import sys
import unittest.mock as mock

class TestMedianMad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Store original modules to restore later
        cls.original_numpy = sys.modules.get('numpy')
        cls.original_scipy = sys.modules.get('scipy')

        # Mock the modules
        sys.modules['numpy'] = MockNumpy()
        mock_scipy = mock.MagicMock()
        sys.modules['scipy'] = mock_scipy
        sys.modules['scipy.spatial'] = mock_scipy.spatial
        sys.modules['scipy.spatial.distance'] = mock_scipy.spatial.distance

        # Import the function after mocking
        global median_mad
        from eeg_utils import median_mad

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        if cls.original_numpy is None:
            del sys.modules['numpy']
        else:
            sys.modules['numpy'] = cls.original_numpy

        if cls.original_scipy is None:
            del sys.modules['scipy']
            if 'scipy.spatial' in sys.modules: del sys.modules['scipy.spatial']
            if 'scipy.spatial.distance' in sys.modules: del sys.modules['scipy.spatial.distance']
        else:
            sys.modules['scipy'] = cls.original_scipy
    def test_basic(self):
        data = [1, 2, 3, 4, 5]
        med, mad = median_mad(data)
        self.assertEqual(med, 3.0)
        self.assertEqual(mad, 1.0)

    def test_with_infinities_and_nans(self):
        data = [1, 2, float('inf'), 4, 5, float('nan')]
        med, mad = median_mad(data)
        self.assertEqual(med, 3.0)
        self.assertEqual(mad, 1.5)

    def test_empty(self):
        data = []
        med, mad = median_mad(data)
        self.assertTrue(math.isnan(med))
        self.assertTrue(math.isnan(mad))

if __name__ == '__main__':
    unittest.main()
