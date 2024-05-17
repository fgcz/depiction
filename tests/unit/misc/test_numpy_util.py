import unittest

import numpy as np

import ionmapper.misc.numpy_util as numpy_util
from ionmapper.misc.numpy_util import NumpyUtil


class TestNumpyUtil(unittest.TestCase):
    def setUp(self):
        self.mock_full_array = np.array([10, 20, 30, 40, 50])

    def test_search_sorted_closest_when_internal_values(self):
        closest_indices = NumpyUtil.search_sorted_closest(self.mock_full_array, np.array([20, 33, 37]))
        np.testing.assert_array_equal([1, 2, 3], closest_indices)

    def test_search_sorted_closest_when_left_boundary(self):
        closest_indices = NumpyUtil.search_sorted_closest(self.mock_full_array, np.array([10, 11]))
        np.testing.assert_array_equal([0, 0], closest_indices)

    def test_search_sorted_closest_when_left_out_of_boundary(self):
        closest_indices = NumpyUtil.search_sorted_closest(self.mock_full_array, np.array([-2, 9]))
        np.testing.assert_array_equal([0, 0], closest_indices)

    def test_search_sorted_closest_when_right_boundary(self):
        closest_indices = NumpyUtil.search_sorted_closest(self.mock_full_array, np.array([49, 50]))
        np.testing.assert_array_equal([4, 4], closest_indices)

    def test_search_sorted_closest_when_right_out_of_boundary(self):
        closest_indices = NumpyUtil.search_sorted_closest(self.mock_full_array, np.array([51, 60]))
        np.testing.assert_array_equal([4, 4], closest_indices)

    def test_get_sorted_indices_within_distance_default_1(self):
        array = np.array([10, 20, 30, 40, 50])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 30, 10)
        np.testing.assert_array_equal([1, 2, 3], indices)

    def test_get_sorted_indices_within_distance_default_2(self):
        array = np.array([10, 20, 30, 40, 50])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 30, 9)
        np.testing.assert_array_equal([2], indices)

    def test_get_sorted_indices_when_input_empty(self):
        array = np.array([])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 30, 10)
        np.testing.assert_array_equal([], indices)

    def test_get_sorted_indices_when_output_empty(self):
        array = np.array([10, 20, 30, 40, 50])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 35, 2)
        np.testing.assert_array_equal([], indices)

    def test_get_sorted_indices_when_left_out_of_boundary(self):
        array = np.array([10, 20, 30, 40, 50])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 5, 10)
        np.testing.assert_array_equal([0], indices)

    def test_get_sorted_indices_when_right_out_of_boundary(self):
        array = np.array([10, 20, 30, 40, 50])
        indices = NumpyUtil.get_sorted_indices_within_distance(array, 55, 10)
        np.testing.assert_array_equal([4], indices)

    def test_get_first_index_when_numeric(self):
        array = np.array([10, 20.0, 30, 40, 50])
        self.assertEqual(1, numpy_util.get_first_index(array, 20))
        self.assertEqual(5, numpy_util.get_first_index(array, 60))

    def test_get_first_index_when_boolean(self):
        array = np.array([True, False, True, False, True])
        self.assertEqual(0, numpy_util.get_first_index(array, True))
        self.assertEqual(1, numpy_util.get_first_index(array, False))


if __name__ == "__main__":
    unittest.main()
