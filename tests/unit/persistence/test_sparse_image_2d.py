import unittest
import warnings
from functools import cached_property
from unittest.mock import patch, ANY

import numpy as np

from ionmapper.image.sparse_image_2d import SparseImage2d


class TestSparseImage2d(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("error")
        self.mock_values = np.array([[2], [4], [6]], dtype=float)
        self.mock_coordinates = np.array([[0, 0], [1, 1], [1, 0]])

    @cached_property
    def mock_image(self) -> SparseImage2d:
        return SparseImage2d(values=self.mock_values, coordinates=self.mock_coordinates)

    def test_n_nonzero(self):
        self.assertEqual(3, self.mock_image.n_nonzero)

    def test_n_channels(self):
        self.assertEqual(1, self.mock_image.n_channels)

    def test_sparse_values(self):
        np.testing.assert_array_equal(self.mock_values, self.mock_image.sparse_values)

    def test_sparse_coordinates(self):
        np.testing.assert_array_equal(self.mock_coordinates, self.mock_image.sparse_coordinates)

    def test_dtype(self):
        self.assertEqual(float, self.mock_image.dtype)

    def test_offset(self):
        self.mock_coordinates += [3, 5]
        np.testing.assert_array_equal([3, 5], self.mock_image.offset)

    def test_dimensions(self):
        self.mock_coordinates = np.array([[0, 0], [4, 1], [1, 7]])
        self.assertEqual((5, 8), self.mock_image.dimensions)

    def test_channel_names_when_provided(self):
        self.mock_values = np.array([[2, 3], [4, 5], [6, 7]])
        mock_image = SparseImage2d(
            values=self.mock_values,
            coordinates=self.mock_coordinates,
            channel_names=["A", "B"],
        )
        self.assertListEqual(["A", "B"], mock_image.channel_names)

    def test_channel_names_when_default(self):
        self.mock_values = np.array([[2, 3], [4, 5], [6, 7]])
        self.assertListEqual(["Channel 0", "Channel 1"], self.mock_image.channel_names)

    def test_get_dense_array(self):
        dense_values = self.mock_image.get_dense_array()
        np.testing.assert_array_equal(np.array([[[0], [4]], [[2], [6]]]), dense_values)

    def test_get_dense_array_when_offset(self):
        self.mock_coordinates[:, 0] += 3
        self.mock_coordinates[:, 1] += 5
        dense_values = self.mock_image.get_dense_array()
        np.testing.assert_array_equal(np.array([[[0], [4]], [[2], [6]]]), dense_values)

    def test_get_dense_array_when_multichannel(self):
        self.mock_values = np.array([[2, 3], [4, 5], [6, 7]])
        dense_values = self.mock_image.get_dense_array()
        np.testing.assert_array_equal(np.array([[[0, 0], [4, 5]], [[2, 3], [6, 7]]]), dense_values)

    def test_get_dense_array_when_bg_value(self):
        dense_array = self.mock_image.get_dense_array(bg_value=1234)
        np.testing.assert_array_equal(np.array([[[1234], [4]], [[2], [6]]]), dense_array)

    def test_get_dense_array_when_bg_value_and_offset(self):
        self.mock_coordinates[:, 0] += 3
        self.mock_coordinates[:, 1] += 5
        dense_array = self.mock_image.get_dense_array(bg_value=1234)
        np.testing.assert_array_equal(np.array([[[1234], [4]], [[2], [6]]]), dense_array)

    def test_get_single_channel_dense_array_when_dtype_float(self):
        for bg_value in (0, 2.5, np.nan):
            with self.subTest(bg_value=bg_value):
                dense_values = self.mock_image.get_single_channel_dense_array(i_channel=0, bg_value=bg_value)
                np.testing.assert_array_equal(np.array([[bg_value, 4], [2, 6]]), dense_values)

    def test_get_single_channel_dense_array_when_dtype_int(self):
        for bg_value in (0, 2, np.nan):
            with self.subTest(bg_value=bg_value):
                self.mock_values = self.mock_values.astype(int)
                dense_values = self.mock_image.get_single_channel_dense_array(i_channel=0, bg_value=bg_value)
                if np.isfinite(bg_value):
                    self.assertEqual(np.dtype(int), dense_values.dtype)
                np.testing.assert_array_equal(np.array([[bg_value, 4], [2, 6]]), dense_values)

    def test_retain_channels(self):
        sparse_values_full = np.array([[4.0, 5, 6, 7], [4, 5, 6, 7], [8, 8, 8, 8], [9, 9, 9, 9]])
        coordinates = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        channel_names = ["A", "B", "C", "D"]

        image = SparseImage2d(
            values=sparse_values_full,
            coordinates=coordinates,
            channel_names=channel_names,
        )

        retained = image.retain_channels([0, 2])
        self.assertEqual(["A", "C"], retained.channel_names)
        np.testing.assert_array_equal(coordinates, retained.sparse_coordinates)
        np.testing.assert_array_equal(np.array([[4.0, 6], [4, 6], [8, 8], [9, 9]]), retained.sparse_values)

    @patch("ionmapper.image.sparse_image_2d.matplotlib.pyplot")
    @patch("seaborn.color_palette")
    def test_save_single_channel_image(self, mock_color_palette, mock_matplotlib_pyplot):
        self.mock_image.save_single_channel_image(0, "path")
        mock_matplotlib_pyplot.imsave.assert_called_once_with(
            "path", ANY, cmap=mock_color_palette.return_value, origin="lower"
        )
        np.testing.assert_equal(
            np.array([[0, 4], [2, 6]]),
            mock_matplotlib_pyplot.imsave.call_args[0][1],
        )

    def test_to_dense_xarray(self):
        result = self.mock_image.to_dense_xarray()
        self.assertListEqual(["Channel 0"], list(result.coords["c"]))
        np.testing.assert_array_equal(np.array([[[0], [4]], [[2], [6]]]), result.values)
        self.assertTupleEqual(("y", "x", "c"), result.dims)
        self.assertListEqual([0, 1], list(result.coords["x"]))
        self.assertListEqual([0, 1], list(result.coords["y"]))

    def test_to_dense_xarray_when_bg_value_nan(self):
        result = self.mock_image.to_dense_xarray(bg_value=np.nan)
        np.testing.assert_array_equal(np.array([[[np.nan], [4]], [[2], [6]]]), result.values)
        self.assertListEqual(["Channel 0"], list(result.coords["c"]))
        self.assertTupleEqual(("y", "x", "c"), result.dims)
        self.assertListEqual([0, 1], list(result.coords["x"]))
        self.assertListEqual([0, 1], list(result.coords["y"]))

    def test_combine_in_parallel(self):
        coordinates = np.array([[0, 0], [1, 1], [1, 0]])
        image1 = SparseImage2d(
            values=np.array([[2, 3], [2, 4], [2, 5]]),
            coordinates=coordinates,
            channel_names=["A", "B"],
        )
        image2 = SparseImage2d(
            values=np.array([[12, 13], [12, 14], [12, 15]]),
            coordinates=coordinates,
            channel_names=["B", "C"],
        )
        image3 = SparseImage2d(
            values=np.array([[22, 23], [22, 24], [22, 25]]),
            coordinates=coordinates,
            channel_names=["D", "E"],
        )
        combined = SparseImage2d.combine_in_parallel(images=[image1, image2, image3])
        self.assertEqual(6, combined.n_channels)
        np.testing.assert_array_equal(
            np.array([[2, 12, 22, 3, 13, 23], [2, 12, 22, 4, 14, 24], [2, 12, 22, 5, 15, 25]]),
            combined.sparse_values,
        )

    def test_from_dense_array(self):
        dense_array = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        image = SparseImage2d.from_dense_array(dense_values=dense_array, offset=np.array([0, 0, 0]))
        np.testing.assert_array_equal(np.array([[1], [2], [3], [4], [5], [6]]), image.sparse_values)
        np.testing.assert_array_equal(
            np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=int),
            image.sparse_coordinates,
            strict=True,
        )

    def test_from_dense_array_when_offset(self):
        dense_array = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        image = SparseImage2d.from_dense_array(dense_values=dense_array, offset=np.array([3, 5, 0]))
        np.testing.assert_array_equal(np.array([[1], [2], [3], [4], [5], [6]]), image.sparse_values)
        np.testing.assert_array_equal(
            np.array([[3, 5], [4, 5], [5, 5], [3, 6], [4, 6], [5, 6]], dtype=int),
            image.sparse_coordinates,
            strict=True,
        )

    def test_from_dense_array_when_offset_is_float_array_of_ints(self):
        dense_array = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        image = SparseImage2d.from_dense_array(dense_values=dense_array, offset=np.array([3.0, 5.0, 0.0]))
        np.testing.assert_array_equal(np.array([[1], [2], [3], [4], [5], [6]]), image.sparse_values)
        # This check ensures that the coordinates are still integers, as it can cause errors if this is not the case
        np.testing.assert_array_equal(
            np.array([[3, 5], [4, 5], [5, 5], [3, 6], [4, 6], [5, 6]], dtype=int),
            image.sparse_coordinates,
            strict=True,
        )

    def test_from_dense_array_when_channel_names(self):
        dense_array = np.array([[[1], [2], [3]], [[4], [5], [6]]])
        image = SparseImage2d.from_dense_array(
            dense_values=dense_array, offset=np.array([0, 0, 0]), channel_names=["A"]
        )
        self.assertListEqual(["A"], image.channel_names)

    def test_with_channel_names(self):
        image = self.mock_image.with_channel_names(["X"])
        self.assertListEqual(["X"], image.channel_names)
        np.testing.assert_array_equal(self.mock_image.sparse_values, image.sparse_values)
        np.testing.assert_array_equal(self.mock_image.sparse_coordinates, image.sparse_coordinates)

    def test_str(self):
        self.assertEqual(
            "SparseImage2d with n_nonzero=3, n_channels=1, offset=(0, 0)",
            str(self.mock_image),
        )


if __name__ == "__main__":
    unittest.main()
