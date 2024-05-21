import unittest

import numpy as np
import xarray

from ionplotter.image.image_normalization import ImageNormalizationVariant, ImageNormalization


class TestImageNormalization(unittest.TestCase):
    def test_normalize_xarray_single_vec_norm(self) -> None:
        images = xarray.DataArray(
            data=[[[2, 0], [0, 2]], [[1, 1], [4, 1]], [[0, 0], [0, 0]]],
            dims=["y", "x", "c"],
        )
        norm_vec = ImageNormalization().normalize_xarray(images, variant=ImageNormalizationVariant.VEC_NORM)
        self.assertEqual(norm_vec.shape, (3, 2, 2))
        expected = xarray.DataArray(
            data=[[[1, 0], [0, 1]], [[0.707107, 0.707107], [0.970143, 0.242536]], [[np.nan, np.nan], [np.nan, np.nan]]],
            dims=["y", "x", "c"],
        )
        xarray.testing.assert_allclose(expected, norm_vec)

    def test_normalize_xarray_multiple_vec_norm(self) -> None:
        images = xarray.DataArray(
            data=[[[[2, 0]]], [[[0, 3]]]],
            dims=["whatever", "y", "x", "c"],
        )
        norm_vec = ImageNormalization().normalize_xarray(images, variant=ImageNormalizationVariant.VEC_NORM)
        self.assertEqual(norm_vec.shape, (2, 1, 1, 2))
        expected = xarray.DataArray(
            data=[[[[1, 0]]], [[[0, 1]]]],
            dims=["whatever", "y", "x", "c"],
            coords={"whatever": [0, 1]},
        )
        xarray.testing.assert_allclose(expected, norm_vec)


if __name__ == "__main__":
    unittest.main()
