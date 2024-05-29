import unittest
from functools import cached_property

import numpy as np
import pandas as pd
import xarray
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage


class TestMultiChannelImage(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_coords = {"c": ["Channel A"]}

    @cached_property
    def mock_data(self) -> DataArray:
        return DataArray(
            [[[2.0], [4]], [[6], [8]], [[10], [12]]],
            dims=("y", "x", "c"),
            coords=self.mock_coords,
            attrs={"bg_value": 0},
        )

    @cached_property
    def mock_image(self) -> MultiChannelImage:
        return MultiChannelImage(data=self.mock_data)

    def test_from_numpy_sparse(self) -> None:
        values = np.array([[1, 2, 3], [4, 5, 6]])
        coordinates = np.array([[0, 0], [1, 1]])
        image = MultiChannelImage.from_numpy_sparse(
            values=values, coordinates=coordinates, channel_names=["A", "B", "C"]
        )
        self.assertListEqual(["A", "B", "C"], image.channel_names)
        values = image.data_spatial.sel(c="B")
        xarray.testing.assert_equal(
            DataArray([[2, 0], [0, 5]], dims=("y", "x"), coords={"c": "B"}, name="values"), values
        )

    def test_n_channels(self) -> None:
        self.assertEqual(1, self.mock_image.n_channels)

    def test_dtype(self) -> None:
        self.assertEqual(float, self.mock_image.dtype)

    def test_bg_value(self) -> None:
        self.assertEqual(0.0, self.mock_image.bg_value)

    def test_bg_mask_when_0(self) -> None:
        self.mock_data[1, :, :] = 0
        bg_mask = self.mock_image.bg_mask
        expected_bg_mask = DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
        xarray.testing.assert_equal(expected_bg_mask, bg_mask)

    def test_bg_mask_when_nan(self) -> None:
        self.mock_data[1, :, :] = np.nan
        self.mock_data.attrs["bg_value"] = np.nan
        bg_mask = self.mock_image.bg_mask
        expected_bg_mask = DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
        xarray.testing.assert_equal(expected_bg_mask, bg_mask)

    def test_dimensions(self):
        self.assertEqual((2, 3), self.mock_image.dimensions)

    def test_channel_names_when_set(self) -> None:
        self.assertListEqual(["Channel A"], self.mock_image.channel_names)

    def test_channel_names_when_not_set(self) -> None:
        self.mock_coords = {}
        self.assertListEqual(["0"], self.mock_image.channel_names)

    def test_data_spatial(self):
        xarray.testing.assert_identical(self.mock_data, self.mock_image.data_spatial)

    def test_data_flat(self):
        self.mock_data[0, 0, 0] = 0
        self.mock_data[1, 0, 0] = np.nan
        expected = DataArray(
            [[4., 8, 10, 12]],
            dims=("c", "i"),
            coords={"c": ["Channel A"],
                    "i": pd.MultiIndex.from_tuples([(0, 1), (1, 1), (2, 0), (2, 1)], names=("y", "x"))},
            attrs={"bg_value": 0}
        )
        xarray.testing.assert_identical(expected, self.mock_image.data_flat)

    def test_with_channel_names(self) -> None:
        image = self.mock_image.with_channel_names(channel_names=["New Channel Name"])
        self.assertListEqual(["New Channel Name"], image.channel_names)
        # TODO check equal contents

    def test_str(self) -> None:
        expected = "MultiChannelImage(size_y=3, size_x=2, n_channels=1)"
        self.assertEqual(expected, str(self.mock_image))


if __name__ == "__main__":
    unittest.main()
