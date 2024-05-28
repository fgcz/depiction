import unittest
from functools import cached_property

import numpy as np
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
        values = image.get_channel_array("B")
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

    def test_channel_names_when_set(self) -> None:
        self.assertListEqual(["Channel A"], self.mock_image.channel_names)

    def test_channel_names_when_not_set(self) -> None:
        self.mock_coords = {}
        self.assertListEqual(["0"], self.mock_image.channel_names)

    def test_get_channel_array_when_str_exists(self) -> None:
        values = self.mock_image.get_channel_array("Channel A")
        xarray.testing.assert_equal(
            DataArray([[2.0, 4], [6, 8], [10, 12]], dims=("y", "x"), coords={"c": "Channel A"}), values
        )

    def test_get_channel_array_when_str_missing(self) -> None:
        with self.assertRaises(KeyError):
            self.mock_image.get_channel_array("Channel C")

    def test_get_channel_array_when_list_size_1(self) -> None:
        values = self.mock_image.get_channel_array(["Channel A"])
        xarray.testing.assert_equal(self.mock_data, values)

    def test_get_channel_array_when_list_multiple(self) -> None:
        values = self.mock_image.get_channel_array(["Channel A", "Channel A"])
        expected = DataArray(
            [[[2.0, 2], [4, 4]], [[6, 6], [8, 8]], [[10, 10], [12, 12]]],
            dims=("y", "x", "c"),
            coords={"c": ["Channel A", "Channel A"]},
        )
        xarray.testing.assert_equal(expected, values)

    def test_get_channel_flat_array_when_str(self):
        self.mock_data[0, 1, 0] = 0
        self.mock_data[1, :, 0] = 0
        values = self.mock_image.get_channel_flat_array("Channel A")
        np.testing.assert_array_equal(np.array([2.0, 10, 12]), values.values)
        np.testing.assert_array_equal([0, 2, 2], values.coords["y"])
        np.testing.assert_array_equal([0, 0, 1], values.coords["x"])
        self.assertListEqual([(0, 0), (2, 0), (2, 1)], values.coords["i"].values.tolist())

    def test_get_channel_flat_array_when_list_multiple(self):
        self.mock_data[0, 1, 0] = 0
        self.mock_data[1, :, 0] = 0
        values = self.mock_image.get_channel_flat_array(["Channel A", "Channel A"])
        np.testing.assert_array_equal(np.array([[2.0, 10, 12], [2.0, 10, 12]]), values.values)
        np.testing.assert_array_equal([0, 2, 2], values.coords["y"])
        np.testing.assert_array_equal([0, 0, 1], values.coords["x"])
        np.testing.assert_array_equal(["Channel A", "Channel A"], values.coords["c"].values)
        self.assertListEqual([(0, 0), (2, 0), (2, 1)], values.coords["i"].values.tolist())

    def test_with_channel_names(self) -> None:
        image = self.mock_image.with_channel_names(channel_names=["New Channel Name"])
        self.assertListEqual(["New Channel Name"], image.channel_names)
        # TODO check equal contents

    def test_str(self) -> None:
        expected = "MultiChannelImage(size_y=3, size_x=2, n_channels=1)"
        self.assertEqual(expected, str(self.mock_image))


if __name__ == "__main__":
    unittest.main()
