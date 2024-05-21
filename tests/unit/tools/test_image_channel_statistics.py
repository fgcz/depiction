import unittest
import xarray
import polars as pl
import polars.testing
from ionplotter.tools.image_channel_statistics import ImageChannelStatistics


class TestImageChannelStatistics(unittest.TestCase):
    def test_compute_xarray(self) -> None:
        # dummy image with channel (x, y, c) shape
        image = xarray.DataArray(
            data=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            dims=["c", "y", "x"],
            coords={"c": ["a", "b"]},
        )
        result = ImageChannelStatistics.compute_xarray(image)
        expected_result = pl.DataFrame(
            {
                "channel": ["a", "b"],
                "int_mean": [2.5, 6.5],
                "int_std": [1.118034, 1.118034],
                "int_sum": [10, 26],
                "int_p25": [1.75, 5.75],
                "int_p50": [2.5, 6.5],
                "int_p75": [3.25, 7.25],
                "int_min": [1, 5],
                "int_max": [4, 8],
            }
        )
        pl.testing.assert_frame_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
