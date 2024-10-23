import numpy as np
import pytest
import xarray
from xarray import DataArray

from depiction.persistence.format_ome_tiff import OmeTiff
from depiction.persistence.pixel_size import PixelSize


@pytest.fixture
def sample_data() -> DataArray:
    sizes = (2, 3, 4)
    return DataArray(
        np.arange(np.prod(sizes), dtype=float).reshape(sizes),
        dims=["c", "y", "x"],
        attrs={"pixel_size": PixelSize(10.0, 20.0, "micrometer")},
        coords={"c": ["a", "b"]},
    )


def test_round_trip(sample_data, tmp_path):
    out_path = tmp_path / "test.ome.tiff"

    # write the file
    OmeTiff.write(sample_data, out_path)
    assert out_path.stat().st_size > 0

    # read the file
    read_data = OmeTiff.read(out_path)

    # check the data
    xarray.testing.assert_identical(read_data, sample_data)
