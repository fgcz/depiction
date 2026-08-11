import numpy as np
import pytest
import xarray
from xarray import DataArray

from depiction.image.ome_tiff import OmeTiff
from depiction_io.pixel_size import PixelSize


@pytest.fixture(params=[PixelSize(10.0, 20.0, "micrometer"), None], ids=["declared", "unknown"])
def sample_data(request) -> DataArray:
    sizes = (2, 3, 4)
    return DataArray(
        np.arange(np.prod(sizes), dtype=float).reshape(sizes),
        dims=["c", "y", "x"],
        attrs={"pixel_size": request.param},
        coords={"c": ["a", "b"]},
    )


def test_round_trip(sample_data, tmp_path):
    """Both cases, because an unknown pixel size has to survive as an unknown one.

    `read` used to build a `PixelSize(None, None, "micrometer")` from a file that declares no
    physical size, which is neither a size nor an admission that there isn't one.
    """
    out_path = tmp_path / "test.ome.tiff"

    # write the file
    OmeTiff.write(sample_data, out_path)
    assert out_path.stat().st_size > 0

    # read the file
    read_data = OmeTiff.read(out_path)

    # check the data
    xarray.testing.assert_identical(read_data, sample_data)
    assert read_data.attrs["pixel_size"] == sample_data.attrs["pixel_size"]
