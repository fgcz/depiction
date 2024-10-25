import pytest
import xarray
from pathlib import Path
from pytest_mock import MockerFixture
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence.format_ome_tiff import OmeTiff


@pytest.fixture
def mock_data() -> DataArray:
    """Dense mock data without any missing values."""
    return DataArray(
        [[[2.0, 5], [4, 5]], [[6, 5], [8, 5]], [[10, 5], [12, 5]]],
        dims=("y", "x", "c"),
        coords={"c": ["Channel A", "Channel B"]},
    )


def test_read_image(mocker: MockerFixture, mock_data: DataArray) -> None:
    mock_read = mocker.patch.object(OmeTiff, "read", return_value=mock_data)
    mock_foreground = xarray.ones_like(mock_data.isel(c=0), dtype=bool).drop_vars("c")
    mocker.patch.object(MultiChannelImage, "_compute_is_foreground", return_value=mock_foreground)
    image = OmeTiff.read_image(Path("test.ome.tiff"))
    xarray.testing.assert_equal(image.data_spatial, mock_data)
    mock_read.assert_called_once_with(Path("test.ome.tiff"))
    xarray.testing.assert_equal(image.fg_mask, mock_foreground)
