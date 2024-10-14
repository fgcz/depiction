import pytest
import xarray
from xarray import DataArray

from depiction.image.feature_selection import FeatureSelectionIQR, select_features, FeatureSelectionCV
from depiction.image.multi_channel_image import MultiChannelImage


@pytest.fixture()
def image() -> MultiChannelImage:
    data = DataArray(
        [[[1.0, 2, 0.0], [1.0, 5, 0.5], [1.0, 10, 0.0], [1.0, 20, 0.5], [1.0, 30, 0.0], [1.0, 40, 0.5]]],
        dims=("y", "x", "c"),
        coords={"c": ["channel1", "channel2", "channel3"]},
    )
    return MultiChannelImage(
        data=data,
        is_foreground=xarray.ones_like(data.isel(c=0), dtype=bool).drop_vars("c"),
    )


def test_select_features_cv(image):
    fs = FeatureSelectionCV.model_validate(dict(n_features=2))
    selection = select_features(feature_selection=fs, image=image)
    assert selection == ["channel3", "channel2"]


def test_select_features_iqr(image):
    fs = FeatureSelectionIQR.model_validate(dict(n_features=2))
    selection = select_features(feature_selection=fs, image=image)
    assert selection == ["channel2", "channel3"]
