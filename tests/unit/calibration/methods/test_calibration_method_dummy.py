import numpy as np
import pytest
import xarray
from xarray import DataArray

from depiction.calibration.methods.calibration_method_dummy import CalibrationMethodDummy
from depiction.image import MultiChannelImage


@pytest.fixture
def dummy_calibration():
    """Fixture to create a CalibrationMethodDummy instance."""
    return CalibrationMethodDummy()


@pytest.fixture
def sample_spectrum_data():
    """Fixture to create sample spectrum data."""
    mz_arr = np.array([100.0, 200.0, 300.0])
    int_arr = np.array([1000.0, 2000.0, 3000.0])
    return mz_arr, int_arr


@pytest.fixture
def sample_image_features():
    """Fixture to create sample MultiChannelImage."""
    data = np.random.rand(10, 10, 3)  # 10x10 image with 3 channels
    return MultiChannelImage.from_spatial(
        DataArray(data, dims=("y", "x", "c"), coords={"c": ["a", "b", "c"]}), bg_value=0
    )


def test_extract_spectrum_features(dummy_calibration, sample_spectrum_data):
    """Test extract_spectrum_features returns expected DataArray."""
    mz_arr, int_arr = sample_spectrum_data
    result = dummy_calibration.extract_spectrum_features(mz_arr, int_arr)

    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert result.values.tolist() == [0]


def test_preprocess_image_features(dummy_calibration, sample_image_features):
    """Test preprocess_image_features returns input unchanged."""
    result = dummy_calibration.preprocess_image_features(sample_image_features)

    assert isinstance(result, MultiChannelImage)
    xarray.testing.assert_identical(result.data_spatial, sample_image_features.data_spatial)
    assert result is sample_image_features  # Should return the same object


def test_fit_spectrum_model(dummy_calibration):
    """Test fit_spectrum_model returns input unchanged."""
    features = DataArray([1.0, 2.0], dims=["c"])
    result = dummy_calibration.fit_spectrum_model(features)

    assert isinstance(result, DataArray)
    assert np.array_equal(result.values, features.values)
    assert result is features  # Should return the same object


def test_apply_spectrum_model(dummy_calibration, sample_spectrum_data):
    """Test apply_spectrum_model returns input arrays unchanged."""
    mz_arr, int_arr = sample_spectrum_data
    model_coef = DataArray([0], dims=["c"])

    result_mz, result_int = dummy_calibration.apply_spectrum_model(mz_arr, int_arr, model_coef)

    assert isinstance(result_mz, np.ndarray)
    assert isinstance(result_int, np.ndarray)
    assert np.array_equal(result_mz, mz_arr)
    assert np.array_equal(result_int, int_arr)
    assert result_mz is mz_arr  # Should return the same object
    assert result_int is int_arr  # Should return the same object


def test_repr(dummy_calibration):
    """Test __repr__ returns expected string."""
    expected = "CalibrationMethodDummy()"
    assert repr(dummy_calibration) == expected


@pytest.mark.parametrize(
    "input_data",
    [
        (np.array([]), np.array([])),  # Empty arrays
        (np.array([1.0]), np.array([1.0])),  # Single element
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])),  # Multiple elements
    ],
)
def test_apply_spectrum_model_with_different_sizes(dummy_calibration, input_data):
    """Test apply_spectrum_model with different input array sizes."""
    mz_arr, int_arr = input_data
    model_coef = DataArray([0], dims=["c"])

    result_mz, result_int = dummy_calibration.apply_spectrum_model(mz_arr, int_arr, model_coef)

    assert np.array_equal(result_mz, mz_arr)
    assert np.array_equal(result_int, int_arr)
