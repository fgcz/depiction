import numpy as np
import pytest
from xarray import DataArray

from depiction.calibration.methods.calibration_method_constant_global_shift import CalibrationMethodConstantGlobalShift
from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from depiction.image import MultiChannelImage


@pytest.fixture
def reference_mz_values():
    """Fixture for reference m/z values."""
    return np.array([100.0, 200.0, 300.0])


@pytest.fixture
def sample_spectrum_data():
    """Fixture for sample spectrum data."""
    mz_arr = np.array([100.5, 200.5, 300.5])  # 0.5 shift from reference
    int_arr = np.array([1000.0, 2000.0, 3000.0])
    return mz_arr, int_arr


@pytest.fixture
def constant_shift_calibration(reference_mz_values):
    """Fixture for CalibrationMethodConstantGlobalShift instance."""
    return CalibrationMethodConstantGlobalShift(
        ref_mz_arr=reference_mz_values, max_distance=2.0, max_distance_unit="mz"
    )


@pytest.fixture
def sample_image_features():
    """Fixture for sample MultiChannelImage with known shift values."""
    # Create a 3x3 image with constant shift of 0.5
    data = np.full((3, 3, 1), 0.5)  # Using 0.5 as our known shift
    return MultiChannelImage.from_spatial(DataArray(data, dims=("y", "x", "c"), coords={"c": ["a"]}), bg_value=0)


def test_initialization(constant_shift_calibration, reference_mz_values):
    """Test initialization of CalibrationMethodConstantGlobalShift."""
    assert np.array_equal(constant_shift_calibration._ref_mz_arr, reference_mz_values)
    assert constant_shift_calibration._max_distance == 2.0
    assert constant_shift_calibration._max_distance_unit == "mz"


def test_extract_spectrum_features(constant_shift_calibration, sample_spectrum_data, mocker):
    """Test extract_spectrum_features method."""
    mz_arr, int_arr = sample_spectrum_data
    expected_distances = np.array([0.5, 0.5, 0.5])  # Simulated distances

    # Mock the ReferencePeakDistances.get_distances_max_peak_in_window method
    mocker.patch.object(ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=expected_distances)

    result = constant_shift_calibration.extract_spectrum_features(mz_arr, int_arr)

    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert np.array_equal(result.values, expected_distances)


def test_preprocess_image_features(constant_shift_calibration, sample_image_features):
    """Test preprocess_image_features method."""
    result = constant_shift_calibration.preprocess_image_features(sample_image_features)

    assert isinstance(result, MultiChannelImage)
    assert result.sizes == {"y": 3, "x": 3, "c": 1}  # Same spatial dimensions, single channel
    assert np.allclose(result.data_spatial, 0.5)  # Should contain the median value
    assert np.array_equal(result.fg_mask, sample_image_features.fg_mask)
    assert result.is_foreground_label == sample_image_features.is_foreground_label


def test_fit_spectrum_model(constant_shift_calibration):
    """Test fit_spectrum_model method."""
    features = DataArray([0.5], dims=["c"])
    result = constant_shift_calibration.fit_spectrum_model(features)

    assert isinstance(result, DataArray)
    assert result is features  # Should return the same object
    assert np.array_equal(result.values, features.values)


def test_apply_spectrum_model(constant_shift_calibration, sample_spectrum_data):
    """Test apply_spectrum_model method."""
    mz_arr, int_arr = sample_spectrum_data
    model_coef = DataArray([0.5], dims=["c"])  # Global shift of 0.5

    result_mz, result_int = constant_shift_calibration.apply_spectrum_model(mz_arr, int_arr, model_coef)

    assert isinstance(result_mz, np.ndarray)
    assert isinstance(result_int, np.ndarray)
    assert np.array_equal(result_mz, mz_arr - 0.5)  # Should subtract the shift
    assert np.array_equal(result_int, int_arr)  # Intensities should remain unchanged


def test_repr(constant_shift_calibration):
    """Test __repr__ method."""
    expected = "CalibrationMethodConstantGlobalShift(max_distance=2.0, max_distance_unit=mz)"
    assert repr(constant_shift_calibration) == expected


@pytest.mark.parametrize(
    "max_distance,max_distance_unit",
    [
        (1.0, "mz"),
        (5.0, "mz"),
        (2.0, "ppm"),
    ],
)
def test_different_initialization_parameters(reference_mz_values, max_distance, max_distance_unit):
    """Test initialization with different parameters."""
    calibration = CalibrationMethodConstantGlobalShift(
        ref_mz_arr=reference_mz_values, max_distance=max_distance, max_distance_unit=max_distance_unit
    )
    assert calibration._max_distance == max_distance
    assert calibration._max_distance_unit == max_distance_unit


def test_preprocess_image_features_with_nan(constant_shift_calibration):
    """Test preprocess_image_features with NaN values."""
    # Create data with some NaN values
    data = np.full((3, 3, 1), 0.5)
    data[0, 0, 0] = np.nan

    features = MultiChannelImage.from_spatial(DataArray(data, dims=("y", "x", "c"), coords={"c": ["a"]}), bg_value=0)

    result = constant_shift_calibration.preprocess_image_features(features)

    assert isinstance(result, MultiChannelImage)
    assert np.allclose(result.data_spatial, 0.5, equal_nan=True)  # Should handle NaN values correctly


def test_extract_spectrum_features_empty_arrays(constant_shift_calibration, mocker):
    """Test extract_spectrum_features with empty arrays."""
    mocker.patch.object(ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=np.array([]))

    result = constant_shift_calibration.extract_spectrum_features(np.array([]), np.array([]))
    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert len(result.values) == 0


def test_apply_spectrum_model_edge_cases(constant_shift_calibration):
    """Test apply_spectrum_model with edge cases."""
    # Test with empty arrays
    empty_mz = np.array([])
    empty_int = np.array([])
    model_coef = DataArray([0.5], dims=["c"])

    result_mz, result_int = constant_shift_calibration.apply_spectrum_model(empty_mz, empty_int, model_coef)
    assert len(result_mz) == 0
    assert len(result_int) == 0

    # Test with very small/large numbers
    small_mz = np.array([1e-10])
    large_int = np.array([1e10])
    result_mz, result_int = constant_shift_calibration.apply_spectrum_model(small_mz, large_int, model_coef)
    assert np.isfinite(result_mz).all()
    assert np.array_equal(result_int, large_int)
