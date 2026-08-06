import numpy as np
import pytest
from xarray import DataArray

from depiction.calibration.methods.calibration_method_regress_shift import CalibrationMethodRegressShift
from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances
from depiction.image import MultiChannelImage
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.tools.calibrate.spatial_smoothing_config import SpatialSmoothingType


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
def regress_shift_calibration(reference_mz_values):
    """Fixture for CalibrationMethodRegressShift instance."""
    return CalibrationMethodRegressShift(
        ref_mz_arr=reference_mz_values,
        max_distance=2.0,
        max_distance_unit="mz",
        model_type="linear",
        model_unit="mz",
        input_smoothing=None,
        min_points=3,
        peak_filtering=None,
    )


@pytest.fixture
def sample_image_features():
    """Fixture for sample MultiChannelImage."""
    data = np.full((3, 3, 3), 0.5)  # 3 channels of shift data
    return MultiChannelImage.from_spatial(
        DataArray(data, dims=("y", "x", "c"), coords={"c": ["a", "b", "c"]}), bg_value=0
    )


def test_initialization(regress_shift_calibration, reference_mz_values):
    """Test initialization of CalibrationMethodRegressShift."""
    assert np.array_equal(regress_shift_calibration._ref_mz_arr, reference_mz_values)
    assert regress_shift_calibration._max_distance == 2.0
    assert regress_shift_calibration._max_distance_unit == "mz"
    assert regress_shift_calibration._model_type == "linear"
    assert regress_shift_calibration._model_unit == "mz"
    assert regress_shift_calibration._min_points == 3
    assert regress_shift_calibration._input_smoothing is None
    assert regress_shift_calibration._peak_filtering is None


def test_extract_spectrum_features_mz_unit(regress_shift_calibration, sample_spectrum_data, mocker):
    """Test extract_spectrum_features method with mz unit."""
    mz_arr, int_arr = sample_spectrum_data
    initial_distances = np.array([0.5, 0.5, 0.5])
    final_distances = np.array([0.4, 0.5, 0.6])  # Simulated varying distances

    # Mock both calls to ReferencePeakDistances
    mock_distances = mocker.patch.object(
        ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=initial_distances
    )
    mock_nearest = mocker.patch.object(
        ReferencePeakDistances, "get_distances_nearest", return_value=final_distances - np.median(initial_distances)
    )

    result = regress_shift_calibration.extract_spectrum_features(mz_arr, int_arr)

    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert np.allclose(result.values, final_distances)
    mock_distances.assert_called_once()
    mock_nearest.assert_called_once()


def test_extract_spectrum_features_ppm_unit(mocker, reference_mz_values):
    """Test extract_spectrum_features method with ppm unit."""
    calibration = CalibrationMethodRegressShift(
        ref_mz_arr=reference_mz_values,
        max_distance=2.0,
        max_distance_unit="mz",
        model_type="linear",
        model_unit="ppm",
        input_smoothing=None,
    )

    mz_arr = np.array([100.5, 200.5, 300.5])
    int_arr = np.array([1000.0, 2000.0, 3000.0])
    distances = np.array([0.5, 0.5, 0.5])

    with (
        mocker.patch.object(ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=distances),
        mocker.patch.object(ReferencePeakDistances, "get_distances_nearest", return_value=distances),
    ):
        result = calibration.extract_spectrum_features(mz_arr, int_arr)

        assert isinstance(result, DataArray)
        assert result.dims == ("c",)
        # Check PPM conversion
        expected_ppm = distances / reference_mz_values * 1e6
        assert np.allclose(result.values, expected_ppm)


def test_insufficient_points(regress_shift_calibration, sample_spectrum_data, mocker):
    """Test behavior when insufficient points are available."""
    mz_arr, int_arr = sample_spectrum_data
    distances = np.array([0.5, np.nan, np.nan])  # Only one valid point

    mocker.patch.object(ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=distances)

    result = regress_shift_calibration.extract_spectrum_features(mz_arr, int_arr)
    assert np.all(np.isnan(result.values))


def test_preprocess_image_features_with_smoothing(regress_shift_calibration, sample_image_features, mocker):
    """Test preprocess_image_features with smoothing."""
    mock_smoothing = mocker.Mock(spec=SpatialSmoothingType)
    mock_smoothing.smooth_image.return_value = sample_image_features

    calibration = CalibrationMethodRegressShift(
        ref_mz_arr=regress_shift_calibration._ref_mz_arr,
        max_distance=2.0,
        max_distance_unit="mz",
        model_type="linear",
        model_unit="mz",
        input_smoothing=mock_smoothing,
    )

    result = calibration.preprocess_image_features(sample_image_features)
    mock_smoothing.smooth_image.assert_called_once_with(sample_image_features)
    assert result is sample_image_features


def test_fit_spectrum_model(regress_shift_calibration, mocker):
    """Test fit_spectrum_model method."""
    features = DataArray([0.4, 0.5, 0.6], dims=["c"])

    # Create a mock model with the correct coefficient shape (slope, intercept)
    mock_model = mocker.Mock(coef=np.array([0.001, 0.3]))

    # Patch the fit_model function
    mocker.patch("depiction.calibration.methods.calibration_method_regress_shift.fit_model", return_value=mock_model)

    result = regress_shift_calibration.fit_spectrum_model(features)
    assert isinstance(result, DataArray)
    assert np.array_equal(result.values, mock_model.coef)


def test_apply_spectrum_model(regress_shift_calibration, sample_spectrum_data):
    """Test apply_spectrum_model method."""
    mz_arr, int_arr = sample_spectrum_data
    model_coef = DataArray([0.1, 0.2], dims=["c"])

    result_mz, result_int = regress_shift_calibration.apply_spectrum_model(mz_arr, int_arr, model_coef)

    assert isinstance(result_mz, np.ndarray)
    assert isinstance(result_int, np.ndarray)
    assert result_mz.shape == mz_arr.shape
    assert np.array_equal(result_int, int_arr)


def test_apply_spectrum_model_ppm_unit(reference_mz_values, sample_spectrum_data):
    """Test apply_spectrum_model with ppm unit."""
    calibration = CalibrationMethodRegressShift(
        ref_mz_arr=reference_mz_values,
        max_distance=2.0,
        max_distance_unit="mz",
        model_type="linear",
        model_unit="ppm",
        input_smoothing=None,
        min_points=3,
    )

    mz_arr, int_arr = sample_spectrum_data
    model_coef = DataArray([1.0, 0.0], dims=["c"])  # slope=1, intercept=0

    result_mz, result_int = calibration.apply_spectrum_model(mz_arr, int_arr, model_coef)

    # The LinearModel uses y = ax + b formula, where:
    # x is the input m/z
    # a, b are the coefficients (slope, intercept)
    # y is the predicted PPM shift
    predicted_ppm = model_coef.values[0] * mz_arr + model_coef.values[1]

    # Convert PPM to mass shift: (ppm/1e6) * mass
    mass_shifts = predicted_ppm / 1e6 * mz_arr

    # Expected result is original mass minus the shift
    expected_mz = mz_arr - mass_shifts

    assert np.allclose(result_mz, expected_mz, rtol=0.001)
    assert np.array_equal(result_int, int_arr)


def test_peak_filtering(reference_mz_values, sample_spectrum_data, mocker):
    """Test peak filtering functionality."""
    mock_filter = mocker.Mock(spec=PeakFilteringType)
    filtered_mz = np.array([100.5, 200.5])
    filtered_int = np.array([1000.0, 2000.0])
    mock_filter.filter_peaks.return_value = (filtered_mz, filtered_int)

    calibration = CalibrationMethodRegressShift(
        ref_mz_arr=reference_mz_values,
        max_distance=2.0,
        max_distance_unit="mz",
        model_type="linear",
        model_unit="mz",
        input_smoothing=None,
        peak_filtering=mock_filter,
    )

    mz_arr, int_arr = sample_spectrum_data

    # Mock the static method properly
    mock_distances = mocker.patch.object(
        ReferencePeakDistances, "get_distances_max_peak_in_window", return_value=np.array([0.5, 0.5, 0.5, 0.5])
    )

    mock_nearest = mocker.patch.object(
        ReferencePeakDistances, "get_distances_nearest", return_value=np.array([0.5, 0.5])
    )

    calibration.extract_spectrum_features(mz_arr, int_arr)

    # Verify the mocks were called
    mock_filter.filter_peaks.assert_called_once()
    mock_distances.assert_called_once()
    mock_nearest.assert_called_once()


def test_repr(regress_shift_calibration):
    """Test __repr__ method."""
    expected = (
        "CalibrationMethodRegressShift(max_distance=2.0, max_distance_unit=mz, "
        "model_type=linear, model_unit=mz, input_smoothing=None, peak_filtering=None, min_points=3)"
    )
    assert repr(regress_shift_calibration) == expected
