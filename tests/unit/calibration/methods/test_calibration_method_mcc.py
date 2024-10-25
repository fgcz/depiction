import numpy as np
import pytest
import statsmodels.api as sm
from xarray import DataArray

from depiction.calibration.methods.calibration_method_mcc import CalibrationMethodMassClusterCenterModel
from depiction.image import MultiChannelImage


@pytest.fixture
def mccm_calibration():
    """Fixture for basic MCCM calibration instance."""
    return CalibrationMethodMassClusterCenterModel(
        model_smoothing_activated=False,
        model_smoothing_kernel_size=27,
        model_smoothing_kernel_std=10.0,
        max_pairwise_distance=500,
    )


@pytest.fixture
def sample_spectrum_data():
    """Fixture for sample spectrum data that should produce predictable deltas."""
    # Creating peaks that are l_none (1.000482) apart to test MCC calculation
    base = 100.0
    l_none = 1.000482
    mz_arr = np.array([base, base + l_none, base + 2 * l_none, base + 3 * l_none])
    int_arr = np.array([1000.0, 2000.0, 1500.0, 1800.0])
    return mz_arr, int_arr


@pytest.fixture
def sample_image_features():
    """Fixture for sample MultiChannelImage."""
    data = np.full((3, 3, 2), 0.1)  # 2 channels for [intercept, slope]
    return MultiChannelImage.from_spatial(
        DataArray(data, dims=("y", "x", "c"), coords={"c": ["intercept", "slope"]}), bg_value=0
    )


def test_initialization(mccm_calibration):
    """Test initialization of CalibrationMethodMassClusterCenterModel."""
    assert not mccm_calibration._model_smoothing_activated
    assert mccm_calibration._model_smoothing_kernel_size == 27
    assert mccm_calibration._model_smoothing_kernel_std == 10.0
    assert mccm_calibration._max_pairwise_distance == 500


def test_compute_distance_from_MCC(mccm_calibration):
    """Test compute_distance_from_MCC method."""
    # Test with values that should produce known results
    l_none = 1.000482
    test_deltas = np.array(
        [
            0.2,  # Should remain positive (< 0.5)
            0.7,  # Should become negative (-0.3)
            l_none,  # Should be close to 0
            1.5 * l_none,  # Should give predictable result
        ]
    )

    result = mccm_calibration.compute_distance_from_MCC(test_deltas, l_none)

    assert len(result) == len(test_deltas)
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(-0.3)
    assert abs(result[2]) < 1e-6  # Should be very close to 0


def test_extract_spectrum_features(mccm_calibration, sample_spectrum_data, mocker):
    """Test extract_spectrum_features method."""
    mz_arr, int_arr = sample_spectrum_data

    # Mock the RLM fit to return known coefficients
    mock_results = mocker.Mock()
    mock_results.params = np.array([0.001])  # Small slope
    mock_fit = mocker.Mock(return_value=mock_results)
    mocker.patch.object(sm.RLM, "fit", mock_fit)

    result = mccm_calibration.extract_spectrum_features(mz_arr, int_arr)

    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert len(result.values) == 2  # [intercept, slope]
    assert isinstance(result.values[0], float)  # intercept
    assert isinstance(result.values[1], float)  # slope


def test_preprocess_image_features_no_smoothing(mccm_calibration, sample_image_features):
    """Test preprocess_image_features without smoothing."""
    result = mccm_calibration.preprocess_image_features(sample_image_features)
    assert result is sample_image_features  # Should return same object when smoothing is off


def test_preprocess_image_features_with_smoothing(sample_image_features):
    """Test preprocess_image_features with smoothing enabled."""
    calibration = CalibrationMethodMassClusterCenterModel(
        model_smoothing_activated=True,
        model_smoothing_kernel_size=3,  # Small kernel for testing
        model_smoothing_kernel_std=1.0,
        max_pairwise_distance=500,
    )

    result = calibration.preprocess_image_features(sample_image_features)

    assert isinstance(result, MultiChannelImage)
    assert result.sizes == sample_image_features.sizes
    assert not np.array_equal(result.data_spatial, sample_image_features.data_spatial)


def test_fit_spectrum_model(mccm_calibration):
    """Test fit_spectrum_model method."""
    features = DataArray([0.1, 0.2], dims=["c"])
    result = mccm_calibration.fit_spectrum_model(features)
    assert result is features  # Should return input unchanged


def test_apply_spectrum_model(mccm_calibration):
    """Test apply_spectrum_model method."""
    spectrum_mz_arr = np.array([100.0, 200.0, 300.0])
    spectrum_int_arr = np.array([1000.0, 2000.0, 3000.0])
    model_coef = DataArray([0.1, 0.001], dims=["c"])  # [intercept, slope]

    result_mz, result_int = mccm_calibration.apply_spectrum_model(spectrum_mz_arr, spectrum_int_arr, model_coef)

    # Calculate expected results
    intercept, slope = model_coef.values
    expected_mz = spectrum_mz_arr * (1 - slope) - intercept

    assert np.allclose(result_mz, expected_mz)
    assert np.array_equal(result_int, spectrum_int_arr)


def test_extract_spectrum_features_zero_division(mccm_calibration):
    """Test extract_spectrum_features handling of ZeroDivisionError."""
    # Create data that might cause zero division
    mz_arr = np.array([100.0, 100.0])  # Identical masses
    int_arr = np.array([1000.0, 1000.0])

    result = mccm_calibration.extract_spectrum_features(mz_arr, int_arr)
    assert isinstance(result, DataArray)
    assert result.dims == ("c",)
    assert len(result.values) == 2
    assert result.values[1] == 0  # Slope should be 0 in case of zero division


def test_compute_distance_from_MCC_edge_cases(mccm_calibration):
    """Test compute_distance_from_MCC with edge cases."""
    l_none = 1.000482
    test_deltas = np.array(
        [
            0.0,  # Zero
            0.5,  # Exactly half
            l_none - 0.001,  # Just under l_none
            l_none + 0.001,  # Just over l_none
        ]
    )

    result = mccm_calibration.compute_distance_from_MCC(test_deltas, l_none)
    assert len(result) == len(test_deltas)
    assert all(np.abs(result) <= 0.5)  # All results should be in [-0.5, 0.5]


def test_repr(mccm_calibration):
    """Test __repr__ method."""
    expected = (
        "CalibrationMethodMassClusterCenterModel("
        "model_smoothing_activated=False, "
        "model_smoothing_kernel_size=27, "
        "model_smoothing_kernel_std=10.0, "
        "max_pairwise_distance=500)"
    )
    assert repr(mccm_calibration) == expected
