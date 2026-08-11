import numpy as np
import pytest

from depiction.calibration.spectrum.reference_peak_distances import ReferencePeakDistances


@pytest.fixture
def peak_data_mz_window():
    """Fixture for peak data used in window tests."""
    peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0])
    peak_int_arr = np.array([5.0, 5, 2, 2, 5, 5.1, 5])
    ref_mz_arr = np.array([90, 100.0, 101.5, 102.4])
    return peak_mz_arr, peak_int_arr, ref_mz_arr


@pytest.fixture
def peak_data_nearest():
    """Fixture for peak data used in nearest tests."""
    peak_mz_arr = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0, 106.0])
    ref_mz_arr = np.array([100.6, 102.0, 103.9, 105.5])
    return peak_mz_arr, ref_mz_arr


def test_get_distances_max_peak_in_window_when_mz(peak_data_mz_window):
    """Test get_distances_max_peak_in_window with mz unit."""
    peak_mz_arr, peak_int_arr, ref_mz_arr = peak_data_mz_window

    distances = ReferencePeakDistances.get_distances_max_peak_in_window(
        peak_mz_arr=peak_mz_arr,
        peak_int_arr=peak_int_arr,
        ref_mz_arr=ref_mz_arr,
        max_distance=0.9,
        max_distance_unit="mz",
    )

    expected = np.array([np.nan, 0.0, 0.5, 0.1])
    np.testing.assert_array_almost_equal(expected, distances, decimal=8)


def test_get_distances_max_peak_in_window_when_ppm(peak_data_mz_window):
    """Test get_distances_max_peak_in_window with ppm unit."""
    peak_mz_arr, peak_int_arr, ref_mz_arr = peak_data_mz_window

    distances = ReferencePeakDistances.get_distances_max_peak_in_window(
        peak_mz_arr=peak_mz_arr,
        peak_int_arr=peak_int_arr,
        ref_mz_arr=ref_mz_arr,
        max_distance=9e3,
        max_distance_unit="ppm",
    )

    expected = np.array([np.nan, 0.0, 0.5, 0.1])
    np.testing.assert_array_almost_equal(expected, distances, decimal=8)


@pytest.mark.parametrize("max_distance_unit,max_distance", [("mz", 0.9), ("ppm", 9e3)])
def test_get_distances_max_peak_in_window_when_window_empty(max_distance_unit, max_distance):
    """A reference with no peak in its window reports nan, whichever side it falls on.

    `peak_mz_arr` is a view into a longer array whose next element sits exactly on the
    out-of-range reference, which makes the old failure deterministic: for a reference above
    the last peak the code indexed one past the end, and with bounds checking off under njit
    that read the 200.0 and reported a distance of 0.0 instead of nan. With a standalone
    array the same read returns whatever is adjacent in memory, which the distance gate
    usually -- but not always -- rejects.
    """
    backing = np.array([100.0, 100.5, 101.0, 200.0])
    peak_mz_arr = backing[:3]
    peak_int_arr = np.array([5.0, 5.0, 5.0])
    ref_mz_arr = np.array([90.0, 100.0, 200.0])

    distances = ReferencePeakDistances.get_distances_max_peak_in_window(
        peak_mz_arr=peak_mz_arr,
        peak_int_arr=peak_int_arr,
        ref_mz_arr=ref_mz_arr,
        max_distance=max_distance,
        max_distance_unit=max_distance_unit,
    )

    np.testing.assert_array_almost_equal(np.array([np.nan, 0.0, np.nan]), distances, decimal=8)


def test_get_distances_max_peak_in_window_when_invalid():
    """Test get_distances_max_peak_in_window with invalid unit."""
    mock_peak_mz_arr = np.array([10.0, 20, 30])
    mock_peak_int_arr = np.array([1, 2, 3])
    mock_ref_arr = np.array([10.0])

    with pytest.raises(ValueError, match="badunit"):
        ReferencePeakDistances.get_distances_max_peak_in_window(
            peak_mz_arr=mock_peak_mz_arr,
            peak_int_arr=mock_peak_int_arr,
            ref_mz_arr=mock_ref_arr,
            max_distance=0.3,
            max_distance_unit="badunit",
        )


def test_get_distances_nearest_when_mz(peak_data_nearest):
    """Test get_distances_nearest with mz unit."""
    peak_mz_arr, ref_mz_arr = peak_data_nearest

    distances = ReferencePeakDistances.get_distances_nearest(
        peak_mz_arr=peak_mz_arr,
        ref_mz_arr=ref_mz_arr,
        max_distance=0.3,
        max_distance_unit="mz",
    )

    expected = np.array([-0.1, 0, 0.1, np.nan])
    np.testing.assert_array_almost_equal(expected, distances, decimal=8)


def test_get_distances_nearest_when_ppm(peak_data_nearest):
    """Test get_distances_nearest with ppm unit."""
    peak_mz_arr, ref_mz_arr = peak_data_nearest

    distances = ReferencePeakDistances.get_distances_nearest(
        peak_mz_arr=peak_mz_arr,
        ref_mz_arr=ref_mz_arr,
        max_distance=3e3,
        max_distance_unit="ppm",
    )

    expected = np.array([-0.1, 0, 0.1, np.nan])
    np.testing.assert_array_almost_equal(expected, distances, decimal=8)


def test_get_distances_nearest_when_invalid():
    """Test get_distances_nearest with invalid unit."""
    mock_peak_mz_arr = np.array([10.0, 20, 30])
    mock_ref_arr = np.array([10.0])

    with pytest.raises(ValueError, match="badunit"):
        ReferencePeakDistances.get_distances_nearest(
            peak_mz_arr=mock_peak_mz_arr,
            ref_mz_arr=mock_ref_arr,
            max_distance=0.3,
            max_distance_unit="badunit",
        )
