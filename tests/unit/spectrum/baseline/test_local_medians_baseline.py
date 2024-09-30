from unittest.mock import MagicMock

import numpy as np
import pytest

from depiction.spectrum.baseline import LocalMediansBaseline


@pytest.fixture(autouse=True)
def mock_environ(monkeypatch, treat_warnings_as_error):
    monkeypatch.setenv("NUMBA_DEBUGINFO", "1")


@pytest.fixture
def mock_window_size():
    return 5


@pytest.fixture
def mock_window_unit():
    return "index"


@pytest.fixture
def mock_baseline(mock_window_size, mock_window_unit):
    print("Using window unit", mock_window_unit)
    return LocalMediansBaseline(window_size=mock_window_size, window_unit=mock_window_unit)


def test_evaluate_baseline_when_unit_index(mock_baseline) -> None:
    mock_mz_arr = MagicMock(name="mock_mz_arr")
    mock_int_arr = np.array([0, 0, 10, 10, 10, 10, 0, 0, 0, 10, 10])
    int_baseline = mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    np.testing.assert_array_equal([10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0], int_baseline)

    # the operation should be symmetric
    mock_int_arr_rev = np.flip(mock_int_arr)
    int_baseline_rev = mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr_rev)
    np.testing.assert_array_equal([0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10], int_baseline_rev)


@pytest.mark.parametrize("mock_window_unit", ["ppm"])
def test_evaluate_baseline_when_unit_ppm(mock_baseline) -> None:
    mock_mz_arr = np.linspace(10, 100, 20)
    mock_int_arr = np.ones(20)
    int_baseline = mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    np.testing.assert_array_equal(np.ones(20), int_baseline)


@pytest.mark.parametrize("mock_window_unit", ["ppm"])
@pytest.mark.parametrize("mock_window_size", [500])
def test_evaluate_baseline_when_unit_ppm_correct_left(mock_baseline) -> None:
    # to keep it simple, construct to have an almost constant ppm error and then count
    n_values = 20
    mock_mz_arr = np.zeros(n_values)
    mock_mz_arr[0] = 200
    mock_int_arr = np.ones(n_values)
    mock_int_arr[:2] = 0
    ppm_distance = 150  # i.e. not symmetric but that's not the point here
    for i in range(1, n_values):
        mz_error = ppm_distance / 1e6 * mock_mz_arr[i - 1]
        mock_mz_arr[i] = mock_mz_arr[i - 1] + mz_error

    int_baseline = mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    expected_arr = np.ones(n_values)
    expected_arr[:3] = 0
    np.testing.assert_array_equal(expected_arr, int_baseline)


@pytest.mark.parametrize("mock_window_unit", ["ppm"])
@pytest.mark.parametrize("mock_window_size", [500])
def test_evaluate_baseline_when_unit_ppm_correct_right(mock_baseline) -> None:
    # to keep it simple, construct to have an almost constant ppm error and then count
    n_values = 20
    mock_mz_arr = np.zeros(n_values)
    mock_mz_arr[0] = 200
    mock_int_arr = np.ones(n_values)
    mock_int_arr[-2:] = 0
    ppm_distance = 200  # i.e. not symmetric but that's not the point here
    for i in range(1, n_values):
        mz_error = ppm_distance / 1e6 * mock_mz_arr[i - 1]
        mock_mz_arr[i] = mock_mz_arr[i - 1] + mz_error

    int_baseline = mock_baseline.evaluate_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    expected_arr = np.ones(n_values)
    expected_arr[-3:] = 0
    np.testing.assert_array_equal(expected_arr, int_baseline)


def test_subtract_baseline(mocker, mock_baseline) -> None:
    method_evaluate_baseline = mocker.patch.object(LocalMediansBaseline, "evaluate_baseline")
    method_evaluate_baseline.return_value = np.array([20, 20, 30, 30, 30])
    mock_int_arr = np.array([50, 10, 10, 10, 50])
    mock_mz_arr = MagicMock(name="mock_mz_arr")
    int_arr = mock_baseline.subtract_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    np.testing.assert_array_equal([30, 0, 0, 0, 20], int_arr)
