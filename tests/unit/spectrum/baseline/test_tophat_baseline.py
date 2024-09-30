import numpy as np
import pytest

import depiction.spectrum.baseline.tophat_baseline as test_module
from depiction.spectrum.baseline import TophatBaseline


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
    return TophatBaseline(window_size=mock_window_size, window_unit=mock_window_unit)


def test_compute_erosion():
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    element_size = 5
    eroded = test_module._compute_erosion(x, element_size)
    np.testing.assert_array_equal([10, 10, 10, 20, 30, 40, 50, 60, 70, 80], eroded)


def test_compute_dilation():
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    element_size = 5
    dilation = test_module._compute_dilation(x, element_size)
    np.testing.assert_array_equal([30, 40, 50, 60, 70, 80, 90, 100, 100, 100], dilation)


def test_compute_opening():
    x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    element_size = 5
    opening = test_module._compute_opening(x, element_size)
    np.testing.assert_array_equal([10, 20, 30, 40, 50, 60, 70, 80, 80, 80], opening)


def test_optimize_structuring_element_size():
    x = np.array([10, 20, 30, 10, 10, 10, 20, 10, 10, 10, 10, 10, 10, 10, 50, 10, 10], dtype=float)
    element_size = test_module._optimize_structuring_element_size(x, tolerance=1e-6)
    assert element_size == 3

    # sanity check
    np.testing.assert_array_equal(
        test_module._compute_opening(x, element_size),
        test_module._compute_opening(x, element_size + 2),
    )
    assert not np.array_equal(
        test_module._compute_opening(x, element_size),
        test_module._compute_opening(x, element_size - 2),
    )


def test_evaluate_baseline(mock_baseline, mocker):
    mock_mz_arr = mocker.MagicMock(name="mock_mz_arr")
    x = np.array([10, 20, 30, 10, 10, 10, 20, 10, 10, 10, 10, 10, 10, 10, 50, 10, 10])
    baseline = mock_baseline.evaluate_baseline(mock_mz_arr, x)
    np.testing.assert_array_equal(np.full_like(x, 10.0), baseline)


def test_subtract_baseline(mock_baseline, mocker):
    mocker.patch.object(TophatBaseline, "evaluate_baseline", return_value=np.array([20, 20, 30, 30, 30]))
    mock_int_arr = np.array([50, 10, 10, 10, 50])
    mock_mz_arr = mocker.MagicMock(name="mock_mz_arr")
    int_arr = mock_baseline.subtract_baseline(mz_arr=mock_mz_arr, int_arr=mock_int_arr)
    np.testing.assert_array_equal([30, 0, 0, 0, 20], int_arr)


def test_get_element_size_when_index(mock_baseline, mocker):
    mock_mz_arr = mocker.MagicMock(name="mock_mz_arr", spec=[])
    assert mock_baseline.get_element_size(mz_arr=mock_mz_arr) == 5


def test_get_element_size_when_ppm(mock_window_size, mock_window_unit):
    mock_window_unit = "ppm"
    mock_window_size = 500
    mock_baseline = TophatBaseline(window_size=mock_window_size, window_unit=mock_window_unit)
    mock_mz_arr = np.arange(995, 1005, 0.05)
    assert mock_baseline.get_element_size(mz_arr=mock_mz_arr) == 10


def test_get_element_size_when_invalid(mocker, mock_window_size):
    mock_baseline = TophatBaseline(window_size=mock_window_size, window_unit="mz")
    with pytest.raises(ValueError):
        mock_baseline.get_element_size(mz_arr=mocker.MagicMock(name="mock_mz_arr", spec=[]))
