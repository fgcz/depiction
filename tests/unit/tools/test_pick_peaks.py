import importlib.util
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from depiction.persistence import ImzmlReadFile, ImzmlWriteFile
from depiction.tools.pick_peaks import (
    PickPeaksConfig,
    PeakPickerBasicInterpolatedConfig,
    get_peak_picker,
    PeakPickerMSPeakPickerConfig,
    PeakPickerFindMFPyConfig,
)


@pytest.fixture()
def mock_filtering(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(name="mock_filtering", spec=[])


@pytest.fixture()
def mock_input_file(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(name="mock_input_file", spec=ImzmlReadFile)


@pytest.fixture()
def mock_output_file(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(name="mock_output_file", spec=ImzmlWriteFile)


def test_get_peak_picker_when_basic_interpolated(mock_filtering: MagicMock) -> None:
    config = PickPeaksConfig(
        peak_picker=PeakPickerBasicInterpolatedConfig(
            min_prominence=0.1,
            min_distance=0.2,
            min_distance_unit="mz",
            peak_filtering=mock_filtering,
        ),
        n_jobs=1,
    )
    picker = get_peak_picker(config, mock_filtering)
    assert picker.min_prominence == 0.1
    assert picker.min_distance == 0.2
    assert picker.min_distance_unit == "mz"
    assert picker.peak_filtering == mock_filtering


def test_get_peak_picker_when_ms_peak_picker(mock_filtering: MagicMock) -> None:
    config = PickPeaksConfig(
        peak_picker=PeakPickerMSPeakPickerConfig(fit_type="quadratic", peak_filtering=mock_filtering),
        n_jobs=1,
    )
    picker = get_peak_picker(config, mock_filtering)
    assert picker.fit_type == "quadratic"
    assert picker.peak_filtering == mock_filtering


@pytest.mark.skipif(importlib.util.find_spec("findmfpy") is None, reason="findmfpy not installed")
def test_get_peak_picker_when_find_mf_peak_picker(mock_filtering: MagicMock) -> None:
    config = PickPeaksConfig(
        peak_picker=PeakPickerFindMFPyConfig(
            resolution=0.1,
            width=0.2,
            int_width=0.3,
            int_threshold=0.4,
            area=True,
            max_peaks=10,
        ),
        n_jobs=1,
    )
    picker = get_peak_picker(config, mock_filtering)
    assert picker.resolution == 0.1
    assert picker.width == 0.2
    assert picker.int_width == 0.3
    assert picker.int_threshold == 0.4
    assert picker.area
    assert picker.max_peaks == 10


if __name__ == "__main__":
    pytest.main()
