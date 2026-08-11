import importlib.util
import sys
from unittest.mock import MagicMock

import pytest
from depiction_io import GenericReadFile, ImzmlWriteFile
from depiction.tools.pick_peaks.config import (
    PeakPickerBasicInterpolatedConfig,
    PeakPickerFindMFPyConfig,
    PeakPickerMSPeakPickerConfig,
    PickPeaksConfig,
)
from depiction.tools.pick_peaks.pick_peaks import get_peak_picker
from pytest_mock import MockerFixture


@pytest.fixture()
def mock_filtering(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(name="mock_filtering", spec=[])


@pytest.fixture()
def mock_input_file(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(name="mock_input_file", spec=GenericReadFile)


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
    # No `importorskip("ms_peak_picker")`: MSPeakPicker defers that import into `pick_peaks`,
    # so constructing one works whether or not the optional extra is installed.
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


def test_get_peak_picker_when_find_mf_peak_picker_and_not_installed(
    mocker: MockerFixture, mock_filtering: MagicMock
) -> None:
    # `None` in sys.modules is what the import machinery treats as "this module is
    # unavailable", so this reproduces a missing `findmf` extra whether or not `findmfpy`
    # happens to be installed in the environment running the test. The wrapper module has to
    # go too, or an earlier test's import of it satisfies this one from cache; `patch.dict`
    # restores the whole mapping afterwards, including that eviction.
    mocker.patch.dict(sys.modules, {"findmfpy": None})
    sys.modules.pop("depiction.spectrum.peak_picking.findmf_peak_picker", None)
    config = PickPeaksConfig(peak_picker=PeakPickerFindMFPyConfig(), n_jobs=1)
    with pytest.raises(ModuleNotFoundError, match="uv sync --extra findmf"):
        get_peak_picker(config, mock_filtering)


if __name__ == "__main__":
    pytest.main()
