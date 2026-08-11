from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from depiction_io import ImzmlModeEnum, ImzmlWriter
from depiction_io.pixel_size import PixelSize

MZ_ARR = np.array([100.0, 200.0])
INT_ARR = np.array([1.0, 2.0])


@pytest.fixture
def writer(tmp_path: Path) -> ImzmlWriter:
    """A writer over a real, if tiny, file: the imzy writer validates what it is handed and
    writes as it goes, so a MagicMock in its place tests nothing that matters."""
    return ImzmlWriter.open(path=tmp_path / "test.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)


def test_open_when_continuous(tmp_path: Path) -> None:
    path = tmp_path / "test.imzML"
    writer = ImzmlWriter.open(path=path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
    assert writer.imzml_path == path
    assert writer.ibd_path == path.with_suffix(".ibd")
    assert writer.imzml_mode == ImzmlModeEnum.CONTINUOUS


def test_open_when_processed(tmp_path: Path) -> None:
    writer = ImzmlWriter.open(path=tmp_path / "test.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)
    assert writer.imzml_mode == ImzmlModeEnum.PROCESSED


def test_open_forwards_the_pixel_size(tmp_path: Path) -> None:
    writer = ImzmlWriter.open(
        path=tmp_path / "test.imzML",
        imzml_mode=ImzmlModeEnum.PROCESSED,
        pixel_size=PixelSize(size_x=50.0, size_y=20.0, unit="micrometer"),
    )
    assert writer._imzml_writer.pixel_size == (50.0, 20.0)


def test_open_without_a_pixel_size(writer: ImzmlWriter) -> None:
    assert writer._imzml_writer.pixel_size is None


def test_open_when_pixel_size_is_not_in_micrometer(tmp_path: Path) -> None:
    # imzy writes `IMS:1000046`/`IMS:1000047` with no unit attribute and the parser reads them back
    # as micrometer, so writing anything else would relabel the number rather than convert it.
    with pytest.raises(ValueError, match="micrometer"):
        ImzmlWriter.open(
            path=tmp_path / "test.imzML",
            imzml_mode=ImzmlModeEnum.PROCESSED,
            pixel_size=PixelSize(size_x=1.0, size_y=1.0, unit="millimeter"),
        )


def test_close(writer: ImzmlWriter, mocker: MockerFixture) -> None:
    mock_close = mocker.patch.object(writer._imzml_writer, "close")
    writer.close()
    mock_close.assert_called_once_with()


def test_discard(writer: ImzmlWriter, mocker: MockerFixture) -> None:
    mock_discard = mocker.patch.object(writer._imzml_writer, "discard")
    writer.discard()
    mock_discard.assert_called_once_with()


def test_discard_leaves_nothing_behind(writer: ImzmlWriter) -> None:
    writer.add_spectrum(MZ_ARR, INT_ARR, (1, 2))
    writer.discard()
    assert list(writer.imzml_path.parent.iterdir()) == []


def test_close_without_any_spectrum_raises(writer: ImzmlWriter) -> None:
    # Documents a behaviour change from the pyimzml writer, which produced a malformed file
    # in this situation instead.
    with pytest.raises(ValueError, match="without any spectra"):
        writer.close()


def test_deactivate_alignment_tracker(writer: ImzmlWriter) -> None:
    writer.deactivate_alignment_tracker()
    assert writer._imzml_alignment_tracker is None


def test_add_spectrum(writer: ImzmlWriter) -> None:
    writer.add_spectrum(MZ_ARR, INT_ARR, (1, 2))
    writer.close()
    assert writer.imzml_path.exists()
    assert writer.ibd_path.exists()


def test_add_spectrum_when_lengths_differ(writer: ImzmlWriter) -> None:
    with pytest.raises(ValueError, match="must be equal"):
        writer.add_spectrum(MZ_ARR, np.array([1.0]), (1, 2))


def test_add_spectrum_when_empty(writer: ImzmlWriter) -> None:
    # The guard that stops imzy from warning and silently skipping the pixel, which would
    # leave n_spectra out of step with the coordinates the caller thinks it wrote.
    with pytest.raises(ValueError, match="empty spectrum"):
        writer.add_spectrum(np.array([]), np.array([]), (1, 2))


def test_add_spectrum_when_declined(writer: ImzmlWriter, mocker: MockerFixture) -> None:
    # Nothing should be able to reach a silent skip, so a False return is an error even
    # though no known input produces one once the empty case is caught above.
    mocker.patch.object(writer._imzml_writer, "add_spectrum", return_value=False)
    with pytest.raises(RuntimeError, match="declined"):
        writer.add_spectrum(MZ_ARR, INT_ARR, (1, 2))


def test_add_spectrum_tracks_alignment(writer: ImzmlWriter, mocker: MockerFixture) -> None:
    mock_track = mocker.patch.object(writer._imzml_alignment_tracker, "track_mz_array")
    writer.add_spectrum(MZ_ARR, INT_ARR, (1, 2))
    mock_track.assert_called_once()


def test_add_spectrum_when_alignment_not_satisfied(tmp_path: Path) -> None:
    writer = ImzmlWriter.open(path=tmp_path / "test.imzML", imzml_mode=ImzmlModeEnum.CONTINUOUS)
    writer.add_spectrum(MZ_ARR, INT_ARR, (1, 1))
    with pytest.raises(ValueError, match="must be identical to the m/z array of all other spectra"):
        writer.add_spectrum(np.array([100.0, 300.0]), INT_ARR, (2, 1))


def test_is_aligned(writer: ImzmlWriter) -> None:
    # A fresh tracker reports False until it has seen a spectrum, which is why the
    # continuous-mode check in `add_spectrum` only fires once one has been tracked.
    assert not writer.is_aligned
    writer.add_spectrum(MZ_ARR, INT_ARR, (1, 2))
    assert writer.is_aligned


def test_copy_spectra(writer: ImzmlWriter, mocker: MockerFixture) -> None:
    mock_add_spectrum = mocker.patch.object(ImzmlWriter, "add_spectrum")
    mock_reader = mocker.MagicMock(name="mock_reader", spec=["get_spectrum_with_coords"])
    mock_reader.get_spectrum_with_coords.side_effect = [("a", "b", "C1"), ("c", "d", "C2")]
    writer.copy_spectra(reader=mock_reader, spectra_indices=[10, 20])
    assert mock_add_spectrum.mock_calls == [mocker.call("a", "b", "C1"), mocker.call("c", "d", "C2")]
