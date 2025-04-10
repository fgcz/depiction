import os
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from unittest.mock import MagicMock, call

from depiction_io.persistence import ImzmlWriter, ImzmlModeEnum


@pytest.fixture
def mock_wrapped_imzml_writer():
    return MagicMock(name="mock_wrapped_imzml_writer")


@pytest.fixture
def mock_imzml_alignment_tracker():
    return MagicMock(name="mock_imzml_alignment_tracker")


@pytest.fixture
def mock_mz_arr():
    return MagicMock(name="mock_mz_arr")


@pytest.fixture
def mock_int_arr():
    return MagicMock(name="mock_int_arr")


@pytest.fixture
def mock_coordinates():
    return MagicMock(name="mock_coordinates")


@pytest.fixture
def imzml_writer(mock_wrapped_imzml_writer, mock_imzml_alignment_tracker):
    return ImzmlWriter(
        wrapped_imzml_writer=mock_wrapped_imzml_writer,
        imzml_alignment_tracker=mock_imzml_alignment_tracker,
    )


def test_open_when_continuous():
    with TemporaryDirectory() as tmpdir:
        mock_path = os.path.join(tmpdir, "test.imzML")
        writer = ImzmlWriter.open(path=mock_path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
        assert Path(mock_path) == writer.imzml_path
        assert ImzmlModeEnum.CONTINUOUS == writer.imzml_mode


def test_open_when_processed():
    with TemporaryDirectory() as tmpdir:
        mock_path = os.path.join(tmpdir, "test.imzML")
        writer = ImzmlWriter.open(path=mock_path, imzml_mode=ImzmlModeEnum.PROCESSED)
        assert Path(mock_path) == writer.imzml_path
        assert ImzmlModeEnum.PROCESSED == writer.imzml_mode


def test_close(imzml_writer, mock_wrapped_imzml_writer):
    imzml_writer.close()
    mock_wrapped_imzml_writer.close.assert_called_once_with()


def test_deactivate_alignment_tracker(imzml_writer):
    imzml_writer.deactivate_alignment_tracker()
    assert imzml_writer._imzml_alignment_tracker is None


def test_imzml_mode(imzml_writer, mock_wrapped_imzml_writer, mocker):
    mock_from_pyimzml_str = mocker.patch.object(ImzmlModeEnum, "from_pyimzml_str")
    mode = imzml_writer.imzml_mode
    mock_from_pyimzml_str.assert_called_once_with(mock_wrapped_imzml_writer.mode)
    assert mock_from_pyimzml_str.return_value == mode


def test_imzml_path(imzml_writer, mock_wrapped_imzml_writer):
    mock_wrapped_imzml_writer.filename = "test.imzML"
    assert Path("test.imzML") == imzml_writer.imzml_path


def test_ibd_path(imzml_writer, mock_wrapped_imzml_writer):
    mock_wrapped_imzml_writer.ibd_filename = "test.ibd"
    assert Path("test.ibd") == imzml_writer.ibd_path


def test_is_aligned(imzml_writer, mock_imzml_alignment_tracker):
    assert mock_imzml_alignment_tracker.is_aligned == imzml_writer.is_aligned


def test_add_spectrum_when_continuous_when_no_tracker(
    mock_wrapped_imzml_writer, mock_mz_arr, mock_int_arr, mock_coordinates, mocker
):
    mocker.patch.object(ImzmlWriter, "imzml_mode", ImzmlModeEnum.CONTINUOUS)
    writer = ImzmlWriter(
        wrapped_imzml_writer=mock_wrapped_imzml_writer,
        imzml_alignment_tracker=None,
    )
    writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
    mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(mock_mz_arr, mock_int_arr, mock_coordinates)


def test_add_spectrum_when_continuous_when_with_tracker(
    imzml_writer,
    mock_wrapped_imzml_writer,
    mock_imzml_alignment_tracker,
    mock_mz_arr,
    mock_int_arr,
    mock_coordinates,
    mocker,
):
    mocker.patch.object(ImzmlWriter, "imzml_mode", ImzmlModeEnum.CONTINUOUS)
    imzml_writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
    mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(mock_mz_arr, mock_int_arr, mock_coordinates)
    mock_imzml_alignment_tracker.track_mz_array.assert_called_once_with(mock_mz_arr)


def test_add_spectrum_when_processed_when_no_tracker(
    mock_wrapped_imzml_writer, mock_mz_arr, mock_int_arr, mock_coordinates, mocker
):
    mocker.patch.object(ImzmlWriter, "imzml_mode", ImzmlModeEnum.PROCESSED)
    writer = ImzmlWriter(
        wrapped_imzml_writer=mock_wrapped_imzml_writer,
        imzml_alignment_tracker=None,
    )
    writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
    mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(mock_mz_arr, mock_int_arr, mock_coordinates)


def test_add_spectrum_when_processed_when_with_tracker(
    imzml_writer,
    mock_wrapped_imzml_writer,
    mock_imzml_alignment_tracker,
    mock_mz_arr,
    mock_int_arr,
    mock_coordinates,
    mocker,
):
    mocker.patch.object(ImzmlWriter, "imzml_mode", ImzmlModeEnum.PROCESSED)
    imzml_writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
    mock_wrapped_imzml_writer.addSpectrum.assert_called_once_with(mock_mz_arr, mock_int_arr, mock_coordinates)
    mock_imzml_alignment_tracker.track_mz_array.assert_called_once_with(mock_mz_arr)


def test_add_spectrum_when_alignment_not_satisfied(
    imzml_writer, mock_mz_arr, mock_int_arr, mock_coordinates, mock_imzml_alignment_tracker, mocker
):
    mocker.patch.object(ImzmlWriter, "imzml_mode", ImzmlModeEnum.CONTINUOUS)
    mock_imzml_alignment_tracker.is_aligned = False
    with pytest.raises(ValueError) as error:
        imzml_writer.add_spectrum(mz_arr=mock_mz_arr, int_arr=mock_int_arr, coordinates=mock_coordinates)
    assert "The m/z array of the first spectrum must be identical to the m/z array of all other spectra!" in str(
        error.value
    )


def test_copy_spectra(imzml_writer, mocker):
    mock_add_spectrum = mocker.patch.object(ImzmlWriter, "add_spectrum")
    mock_reader = MagicMock(name="mock_reader")
    mock_reader.get_spectrum_with_coords.side_effect = [("a", "b", "C1"), ("c", "d", "C2")]
    imzml_writer.copy_spectra(reader=mock_reader, spectra_indices=[10, 20])
    assert mock_add_spectrum.mock_calls == [call("a", "b", "C1"), call("c", "d", "C2")]


if __name__ == "__main__":
    pytest.main()
