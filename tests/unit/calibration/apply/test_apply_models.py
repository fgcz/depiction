import numpy as np
import pytest
from xarray import DataArray

from depiction.calibration.apply.apply_models import ApplyModels
from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction_io.parallel_ops import ParallelConfig


@pytest.fixture
def model_coefficients():
    """Fixture for sample model coefficients."""
    data = np.array([[[0.1]], [[0.2]], [[0.3]]])  # 3 spectra, 1 coefficient each
    return MultiChannelImage.from_spatial(DataArray(data, dims=("y", "x", "c"), coords={"c": ["shift"]}), bg_value=0)


@pytest.fixture
def sample_spectrum_data():
    """Fixture for sample spectrum data."""
    mz_arr = np.array([100.0, 200.0])
    int_arr = np.array([1000.0, 2000.0])
    coords = {"id": 0}
    return mz_arr, int_arr, coords


def test_write_to_file(mocker, model_coefficients, sample_spectrum_data):
    """Test writing calibrated spectra to file."""
    # Mock calibration method
    mock_calibration = mocker.Mock(spec=CalibrationMethod)
    mock_calibration.apply_spectrum_model.return_value = (
        sample_spectrum_data[0] + 0.1,  # shifted mz values
        sample_spectrum_data[1],  # unchanged intensities
    )

    # Mock reader and writer
    mock_reader = mocker.Mock()
    mock_reader.get_spectrum_with_coords.return_value = sample_spectrum_data

    mock_writer = mocker.Mock()

    # Mock read and write files
    mock_read_file = mocker.Mock()
    mock_read_file.get_reader.return_value = mock_reader

    mock_write_file = mocker.Mock()
    mock_write_file.get_writer.return_value = mock_writer

    # Mock WriteSpectraParallel
    mock_write_parallel = mocker.Mock()
    mocker.patch("depiction.parallel_ops.WriteSpectraParallel.from_config", return_value=mock_write_parallel)

    # Create ApplyModels instance and call write_to_file
    parallel_config = ParallelConfig(n_jobs=1)
    apply_models = ApplyModels(mock_calibration, parallel_config)
    apply_models.write_to_file(mock_read_file, mock_write_file, model_coefficients)

    # Verify the parallel writer was called correctly
    mock_write_parallel.map_chunked_to_file.assert_called_once()
    call_args = mock_write_parallel.map_chunked_to_file.call_args[1]
    assert call_args["read_file"] == mock_read_file
    assert call_args["write_file"] == mock_write_file
    assert call_args["operation"] == apply_models.calibrate_spectra


def test_calibrate_spectra(mocker, model_coefficients, sample_spectrum_data):
    """Test calibration of individual spectra."""
    # Mock calibration method
    mock_calibration = mocker.Mock(spec=CalibrationMethod)
    mock_calibration.apply_spectrum_model.return_value = (sample_spectrum_data[0] + 0.1, sample_spectrum_data[1])

    # Mock reader and writer
    mock_reader = mocker.Mock()
    mock_reader.get_spectrum_with_coords.return_value = sample_spectrum_data

    mock_writer = mocker.Mock()

    # Test calibrate_spectra
    spectra_indices = [0, 1]
    ApplyModels.calibrate_spectra(
        reader=mock_reader,
        spectra_indices=spectra_indices,
        writer=mock_writer,
        calibration=mock_calibration,
        all_model_coefs=model_coefficients,
    )

    # Verify interactions
    assert mock_reader.get_spectrum_with_coords.call_count == len(spectra_indices)
    assert mock_writer.add_spectrum.call_count == len(spectra_indices)
    assert mock_calibration.apply_spectrum_model.call_count == len(spectra_indices)


def test_error_handling(mocker, model_coefficients):
    """Test error handling during spectrum processing."""
    # Mock reader that raises an error
    mock_reader = mocker.Mock()
    mock_reader.get_spectrum_with_coords.side_effect = ValueError("Spectrum not found")

    mock_writer = mocker.Mock()
    mock_calibration = mocker.Mock(spec=CalibrationMethod)

    # Test that error is propagated
    with pytest.raises(ValueError, match="Spectrum not found"):
        ApplyModels.calibrate_spectra(
            reader=mock_reader,
            spectra_indices=[0],
            writer=mock_writer,
            calibration=mock_calibration,
            all_model_coefs=model_coefficients,
        )
