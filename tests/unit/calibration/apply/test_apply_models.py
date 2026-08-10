import numpy as np
import pytest
from xarray import DataArray

from depiction.calibration.apply.apply_models import ApplyModels
from depiction.calibration.methods.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig


@pytest.fixture
def model_coefficients():
    """Fixture for sample model coefficients: a 2x2 image, one distinguishable coefficient per pixel.

    The coefficient of pixel (x, y) is (0, 0) -> 10, (1, 0) -> 20, (0, 1) -> 30, (1, 1) -> 40.
    """
    coordinates = DataArray([[0, 0], [1, 0], [0, 1], [1, 1]], dims=("i", "d"), coords={"d": ["x", "y"]})
    values = DataArray([[10.0], [20.0], [30.0], [40.0]], dims=("i", "c"), coords={"c": ["shift"]})
    return MultiChannelImage.from_flat(values, coordinates=coordinates)


@pytest.fixture
def sample_spectrum_data():
    """Fixture for sample spectrum data."""
    mz_arr = np.array([100.0, 200.0])
    int_arr = np.array([1000.0, 2000.0])
    coords = np.array([0, 0])
    return mz_arr, int_arr, coords


def apply_coefficients(mocker, model_coefficients, coordinates):
    """Calibrates a file whose spectra come in the given (x, y) order, returning the coefficient applied to each."""
    mz_arr = np.array([100.0, 200.0])
    int_arr = np.array([1000.0, 2000.0])
    mock_reader = mocker.Mock()
    mock_reader.get_spectrum_with_coords.side_effect = lambda i: (mz_arr, int_arr, np.array(coordinates[i]))
    mock_calibration = mocker.Mock(spec=CalibrationMethod)
    mock_calibration.apply_spectrum_model.return_value = (mz_arr, int_arr)

    ApplyModels.calibrate_spectra(
        reader=mock_reader,
        spectra_indices=list(range(len(coordinates))),
        writer=mocker.Mock(),
        calibration=mock_calibration,
        all_model_coefs=model_coefficients,
    )

    return [call.kwargs["model_coef"].values.item() for call in mock_calibration.apply_spectrum_model.call_args_list]


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


def test_calibrate_spectra_uses_the_model_of_the_matching_pixel(mocker, model_coefficients):
    """Each spectrum gets the model fitted for its own pixel, also when the file is not row-major."""
    applied = apply_coefficients(mocker, model_coefficients, [(0, 1), (0, 0), (1, 0), (1, 1)])
    assert applied == [30.0, 10.0, 20.0, 40.0]


def test_calibrate_spectra_matches_positional_order_for_a_row_major_file(mocker, model_coefficients):
    """For a row-major file the assignment is the flat model order, i.e. unchanged for such acquisitions."""
    applied = apply_coefficients(mocker, model_coefficients, [(0, 0), (1, 0), (0, 1), (1, 1)])
    assert applied == model_coefficients.data_flat.values.ravel().tolist()


def test_calibrate_spectra_when_pixel_outside_model_image(mocker, model_coefficients):
    """A pixel without a fitted model fails loudly instead of silently borrowing another pixel's model."""
    with pytest.raises(KeyError):
        apply_coefficients(mocker, model_coefficients, [(5, 5)])


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
