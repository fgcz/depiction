import pytest

from depiction.calibration.perform_calibration import PerformCalibration
from depiction.parallel_ops import ParallelConfig


@pytest.fixture()
def parallel_config():
    return ParallelConfig.no_parallelism()


@pytest.fixture()
def calibration_method(mocker):
    return mocker.MagicMock(name="calibration_method")


@pytest.fixture()
def coefficient_output_file(mocker):
    return mocker.MagicMock(name="coefficient_output_file")


@pytest.fixture()
def perform_calibration(calibration_method, parallel_config, coefficient_output_file):
    return PerformCalibration(
        calibration=calibration_method, parallel_config=parallel_config, coefficient_output_file=coefficient_output_file
    )


def test_calibrate_image(mocker, perform_calibration, calibration_method):
    mock_extract_features = mocker.patch("depiction.calibration.perform_calibration.ExtractFeatures")
    mock_fit_models = mocker.patch("depiction.calibration.perform_calibration.FitModels")
    mock_apply_models = mocker.patch("depiction.calibration.perform_calibration.ApplyModels")
    mock_write_data_array = mocker.patch.object(perform_calibration, "_write_data_array")

    mock_read_peaks = mocker.MagicMock(name="mock_read_peaks")
    mock_write_file = mocker.MagicMock(name="mock_write_file")
    mock_read_full = mocker.MagicMock(name="mock_read_full")

    perform_calibration.calibrate_image(
        read_peaks=mock_read_peaks, write_file=mock_write_file, read_full=mock_read_full
    )

    mock_extract_features.assert_called_once_with(
        perform_calibration._calibration, perform_calibration._parallel_config
    )
    mock_extract_features.return_value.get_image.assert_called_once_with(mock_read_peaks)
    mock_fit_models.assert_called_once_with(perform_calibration._calibration, perform_calibration._parallel_config)
    mock_fit_models.return_value.get_image.assert_called_once_with(
        calibration_method.preprocess_image_features.return_value
    )
    calibration_method.preprocess_image_features.assert_called_once_with(
        all_features=mock_extract_features.return_value.get_image.return_value
    )
    mock_apply_models.assert_called_once_with(perform_calibration._calibration, perform_calibration._parallel_config)
    mock_apply_models.return_value.write_to_file.assert_called_once_with(
        read_file=mock_read_full,
        write_file=mock_write_file,
        all_model_coefs=mock_fit_models.return_value.get_image.return_value,
    )
    assert mock_write_data_array.mock_calls == [
        mocker.call(mock_extract_features().get_image.return_value, group="features_raw"),
        mocker.call(calibration_method.preprocess_image_features.return_value, group="features_processed"),
        mocker.call(mock_fit_models().get_image.return_value, group="model_coefs"),
    ]


@pytest.mark.parametrize("coefficient_output_file", [None, "/path/to/output"])
def test_write_data_array(mocker, coefficient_output_file, perform_calibration):
    mock_image = mocker.MagicMock(name="mock_image")
    perform_calibration._write_data_array(mock_image, group="features_raw")
    if coefficient_output_file is None:
        mock_image.write_hdf5.assert_not_called()
    else:
        mock_image.write_hdf5.assert_called_once_with(path=coefficient_output_file, mode="a", group="features_raw")
