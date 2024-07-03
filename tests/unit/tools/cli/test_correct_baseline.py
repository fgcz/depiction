from pathlib import Path

import pytest

from depiction.tools.cli.correct_baseline import run
from depiction.tools.correct_baseline import BaselineVariants


def test_run_when_variant_zero(mocker) -> None:
    mock_copyfile = mocker.patch("shutil.copyfile")
    mock_logger = mocker.patch("depiction.tools.cli.correct_baseline.logger")
    mock_input_imzml = Path("/dev/null/hello.imzML")
    mock_output_imzml = mocker.MagicMock(name="mock_output_imzml")

    run(input_imzml=mock_input_imzml, output_imzml=mock_output_imzml, baseline_variant=BaselineVariants.Zero)

    assert mock_copyfile.mock_calls == [
        mocker.call(mock_input_imzml, mock_output_imzml),
        mocker.call(Path("/dev/null/hello.ibd"), mocker.ANY),
    ]
    mock_output_imzml.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_logger.info.assert_called_once()


def test_run_when_other_variant(mocker) -> None:
    mock_logger = mocker.patch("loguru.logger")
    mock_imzml_mode = mocker.MagicMock(name="mock_imzml_mode", spec=[])
    construct_imzml_read_file = mocker.patch("depiction.tools.cli.correct_baseline.ImzmlReadFile")
    construct_imzml_read_file.return_value.imzml_mode = mock_imzml_mode
    construct_imzml_write_file = mocker.patch("depiction.tools.cli.correct_baseline.ImzmlWriteFile")
    construct_correct_baseline = mocker.patch("depiction.tools.cli.correct_baseline.CorrectBaseline.from_variant")
    mock_input_imzml = Path("/dev/null/hello.imzML")
    mock_output_imzml = mocker.MagicMock(name="mock_output_imzml")

    run(input_imzml=mock_input_imzml, output_imzml=mock_output_imzml, baseline_variant=BaselineVariants.TopHat)

    construct_correct_baseline.assert_called_once_with(
        parallel_config=mocker.ANY, variant=BaselineVariants.TopHat, window_size=5000, window_unit="ppm"
    )
    construct_correct_baseline.return_value.evaluate_file.assert_called_once_with(
        construct_imzml_read_file.return_value, construct_imzml_write_file.return_value
    )
    construct_imzml_read_file.assert_called_once_with(mock_input_imzml)
    construct_imzml_write_file.assert_called_once_with(mock_output_imzml, imzml_mode=mock_imzml_mode)
    mock_output_imzml.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


if __name__ == "__main__":
    pytest.main()
