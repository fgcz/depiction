import pytest
from pytest_mock import MockerFixture

from depiction.parallel_ops import WriteSpectraParallel, ParallelConfig
from depiction.persistence import ImzmlReader, ImzmlWriter
from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
from depiction.tools.filter_peaks import (
    FilterPeaksConfig,
    FilterNHighestIntensityPartitionedConfig,
    filter_peaks,
    _filter_chunk,
)


def test_filter_peaks_when_n_highest_intensity_partitioned(mocker: MockerFixture) -> None:
    mock_write_parallel = mocker.MagicMock(name="mock_write_parallel", spec=WriteSpectraParallel)
    mock_input_file = mocker.MagicMock(name="mock_input_file", spec=[])
    mock_output_file = mocker.MagicMock(name="mock_output_file", spec=[])
    mock_from_config = mocker.patch(
        "depiction.tools.filter_peaks.WriteSpectraParallel.from_config", return_value=mock_write_parallel
    )
    config = FilterPeaksConfig(
        filters=[FilterNHighestIntensityPartitionedConfig(max_count=10, n_partitions=20)], n_jobs=30
    )
    filter_peaks(config=config, input_file=mock_input_file, output_file=mock_output_file)
    mock_from_config.assert_called_once_with(ParallelConfig(n_jobs=30))
    mock_write_parallel.map_chunked_to_file.assert_called_once_with(
        read_file=mock_input_file,
        write_file=mock_output_file,
        operation=_filter_chunk,
        bind_args={"peaks_filter": FilterNHighestIntensityPartitioned(max_count=10, n_partitions=20)},
    )


def test_filter_chunk(mocker: MockerFixture) -> None:
    mock_reader = mocker.MagicMock(name="mock_reader", spec=ImzmlReader)
    mock_reader.get_spectrum_with_coords.side_effect = [("m1", "i1", "c1"), ("m2", "i2", "c2")]
    mock_peaks_filter = mocker.MagicMock(name="mock_peaks_filter", spec=FilterNHighestIntensityPartitioned)
    mock_peaks_filter.filter_peaks.side_effect = lambda mz_arr, int_arr, _1, _2: (mz_arr, int_arr)
    mock_writer = mocker.MagicMock(name="mock_writer", spec=ImzmlWriter)
    _filter_chunk(mock_reader, [5, 6], mock_writer, mock_peaks_filter)
    assert mock_reader.get_spectrum_with_coords.mock_calls == [mocker.call(5), mocker.call(6)]
    assert mock_peaks_filter.filter_peaks.mock_calls == [
        mocker.call("m1", "i1", "m1", "i1"),
        mocker.call("m2", "i2", "m2", "i2"),
    ]
    assert mock_writer.add_spectrum.mock_calls == [mocker.call("m1", "i1", "c1"), mocker.call("m2", "i2", "c2")]


if __name__ == "__main__":
    pytest.main()
