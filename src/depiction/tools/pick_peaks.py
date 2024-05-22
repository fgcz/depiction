import argparse

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.spectrum.peak_filtering import FilterByIntensity
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker
from depiction.persistence import ImzmlWriteFile, ImzmlReadFile, ImzmlWriter, ImzmlReader, ImzmlModeEnum


class PickPeaks:
    def __init__(self, peak_picker, parallel_config: ParallelConfig) -> None:
        self._peak_picker = peak_picker
        self._parallel_config = parallel_config

    def evaluate_file(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile) -> None:
        parallel = WriteSpectraParallel.from_config(self._parallel_config)
        parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._operation,
            bind_args=dict(peak_picker=self._peak_picker),
        )

    @classmethod
    def _operation(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        writer: ImzmlWriter,
        peak_picker,
    ) -> None:
        for spectrum_index in spectra_ids:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(spectrum_index)
            peak_mz, peak_int = peak_picker.pick_peaks(mz_arr, int_arr)
            writer.add_spectrum(peak_mz, peak_int, coords)


def debug_diagnose_threshold_correspondence(
    peak_filtering: FilterByIntensity,
    peak_picker: BasicInterpolatedPeakPicker,
    input_imzml: ImzmlReadFile,
    n_points: int,
) -> None:
    unfiltered_peak_picker = BasicInterpolatedPeakPicker(
        min_prominence=peak_picker.min_prominence,
        min_distance=peak_picker.min_distance,
        min_distance_unit=peak_picker.min_distance_unit,
        peak_filtering=None,
    )

    # TODO remove/consolidate this debugging functionality
    with input_imzml.reader() as reader:
        for i_spectrum in range(0, input_imzml.n_spectra, input_imzml.n_spectra // n_points):
            spec_mz_arr, spec_int_arr = reader.get_spectrum(i_spectrum)
            _, peak_int_arr = unfiltered_peak_picker.pick_peaks(spec_mz_arr, spec_int_arr)
            peak_filtering.debug_diagnose_threshold_correspondence(
                spectrum_int_arr=spec_int_arr, peak_int_arr=peak_int_arr
            )


def pick_peaks(
    input_imzml_path: str,
    output_imzml_path: str,
    n_jobs: int,
    peak_picker: str,
    min_prominence: float,
    min_distance: float,
    min_distance_unit: str,
    min_peak_intensity: float,
    min_peak_intensity_unit: str,
) -> None:
    parallel_config = ParallelConfig(n_jobs=n_jobs, task_size=None)
    if peak_picker != "basic_interpolated":
        raise ValueError(f"Unknown peak picker: {peak_picker}")
    peak_filtering = FilterByIntensity(min_intensity=min_peak_intensity, normalization=min_peak_intensity_unit)
    peak_picker = BasicInterpolatedPeakPicker(
        min_prominence=min_prominence,
        min_distance=min_distance,
        min_distance_unit=min_distance_unit,
        peak_filtering=peak_filtering,
        # peak_filtering=FilterNHighestIntensityPartitioned(max_count=120*3, n_partitions=8),
        # peak_filtering=FilterByIntensity(min_intensity=min_peak_intensity, normalization="vec_norm"),
    )
    input_imzml = ImzmlReadFile(input_imzml_path)
    debug_diagnose_threshold_correspondence(
        peak_filtering=peak_filtering, peak_picker=peak_picker, input_imzml=input_imzml, n_points=10
    )

    pick = PickPeaks(
        peak_picker=peak_picker,
        parallel_config=parallel_config,
    )
    pick.evaluate_file(
        read_file=input_imzml,
        write_file=ImzmlWriteFile(output_imzml_path, imzml_mode=ImzmlModeEnum.PROCESSED),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, dest="input_imzml_path")
    parser.add_argument("--output", type=str, required=True, dest="output_imzml_path")
    parser.add_argument("--n_jobs", type=int, default=20, dest="n_jobs")
    subparsers = parser.add_subparsers(dest="peak_picker")
    parser_bi = subparsers.add_parser("basic_interpolated")
    parser_bi.add_argument("--min_prominence", type=float, default=0.01)
    parser_bi.add_argument("--min_distance", type=float, default=0.5)
    parser_bi.add_argument("--min_distance_unit", type=str, default="mz")
    parser_bi.add_argument("--min_peak_intensity", type=float, default=0.0005)
    parser_bi.add_argument("--min_peak_intensity_unit", type=str, default="tic", choices=["tic", "median", "vec_norm"])

    args = vars(parser.parse_args())
    pick_peaks(**args)


if __name__ == "__main__":
    main()
