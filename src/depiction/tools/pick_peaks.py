from __future__ import annotations

from typing import Literal, Any

from loguru import logger
from pydantic import BaseModel, Field

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlModeEnum
from depiction.persistence import ImzmlWriteFile, ImzmlReadFile, ImzmlWriter, ImzmlReader
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker
from depiction.spectrum.peak_picking.ms_peak_picker_wrapper import MSPeakPicker
from depiction.tools.filter_peaks import FilterPeaksConfig, get_peak_filter


class PeakPickerBasicInterpolatedConfig(BaseModel):
    peak_picker_type: Literal["BasicInterpolated"] = "BasicInterpolated"
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: Literal["index", "mz"] | None = None

    # TODO ensure min_distance are both either present or missing
    # (ideally we would just have a better typing support here and provide as tuple,
    #  but postpone for later)


class PeakPickerMSPeakPickerConfig(BaseModel):
    peak_picker_type: Literal["MSPeakPicker"] = "MSPeakPicker"
    fit_type: Literal["quadratic"] = "quadratic"


class PeakPickerFindMFPyConfig(BaseModel):
    peak_picker_type: Literal["FindMFPy"] = "FindMFPy"
    resolution: float = 10000.0
    width: float = 2.0
    int_width: float = 2.0
    int_threshold: float = 10.0
    area: bool = True
    max_peaks: int = 0


class PickPeaksConfig(BaseModel, use_enum_values=True, validate_default=True):
    peak_picker: PeakPickerBasicInterpolatedConfig | PeakPickerMSPeakPickerConfig | PeakPickerFindMFPyConfig = Field(
        ..., discriminator="peak_picker_type"
    )
    n_jobs: int
    force_peak_picker: bool = False
    peak_filtering: FilterPeaksConfig | None = None


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
            if len(peak_mz) > 0:
                writer.add_spectrum(peak_mz, peak_int, coords)
            else:
                logger.warning(f"Dropped spectrum {spectrum_index} as no peaks were found")


# def debug_diagnose_threshold_correspondence(
#    peak_filtering: FilterByIntensity,
#    peak_picker: BasicInterpolatedPeakPicker,
#    input_imzml: ImzmlReadFile,
#    n_points: int,
# ) -> None:
#    unfiltered_peak_picker = BasicInterpolatedPeakPicker(
#        min_prominence=peak_picker.min_prominence,
#        min_distance=peak_picker.min_distance,
#        min_distance_unit=peak_picker.min_distance_unit,
#        peak_filtering=None,
#    )
#
#    # TODO remove/consolidate this debugging functionality
#    with input_imzml.reader() as reader:
#        for i_spectrum in range(0, input_imzml.n_spectra, input_imzml.n_spectra // n_points):
#            spec_mz_arr, spec_int_arr = reader.get_spectrum(i_spectrum)
#            _, peak_int_arr = unfiltered_peak_picker.pick_peaks(spec_mz_arr, spec_int_arr)
#            peak_filtering.debug_diagnose_threshold_correspondence(
#                spectrum_int_arr=spec_int_arr, peak_int_arr=peak_int_arr
#            )


def get_peak_picker(config: PickPeaksConfig, peak_filtering: PeakFilteringType | None) -> Any:
    match config.peak_picker:
        case PeakPickerBasicInterpolatedConfig() as peak_picker_config:
            return BasicInterpolatedPeakPicker(
                min_prominence=peak_picker_config.min_prominence,
                min_distance=peak_picker_config.min_distance,
                min_distance_unit=peak_picker_config.min_distance_unit,
                peak_filtering=peak_filtering,
            )
        case PeakPickerMSPeakPickerConfig() as peak_picker_config:
            return MSPeakPicker(fit_type=peak_picker_config.fit_type, peak_filtering=peak_filtering)
        case PeakPickerFindMFPyConfig() as peak_picker_config:
            # TODO refactor this later?
            # NOTE: importing this here since it has non-standard dependencies
            from depiction.spectrum.peak_picking.findmf_peak_picker import FindMFPeakPicker

            return FindMFPeakPicker(
                resolution=peak_picker_config.resolution,
                width=peak_picker_config.width,
                int_width=peak_picker_config.int_width,
                int_threshold=peak_picker_config.int_threshold,
                area=peak_picker_config.area,
                max_peaks=peak_picker_config.max_peaks,
                peak_filtering=peak_filtering,
            )
        case _:
            raise ValueError(f"Unsupported peak picker type: {config.peak_picker.peak_picker_type}")


def pick_peaks(
    config: PickPeaksConfig,
    input_file: ImzmlReadFile,
    output_file: ImzmlWriteFile,
) -> None:
    peak_filtering = get_peak_filter(config.peak_filtering)
    peak_picker = get_peak_picker(config, peak_filtering)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs)

    if config.peak_picker is None or (
        not config.force_peak_picker and input_file.imzml_mode == ImzmlModeEnum.PROCESSED
    ):
        logger.info("Peak picking is deactivated")
        input_file.copy_to(output_file.imzml_file)
    else:
        pick_peaks = PickPeaks(peak_picker=peak_picker, parallel_config=parallel_config)
        pick_peaks.evaluate_file(read_file=input_file, write_file=output_file)
