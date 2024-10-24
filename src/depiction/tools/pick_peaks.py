from __future__ import annotations

from typing import Literal, Any, Self, Annotated

from depiction.parallel_ops import ParallelConfig, WriteSpectraParallel
from depiction.persistence import ImzmlModeEnum
from depiction.persistence import ImzmlWriteFile, ImzmlReadFile, ImzmlWriter, ImzmlReader
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction.spectrum.peak_picking import BasicInterpolatedPeakPicker, BasicPeakPicker
from depiction.spectrum.peak_picking.ms_peak_picker_wrapper import MSPeakPicker
from depiction.tools.filter_peaks.config import FilterPeaksConfig
from depiction.tools.filter_peaks.filter_peaks import get_peak_filter
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class PeakPickerBasicInterpolatedConfig(BaseModel):
    peak_picker_type: Literal["BasicInterpolated"] = "BasicInterpolated"
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: Literal["index", "mz"] | None = None

    @model_validator(mode="after")
    def validate_min_distance(self) -> Self:
        if self.min_distance is not None and self.min_distance_unit is None:
            raise ValueError("min_distance_unit must be provided if min_distance is set")
        if self.min_distance_unit is not None and self.min_distance is None:
            raise ValueError("min_distance must be provided if min_distance_unit is set")
        return self


class PeakPickerBasicUninterpolatedConfig(BaseModel):
    peak_picker_type: Literal["BasicUninterpolated"] = "BasicUninterpolated"
    min_prominence: float
    min_distance: int | float | None = None
    min_distance_unit: Literal["index", "mz"] | None = None
    # TODO make optional later
    smooth_sigma: Annotated[float, Field(gt=0)] = 0.0

    @model_validator(mode="after")
    def validate_min_distance(self) -> Self:
        if self.min_distance is not None and self.min_distance_unit is None:
            raise ValueError("min_distance_unit must be provided if min_distance is set")
        if self.min_distance_unit is not None and self.min_distance is None:
            raise ValueError("min_distance must be provided if min_distance_unit is set")
        return self


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
    peak_picker: (
        PeakPickerBasicInterpolatedConfig
        | PeakPickerBasicUninterpolatedConfig
        | PeakPickerMSPeakPickerConfig
        | PeakPickerFindMFPyConfig
    ) = Field(..., discriminator="peak_picker_type")
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
            if len(peak_mz) > 1:
                writer.add_spectrum(peak_mz, peak_int, coords)
            else:
                logger.warning(
                    f"Dropped spectrum {spectrum_index} as insufficient ({len(peak_mz)} <= 1) peaks were found"
                )


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
        case PeakPickerBasicUninterpolatedConfig() as peak_picker_config:
            return BasicPeakPicker(
                smooth_sigma=peak_picker_config.smooth_sigma,
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


def get_peak_picker_from_config(config: PickPeaksConfig) -> Any:
    peak_filtering = get_peak_filter(config.peak_filtering) if config.peak_filtering else None
    peak_picker = get_peak_picker(config, peak_filtering)
    return peak_picker


def pick_peaks(
    config: PickPeaksConfig,
    input_file: ImzmlReadFile,
    output_file: ImzmlWriteFile,
) -> None:
    peak_picker = get_peak_picker_from_config(config)
    parallel_config = ParallelConfig(n_jobs=config.n_jobs)

    if config.peak_picker is None or (
        not config.force_peak_picker and input_file.imzml_mode == ImzmlModeEnum.PROCESSED
    ):
        logger.info("Peak picking is deactivated")
        input_file.copy_to(output_file.imzml_file)
    else:
        logger.info(f"Using peak picker: {peak_picker=}")
        pick_peaks = PickPeaks(peak_picker=peak_picker, parallel_config=parallel_config)
        pick_peaks.evaluate_file(read_file=input_file, write_file=output_file)
